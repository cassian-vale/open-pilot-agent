import asyncio
import json
import re
import sys
from pathlib import Path
from typing import AsyncGenerator, TypedDict, Annotated, List, Union, Optional, Dict, Any

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import convert_to_openai_messages

dir_name = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(dir_name))

from base_agent import BaseAgent
from applications.query_rewriting.query_rewriting_prompt import (
    QUERY_REWRITE_SYSTEM_MESSAGE,
    QUERY_REWRITE_PROMPT
)
from llm_api.llm_client_chat_model import LLMClientChatModel
from utils.time_count import timer
from utils.stream_chunk import StreamChunk
from utils.schema_parse import SchemaParser


# ===== 输出结构定义 =====
class QueryRewriteOutput(BaseModel):
    success: bool = Field(description="查询改写是否成功")
    original_query: str = Field(description="原始查询")
    rewritten_queries: List[Dict[str, str]] = Field(description="改写后的查询列表，每个包含query和strategy")
    optimization_notes: str = Field(description="优化说明")
    validation_errors: List[str] = Field(default_factory=list, description="验证错误信息")
    confidence: float = Field(description="整体置信度", ge=0, le=1)
    statistics: Dict[str, Any] = Field(description="统计信息")


# ===== 改写查询项定义 =====
class RewrittenQueryItem(BaseModel):
    rewritten_query: str = Field(description="改写后的查询文本")
    rewritten_strategy: str = Field(description="使用的改写策略")


# ===== LLM输出结构定义 =====
class LLMRewriteOutput(BaseModel):
    """LLM输出的查询改写结果"""
    rewritten_queries: List[RewrittenQueryItem] = Field(
        description="改写后的查询列表，按优化效果从好到差排序，每个查询包含查询文本和使用的策略"
    )
    optimization_notes: str = Field(
        description="优化说明和主要采用的策略总结"
    )


# ===== 状态定义 =====
class QueryRewriteState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], "消息历史"]
    current_query: str
    conversation_history: List[Dict[str, Any]]
    max_rewrites: int
    preserve_system: bool
    domain_context: Optional[str]
    rewritten_queries: List[Dict[str, str]]  # 改为字典列表，包含query和strategy
    optimization_notes: str
    validation_errors: List[str]
    statistics: Dict[str, Any]
    final_output: Optional[dict]


# ===== 查询改写Agent主类 =====
class QueryRewriteAgent(BaseAgent):
    def __init__(
            self,
            name: str = "query-rewrite-agent",
            # openai client init config
            base_url: str = "https://api.deepseek.com/v1",
            api_key: Optional[str] = None,
            timeout: float = 60.0,
            max_retries: int = 3,
            # openai client run config
            model: str = "deepseek-chat",
            max_tokens: Optional[int] = None,
            temperature: float = 0.3,  # 稍高的温度以产生多样性
            top_p: float = 1.0,
            stream: bool = False,
            enable_thinking: bool = False,
            default_max_rewrites: int = 5
    ):
        # 调用父类初始化
        super().__init__(
            name=name,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            stream=stream,
            enable_thinking=enable_thinking,
        )

        self.default_max_rewrites = default_max_rewrites
        
        # 初始化SchemaParser
        self.schema_parser = SchemaParser(LLMRewriteOutput)
        
        # 构建工作流图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        graph = StateGraph(QueryRewriteState)

        def initialize_node(state: QueryRewriteState, config: RunnableConfig) -> QueryRewriteState:
            """初始化节点：准备查询改写任务"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 初始化查询改写任务"):
                self.logger.info(f"request_id: {request_id}, 开始查询改写, 当前查询: '{state['current_query']}'")
                
                # 设置默认值
                if not state.get("max_rewrites") or state["max_rewrites"] <= 0:
                    state["max_rewrites"] = self.default_max_rewrites
                
                if not state.get("domain_context"):
                    state["domain_context"] = "通用领域"
                
                if state.get("preserve_system") is None:
                    state["preserve_system"] = True
                
                # 处理对话历史
                processed_history = self._process_conversation_history(
                    state.get("conversation_history", []),
                    state["preserve_system"]
                )
                state["conversation_history"] = processed_history
                
                self.logger.debug(f"request_id: {request_id}, 最大改写数: {state['max_rewrites']}, 历史消息数: {len(processed_history)}")
                
                return state

        async def rewrite_node(state: QueryRewriteState, config: RunnableConfig) -> QueryRewriteState:
            """改写节点：执行查询改写"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 执行查询改写"):
                # 构建提示词
                prompt_text = self._build_rewrite_prompt(
                    state["current_query"],
                    state["conversation_history"],
                    state["domain_context"],
                    state["max_rewrites"]
                )

                self.logger.info(f"request_id: {request_id}, 调用LLM进行查询改写, 查询长度: {len(state['current_query'])}")

                system_message_content = QUERY_REWRITE_SYSTEM_MESSAGE
                messages = [SystemMessage(content=system_message_content), HumanMessage(content=prompt_text)]

                # 调用LLM
                llm_client = self.get_llm_client(run_config)
                chat_model = LLMClientChatModel(llm_client=llm_client)
                try:
                    response = await chat_model.ainvoke(messages, config=config)

                    chat_completion = response.chat_completion.to_dict()
                    choices = chat_completion.get("choices", [])
                    
                    # 初始化最终输出结构
                    final_output = {
                        "metadata": {
                            "usage": chat_completion.get("usage", {}),
                            # "messages": []
                        }
                    }

                    if len(choices) > 0:
                        content = choices[0].get("message", {}).get("content", "")
                        reasoning_content = choices[0].get("message", {}).get("reasoning_content", "")

                        self.logger.debug(f"request_id: {request_id}, LLM响应长度: {len(content)}")

                        # 使用SchemaParser解析响应
                        rewrite_result = self.schema_parser.parse_response_to_json(content)
                        
                        final_output["output"] = rewrite_result
                        
                        # 生成统计信息
                        statistics = self._generate_statistics(rewrite_result, state["current_query"])
                        
                        # 更新消息历史
                        new_messages = state["messages"] + messages + [AIMessage(content=content)]
                        # final_output["metadata"]["messages"] = convert_to_openai_messages(new_messages)
                        
                        final_output["content"] = content
                        final_output["reasoning_content"] = reasoning_content

                        self.logger.info(f"request_id: {request_id}, LLM Parse Output: {json.dumps(rewrite_result, ensure_ascii=False)}")
                        
                        return {
                            **state,
                            "messages": new_messages,
                            "rewritten_queries": rewrite_result["rewritten_queries"],
                            "optimization_notes": rewrite_result["optimization_notes"],
                            "statistics": statistics,
                            "final_output": final_output
                        }
                    else:
                        raise ValueError("LLM api输出错误, choices为空")
                except asyncio.CancelledError:
                    self.logger.warning(f"⛔ request_id: {request_id}, 任务被中断，已停止 LLM 请求")
                    raise

        def validate_node(state: QueryRewriteState, config: RunnableConfig) -> QueryRewriteState:
            """验证节点：验证改写的查询"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 验证改写的查询"):
                validation_errors = []
                rewritten_queries = state["rewritten_queries"]
                
                # 验证基本格式
                if not isinstance(rewritten_queries, list):
                    validation_errors.append("改写查询格式错误：应该是一个列表")
                
                if len(rewritten_queries) == 0:
                    validation_errors.append("未生成任何改写查询")
                
                # 验证每个改写查询项
                for i, query_item in enumerate(rewritten_queries):
                    if not isinstance(query_item, dict):
                        validation_errors.append(f"第{i+1}个改写查询项不是字典类型")
                        continue
                    
                    # 验证必要字段
                    if "rewritten_query" not in query_item:
                        validation_errors.append(f"第{i+1}个改写查询项缺少'rewritten_query'字段")
                    
                    if "rewritten_strategy" not in query_item:
                        validation_errors.append(f"第{i+1}个改写查询项缺少'rewritten_strategy'字段")
                    
                    # 验证查询文本
                    query_text = query_item.get("rewritten_query", "")
                    if not isinstance(query_text, str):
                        validation_errors.append(f"第{i+1}个改写查询文本不是字符串类型")
                    elif not query_text.strip():
                        validation_errors.append(f"第{i+1}个改写查询文本为空")
                    elif len(query_text) < 3:
                        validation_errors.append(f"第{i+1}个改写查询'{query_text}'过短")
                    elif len(query_text) > 500:
                        validation_errors.append(f"第{i+1}个改写查询过长({len(query_text)}字符)")
                    
                    # 验证策略描述
                    strategy = query_item.get("rewritten_strategy", "")
                    if not isinstance(strategy, str):
                        validation_errors.append(f"第{i+1}个策略描述不是字符串类型")
                    elif not strategy.strip():
                        validation_errors.append(f"第{i+1}个策略描述为空")
                
                if not validation_errors:
                    self.logger.info(f"request_id: {request_id}, 查询改写验证通过, 生成{len(rewritten_queries)}个改写版本")
                else:
                    error_count = len(validation_errors)
                    self.logger.warning(f"request_id: {request_id}, 发现 {error_count} 个验证问题")
                    for error in validation_errors[:3]:
                        self.logger.debug(f"查询改写验证问题: {error}")

                state["validation_errors"] = validation_errors
                return state

        def finalize_node(state: QueryRewriteState, config: RunnableConfig) -> QueryRewriteState:
            """最终处理节点：汇总结果"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 结果汇总"):
                success = len(state["validation_errors"]) == 0
                
                # 计算置信度
                base_confidence = max(0.0, 1.0 - len(state["validation_errors"]) * 0.15)
                
                # 根据改写数量和质量调整置信度
                rewrite_count = len(state["rewritten_queries"])
                expected_count = state["max_rewrites"]
                count_ratio = min(rewrite_count / expected_count, 1.0) if expected_count > 0 else 1.0
                
                # 多样性评估（基于查询相似度）
                diversity_score = self._calculate_diversity_score([item["rewritten_query"] for item in state["rewritten_queries"]])
                confidence = base_confidence * 0.6 + count_ratio * 0.2 + diversity_score * 0.2
                
                # 构建最终输出，保留metadata信息
                final_output = state.get("final_output", {})
                final_output.update({
                    # "success": success,
                    # "original_query": state["current_query"],
                    # "rewritten_queries": state["rewritten_queries"],
                    # "optimization_notes": state["optimization_notes"],
                    # "validation_errors": state["validation_errors"],
                    "confidence": confidence,
                    # "statistics": state["statistics"],
                    # "query_length": len(state["current_query"]),
                    # "rewrite_count": rewrite_count,
                    # "max_rewrites_set": state["max_rewrites"],
                    # "history_messages_count": len(state["conversation_history"])
                })

                status_msg = "成功" if success else f"有{len(state['validation_errors'])}个警告"
                self.logger.success(f"request_id: {request_id}, 查询改写完成, 状态: {status_msg}, 置信度: {confidence:.2f}, 生成{rewrite_count}个改写版本")

                return {
                    **state,
                    "final_output": final_output
                }

        # 添加节点
        graph.add_node("initialize", initialize_node)
        graph.add_node("rewrite", rewrite_node)
        graph.add_node("validate", validate_node)
        graph.add_node("finalize", finalize_node)

        # 设置工作流
        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "rewrite")
        graph.add_edge("rewrite", "validate")
        graph.add_edge("validate", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _convert_llm_output(self, llm_output: LLMRewriteOutput, original_query: str) -> Dict[str, Any]:
        """转换LLM输出格式"""
        try:
            # 转换RewrittenQueryItem对象为字典
            rewritten_queries = []
            for item in llm_output.rewritten_queries:
                rewritten_queries.append({
                    "rewritten_query": item.rewritten_query,
                    "rewritten_strategy": item.rewritten_strategy
                })
            
            return {
                "rewritten_queries": rewritten_queries,
                "optimization_notes": llm_output.optimization_notes
            }
        except Exception as e:
            self.logger.error(f"转换LLM输出失败: {e}")
            # 返回默认结果
            return {
                "rewritten_queries": [{
                    "rewritten_query": original_query,
                    "rewritten_strategy": "原始查询"
                }],
                "optimization_notes": "解析失败，使用原始查询"
            }

    def _process_conversation_history(self, history: List[Dict[str, Any]], preserve_system: bool) -> List[Dict[str, Any]]:
        """处理对话历史"""
        if not history:
            return []
        
        processed_history = []
        for msg in history:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # 根据preserve_system决定是否保留system消息
                if role == "system" and not preserve_system:
                    continue
                
                if content and isinstance(content, str):
                    processed_history.append({
                        "role": role,
                        "content": content[:1000]  # 限制长度
                    })
        
        return processed_history[-10:]  # 只保留最近10条消息

    def _build_rewrite_prompt(self, current_query: str, conversation_history: List[Dict[str, Any]], 
                            domain_context: str, max_rewrites: int) -> str:
        """构建查询改写提示词"""
        
        # 格式化对话历史
        history_text = "无"
        if conversation_history:
            history_lines = []
            for msg in conversation_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines)
        
        return QUERY_REWRITE_PROMPT.format(
            current_query=current_query,
            conversation_history=history_text,
            domain_context=domain_context,
            max_rewrites=max_rewrites
        )

    def _generate_statistics(self, rewrite_result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """生成统计信息"""
        queries = [item["rewritten_query"] for item in rewrite_result["rewritten_queries"]]
        strategies = [item["rewritten_strategy"] for item in rewrite_result["rewritten_queries"]]
        original_length = len(original_query)
        
        return {
            "total_rewrites": len(queries),
            "average_query_length": round(sum(len(q) for q in queries) / len(queries), 2) if queries else 0,
            "original_query_length": original_length,
            "length_change_ratio": round((sum(len(q) for q in queries) / len(queries) - original_length) / original_length, 4) if original_length > 0 else 0,
            "unique_strategies": len(set(strategies)),
            "diversity_score": self._calculate_diversity_score(queries)
        }

    def _calculate_diversity_score(self, queries: List[str]) -> float:
        """计算查询多样性得分"""
        if len(queries) <= 1:
            return 0.0
        
        # 简单的多样性评估：基于词汇重叠度
        total_similarity = 0
        count = 0
        
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                words_i = set(queries[i].lower().split())
                words_j = set(queries[j].lower().split())
                
                if words_i and words_j:
                    overlap = len(words_i & words_j) / len(words_i | words_j)
                    total_similarity += overlap
                    count += 1
        
        if count == 0:
            return 1.0
        
        average_similarity = total_similarity / count
        return max(0.0, 1.0 - average_similarity)

    async def run(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None, 
                 max_rewrites: Optional[int] = None, preserve_system: bool = True,
                 domain_context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        执行查询改写流程

        :param query: 当前查询
        :param conversation_history: 对话历史 (OpenAI messages格式)
        :param max_rewrites: 最大改写数量
        :param preserve_system: 是否保留system消息
        :param domain_context: 领域上下文
        :return: 结构化输出字典
        """
        if not query.strip():
            raise ValueError("查询不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔧 request_id: {request_id}, 开始处理查询改写请求, 查询: '{query}'")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "current_query": query,
            "conversation_history": conversation_history or [],
            "max_rewrites": max_rewrites or self.default_max_rewrites,
            "preserve_system": preserve_system,
            "domain_context": domain_context,
            "rewritten_queries": [],
            "optimization_notes": "",
            "validation_errors": [],
            "statistics": {},
            "final_output": None
        }

        with timer(self.logger, f"request_id: {request_id}, 完整查询改写流程"):
            # 传递运行时配置
            config = {"configurable": run_config} if run_config else {}
            final_state = await self.graph.ainvoke(inputs, config=config)
            output = final_state.get("final_output", {})
            self.logger.success(f"🎉 request_id: {request_id}, 查询改写完成")

        return output
    
    async def run_stream(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None,
                        max_rewrites: Optional[int] = None, preserve_system: bool = True,
                        domain_context: Optional[str] = None, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """
        流式执行查询改写流程

        :param query: 当前查询
        :param conversation_history: 对话历史 (OpenAI messages格式)
        :param max_rewrites: 最大改写数量
        :param preserve_system: 是否保留system消息
        :param domain_context: 领域上下文
        :return: 流式输出生成器
        """
        if not query.strip():
            raise ValueError("查询不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔧 request_id: {request_id}, 开始流式处理查询改写请求, 查询: '{query}'")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "current_query": query,
            "conversation_history": conversation_history or [],
            "max_rewrites": max_rewrites or self.default_max_rewrites,
            "preserve_system": preserve_system,
            "domain_context": domain_context,
            "rewritten_queries": [],
            "optimization_notes": "",
            "validation_errors": [],
            "statistics": {},
            "final_output": None
        }

        with timer(self.logger, f"request_id: {request_id}, 完整流式查询改写流程"):
            # 传递运行时配置
            config = {"configurable": run_config} if run_config else {}
            
            async for event in self.graph.astream_events(inputs, config=config):
                event_type = event.get("event", "")
                
                 # 处理LLM流式输出
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", None)
                    if chunk and hasattr(chunk, "chat_completion_chunk") and chunk.chat_completion_chunk:
                        chat_completion_chunk = chunk.chat_completion_chunk.to_dict()
                        choices = chat_completion_chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            reasoning_content = delta.get("reasoning_content", "")
                            
                            # 输出思考内容
                            if reasoning_content:
                                yield StreamChunk(
                                    type="thinking",
                                    content=reasoning_content
                                )
                            # 输出生成内容
                            elif content:
                                yield StreamChunk(
                                    type="content", 
                                    content=content
                                )
                
                # 处理节点开始事件
                elif event_type == "on_chain_start":
                    name = event.get("name", "")
                    if name == "initialize":
                        yield StreamChunk(
                            type="processing",
                            content="开始初始化查询改写任务..."
                        )
                    elif name == "rewrite":
                        yield StreamChunk(
                            type="processing",
                            content="正在分析查询和对话历史，生成优化版本..."
                        )
                    elif name == "validate":
                        yield StreamChunk(
                            type="processing", 
                            content="正在验证改写的查询质量..."
                        )
                    elif name == "finalize":
                        yield StreamChunk(
                            type="processing",
                            content="正在汇总最终结果..."
                        )
                
                # 处理图结束事件, 输出最终结果
                elif event_type == "on_chain_end" and event.get("name", "") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    final_output = output.get("final_output", {})
                        
                    yield StreamChunk(
                        type="final",
                        content="",
                        metadata=final_output
                    )

            self.logger.success(f"🎉 request_id: {request_id}, 流式查询改写完成")


# ===== 使用示例 =====
async def main():
    """使用示例"""
    
    # 测试查询和对话历史
    TEST_QUERY = "它怎么安装？"
    TEST_HISTORY = [
        {"role": "user", "content": "我想了解Docker容器技术"},
        {"role": "assistant", "content": "Docker是一种容器化平台，可以帮助您打包、分发和运行应用程序。"},
        {"role": "user", "content": "那Docker Compose呢？"},
        {"role": "assistant", "content": "Docker Compose是一个用于定义和运行多容器Docker应用程序的工具。"},
        {"role": "system", "content": "你是一个技术助手，专门回答Docker相关问题"}
    ]
    
    # 初始化Agent
    agent = QueryRewriteAgent(
        name="test-query-rewrite-agent",
        base_url="https://api.deepseek.com/v1",
        api_key="YOUR_API_KEY",  # 替换为实际API密钥
        temperature=0.3,
        default_max_rewrites=4
    )
    
    # 流式处理示例
    print("=== 流式查询改写 ===")
    async for chunk in agent.run_stream(
        TEST_QUERY, 
        conversation_history=TEST_HISTORY,
        domain_context="Docker容器技术",
        max_rewrites=3,
        preserve_system=False,
        request_id="test-rewrite-001"
    ):
        if chunk.type == "thinking":
            print(f"🤔 {chunk.content}")
        elif chunk.type == "content":
            print(f"{chunk.content}", end="", flush=True)
        elif chunk.type == "processing":
            print(f"🔄 {chunk.content}")
        elif chunk.type == "final":
            result = chunk.metadata
            status = "成功" if result["success"] else f"有{len(result['validation_errors'])}个警告"
            print(f"\n✅ 查询改写完成: {status}, 置信度: {result['confidence']:.2f}")
            print(f"📊 实际生成改写数: {result['rewrite_count']}/{result['max_rewrites_set']}")
            print(f"📋 改写的查询:")
            for i, query_item in enumerate(result["rewritten_queries"], 1):
                print(f"  {i}. [{query_item['rewritten_strategy']}] {query_item['rewritten_query']}")
            print(f"📝 优化说明: {result['optimization_notes']}")
    
    print("\n" + "="*50 + "\n")

    quit()
    
    # 同步处理示例
    print("=== 同步查询改写 ===")
    result = await agent.run(
        TEST_QUERY,
        conversation_history=TEST_HISTORY,
        domain_context="Docker容器技术",
        max_rewrites=4,
        preserve_system=True,
        request_id="test-rewrite-002"
    )
    
    # 打印结果
    print(f"📊 查询改写结果:")
    print(f"  状态: {'✅ 成功' if result['success'] else '⚠️ 有警告'}")
    print(f"  置信度: {result['confidence']:.2f}")
    print(f"  原始查询: '{result['original_query']}'")
    print(f"  改写数量: {result['rewrite_count']}/{result['max_rewrites_set']}")
    print(f"  历史消息数: {result['history_messages_count']}")
    
    if result['validation_errors']:
        print(f"  验证警告: {len(result['validation_errors'])} 个")
        for error in result['validation_errors'][:2]:
            print(f"    - {error}")
    
    print(f"\n📋 改写的查询 (按优化效果排序):")
    for i, query_item in enumerate(result['rewritten_queries'], 1):
        print(f"  {i}. [{query_item['rewritten_strategy']}] {query_item['rewritten_query']}")
    
    print(f"\n📝 优化说明:")
    print(result['optimization_notes'])
    
    print(f"\n📈 统计信息:")
    stats = result['statistics']
    print(f"  - 平均查询长度: {stats['average_query_length']} 字符")
    print(f"  - 长度变化率: {stats['length_change_ratio']:+.2%}")
    print(f"  - 独特策略数: {stats['unique_strategies']}")
    print(f"  - 多样性得分: {stats['diversity_score']:.2f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())