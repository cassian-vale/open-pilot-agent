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

from applications.keyword_generation.keyword_prompt import (
    KEYWORD_GENERATION_PROMPT,
    KEYWORD_GENERATION_SYSTEM_MESSAGE
)
from base_agent import BaseAgent
from llm_api.llm_client_chat_model import LLMClientChatModel
from utils.time_count import timer
from utils.stream_chunk import StreamChunk


# ===== 输出结构定义 =====
class KeywordGenerationOutput(BaseModel):
    success: bool = Field(description="关键词生成是否成功")
    keywords: List[str] = Field(description="生成的关键词列表，按重要性排序")
    validation_errors: List[str] = Field(default_factory=list, description="验证错误信息")
    confidence: float = Field(description="整体置信度", ge=0, le=1)
    keyword_analysis: str = Field(description="关键词分析说明")
    statistics: Dict[str, Any] = Field(description="统计信息")


# ===== 状态定义 =====
class KeywordGenerationState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], "消息历史"]
    content: str
    domain_context: Optional[str]
    max_keywords: int
    generated_keywords: List[str]
    validation_errors: List[str]
    keyword_analysis: str
    statistics: Dict[str, Any]
    final_output: Optional[dict]


# ===== 关键词生成Agent主类 =====
class KeywordGenerationAgent(BaseAgent):
    def __init__(
            self,
            name: str = "keyword-generation-agent",
            # openai client init config
            base_url: str = "https://api.deepseek.com/v1",
            api_key: Optional[str] = None,
            timeout: float = 60.0,
            max_retries: int = 3,
            # openai client run config
            model: str = "deepseek-chat",
            max_tokens: Optional[int] = None,
            temperature: float = 0.1,
            top_p: float = 1.0,
            stream: bool = False,
            enable_thinking: bool = False,
            default_max_keywords: int = 10
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

        self.default_max_keywords = default_max_keywords
        # 构建工作流图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        graph = StateGraph(KeywordGenerationState)

        def initialize_node(state: KeywordGenerationState, config: RunnableConfig) -> KeywordGenerationState:
            """初始化节点：准备关键词生成任务"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 初始化关键词生成任务"):
                self.logger.info(f"request_id: {request_id}, 开始关键词生成, 内容长度: {len(state['content'])}")
                
                # 设置默认领域上下文和最大关键词数
                if not state.get("domain_context"):
                    state["domain_context"] = "通用领域"
                
                if not state.get("max_keywords") or state["max_keywords"] <= 0:
                    state["max_keywords"] = self.default_max_keywords
                
                self.logger.debug(f"request_id: {request_id}, 最大关键词数: {state['max_keywords']}, 领域: {state['domain_context']}")
                
                return state

        async def generate_node(state: KeywordGenerationState, config: RunnableConfig) -> KeywordGenerationState:
            """生成节点：执行关键词生成"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 执行关键词生成"):
                # 构建提示词
                prompt_text = self._build_generation_prompt(
                    state["content"], 
                    state["domain_context"],
                    state["max_keywords"]
                )

                self.logger.info(f"request_id: {request_id}, 调用LLM进行关键词生成, 内容长度: {len(state['content'])}")

                messages = [SystemMessage(content=KEYWORD_GENERATION_SYSTEM_MESSAGE), HumanMessage(content=prompt_text)]

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

                        # 提取关键词结果
                        generated_keywords = self._parse_generation_response(content)
                        
                        # 生成关键词分析
                        keyword_analysis = self._generate_keyword_analysis(generated_keywords, state["content"])
                        
                        # 生成统计信息
                        statistics = self._generate_statistics(generated_keywords, state["content"])
                        
                        # 更新消息历史
                        new_messages = state["messages"] + messages + [AIMessage(content=content)]
                        # final_output["metadata"]["messages"] = convert_to_openai_messages(new_messages)
                        
                        final_output["content"] = content
                        final_output["reasoning_content"] = reasoning_content

                        self.logger.info(f"request_id: {request_id}, LLM Parse Output: {json.dumps(generated_keywords, ensure_ascii=False)}")
                        
                        return {
                            **state,
                            "messages": new_messages,
                            "generated_keywords": generated_keywords,
                            "keyword_analysis": keyword_analysis,
                            "statistics": statistics,
                            "final_output": final_output
                        }
                    else:
                        raise ValueError("LLM api输出错误, choices为空")
                
                except asyncio.CancelledError:
                    self.logger.warning(f"⛔ request_id: {request_id}, 任务被中断，已停止 LLM 请求")
                    raise

        def validate_node(state: KeywordGenerationState, config: RunnableConfig) -> KeywordGenerationState:
            """验证节点：验证生成的关键词"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 验证生成的关键词"):
                validation_errors = []
                keywords = state["generated_keywords"]
                
                # 验证关键词格式
                if not isinstance(keywords, list):
                    validation_errors.append("关键词格式错误：应该是一个列表")
                
                if len(keywords) == 0:
                    validation_errors.append("未生成任何关键词")
                
                if len(keywords) > state["max_keywords"] * 1.5:  # 允许一定的灵活性
                    validation_errors.append(f"生成的关键词数量({len(keywords)})超过限制({state['max_keywords']})")
                
                # 验证每个关键词
                for i, keyword in enumerate(keywords):
                    if not isinstance(keyword, str):
                        validation_errors.append(f"第{i+1}个关键词不是字符串类型")
                        continue
                    
                    keyword = keyword.strip()
                    if not keyword:
                        validation_errors.append(f"第{i+1}个关键词为空")
                        continue
                    
                    # 检查是否是短语（包含空格但允许专业复合词）
                    if ' ' in keyword and not self._is_professional_term(keyword):
                        validation_errors.append(f"关键词'{keyword}'可能是短语而非单个词语")
                
                if not validation_errors:
                    self.logger.info(f"request_id: {request_id}, 关键词验证通过, 生成{len(keywords)}个关键词")
                else:
                    error_count = len(validation_errors)
                    self.logger.warning(f"request_id: {request_id}, 发现 {error_count} 个关键词验证问题")
                    for error in validation_errors[:3]:
                        self.logger.debug(f"关键词验证问题: {error}")

                state["validation_errors"] = validation_errors
                return state

        def finalize_node(state: KeywordGenerationState, config: RunnableConfig) -> KeywordGenerationState:
            """最终处理节点：汇总结果"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 结果汇总"):
                success = len(state["validation_errors"]) == 0
                
                # 计算置信度（基于验证错误数量和关键词质量）
                base_confidence = max(0.0, 1.0 - len(state["validation_errors"]) * 0.2)
                
                # 根据关键词数量和质量调整置信度
                keyword_count = len(state["generated_keywords"])
                expected_count = state["max_keywords"]
                count_ratio = min(keyword_count / expected_count, 1.0) if expected_count > 0 else 1.0
                confidence = base_confidence * 0.7 + count_ratio * 0.3
                
                # 构建最终输出，保留metadata信息
                final_output = state.get("final_output", {})
                final_output.update({
                    # "success": success,
                    "output": state["generated_keywords"][:expected_count],
                    # "validation_errors": state["validation_errors"],
                    "confidence": confidence,
                    # "keyword_analysis": state["keyword_analysis"],
                    # "statistics": state["statistics"],
                    # "content_length": len(state["content"]),
                    # "keyword_count": keyword_count,
                    # "max_keywords_set": state["max_keywords"]
                })

                status_msg = "成功" if success else f"有{len(state['validation_errors'])}个警告"
                self.logger.success(f"request_id: {request_id}, 关键词生成完成, 状态: {status_msg}, 置信度: {confidence:.2f}, 生成{keyword_count}个关键词")

                return {
                    **state,
                    "final_output": final_output
                }

        # 添加节点
        graph.add_node("initialize", initialize_node)
        graph.add_node("generate", generate_node)
        graph.add_node("validate", validate_node)
        graph.add_node("finalize", finalize_node)

        # 设置工作流
        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _build_generation_prompt(self, content: str, domain_context: str, max_keywords: int) -> str:
        """构建关键词生成提示词"""
        return KEYWORD_GENERATION_PROMPT.format(
            content=content,
            domain_context=domain_context,
            max_keywords=max_keywords
        )

    def _parse_generation_response(self, content: str) -> List[str]:
        """解析LLM的生成响应"""
        try:
            # 尝试直接解析JSON数组
            keywords = json.loads(content)
            if isinstance(keywords, list):
                # 清理每个关键词
                cleaned_keywords = []
                for keyword in keywords:
                    if isinstance(keyword, str):
                        cleaned_keyword = keyword.strip()
                        if cleaned_keyword:
                            cleaned_keywords.append(cleaned_keyword)
                return cleaned_keywords
        except json.JSONDecodeError:
            # 如果直接解析失败, 尝试提取JSON数组
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                try:
                    keywords = json.loads(json_match.group())
                    if isinstance(keywords, list):
                        self.logger.warning("从响应文本中提取JSON数组成功")
                        return [str(k).strip() for k in keywords if str(k).strip()]
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON提取后解析失败: {e}")
            else:
                self.logger.error("无法从响应中提取有效的JSON数组")
        
        # 如果JSON解析失败，尝试按行分割
        lines = content.strip().split('\n')
        keywords = []
        for line in lines:
            line = line.strip()
            # 移除编号和特殊字符
            line = re.sub(r'^\d+[\.\)\-\*]\s*', '', line)
            line = re.sub(r'^[\-\*]\s*', '', line)
            if line and len(line) < 50:  # 避免过长的"关键词"
                keywords.append(line)
        
        self.logger.warning(f"使用备选解析方法，提取到 {len(keywords)} 个关键词")
        return keywords[:self.default_max_keywords]

    def _generate_keyword_analysis(self, keywords: List[str], original_content: str) -> str:
        """生成关键词分析说明"""
        if not keywords:
            return "未生成有效关键词"
        
        analysis_parts = []
        
        # 基本统计
        analysis_parts.append(f"共生成 {len(keywords)} 个关键词，按重要性排序。")
        
        # 关键词类型分析
        single_word_count = sum(1 for k in keywords if ' ' not in k)
        compound_word_count = len(keywords) - single_word_count
        
        if compound_word_count > 0:
            analysis_parts.append(f"包含 {compound_word_count} 个专业复合词。")
        
        # 重要性分布说明
        if len(keywords) >= 3:
            top_keywords = keywords[:3]
            analysis_parts.append(f"最重要的前3个关键词是：{', '.join(top_keywords)}")
        
        return " ".join(analysis_parts)

    def _generate_statistics(self, keywords: List[str], original_content: str) -> Dict[str, Any]:
        """生成统计信息"""
        content_words = len(original_content.split())
        keyword_chars = sum(len(k) for k in keywords)
        
        return {
            "total_keywords": len(keywords),
            "average_keyword_length": round(keyword_chars / len(keywords), 2) if keywords else 0,
            "content_word_count": content_words,
            "keyword_to_content_ratio": round(len(keywords) / content_words, 4) if content_words > 0 else 0,
            "single_word_keywords": sum(1 for k in keywords if ' ' not in k),
            "compound_word_keywords": sum(1 for k in keywords if ' ' in k),
        }

    def _is_professional_term(self, term: str) -> bool:
        """判断是否为专业复合词（简单的启发式判断）"""
        professional_indicators = [
            # 常见的专业复合词模式
            r'.*[A-Z].*',  # 包含大写字母（如JavaScript）
            r'.*[0-9].*',  # 包含数字（如C++）
            r'.*[+\-*/].*',  # 包含运算符号
            r'^[A-Z].*',  # 首字母大写（可能为专有名词）
        ]
        
        for pattern in professional_indicators:
            if re.match(pattern, term):
                return True
        return False

    async def run(self, content: str, domain_context: Optional[str] = None, max_keywords: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        执行关键词生成流程

        :param content: 需要提取关键词的内容
        :param domain_context: 领域上下文信息
        :param max_keywords: 最大关键词数量
        :return: 结构化输出字典
        """
        if not content.strip():
            raise ValueError("内容不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔧 request_id: {request_id}, 开始处理关键词生成请求, 内容长度: {len(content)}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "content": content,
            "domain_context": domain_context,
            "max_keywords": max_keywords or self.default_max_keywords,
            "generated_keywords": [],
            "validation_errors": [],
            "keyword_analysis": "",
            "statistics": {},
            "final_output": None
        }

        with timer(self.logger, f"request_id: {request_id}, 完整关键词生成流程"):
            # 传递运行时配置
            config = {"configurable": run_config} if run_config else {}
            final_state = await self.graph.ainvoke(inputs, config=config)
            output = final_state.get("final_output", {})
            self.logger.success(f"🎉 request_id: {request_id}, 关键词生成完成")

        return output
    
    async def run_stream(self, content: str, domain_context: Optional[str] = None, max_keywords: Optional[int] = None, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """
        流式执行关键词生成流程

        :param content: 需要提取关键词的内容
        :param domain_context: 领域上下文信息
        :param max_keywords: 最大关键词数量
        :return: 流式输出生成器
        """
        if not content.strip():
            raise ValueError("内容不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔧 request_id: {request_id}, 开始流式处理关键词生成请求, 内容长度: {len(content)}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "content": content,
            "domain_context": domain_context,
            "max_keywords": max_keywords or self.default_max_keywords,
            "generated_keywords": [],
            "validation_errors": [],
            "keyword_analysis": "",
            "statistics": {},
            "final_output": None
        }

        with timer(self.logger, f"request_id: {request_id}, 完整流式关键词生成流程"):
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
                            content="开始初始化关键词生成任务..."
                        )
                    elif name == "generate":
                        yield StreamChunk(
                            type="processing",
                            content="正在分析内容并提取关键词..."
                        )
                    elif name == "validate":
                        yield StreamChunk(
                            type="processing", 
                            content="正在验证生成的关键词..."
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

            self.logger.success(f"🎉 request_id: {request_id}, 流式关键词生成完成")


# ===== 使用示例 =====
async def main():
    """使用示例"""
    
    # 测试内容
    TEST_CONTENT = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    自从人工智能诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。
    人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考，也可能超过人的智能。
    """
    
    # 初始化Agent
    agent = KeywordGenerationAgent(
        name="test-keyword-agent",
        base_url="https://api.deepseek.com/v1",
        api_key="YOUR_API_KEY",  # 替换为实际API密钥
        temperature=0.1,
        default_max_keywords=8
    )
    
    # 流式处理示例
    print("=== 流式关键词生成 ===")
    async for chunk in agent.run_stream(
        TEST_CONTENT, 
        domain_context="人工智能技术",
        max_keywords=6,
        request_id="test-keyword-001"
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
            print(f"\n✅ 关键词生成完成: {status}, 置信度: {result['confidence']:.2f}")
            print(f"📊 实际生成关键词数: {result['keyword_count']}/{result['max_keywords_set']}")
            print(f"📋 关键词列表: {', '.join(result['keywords'])}")
            print(f"📝 分析: {result['keyword_analysis']}")
    
    print("\n" + "="*50 + "\n")

    quit()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())