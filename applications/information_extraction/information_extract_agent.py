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
from llm_api.llm_client_chat_model import LLMClientChatModel
from applications.information_extraction.information_extract_prompt import ie_prompt, ie_system_message
from utils.schema_validate import SchemaValidator
from utils.time_count import timer
from utils.stream_chunk import StreamChunk


# ===== 输出结构定义 =====
class ExtractionOutput(BaseModel):
    success: bool = Field(description="抽取是否成功")
    extraction_result: Dict[str, Any] = Field(description="抽取的结构化结果")
    validation_errors: List[str] = Field(default_factory=list, description="验证错误信息")
    confidence: float = Field(description="整体置信度", ge=0, le=1)


# ===== 状态定义 =====
class ExtractionState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], "消息历史"]
    original_text: str
    extraction_schema: Dict[str, Any]
    extraction_result: Dict[str, Any]
    validation_errors: List[str]
    final_output: Optional[dict]


# ===== 信息抽取Agent主类 =====
class InformationExtractionAgent(BaseAgent):
    def __init__(
            self,
            name: str = "information-extraction-agent",
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

        # 初始化组件
        self.validator = SchemaValidator()

        # 构建工作流图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        graph = StateGraph(ExtractionState)

        def initialize_node(state: ExtractionState, config: RunnableConfig) -> ExtractionState:
            """初始化节点：准备抽取任务"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 初始化信息抽取任务"):
                schema_fields = list(state["extraction_schema"].keys())
                self.logger.info(f"request_id: {request_id}, 开始结构化信息抽取, Schema包含 {len(schema_fields)} 个字段: {schema_fields}")
                
                return state

        async def extract_node(state: ExtractionState, config: RunnableConfig) -> ExtractionState:
            """抽取节点：执行信息抽取"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 执行信息抽取"):
                # 构建提示词
                prompt_text = self._build_extraction_prompt(
                    state["original_text"], 
                    state["extraction_schema"]
                )

                self.logger.info(f"request_id: {request_id}, 调用LLM进行信息抽取, 文本长度: {len(state['original_text'])}")

                messages = [SystemMessage(content=ie_system_message), HumanMessage(content=prompt_text)]

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
                        
                        # 更新消息历史
                        new_messages = state["messages"] + messages + [AIMessage(content=content)]
                        # final_output["metadata"]["messages"] = convert_to_openai_messages(new_messages)

                        final_output["content"] = content
                        final_output["reasoning_content"] = reasoning_content

                        self.logger.debug(f"request_id: {request_id}, LLM响应长度: {len(content)}")

                        # 提取JSON结果
                        extraction_result = self._parse_extraction_response(content)

                        final_output["output"] = extraction_result   
                        
                        self.logger.info(f"request_id: {request_id}, LLM Parse Output: {json.dumps(extraction_result, ensure_ascii=False)}")
                        
                        return {
                            **state,
                            "messages": new_messages,
                            "extraction_result": extraction_result,
                            "final_output": final_output
                        }
                    else:
                        raise ValueError("LLM api输出错误, choices为空")
                    
                except asyncio.CancelledError:
                    self.logger.warning(f"⛔ request_id: {request_id}, 任务被中断，已停止 LLM 请求")
                    raise

        def validate_node(state: ExtractionState, config: RunnableConfig) -> ExtractionState:
            """验证节点：验证抽取结果"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 验证抽取结果"):
                validation_result = self.validator.validate_data(
                    state["extraction_result"], 
                    state["extraction_schema"]
                )

                if validation_result["valid"]:
                    self.logger.info(f"request_id: {request_id}, 验证通过, 使用清理后的数据")
                    # 使用验证后的数据（经过Pydantic清理和转换）
                    state["extraction_result"] = validation_result["data"]
                else:
                    error_count = len(validation_result["errors"])
                    self.logger.warning(f"request_id: {request_id}, 发现 {error_count} 个验证错误")
                    for error in validation_result["errors"][:3]:  # 只记录前3个错误
                        self.logger.debug(f"验证错误: {error}")
                    state["validation_errors"].extend(validation_result["errors"])

                return state

        def finalize_node(state: ExtractionState, config: RunnableConfig) -> ExtractionState:
            """最终处理节点：汇总结果"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 结果汇总"):
                success = len(state["validation_errors"]) == 0
                
                # 计算置信度（基于验证错误数量）
                confidence = max(0.0, 1.0 - len(state["validation_errors"]) * 0.1)
                
                # 构建最终输出，保留metadata信息
                final_output = state.get("final_output", {})
                final_output.update({
                    # "success": success,
                    # "extraction_result": state["extraction_result"],
                    # "validation_errors": state["validation_errors"],
                    "confidence": confidence,
                    # "original_text_length": len(state["original_text"]),
                    # "schema_fields": list(state["extraction_schema"].keys())
                })

                status_msg = "成功" if success else f"有{len(state['validation_errors'])}个错误"
                self.logger.success(f"request_id: {request_id}, 信息抽取完成, 状态: {status_msg}, 置信度: {confidence:.2f}")

                return {
                    **state,
                    "final_output": final_output
                }

        # 添加节点
        graph.add_node("initialize", initialize_node)
        graph.add_node("extract", extract_node)
        graph.add_node("validate", validate_node)
        graph.add_node("finalize", finalize_node)

        # 设置工作流
        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "extract")
        graph.add_edge("extract", "validate")
        graph.add_edge("validate", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _build_extraction_prompt(self, text: str, schema: Dict[str, Any]) -> str:
        """构建信息抽取提示词"""
        # 生成Schema文档描述
        schema_doc = self.validator.generate_schema_description(schema)
        
        # 生成智能示例
        example = self.validator.generate_example_data(schema)
        
        prompt = ie_prompt.format(
            text=text, 
            schema_doc=schema_doc, 
            example=json.dumps(example, ensure_ascii=False, indent=2)
        )
        
        return prompt

    def _parse_extraction_response(self, content: str) -> Dict[str, Any]:
        """解析LLM的抽取响应"""
        try:
            # 尝试直接解析
            extraction_result = json.loads(content)
            return extraction_result
        except json.JSONDecodeError:
            # 如果直接解析失败, 尝试提取JSON对象
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    extraction_result = json.loads(json_match.group())
                    self.logger.warning("从响应文本中提取JSON成功")
                    return extraction_result
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON提取后解析失败: {e}")
            else:
                self.logger.error("无法从响应中提取有效的JSON")
            
            # 返回空结果
            return {}

    async def run(self, text: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行信息抽取流程

        :param text: 输入文本
        :param schema: 抽取schema定义
        :return: 结构化输出字典
        """
        if not text.strip():
            raise ValueError("文本不能为空")
        
        if not schema:
            raise ValueError("Schema不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔍 request_id: {request_id}, 开始处理信息抽取请求, text_length: {len(text)}, schema_fields: {len(schema)}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "original_text": text,
            "extraction_schema": schema,
            "extraction_result": {},
            "validation_errors": [],
            "final_output": None
        }

        with timer(self.logger, f"request_id: {request_id}, 完整信息抽取流程"):
            # 传递运行时配置
            config = {"configurable": run_config} if run_config else {}
            final_state = await self.graph.ainvoke(inputs, config=config)
            output = final_state.get("final_output", {})
            self.logger.success(f"🎉 request_id: {request_id}, 信息抽取完成")

        return output
    
    async def run_stream(self, text: str, schema: Dict[str, Any], **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """
        流式执行信息抽取流程

        :param text: 输入文本
        :param schema: 抽取schema定义
        :return: 流式输出生成器
        """
        if not text.strip():
            raise ValueError("文本不能为空")
        
        if not schema:
            raise ValueError("Schema不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔍 request_id: {request_id}, 开始流式处理信息抽取请求, text_length: {len(text)}, schema_fields: {len(schema)}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "original_text": text,
            "extraction_schema": schema,
            "extraction_result": {},
            "validation_errors": [],
            "final_output": None
        }

        with timer(self.logger, f"request_id: {request_id}, 完整流式信息抽取流程"):
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
                            # 输出结果内容
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
                            content="开始初始化信息抽取任务..."
                        )
                    elif name == "extract":
                        yield StreamChunk(
                            type="processing",
                            content="正在调用LLM进行信息抽取..."
                        )
                    elif name == "validate":
                        yield StreamChunk(
                            type="processing", 
                            content="正在验证抽取结果..."
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

            self.logger.success(f"🎉 request_id: {request_id}, 流式信息抽取完成")


# ===== 使用示例 =====
async def main():
    """使用示例"""
    from data.information_extraction.schema_sample import PRODUCT_REVIEW_SCHEMA
    from data.information_extraction.text_sample import TEST_PRODUCT_REVIEW
    
    # 初始化Agent
    agent = InformationExtractionAgent(
        name="test-ie-agent",
        base_url="https://api.deepseek.com/v1",
        api_key="YOUR_API_KEY",  # 替换为实际API密钥
        temperature=0.1
    )
    
    # 流式处理示例
    async for chunk in agent.run_stream(
        TEST_PRODUCT_REVIEW, 
        PRODUCT_REVIEW_SCHEMA, 
        request_id="test-ie-001"
    ):
        if chunk.type == "thinking":
            print(f"🤔 {chunk.content}")
        if chunk.type == "content":
            print(f"{chunk.content}", end="", flush=True)
        elif chunk.type == "final":
            result = chunk.metadata
            status = "成功" if result["success"] else f"有{len(result['validation_errors'])}个错误"
            print(f"✅ 信息抽取完成: {status}, 置信度: {result['confidence']:.2f}")
    
    # 同步处理示例
    # result = await agent.run(
    #     TEST_PRODUCT_REVIEW, 
    #     PRODUCT_REVIEW_SCHEMA, 
    #     request_id="test-ie-002"
    # )
    
    # # 打印结果
    # print(f"\n📊 信息抽取结果:")
    # print(f"  状态: {'✅ 成功' if result['success'] else '❌ 有错误'}")
    # print(f"  置信度: {result['confidence']:.2f}")
    # print(f"  原文长度: {result['original_text_length']}")
    # print(f"  Schema字段数: {len(result['schema_fields'])}")
    # print(f"  Token使用情况: {result.get('metadata', {}).get('usage', {})}")
    
    # if result['validation_errors']:
    #     print(f"  验证错误: {len(result['validation_errors'])} 个")
    #     for error in result['validation_errors'][:3]:
    #         print(f"    - {error}")
    
    # print(f"\n📋 抽取结果:")
    # print(json.dumps(result['extraction_result'], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())