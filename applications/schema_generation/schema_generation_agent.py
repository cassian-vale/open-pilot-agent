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

from utils.schema_parse import SchemaParser # 引入此行

dir_name = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(dir_name))

from base_agent import BaseAgent
from applications.schema_generation.schema_generation_prompt import (
    SCHEMA_GENERATION_SYSTEM_MESSAGE,
    SCHEMA_GENERATION_PROMPT
)
from llm_api.llm_client_chat_model import LLMClientChatModel
from utils.schema_validate import SchemaValidator
from utils.time_count import timer
from utils.stream_chunk import StreamChunk


# ===== 输出结构定义 =====
class SchemaGenerationOutput(BaseModel):
    success: bool = Field(description="Schema生成是否成功")
    generated_schema: Dict[str, Any] = Field(description="生成的Schema定义")
    validation_errors: List[str] = Field(default_factory=list, description="Schema验证错误信息")
    confidence: float = Field(description="整体置信度", ge=0, le=1)
    schema_description: str = Field(description="Schema的详细说明")


# ===== 状态定义 =====
class SchemaGenerationState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], "消息历史"]
    user_requirements: str
    domain_context: Optional[str]
    generated_schema: Dict[str, Any]
    validation_errors: List[str]
    schema_description: str
    final_output: Optional[dict] # 修改为Optional[dict]


# ===== Schema生成Agent主类 =====
class SchemaGenerationAgent(BaseAgent):
    def __init__(
            self,
            name: str = "schema-generation-agent",
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

        self.schema_parser = SchemaParser()

        # 构建工作流图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        graph = StateGraph(SchemaGenerationState)

        def initialize_node(state: SchemaGenerationState, config: RunnableConfig) -> SchemaGenerationState:
            """初始化节点：准备Schema生成任务"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 初始化Schema生成任务"):
                self.logger.info(f"request_id: {request_id}, 开始Schema生成, 用户需求长度: {len(state['user_requirements'])}")
                
                # 设置默认领域上下文
                if not state.get("domain_context"):
                    state["domain_context"] = "通用数据模型"
                
                return state

        async def generate_node(state: SchemaGenerationState, config: RunnableConfig) -> SchemaGenerationState:
            """生成节点：执行Schema生成"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 执行Schema生成"):
                # 构建提示词
                prompt_text = self._build_generation_prompt(
                    state["user_requirements"], 
                    state["domain_context"]
                )

                self.logger.info(f"request_id: {request_id}, 调用LLM进行Schema生成, 需求长度: {len(state['user_requirements'])}")

                system_message_content = SCHEMA_GENERATION_SYSTEM_MESSAGE
                messages = [SystemMessage(content=system_message_content), HumanMessage(content=prompt_text)]

                # 调用LLM
                llm_client = self.get_llm_client(run_config)
                chat_model = LLMClientChatModel(llm_client=llm_client)
                try:
                    response = await chat_model.ainvoke(messages, config=config)

                    chat_completion = response.chat_completion.to_dict()
                    choices = chat_completion.get("choices", [])
                    
                    # 初始化最终输出结构，包含metadata
                    output_metadata = {
                        "usage": chat_completion.get("usage", {}),
                        # "messages": []
                    }
                    
                    if len(choices) > 0:
                        content = choices[0].get("message", {}).get("content", "")
                        reasoning_content = choices[0].get("message", {}).get("reasoning_content", "")

                        self.logger.debug(f"request_id: {request_id}, LLM响应长度: {len(content)}")

                        # 提取JSON结果
                        generated_schema = self.schema_parser.parse_response_to_json(content)
                        
                        # 生成Schema描述
                        schema_description = self.validator.generate_schema_description(generated_schema)
                        
                        # 更新消息历史
                        new_messages = state["messages"] + messages + [AIMessage(content=content)]
                        # output_metadata["messages"] = convert_to_openai_messages(new_messages)
                        
                        # 构造 output 字典
                        output_data = {
                            "generated_schema": generated_schema,
                            "schema_description": schema_description
                        }

                        # 构建包含四个固定元素的 final_output
                        final_output_structure = {
                            "output": output_data,
                            "content": content,
                            "reasoning_content": reasoning_content,
                            "metadata": output_metadata,
                            "confidence": 0.0 # 暂时设为0，在finalize_node中更新
                        }
                        
                        return {
                            **state,
                            "messages": new_messages,
                            "generated_schema": generated_schema,
                            "schema_description": schema_description,
                            "final_output": final_output_structure # 更新final_output
                        }
                    else:
                        raise ValueError("LLM api输出错误, choices为空")
                except asyncio.CancelledError:
                    self.logger.warning(f"⛔ request_id: {request_id}, 任务被中断，已停止 LLM 请求")
                    raise

        def validate_node(state: SchemaGenerationState, config: RunnableConfig) -> SchemaGenerationState:
            """验证节点：验证生成的Schema"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 验证生成的Schema"):
                # 验证Schema格式
                validation_errors = self.validator.validate_schema(state["generated_schema"], strict=False)

                if not validation_errors:
                    self.logger.info(f"request_id: {request_id}, Schema验证通过")
                else:
                    error_count = len(validation_errors)
                    self.logger.warning(f"request_id: {request_id}, 发现 {error_count} 个Schema验证错误")
                    for error in validation_errors[:3]:  # 只记录前3个错误
                        self.logger.debug(f"Schema验证错误: {error}")

                state["validation_errors"] = validation_errors
                return state

        def finalize_node(state: SchemaGenerationState, config: RunnableConfig) -> SchemaGenerationState:
            """最终处理节点：汇总结果"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 结果汇总"):
                success = len(state["validation_errors"]) == 0
                
                # 计算置信度（基于验证错误数量）
                confidence = max(0.0, 1.0 - len(state["validation_errors"]) * 0.1)
                
                # 更新 final_output 字典
                final_output = state.get("final_output", {})
                
                # 更新 output 部分
                # if "output" in final_output:
                #     final_output["output"].update({
                #         "success": success,
                #         "validation_errors": state["validation_errors"],
                #         "requirements_length": len(state["user_requirements"]),
                #         "schema_field_count": len(state["generated_schema"])
                #     })
                
                # 更新 confidence
                final_output["confidence"] = confidence

                status_msg = "成功" if success else f"有{len(state['validation_errors'])}个警告"
                self.logger.success(f"request_id: {request_id}, Schema生成完成, 状态: {status_msg}, 置信度: {confidence:.2f}")

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

    def _build_generation_prompt(self, requirements: str, domain_context: str) -> str:
        """构建Schema生成提示词"""
        return SCHEMA_GENERATION_PROMPT.format(
            user_requirements=requirements,
            domain_context=domain_context
        )

    async def run(self, user_requirements: str, domain_context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        执行Schema生成流程

        :param user_requirements: 用户需求描述
        :param domain_context: 领域上下文信息
        :return: 结构化输出字典
        """
        if not user_requirements.strip():
            raise ValueError("用户需求不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔧 request_id: {request_id}, 开始处理Schema生成请求, requirements_length: {len(user_requirements)}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "user_requirements": user_requirements,
            "domain_context": domain_context,
            "generated_schema": {},
            "validation_errors": [],
            "schema_description": "",
            "final_output": None
        }

        with timer(self.logger, f"request_id: {request_id}, 完整Schema生成流程"):
            # 传递运行时配置
            config = {"configurable": run_config} if run_config else {}
            final_state = await self.graph.ainvoke(inputs, config=config)
            output = final_state.get("final_output", {})
            self.logger.success(f"🎉 request_id: {request_id}, Schema生成完成")

        return output
    
    async def run_stream(self, user_requirements: str, domain_context: Optional[str] = None, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """
        流式执行Schema生成流程

        :param user_requirements: 用户需求描述
        :param domain_context: 领域上下文信息
        :return: 流式输出生成器
        """
        if not user_requirements.strip():
            raise ValueError("用户需求不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔧 request_id: {request_id}, 开始流式处理Schema生成请求, requirements_length: {len(user_requirements)}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "user_requirements": user_requirements,
            "domain_context": domain_context,
            "generated_schema": {},
            "validation_errors": [],
            "schema_description": "",
            "final_output": None
        }

        with timer(self.logger, f"request_id: {request_id}, 完整流式Schema生成流程"):
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
                            content="开始初始化Schema生成任务..."
                        )
                    elif name == "generate":
                        yield StreamChunk(
                            type="processing",
                            content="正在分析需求并生成Schema..."
                        )
                    elif name == "validate":
                        yield StreamChunk(
                            type="processing", 
                            content="正在验证生成的Schema..."
                        )
                    elif name == "finalize":
                        yield StreamChunk(
                            type="processing",
                            content="正在汇总最终结果..."
                        )
                
                
                # 处理图结束事件, 输出最终结果
                elif event_type == "on_chain_end" and event.get("name", "") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    final_output_data = output.get("final_output", {}) # 确保获取的是整个final_output 
                    
                    yield StreamChunk(
                        type="final",
                        content="",
                        metadata=final_output_data
                    )

            self.logger.success(f"🎉 request_id: {request_id}, 流式Schema生成完成")


# ===== 使用示例 =====
async def main():
    """使用示例"""
    
    # 测试用户需求
    TEST_REQUIREMENTS = """
    我需要一个商品评论的数据Schema, 包含以下信息：
    - 评论ID（唯一标识）
    - 用户信息：用户ID、用户名
    - 商品信息：商品ID、商品名称、商品分类
    - 评论内容：评分（1-5分）、评论标题、详细内容、评论时间
    - 有用性统计：点赞数、点踩数
    - 标签：用户自定义的标签列表
    - 图片信息：图片URL列表
    - 是否匿名评论
    """
    
    # 初始化Agent
    agent = SchemaGenerationAgent(
        name="test-schema-agent",
        base_url="https://api.deepseek.com/v1",
        api_key="YOUR_API_KEY",  # 替换为实际API密钥
        temperature=0.1
    )
    
    # 流式处理示例
    print("=== 流式Schema生成 ===")
    async for chunk in agent.run_stream(
        TEST_REQUIREMENTS, 
        domain_context="电商评论系统",
        request_id="test-schema-001"
    ):
        if chunk.type == "thinking":
            print(f"🤔 {chunk.content}")
        elif chunk.type == "content":
            print(f"{chunk.content}", end="", flush=True)
        elif chunk.type == "processing": # 添加处理processing类型
            print(f"🔄 {chunk.content}")
        elif chunk.type == "final":
            result = json.loads(chunk.content)
            status = "成功" if result["success"] else f"有{len(result['validation_errors'])}个警告"
            print(f"\n✅ Schema生成完成: {status}, 置信度: {result['confidence']:.2f}")
            print(f"📊 生成字段数: {result['schema_field_count']}")
            print(f"📋 生成的Schema:")
            print(json.dumps(result['generated_schema'], ensure_ascii=False, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # 同步处理示例
    print("=== 同步Schema生成 ===")
    result = await agent.run(
        TEST_REQUIREMENTS,
        domain_context="电商评论系统", 
        request_id="test-schema-002"
    )
    
    # 打印结果
    print(f"📊 Schema生成结果:")
    # 从result中提取需要的字段
    output_data = result.get("output", {})
    success = output_data.get("success", False)
    confidence = result.get("confidence", 0.0) # confidence现在直接在final_output的顶层
    validation_errors = output_data.get("validation_errors", [])
    generated_schema = output_data.get("generated_schema", {})
    schema_description = output_data.get("schema_description", "")
    requirements_length = output_data.get("requirements_length", 0)
    schema_field_count = output_data.get("schema_field_count", 0)


    print(f"  状态: {'✅ 成功' if success else '⚠️ 有警告'}")
    print(f"  置信度: {confidence:.2f}")
    print(f"  需求长度: {requirements_length}")
    print(f"  生成字段数: {schema_field_count}")
    
    if validation_errors:
        print(f"  验证警告: {len(validation_errors)} 个")
        for error in validation_errors[:3]:
            print(f"    - {error}")
    
    print(f"\n📋 生成的Schema:")
    print(json.dumps(generated_schema, ensure_ascii=False, indent=2))
    
    print(f"\n📝 Schema说明:")
    print(schema_description)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())