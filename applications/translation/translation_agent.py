import json
import sys
import asyncio
from pathlib import Path
from typing import AsyncGenerator, TypedDict, Annotated, List, Union, Optional, Dict, Any

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import convert_to_openai_messages

dir_name = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(dir_name))

from applications.translation.translation_prompt import (
    TRANSLATION_SYSTEM_MESSAGE,
    TRANSLATION_PROMPT,
    STYLE_GUIDELINES,
    DIRECTION_DESCRIPTIONS
)
from base_agent import BaseAgent
from llm_api.llm_client_chat_model import LLMClientChatModel
from utils.time_count import timer
from utils.stream_chunk import StreamChunk  # 引入标准 StreamChunk


# ===== 输出结构定义 (对应 final_output["output"]) =====
class TranslationOutputContent(BaseModel):
    success: bool = Field(description="翻译是否成功")
    original_text: str = Field(description="原文")
    translated_text: str = Field(description="译文")
    translation_direction: str = Field(description="翻译方向")
    translation_style: str = Field(description="翻译风格")
    quality_score: float = Field(description="翻译质量评分", ge=0, le=10)
    character_count: Dict[str, int] = Field(description="字符统计")
    validation_errors: List[str] = Field(default_factory=list, description="验证错误信息")


# ===== 状态定义 =====
class TranslationState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], "消息历史"]
    original_text: str
    translation_direction: str
    translation_style: str
    translated_text: str
    quality_score: float
    character_count: Dict[str, int]
    validation_errors: List[str]
    # 新增标准输出字段
    final_output: Optional[dict]
    reasoning_content: str
    metadata: Dict[str, Any]


# ===== 翻译Agent主类 =====
class TranslationAgent(BaseAgent):
    def __init__(
            self,
            name: str = "translation-agent",
            # openai client init config
            base_url: str = "https://api.deepseek.com/v1",
            api_key: Optional[str] = None,
            timeout: float = 60.0,
            max_retries: int = 3,
            # openai client run config
            model: str = "deepseek-chat",
            max_tokens: Optional[int] = None,
            temperature: float = 0.3,
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

        # 验证支持的翻译风格
        self.supported_styles = list(STYLE_GUIDELINES.keys())
        self.supported_directions = list(DIRECTION_DESCRIPTIONS.keys())

        # 构建工作流图
        self.graph = self._build_graph()

    def _build_translation_prompt(self, text: str, direction: str, style: str) -> str:
        """构建翻译提示词"""
        style_guidelines = STYLE_GUIDELINES.get(style, STYLE_GUIDELINES["普通"])
        direction_desc = DIRECTION_DESCRIPTIONS.get(direction, direction)
        
        return TRANSLATION_PROMPT.format(
            text=text,
            direction=direction_desc,
            style=style,
            style_guidelines=style_guidelines
        )

    def _validate_inputs(self, text: str, direction: str, style: str) -> List[str]:
        """验证输入参数"""
        errors = []
        
        if not text.strip():
            errors.append("待翻译文本不能为空")
        
        if direction not in self.supported_directions:
            errors.append(f"不支持的翻译方向: {direction}, 支持的翻译方向: {', '.join(self.supported_directions)}")
        
        if style not in self.supported_styles:
            errors.append(f"不支持的翻译风格: {style}, 支持的风格: {', '.join(self.supported_styles)}")
        
        return errors

    def _calculate_quality_score(self, original_text: str, translated_text: str, direction: str) -> float:
        """计算翻译质量评分（简化版）"""
        score = 8.0  # 基础分
        
        if not translated_text.strip():
            return 0.0
        
        original_len = len(original_text)
        translated_len = len(translated_text)
        
        if direction == "中译英":
            if translated_len < original_len * 0.3:
                score -= 2.0
            elif translated_len > original_len * 3:
                score -= 1.0
        else:  # 英译中
            if translated_len > original_len * 2:
                score -= 2.0
            elif translated_len < original_len * 0.3:
                score -= 1.0
        
        problematic_phrases = ["翻译", "interpret", "sorry", "无法翻译"]
        if any(phrase in translated_text.lower() for phrase in problematic_phrases):
            score -= 1.0
        
        return max(0.0, min(10.0, score))

    def _get_character_count(self, original_text: str, translated_text: str) -> Dict[str, int]:
        """获取字符统计"""
        return {
            "original_chars": len(original_text),
            "translated_chars": len(translated_text),
            "original_words": len(original_text.split()),
            "translated_words": len(translated_text.split())
        }

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        graph = StateGraph(TranslationState)

        async def initialize_node(state: TranslationState, config: RunnableConfig) -> TranslationState:
            """初始化节点：验证输入参数"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 初始化翻译任务"):
                self.logger.info(f"request_id: {request_id}, 开始翻译任务, 文本长度: {len(state['original_text'])}, 方向: {state['translation_direction']}, 风格: {state['translation_style']}")
                
                validation_errors = self._validate_inputs(
                    state["original_text"],
                    state["translation_direction"],
                    state["translation_style"]
                )
                
                if validation_errors:
                    self.logger.warning(f"request_id: {request_id}, 输入验证失败: {validation_errors}")
                    return {
                        **state,
                        "validation_errors": validation_errors,
                        "success": False
                    }
                
                return state

        async def translate_node(state: TranslationState, config: RunnableConfig) -> TranslationState:
            """翻译节点：执行翻译"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 执行翻译"):
                if state.get("validation_errors"):
                    return state

                prompt_text = self._build_translation_prompt(
                    state["original_text"],
                    state["translation_direction"],
                    state["translation_style"]
                )

                self.logger.info(f"request_id: {request_id}, 调用LLM进行翻译")

                messages = [
                    SystemMessage(content=TRANSLATION_SYSTEM_MESSAGE), 
                    HumanMessage(content=prompt_text)
                ]

                llm_client = self.get_llm_client(run_config)
                chat_model = LLMClientChatModel(llm_client=llm_client)

                try:

                    response = await chat_model.ainvoke(messages, config=config)
                    
                    chat_completion = response.chat_completion.to_dict()
                    choices = chat_completion.get("choices", [])
                    
                    # 初始化 metadata
                    output_metadata = {
                        "usage": chat_completion.get("usage", {}),
                        # "messages": []
                    }

                    if len(choices) > 0:
                        content = choices[0].get("message", {}).get("content", "")
                        reasoning_content = choices[0].get("message", {}).get("reasoning_content", "")

                        translated_text = content.strip()
                        
                        new_messages = state["messages"] + messages + [AIMessage(content=content)]
                        # output_metadata["messages"] = convert_to_openai_messages(new_messages)
                        
                        self.logger.debug(f"request_id: {request_id}, 翻译完成, 译文长度: {len(translated_text)}")
                        
                        return {
                            **state,
                            "messages": new_messages,
                            "translated_text": translated_text,
                            "reasoning_content": reasoning_content,
                            "metadata": output_metadata
                        }
                    else:
                        raise ValueError("LLM api输出错误, choices为空")
                except asyncio.CancelledError:
                    self.logger.warning(f"⛔ request_id: {request_id}, 任务被中断，已停止 LLM 请求")
                    raise

        async def evaluate_node(state: TranslationState, config: RunnableConfig) -> TranslationState:
            """评估节点：评估翻译质量"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 评估翻译质量"):
                if state.get("validation_errors") or not state.get("translated_text"):
                    return state

                quality_score = self._calculate_quality_score(
                    state["original_text"],
                    state["translated_text"],
                    state["translation_direction"]
                )
                
                character_count = self._get_character_count(
                    state["original_text"],
                    state["translated_text"]
                )
                
                self.logger.debug(f"request_id: {request_id}, 质量评估完成, 评分: {quality_score:.2f}")
                
                return {
                    **state,
                    "quality_score": quality_score,
                    "character_count": character_count
                }

        def finalize_node(state: TranslationState, config: RunnableConfig) -> TranslationState:
            """最终处理节点：汇总结果"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 结果汇总"):
                success = not state.get("validation_errors") and bool(state.get("translated_text"))
                
                # 构建 output 字典（业务数据）
                output_data = {
                    # "success": success,
                    "original_text": state["original_text"],
                    "translated_text": state.get("translated_text", ""),
                    "translation_direction": state["translation_direction"],
                    "translation_style": state["translation_style"],
                    # "quality_score": state.get("quality_score", 0.0),
                    # "character_count": state.get("character_count", {}),
                    # "validation_errors": state.get("validation_errors", [])
                }
                
                # 计算置信度 (使用质量评分归一化)
                confidence = state.get("quality_score", 0.0) / 10.0 if success else 0.0

                # 构建包含四个固定元素的 final_output
                final_output_structure = {
                    "output": output_data,
                    "content": state.get("translated_text", ""),
                    "reasoning_content": state.get("reasoning_content", ""),
                    "metadata": state.get("metadata", {}),
                    "confidence": confidence
                }

                status_msg = "成功" if success else "失败"
                quality_msg = f", 质量评分: {state.get('quality_score', 0):.2f}" if success else ""
                self.logger.success(f"request_id: {request_id}, 翻译完成, 状态: {status_msg}{quality_msg}")

                return {
                    **state,
                    "final_output": final_output_structure
                }

        # 添加节点
        graph.add_node("initialize", initialize_node)
        graph.add_node("translate", translate_node)
        graph.add_node("evaluate", evaluate_node)
        graph.add_node("finalize", finalize_node)

        # 设置工作流
        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "translate")
        graph.add_edge("translate", "evaluate")
        graph.add_edge("evaluate", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    async def run(self, text: str, translation_direction: str, translation_style: str = "普通", **kwargs) -> Dict[str, Any]:
        """
        执行翻译流程

        :param text: 待翻译的文本
        :param translation_direction: 翻译方向
        :param translation_style: 翻译风格
        :return: 结构化输出字典 {output, reasoning_content, metadata, confidence}
        """
        request_id = kwargs.get("request_id")
        self.logger.info(f"🔧 request_id: {request_id}, 开始处理翻译请求, text_length: {len(text)}")

        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "original_text": text,
            "translation_direction": translation_direction,
            "translation_style": translation_style,
            "translated_text": "",
            "quality_score": 0.0,
            "character_count": {},
            "validation_errors": [],
            "final_output": {},
            "reasoning_content": "",
            "metadata": {}
        }

        with timer(self.logger, f"request_id: {request_id}, 完整翻译流程"):
            config = {"configurable": run_config} if run_config else {}
            final_state = await self.graph.ainvoke(inputs, config=config)
            output = final_state.get("final_output", {})
            self.logger.success(f"🎉 request_id: {request_id}, 翻译完成")

        return output
    
    async def run_stream(self, text: str, translation_direction: str, translation_style: str = "普通", **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """
        流式执行翻译流程

        :return: StreamChunk 流式输出生成器
        """
        request_id = kwargs.get("request_id")
        self.logger.info(f"🔧 request_id: {request_id}, 开始流式处理翻译请求")

        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "original_text": text,
            "translation_direction": translation_direction,
            "translation_style": translation_style,
            "translated_text": "",
            "quality_score": 0.0,
            "character_count": {},
            "validation_errors": [],
            "final_output": {},
            "reasoning_content": "",
            "metadata": {}
        }

        with timer(self.logger, f"request_id: {request_id}, 完整流式翻译流程"):
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
                            
                            if reasoning_content:
                                yield StreamChunk(
                                    type="thinking",
                                    content=reasoning_content
                                )
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
                            content="初始化：验证输入参数..."
                        )
                    elif name == "translate":
                        yield StreamChunk(
                            type="processing",
                            content=f"正在执行{translation_direction}翻译（{translation_style}风格）..."
                        )
                    elif name == "evaluate":
                        yield StreamChunk(
                            type="processing", 
                            content="正在评估翻译质量..."
                        )
                    elif name == "finalize":
                        yield StreamChunk(
                            type="processing",
                            content="汇总：生成最终结果..."
                        )
                
                # 处理图结束事件
                elif event_type == "on_chain_end":
                    name = event.get("name", "")
                    if name == "LangGraph":
                        output = event.get("data", {}).get("output", {})
                        final_output_data = output.get("final_output", {})

                        if final_output_data:
                            yield StreamChunk(
                                type="final",
                                content="",
                                metadata=final_output_data
                            )

            self.logger.success(f"🎉 request_id: {request_id}, 流式翻译完成")


# ===== 使用示例 =====
async def main():
    """使用示例"""
    
    # 初始化Agent
    agent = TranslationAgent(
        name="test-translation-agent",
        base_url="https://api.deepseek.com/v1",
        api_key="YOUR_API_KEY",  # 替换为实际API密钥
        temperature=0.3
    )
    
    test_text = "今天天气真好, 我们一起去公园散步吧！"
    direction = "中译英"
    style = "普通"
    
    print(f"\n=== 测试翻译: {test_text} ===")
    
    # 1. 非流式
    print("\n--- 非流式 ---")
    result = await agent.run(
        text=test_text,
        translation_direction=direction,
        translation_style=style,
        request_id="test-trans-001"
    )
    print(f"✅ 结果: {result['output']['translated_text']}")
    print(f"📊 评分: {result['output']['quality_score']}")
    
    # 2. 流式
    print("\n--- 流式 ---")
    async for chunk in agent.run_stream(
        text=test_text,
        translation_direction=direction,
        translation_style=style,
        request_id="test-trans-002"
    ):
        if chunk.type == "thinking":
            print(f"🤔 {chunk.content}")
        elif chunk.type == "translation":
            print(chunk.content, end="", flush=True)
        elif chunk.type == "final":
            final_data = chunk.metadata.get("output", {})
            print(f"\n✅ 流式完成: {final_data.get('translated_text')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())