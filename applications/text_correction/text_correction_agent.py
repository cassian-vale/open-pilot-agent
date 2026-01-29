# coding: utf-8
import json
import re
import sys
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from typing import AsyncGenerator, TypedDict, Annotated, List, Union, Optional, Tuple, Dict, Any, Iterator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import convert_to_openai_messages

# 假设相对路径配置保持不变
dir_name = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(dir_name))

from base_agent import BaseAgent
from preprocess.long_text_preprocessor import LongTextPreprocessor
from applications.text_correction.text_correction_prompt import (
    ctc_system_prompt,
    ctc_user_prompt
)
from utils.time_count import timer
from utils.schema_parse import SchemaParser
from utils.stream_chunk import StreamChunk
from llm_api.llm_client_chat_model import LLMClientChatModel


# ===== 输出结构定义 =====
class CorrectionItem(BaseModel):
    error_type: str = Field(description="错误类型：错别字/形近字错误/音近字错误/拼音串错误等")
    original_text: str = Field(description="原错误文本")
    corrected_text: str = Field(description="修正后的文本")
    reason: str = Field(description="错误原因说明")
    confidence: int = Field(description="置信度, 0-5分", ge=0, le=5)
    sentence_start_idx: int = Field(description="错误在原文中的起始位置")
    sentence_end_idx: int = Field(description="错误在原文中的结束位置")


class CorrectionOutput(BaseModel):
    corrections: List[CorrectionItem] = Field(description="纠错结果列表, 无错误则为空列表")


# ===== 最终响应结构 =====
class TextCorrectionResponse(BaseModel):
    output: Dict[str, Any] = Field(description="业务结果，包含corrections, corrected_text等")
    content: str = Field(default="", description="模型最终输出")
    reasoning_content: str = Field(default="", description="思考过程")
    metadata: Dict[str, Any] = Field(default=None, description="元数据")
    confidence: float = Field(default=1.0, description="整体置信度", ge=0, le=1)


# ===== 状态定义 =====
class CorrectionState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], "消息历史"]
    original_text: str
    processed_chunks: List[Dict[str, Any]]
    corrections: List[Dict[str, Any]]
    current_chunk_index: int
    final_output: Optional[TextCorrectionResponse]
    # [新增] 用于存储累加的 token usage
    usage: Dict[str, int]


# ===== 纠错Agent主类 =====
class TextCorrectionAgent(BaseAgent):
    def __init__(
            self,
            name: str = "text-correction-agent",
            # openai client init config
            base_url: str = "https://api.deepseek.com/v1",
            api_key: Optional[str] = None,
            timeout: float = 60.0,
            max_retries: int = 3,
            # openai client run config
            model: str = "deepseek-chat",
            max_tokens: Optional[int] = None,
            temperature: float = 0.0,
            top_p: float = 1.0,
            stream: bool = False,
            enable_thinking: bool = False,
            # chunk config
            max_chunk_length: int = 512  # 限制每个纠错块的最大长度
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

        # 保存自定义配置
        self.init_config.update({
            "max_chunk_length": max_chunk_length,
        })

        # 初始化组件
        self.preprocessor = LongTextPreprocessor()
        self.output_parser = SchemaParser(CorrectionOutput)

        # 构建工作流图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        graph = StateGraph(CorrectionState)

        def preprocess_node(state: CorrectionState, config: RunnableConfig) -> CorrectionState:
            """预处理节点：文本分块"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 文本预处理分块"):
                # 从config中获取max_chunk_length, 如果没有则使用初始化值
                max_chunk_length = run_config.get("max_chunk_length", self.init_config.get("max_chunk_length", 512))

                chunks = self.preprocessor.prepare_correction_chunks(
                    state["original_text"],
                    max_chunk_length=max_chunk_length
                )

                self.logger.info(f"request_id: {request_id}, 文本分块完成, 共{len(chunks)}个块, 最大块长度{max_chunk_length}")
                for i, chunk in enumerate(chunks):
                    self.logger.debug(f"块 {i+1}: 位置[{chunk['text_start']}-{chunk['text_end']}], 长度{len(chunk['text'])}")
                
                return {
                    **state,
                    "processed_chunks": chunks,
                    "corrections": [],
                    "current_chunk_index": 0,
                    # [新增] 初始化 usage 计数器
                    "usage": {
                        "prompt_tokens": 0, 
                        "completion_tokens": 0, 
                        "total_tokens": 0
                    }
                }

        async def correct_chunk_node(state: CorrectionState, config: RunnableConfig) -> CorrectionState:
            """纠错节点：处理单个文本块"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            current_index = state["current_chunk_index"]
            chunks = state["processed_chunks"]

            if current_index >= len(chunks):
                return state

            current_chunk = chunks[current_index]

            with timer(self.logger, f"request_id: {request_id}, 处理文本块 {current_index + 1}/{len(chunks)}"):
                # 检查块长度是否超过限制
                max_chunk_length = run_config.get("max_chunk_length", self.init_config.get("max_chunk_length", 512))
                if len(current_chunk["text"]) > max_chunk_length:
                    self.logger.warning(f"request_id: {request_id}, 块 {current_index + 1} 长度 {len(current_chunk['text'])} 超过限制 {max_chunk_length}")

                # 构建提示词
                prompt_text = self._get_correction_prompt(current_chunk)

                self.logger.info(f"request_id: {request_id}, 处理块 {current_index + 1}, 长度: {len(current_chunk['text'])}, 位置: {current_chunk['text_start']}-{current_chunk['text_end']}")

                messages = [HumanMessage(content=prompt_text)]

                # 调用LLM
                llm_client = self.get_llm_client(run_config)
                chat_model = LLMClientChatModel(llm_client=llm_client)
                try:
                    response = await chat_model.ainvoke(messages, config=config)
                    
                    # 结果解析
                    chat_completion = response.chat_completion.to_dict()
                    choices = chat_completion.get("choices", [])

                    # [新增] 获取并累加 usage
                    current_usage = chat_completion.get("usage", {}) or {}
                    prev_usage = state.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                    
                    new_usage = {
                        "prompt_tokens": prev_usage.get("prompt_tokens", 0) + current_usage.get("prompt_tokens", 0),
                        "completion_tokens": prev_usage.get("completion_tokens", 0) + current_usage.get("completion_tokens", 0),
                        "total_tokens": prev_usage.get("total_tokens", 0) + current_usage.get("total_tokens", 0)
                    }
                    
                    if len(choices) > 0:
                        content = choices[0].get("message", {}).get("content", "")
                        reasoning_content = choices[0].get("message", {}).get("reasoning_content", "")
                        
                        # 这里的 final_output 仅作中间存储，最终会由 finalize_node 覆盖
                        temp_final_output = state.get("final_output", {}) or {}
                        temp_final_output["reasoning_content"] = reasoning_content
                        
                        self.logger.debug(f"request_id: {request_id}, LLM Response Length: {len(content)}")
                        
                        # 解析纠错结果
                        try:
                            correction_data = self.output_parser.parse_response_to_json(content)
                            corrections = correction_data.get("corrections", [])
                            
                            # 验证并修正位置信息
                            valid_corrections = []
                            for correction in corrections:
                                start_idx = correction.get("sentence_start_idx", 0)
                                end_idx = correction.get("sentence_end_idx", 0)
                                original_text = correction.get("original_text", "")
                                corrected_text = correction.get("corrected_text", "")

                                if not corrected_text:
                                    self.logger.warning(f"request_id: {request_id}, LLM没有输出修正文本")
                                    continue
                                
                                # 如果位置信息看起来不合理, 尝试基于文本匹配修正
                                if start_idx < current_chunk["text_start"] or end_idx > current_chunk["text_end"]:
                                    self.logger.warning(f"request_id: {request_id}, 位置信息异常")
                                    continue
                                
                                chunk_text = current_chunk["text"]
                                corrected_sentence = chunk_text[start_idx-current_chunk["text_start"]: end_idx-current_chunk["text_start"]]
                                correction["corrected_sentence"] = corrected_sentence
                                if original_text in corrected_sentence:
                                    corrected_start = corrected_sentence.find(original_text)
                                    if corrected_start != -1:
                                        corrected_end = corrected_start + len(original_text)
                                        correction["corrected_sentence"] = corrected_sentence[:corrected_start] + corrected_text + corrected_sentence[corrected_end:] 
                                        self.logger.info(f"request_id: {request_id}, 句子修正为: {correction['corrected_sentence']}")

                                        # 为每个纠正添加块信息
                                        correction["chunk_index"] = current_index
                                        correction["chunk_text"] = current_chunk["text"]
                                        valid_corrections.append(correction)
                            
                            self.logger.info(f"request_id: {request_id}, 块 {current_index + 1} 发现 {len(valid_corrections)} 个错误")
                            
                        except Exception as e:
                            import traceback
                            self.logger.error(f"request_id: {request_id}, 解析纠错结果失败: {traceback.format_exc()}")
                            valid_corrections = []
                        
                        # 更新消息历史
                        new_messages = state["messages"] + [HumanMessage(content=prompt_text)] + [AIMessage(content=content)]
                        
                        temp_final_output["content"] = content
                        
                        return {
                            **state,
                            "messages": new_messages,
                            "corrections": state["corrections"] + valid_corrections,
                            "current_chunk_index": current_index + 1,
                            "final_output": temp_final_output,
                            "usage": new_usage # [新增] 更新累加后的 usage
                        }
                    else:
                        raise ValueError("LLM api输出错误, choices为空")
                except asyncio.CancelledError:
                    self.logger.warning(f"⛔ request_id: {request_id}, 任务被中断，已停止 LLM 请求")
                    raise

        def finalize_node(state: CorrectionState, config: RunnableConfig) -> CorrectionState:
            """最终处理节点：汇总结果"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")
            final_output = state.get("final_output", {}) or {}

            with timer(self.logger, f"request_id: {request_id}, 结果汇总"):
                # 应用所有纠正到原文本
                corrected_text = self._apply_corrections(
                    state["original_text"], 
                    state["corrections"]
                )
                
                # 计算整体置信度
                overall_confidence = 1.0
                if state["corrections"]:
                    avg_confidence = sum(c.get("confidence", 0) for c in state["corrections"]) / len(state["corrections"])
                    overall_confidence = avg_confidence / 5.0  # 归一化到0-1
                
                # 构建业务输出
                business_output = {
                    "original_text": state["original_text"],
                    "total_errors": len(state["corrections"]),
                    "corrections": state["corrections"],
                    "corrected_text": corrected_text
                }

                # [修改] 组装最终 metadata，包含累加的 token usage
                metadata = {
                    "usage": state.get("usage", {}),
                    "chunk_count": len(state["processed_chunks"])
                }

                final_output["output"] = business_output
                final_output["confidence"] = overall_confidence
                final_output["metadata"] = metadata

                self.logger.success(f"request_id: {request_id}, 纠错完成, 共处理 {len(state['processed_chunks'])} 个块, 发现 {len(state['corrections'])} 个错误")
                self.logger.info(f"request_id: {request_id}, Total Token Usage: {state.get('usage')}")

                return {
                    **state,
                    "final_output": final_output
                }

        def should_continue(state: CorrectionState) -> str:
            """判断是否继续处理下一个片段"""
            if state["current_chunk_index"] >= len(state["processed_chunks"]):
                return "end"
            return "continue"

        # 添加节点
        graph.add_node("preprocess", preprocess_node)
        graph.add_node("correct_chunk", correct_chunk_node)
        graph.add_node("finalize", finalize_node)

        # 设置工作流
        graph.set_entry_point("preprocess")
        graph.add_edge("preprocess", "correct_chunk")
        
        # 条件边：循环处理所有块
        graph.add_conditional_edges(
            "correct_chunk",
            should_continue,
            {
                "continue": "correct_chunk",
                "end": "finalize"
            }
        )
        
        graph.add_edge("finalize", END)

        return graph.compile()

    def _get_correction_prompt(self, chunk: Dict[str, Any]) -> str:
        """生成纠错提示词"""
        return f"{ctc_system_prompt}\n\n{self.output_parser.schema_generation_prompt}\n\n{ctc_user_prompt.format(text=chunk)}"

    def _apply_corrections(self, original_text: str, corrections: List[Dict]) -> str:
        """应用所有纠正到原文本"""
        if not corrections:
            return original_text

        # 按句子起始位置正序排序
        corrections_sorted = sorted(corrections, key=lambda x: x["sentence_start_idx"])
        
        result = []
        current_pos = 0
        
        for correction in corrections_sorted:
            start = correction["sentence_start_idx"]
            end = correction["sentence_end_idx"]
            corrected = correction["corrected_sentence"]
            
            # 添加当前修正点之前的文本
            if current_pos < start:
                result.append(original_text[current_pos:start])
            
            # 添加修正后的文本
            result.append(corrected)
            current_pos = end
        
        # 添加剩余文本
        if current_pos < len(original_text):
            result.append(original_text[current_pos:])
        
        return "".join(result)

    async def run(self, text: str, **kwargs) -> TextCorrectionResponse:
        """
        执行文本纠错流程
        """
        if not text.strip():
            raise ValueError("文本不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔍 request_id: {request_id}, 开始处理纠错请求, text_length: {len(text)}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "original_text": text,
            "processed_chunks": [],
            "corrections": [],
            "current_chunk_index": 0,
            "final_output": None,
            "usage": {} # 初始化为空，preprocess 节点会填充初始值
        }

        with timer(self.logger, f"request_id: {request_id}, 完整纠错流程"):
            config = {"configurable": run_config} if run_config else {}
            final_state = await self.graph.ainvoke(inputs, config=config)
            output = final_state.get("final_output")
            self.logger.success(f"🎉 request_id: {request_id}, 纠错完成")

        return output
    
    async def run_stream(self, text: str, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """
        流式执行文本纠错流程
        """
        if not text.strip():
            raise ValueError("文本不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔍 request_id: {request_id}, 开始流式处理纠错请求, text_length: {len(text)}")

        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "original_text": text,
            "processed_chunks": [],
            "corrections": [],
            "current_chunk_index": 0,
            "final_output": None,
            "usage": {}
        }

        with timer(self.logger, f"request_id: {request_id}, 完整流式纠错流程"):
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
                
                # 处理节点事件
                elif event_type == "on_chain_start":
                    name = event.get("name", "")
                    if name == "preprocess":
                        yield StreamChunk(
                            type="processing",
                            content="开始文本预处理和分块..."
                        )
                    elif name == "correct_chunk":
                        tags = event.get("tags", [])
                        if tags and "graph:step:" in tags[0]:
                            current_index = event.get("data", {}).get("input", {}).get("current_chunk_index", 0)
                            chunks = event.get("data", {}).get("input", {}).get("processed_chunks", [])
                            if current_index < len(chunks):
                                yield StreamChunk(
                                    type="processing",
                                    content=f"正在处理第 {current_index + 1}/{len(chunks)} 个文本块..."
                                )
                    elif name == "finalize":
                        yield StreamChunk(
                            type="processing", 
                            content="正在汇总最终纠错结果..."
                        )
                
                elif event_type == "on_chain_end":
                    name = event.get("name", "")
                    if name == "preprocess":
                        output = event.get("data", {}).get("output", {})
                        chunk_count = len(output.get("processed_chunks", []))
                        yield StreamChunk(
                            type="processing",
                            content=f"文本预处理完成, 共分成 {chunk_count} 个块"
                        )
                    elif name == "correct_chunk":
                        tags = event.get("tags", [])
                        if tags and "graph:step:" in tags[0]:
                            output = event.get("data", {}).get("output", {})
                            current_index = output.get("current_chunk_index", 0)
                            corrections = output.get("corrections", [])
                            
                            current_corrections = [
                                c for c in corrections 
                                if c.get("chunk_index", -1) == current_index - 1
                            ]
                            
                            if current_corrections:
                                yield StreamChunk(
                                    type="processing",
                                    content=f"第 {current_index} 个文本块处理完成, 发现 {len(current_corrections)} 个错误"
                                )     
                
                    elif name == "LangGraph":
                        output = event.get("data", {}).get("output", {})
                        final_output = output.get("final_output")
                        
                        yield StreamChunk(
                            type="final",
                            content="",
                            metadata=final_output
                        )

            self.logger.success(f"🎉 request_id: {request_id}, 流式纠错完成")


# ===== 使用示例 =====
async def main():
    """使用示例"""
    agent = TextCorrectionAgent(
        name="test-correction-agent",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-xxxx", # 替换你的 key
        max_chunk_length=512
    )
    
    test_text = """陕嘻袁家村太和居瑞斯丽酒店有限公司。"""

    # 同步调用示例
    result = await agent.run(test_text, request_id="test-001")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())