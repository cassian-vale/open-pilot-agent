# coding: utf-8
import asyncio
import json
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import AsyncGenerator, TypedDict, Annotated, List, Union, Optional, Tuple, Dict, Any, Iterator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import convert_to_openai_messages

dir_name = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(dir_name))

from base_agent import BaseAgent
from preprocess.chunk import TextChunker
from utils.time_count import timer
from utils.schema_parse import SchemaParser
from utils.stream_chunk import StreamChunk
from llm_api.llm_client_chat_model import LLMClientChatModel


# ===== 输出结构定义 =====
class QAOutput(BaseModel):
    answer: str = Field(
        description="基于相关句子生成的自然语言答案，简洁准确完整"
    )
    sentence_indices: List[Tuple[int, int]] = Field(
        description="与问题语义相关句子在原文中的索引范围列表，格式如 [[0, 91], [91, 173]]"
    )


# ===== 状态定义 =====
class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], "消息历史"]
    doc_text: str
    query: str
    structured_doc: str
    final_output: Optional[dict]


# ===== Agent 主类（继承基类）=====
class DocQAAgent(BaseAgent):
    def __init__(
            self,
            name: str,
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
            chunk_size: int = 512,
            overlap: int = 100,
            return_sentences: bool = True
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
            enable_thinking=enable_thinking,  # 传递思考模式配置
        )

        # 保存自定义配置
        self.init_config.update(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "return_sentences": return_sentences,
            }
        )

        self.chunker = TextChunker()

        # 初始化输出解析器
        self.output_parser = SchemaParser(QAOutput)

        # 构建工作流图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        graph = StateGraph(AgentState)

        def preprocess_node(state: AgentState, config: RunnableConfig) -> AgentState:
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 文档预处理"):
                # 从config中获取chunk_size，如果没有则使用初始化值
                
                chunk_size = run_config.get("chunk_size", self.init_config.get("chunk_size"))
                overlap_size = run_config.get("overlap", self.init_config.get("overlap"))
                return_sentences = run_config.get("return_sentences", self.init_config.get("return_sentences"))

                chunks = self.chunker.chunk(
                    state["doc_text"],
                    chunk_size=chunk_size,
                    overlap=overlap_size,
                    return_sentences=return_sentences
                )

                structured_doc = self.chunker.add_start_end(chunks)
                self.logger.info(f"request_id: {request_id}, 📄 文档预处理完成")
                return {**state, "structured_doc": structured_doc}


        async def llm_qa_node(state: AgentState, config: RunnableConfig) -> AgentState:
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, LLM问答"):
                # 从config中获取运行时参数
                
                # 构建提示词
                prompt_text = self._get_prompt(
                    text=state["structured_doc"],
                    query=state["query"]
                )

                self.logger.info(f"request_id: {request_id}, LLM Input: {json.dumps({"input_text": prompt_text}, ensure_ascii=False)}")

                messages = [HumanMessage(content=prompt_text)]

                # 调用LLM      
                llm_client = self.get_llm_client(run_config)
                chat_model = LLMClientChatModel(llm_client=llm_client)

                try:
                    response = await chat_model.ainvoke(messages, config=config)
                    # 结果解析  
                    chat_completion = response.chat_completion.to_dict()

                    self.logger.info(f"request_id: {request_id}, LLM Response: {json.dumps(chat_completion, ensure_ascii=False)}")

                    choices = chat_completion.get("choices", [])
                    
                    final_output = {
                        "metadata": {
                            "usage": chat_completion.get("usage", {}),
                            # "messages": []
                        }
                    }

                    if len(choices) > 0:
                        content = choices[0].get("message", {}).get("content", "")
                        messages += [AIMessage(content=content)]
                        # final_output["metadata"]["messages"] = convert_to_openai_messages(messages)
                        final_output["content"] = content
                        reasoning_content = choices[0].get("message", {}).get("reasoning_content", "")
                        final_output["reasoning_content"] = reasoning_content
                        output_json = self.output_parser.parse_response_to_json(content)
                        final_output["output"] = output_json     

                        self.logger.info(f"request_id: {request_id}, LLM Parse Output: {json.dumps(output_json, ensure_ascii=False)}")          

                        return {
                            **state,
                            "messages": messages,
                            "final_output": final_output
                        }
                    else:
                        raise ValueError("LLM api输出错误，choices为空")
                except asyncio.CancelledError:
                    self.logger.warning(f"⛔ request_id: {request_id}, 任务被中断，已停止 LLM 请求")
                    raise
                

        graph.add_node("preprocess", preprocess_node)
        graph.add_node("llm_qa", llm_qa_node)
        graph.set_entry_point("preprocess")
        graph.add_edge("preprocess", "llm_qa")
        graph.add_edge("llm_qa", END)

        return graph.compile()

    def _get_prompt(self, text: str, query: str) -> str:
        """生成 LLM 提示词"""
        return f"""{text}

以上是一篇文章的句子结构化结果（已经对应了各个句子在文章中的索引），我有一个问题是：{query}。

你是一个智能问答助理，请完成以下两个任务：
1. 检索出文章中所有与这个问题直接相关的句子，输出这些句子的索引范围[start, end];要求：输出的每一个句子都必须能够支持回答问题；
2. 基于这些句子，生成一个简洁、准确、完整的自然语言答案。

注意：最终输出的答案的语言需要严格遵循用户问题的语言，除非用户问题里明确提到使用某种语言回答。

{self.output_parser.schema_generation_prompt}
"""

    async def run(self, doc_text: str, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行文档问答流程

        :param doc_text: 输入文档文本
        :param query: 用户问题
        :return: 结构化输出字典
        """
        if not doc_text.strip():
            raise ValueError("文档文本不能为空")
        if not query.strip():
            raise ValueError("问题不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔍 request_id: {request_id}, 开始处理问答请求, query: {query}, doc_text_len: {len(doc_text)}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "doc_text": doc_text,
            "query": query,
            "structured_doc": "",
            "final_output": None
        }

        with timer(self.logger, f"request_id: {request_id}, 完整问答流程"):
            # 传递运行时配置
            config = {"configurable": run_config} if run_config else {}
            final_state = await self.graph.ainvoke(inputs, config=config)
            output = final_state.get("final_output", {})
            self.logger.success(f"🎉 request_id: {request_id}, 问答完成")

        return output


    async def run_stream(self, doc_text: str, query: str, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """
        流式执行文档问答流程

        :param doc_text: 输入文档文本
        :param query: 用户问题
        :return: 流式输出迭代器
        """
        if not doc_text.strip():
            raise ValueError("文档文本不能为空")
        if not query.strip():
            raise ValueError("问题不能为空")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔍 request_id: {request_id}, 开始处理问答流式请求, query: {query}, doc_text_len: {len(doc_text)}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "doc_text": doc_text,
            "query": query,
            "structured_doc": "",
            "final_output": None
        }

        with timer(self.logger, f"request_id: {request_id}, 完整流式问答流程"):
            # 传递运行时配置
            config = {"configurable": run_config} if run_config else {}
            async for event in self.graph.astream_events(inputs, config=config):
                event_type = event.get("event", "")
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", None)
                    if chunk and hasattr(chunk, "chat_completion_chunk") and chunk.chat_completion_chunk:
                        chat_completion_chunk = chunk.chat_completion_chunk.to_dict()
                        choices = chat_completion_chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            reasoning_content = delta.get("reasoning_content", "")
                            if content:
                                yield StreamChunk(
                                    type="content",
                                    content=content
                                )
                            elif reasoning_content:
                                yield StreamChunk(
                                    type="thinking",
                                    content=reasoning_content
                                )                                                         
                elif event_type == "on_chain_end" and event.get("name", "") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    yield StreamChunk(
                        type="final",
                        content="",
                        metadata=output.get("final_output", {})
                    )
            self.logger.success(f"🎉 request_id: {request_id}, 流式问答完成")
