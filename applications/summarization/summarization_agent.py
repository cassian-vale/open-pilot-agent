import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncGenerator, TypedDict, Annotated, List, Union, Optional, Dict, Any

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import convert_to_openai_messages # 引入此行

dir_name = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(dir_name))

from applications.summarization.summarization_prompt import (
    SUMMARIZATION_SYSTEM_MESSAGE,
    SUMMARIZATION_PROMPT,
    SUMMARY_TYPE_GUIDELINES
)
from base_agent import BaseAgent
from preprocess.long_text_preprocessor import LongTextPreprocessor
from llm_api.llm_client_chat_model import LLMClientChatModel
from utils.time_count import timer
from utils.stream_chunk import StreamChunk # 确保 StreamChunk 在这里被正确导入


# ===== 输出结构定义 (BaseModel 用于内部验证，最终输出结构会调整) =====
class SummarizationOutputContent(BaseModel):
    success: bool = Field(description="摘要是否成功")
    original_text: str = Field(description="原文")
    summarized_text: str = Field(description="摘要文本")
    summary_type: str = Field(description="摘要类型")
    target_words: Optional[int] = Field(default=None, description="目标字数，None表示不限制")
    actual_words: int = Field(description="实际字数")
    chunk_count: int = Field(description="分块数量")
    quality_score: float = Field(description="质量评分", ge=0, le=10)
    word_limit_mode: bool = Field(description="是否限制字数模式")


# ===== 状态定义 =====
class SummarizationState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], "消息历史"]
    original_text: str
    target_words: Optional[int]  # None表示不限制字数
    original_target_words: Optional[int]  # 原始目标字数，None表示不限制字数
    summary_type: str
    processed_chunks: List[Dict[str, Any]]
    summarized_text: str
    actual_words: int
    quality_score: float
    final_output: Optional[dict] # 修改为Optional[dict]
    word_limit_mode: bool  # 是否限制字数模式


# ===== 文本摘要Agent主类 =====
class TextSummarizationAgent(BaseAgent):
    def __init__(
            self,
            name: str = "text-summarization-agent",
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
            # chunk config
            max_chunk_length: int = 1000  # 摘要分块可以大一些
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
        
        # 验证支持的摘要类型
        self.supported_types = list(SUMMARY_TYPE_GUIDELINES.keys())

        # 构建工作流图
        self.graph = self._build_graph()

    def _build_summarization_prompt(self, text: str, target_words: Optional[int], summary_type: str, word_limit_mode: bool) -> str:
        """构建摘要提示词"""
        # 使用预处理器分块
        chunks = self.preprocessor.prepare_correction_chunks(
            text, 
            max_chunk_length=self.init_config["max_chunk_length"]
        )
        
        # 构建分块信息提示
        chunk_info = ""
        total_original_chars = len(text)
        
        if word_limit_mode and target_words is not None:
            # 限制字数模式：显示字数分配
            for i, chunk in enumerate(chunks):
                chunk_length = len(chunk["text"])
                # 确保分配的字数至少为1，避免除零错误或过小
                suggested_summary_length = max(1, int(chunk_length * target_words / total_original_chars))
                
                chunk_info += f"\n<chunk{i+1}: 原文{chunk_length}字 | 建议分配{suggested_summary_length}字>\n"
                chunk_info += f"{chunk['text']}\n"
                chunk_info += f"</chunk{i+1}>\n"
        else:
            # 不限制字数模式：只显示原文块
            for i, chunk in enumerate(chunks):
                chunk_info += f"{chunk['text']}"
        
        # 获取摘要类型指南
        type_guidelines = SUMMARY_TYPE_GUIDELINES.get(summary_type, SUMMARY_TYPE_GUIDELINES["要点摘要"])

        chunk_count=len(chunks) if word_limit_mode else 1 # 这里可以简化，始终是len(chunks)
        
        prompt = SUMMARIZATION_PROMPT.format(
            chunk_info=chunk_info,
            target_words=f"约{target_words}字（±20%范围内）" if word_limit_mode else "无字数限制",
            summary_type=summary_type,
            type_guidelines=type_guidelines,
            total_original_chars=total_original_chars,
            chunk_count=f"{chunk_count}块",
        )
        
        return prompt

    def _calculate_quality_score(self, original_text: str, summarized_text: str, target_words: Optional[int], word_limit_mode: bool) -> float:
        """计算摘要质量评分"""
        actual_words = len(summarized_text)
        original_words = len(original_text)
        
        # 基础分
        score = 7.0
        
        if word_limit_mode and target_words is not None:
            # 限制字数模式：检查字数符合度
            word_ratio = actual_words / target_words
            if 0.6 <= word_ratio <= 1.1: # 允许略微超出
                score += 3.0
            elif 1.1 < word_ratio <= 1.3: # 稍微超出
                score += 1.0
            elif word_ratio < 0.6: # 字数过少
                score -= 1.0
        else:
            # 不限制字数模式：检查压缩比合理性
            if original_words == 0: # 避免除零
                return 0.0
            compression_ratio = actual_words / original_words
            if 0.1 <= compression_ratio <= 0.5:  # 合理的压缩比范围
                score += 2.0
            elif 0.05 <= compression_ratio <= 0.7:
                score += 1.0
            else: # 压缩比不合理
                score -= 1.0
        
        return max(0.0, min(10.0, score)) # 确保分数在0-10之间

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        graph = StateGraph(SummarizationState)

        async def preprocess_node(state: SummarizationState, config: RunnableConfig) -> SummarizationState:
            """预处理节点：准备分块信息"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 文本预处理分块"):
                # 验证摘要类型
                if state["summary_type"] not in self.supported_types:
                    raise ValueError(f"不支持的摘要类型: {state['summary_type']}，支持的类型: {', '.join(self.supported_types)}")
                
                # 准备分块
                max_chunk_length = run_config.get("max_chunk_length", self.init_config["max_chunk_length"])
                chunks = self.preprocessor.prepare_correction_chunks(
                    state["original_text"],
                    max_chunk_length=max_chunk_length
                )

                mode_desc = "限制字数模式" if state["word_limit_mode"] else "自由长度模式"
                target_desc = f"目标字数{state['target_words']}" if state["word_limit_mode"] else "无字数限制"
                
                self.logger.info(f"request_id: {request_id}, 文本分块完成, 共{len(chunks)}个块, {mode_desc}, {target_desc}")
                
                return {
                    **state,
                    "processed_chunks": chunks
                }

        async def summarize_node(state: SummarizationState, config: RunnableConfig) -> SummarizationState:
            """摘要节点：一次性生成摘要"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 生成摘要"):
                # 构建提示词
                prompt_text = self._build_summarization_prompt(
                    state["original_text"],
                    state["target_words"],
                    state["summary_type"],
                    state["word_limit_mode"]
                )

                mode_desc = "限制字数" if state["word_limit_mode"] else "自由长度"
                self.logger.info(f"request_id: {request_id}, 调用LLM生成摘要, 模式: {mode_desc}")

                messages = [
                    SystemMessage(content=SUMMARIZATION_SYSTEM_MESSAGE), 
                    HumanMessage(content=prompt_text)
                ]

                # 调用LLM
                llm_client = self.get_llm_client(run_config)
                chat_model = LLMClientChatModel(llm_client=llm_client)
                try:
                    response = await chat_model.ainvoke(messages, config=config)

                    chat_completion = response.chat_completion.to_dict()

                    print(chat_completion)
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
                        summarized_text = content.strip()
                        
                        # 计算质量评分
                        quality_score = self._calculate_quality_score(
                            state["original_text"],
                            summarized_text,
                            state["original_target_words"],
                            state["word_limit_mode"]
                        )
                        
                        # 更新消息历史
                        new_messages = state["messages"] + messages + [AIMessage(content=content)]
                        # output_metadata["messages"] = convert_to_openai_messages(new_messages)
                        
                        # 构造 output 字典
                        output_data = {
                            "summarized_text": summarized_text,
                            "summary_type": state["summary_type"],
                            "target_words": state["original_target_words"] if state["word_limit_mode"] else None,
                            "actual_words": len(summarized_text),
                            "quality_score": quality_score,
                            "word_limit_mode": state["word_limit_mode"]
                        }

                        # 构建包含四个固定元素的 final_output
                        final_output_structure = {
                            "output": output_data,
                            "content": content,
                            "reasoning_content": reasoning_content,
                            "metadata": output_metadata,
                            "confidence": 0.0 # 暂时设为0，在finalize_node中更新
                        }
                        
                        self.logger.debug(f"request_id: {request_id}, 摘要生成完成, 实际字数: {len(summarized_text)}, 质量评分: {quality_score:.2f}")
                        
                        return {
                            **state,
                            "messages": new_messages,
                            "summarized_text": summarized_text,
                            "actual_words": len(summarized_text),
                            "quality_score": quality_score,
                            "final_output": final_output_structure # 更新final_output
                        }
                    else:
                        raise ValueError("LLM api输出错误, choices为空")
                except asyncio.CancelledError:
                    self.logger.warning(f"⛔ request_id: {request_id}, 任务被中断，已停止 LLM 请求")
                    raise

        def finalize_node(state: SummarizationState, config: RunnableConfig) -> SummarizationState:
            """最终处理节点：汇总结果"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 结果汇总"):
                success = bool(state.get("summarized_text"))
                
                # 计算置信度 (简化示例，可根据实际需求更复杂计算)
                # 例如，基于质量评分和是否有摘要内容
                confidence = state.get("quality_score", 0.0) / 10.0 if success else 0.0

                final_output = state.get("final_output", {})

                # 更新 confidence
                final_output["confidence"] = confidence

                mode_desc = "限制字数模式" if state["word_limit_mode"] else "自由长度模式"
                status_msg = "成功" if success else "失败"
                quality_msg = f"，质量评分: {state.get('quality_score', 0):.2f}" if success else ""
                self.logger.success(f"request_id: {request_id}, 摘要生成{status_msg} [{mode_desc}]{quality_msg}")

                return {
                    **state,
                    "final_output": final_output
                }

        # 添加节点
        graph.add_node("preprocess", preprocess_node)
        graph.add_node("summarize", summarize_node)
        graph.add_node("finalize", finalize_node)

        # 设置工作流
        graph.set_entry_point("preprocess")
        graph.add_edge("preprocess", "summarize")
        graph.add_edge("summarize", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    async def run(self, text: str, target_words: Optional[int] = None, summary_type: str = "要点摘要", ratio: float = 1.5, **kwargs) -> Dict[str, Any]:
        """
        执行文本摘要流程

        :param text: 输入文本
        :param target_words: 目标字数，None表示不限制字数
        :param summary_type: 摘要类型 ("要点摘要", "段落摘要", "新闻摘要", "技术摘要", "会议摘要")
        :param ratio: 字数调整比例
        :return: 结构化输出字典
        """
        if not text.strip():
            raise ValueError("文本不能为空")
        
        # 确定模式
        word_limit_mode = target_words is not None
        
        if word_limit_mode:
            if target_words <= 0:
                raise ValueError("目标字数必须大于0")
            if target_words > len(text):
                raise ValueError("目标字数不能超过原文长度")
            # 应用ratio调整
            adjusted_target_words = int(target_words / ratio)
        else:
            adjusted_target_words = None

        request_id = kwargs.get("request_id")

        mode_desc = "限制字数模式" if word_limit_mode else "自由长度模式"
        target_desc = f"target_words: {adjusted_target_words}" if word_limit_mode else "无字数限制"
        
        self.logger.info(f"📝 request_id: {request_id}, 开始处理摘要请求, text_length: {len(text)}, {target_desc}, type: {summary_type}, mode: {mode_desc}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "original_text": text,
            "target_words": adjusted_target_words,
            "original_target_words": target_words,  # 保存原始目标字数
            "summary_type": summary_type,
            "processed_chunks": [],
            "summarized_text": "",
            "actual_words": 0,
            "quality_score": 0.0,
            "final_output": None,
            "word_limit_mode": word_limit_mode
        }

        with timer(self.logger, f"request_id: {request_id}, 完整摘要流程"):
            # 传递运行时配置
            config = {"configurable": run_config} if run_config else {}
            final_state = await self.graph.ainvoke(inputs, config=config)
            output = final_state.get("final_output", {})
            self.logger.success(f"🎉 request_id: {request_id}, 摘要完成")

        return output
    
    async def run_stream(self, text: str, target_words: Optional[int] = None, summary_type: str = "要点摘要", ratio: float = 1.5, **kwargs) -> AsyncGenerator[StreamChunk, None]: # StreamChunk应是utils.stream_chunk.StreamChunk
        """
        流式执行文本摘要流程

        :param text: 输入文本
        :param target_words: 目标字数，None表示不限制字数
        :param summary_type: 摘要类型
        :param ratio: 字数调整比例
        :return: 流式输出生成器
        """
        if not text.strip():
            raise ValueError("文本不能为空")
        
        # 确定模式
        word_limit_mode = target_words is not None
        
        if word_limit_mode:
            if target_words <= 0:
                raise ValueError("目标字数必须大于0")
            if target_words > len(text):
                raise ValueError("目标字数不能超过原文长度")
            # 应用ratio调整
            adjusted_target_words = int(target_words / ratio)
        else:
            adjusted_target_words = None

        request_id = kwargs.get("request_id")

        mode_desc = "限制字数模式" if word_limit_mode else "自由长度模式"
        self.logger.info(f"📝 request_id: {request_id}, 开始流式处理摘要请求, text_length: {len(text)}, mode: {mode_desc}, type: {summary_type}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "original_text": text,
            "target_words": adjusted_target_words,
            "original_target_words": target_words,  # 保存原始目标字数
            "summary_type": summary_type,
            "processed_chunks": [],
            "summarized_text": "",
            "actual_words": 0,
            "quality_score": 0.0,
            "final_output": None,
            "word_limit_mode": word_limit_mode
        }

        with timer(self.logger, f"request_id: {request_id}, 完整流式摘要流程"):
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
                            # 输出摘要内容
                            elif content:
                                yield StreamChunk(
                                    type="content",
                                    content=content
                                )
                
                # 处理节点开始事件
                elif event_type == "on_chain_start":
                    name = event.get("name", "")
                    if name == "preprocess":
                        yield StreamChunk(
                            type="processing", # 使用 "processing" 类型
                            content="开始文本预处理和分块分析..."
                        )
                    elif name == "summarize":
                        if word_limit_mode:
                            yield StreamChunk(
                                type="processing", # 使用 "processing" 类型
                                content=f"正在生成{summary_type}，目标字数: {adjusted_target_words}..."
                            )
                        else:
                            yield StreamChunk(
                                type="processing", # 使用 "processing" 类型
                                content=f"正在生成{summary_type}，自由长度模式..."
                            )
                    elif name == "finalize":
                        yield StreamChunk(
                            type="processing", # 使用 "processing" 类型
                            content="正在汇总最终结果..."
                        )
                
                # 处理节点结束事件 (可以在这里添加一些中间状态的Thinking)
                elif event_type == "on_chain_end":
                    name = event.get("name", "")
                    # 处理图结束事件，输出最终结果
                    if name == "LangGraph":
                        output = event.get("data", {}).get("output", {})
                        final_output_data = output.get("final_output", {}) # 确保获取的是整个final_output               
                        yield StreamChunk(
                            type="final",
                            content="",
                            metadata=final_output_data
                        )

            self.logger.success(f"🎉 request_id: {request_id}, 流式摘要完成")


# ===== 使用示例 =====
async def main():
    """使用示例"""
    
    # 初始化Agent
    agent = TextSummarizationAgent(
        name="test-summarization-agent",
        base_url="https://api.deepseek.com/v1",
        api_key="YOUR_API_KEY",  # 替换为实际API密钥
        temperature=0.3,
        max_chunk_length=500
    )
    
    # 测试文本
    test_text = """
    新华社三亚11月7日电（记者梅常伟）我国第一艘电磁弹射型航空母舰福建舰入列授旗仪式5日在海南三亚某军港举行。中共中央总书记、国家主席、中央军委主席习近平出席入列授旗仪式并登舰视察。

　　十一月的三亚，海阔天高，碧波浩瀚。军港内，福建舰踏海而立、满旗高悬，山东舰伏波相伴，来自海军部队和航母建设单位的代表2000余人在码头整齐列队，气氛隆重热烈。

    下午4时30分许，入列授旗仪式开始，全场高唱中华人民共和国国歌，五星红旗冉冉升起。仪仗礼兵护卫着八一军旗，正步行进到主席台前。习近平将八一军旗授予福建舰舰长、政治委员。福建舰舰长、政治委员向习近平敬礼，从习近平手中接过八一军旗。习近平同他们合影留念。入列授旗仪式在中国人民解放军军歌声中结束。

　　习近平对我国航母建设发展一直很关注。仪式结束后，习近平登上福建舰，听取我国航母建设发展工作汇报，了解航母体系作战能力生成、电磁弹射系统建设运用等情况。

　　宽阔的飞行甲板上，4道阻拦索、3个弹射起飞位格外醒目，歼-35、歼-15T、空警-600等新型舰载机依次停放。习近平听取甲板功能布局介绍，不时驻足察看装备设施。习近平同舰载机飞行员亲切交流，详细询问飞机技战术性能和电磁弹射特点优势，观看舰载机弹射放飞流程演示。身着多种颜色马甲的航空保障人员看到习主席到来，纷纷围拢过来，向习主席问好，报告各自岗位和主要职责。习近平勉励大家不断提升专业技能和打仗本领，为福建舰战斗力建设贡献力量。

　　随后，习近平前往福建舰舰岛，登上塔台，了解飞行指挥和起降运行情况。习近平进入驾驶室，察看值勤战位，在航泊日志上郑重签名。习近平亲自决策福建舰采用电磁弹射技术。他来到弹射综合控制站，仔细观摩工作流程，按下弹射按钮，甲板上空载的动子如离弦之箭弹向舰艏。习近平十分关心舰上官兵生活，专门来到餐厅和士兵舱，察看饮食和住宿保障情况，同士兵们亲切交流，叮嘱各级搞好各方面保障，让广大官兵更好投身部队建设和备战打仗。

　　离别时，全舰官兵依依不舍，在飞行甲板和码头整齐列队，向习主席敬礼，齐声高呼“听党指挥、能打胜仗、作风优良”。

　　蔡奇、张国清出席福建舰入列授旗仪式。张升民主持仪式。

　　福建舰是我国第一艘电磁弹射型航空母舰，也是我国第三艘航空母舰，舷号为“18”，2022年6月下水命名。福建舰由我国完全自主设计建造，其电磁弹射技术处于世界先进水平。

　　中央和国家机关有关部门、军委机关有关部门、南部战区、海军、海南省以及航母建设单位的负责同志参加仪式。

今年9月的相关报道显示，歼-35、歼-15T、空警-600三型舰载机已完成在福建舰上的首次弹射起飞和着舰训练，标志着福建舰具备了电磁弹射和回收能力。

　　电磁弹射具有推力大、效率高、精准控制力道等优势，让舰载机实现“满弹满油”起飞、短距起飞、高效出动，进一步提升航母的综合作战效能。目前全球只有极少数国家能够熟练掌握这一技术。

　　尚未入列时，福建舰就已实现主要舰载机弹射起飞。张军社指出，这说明中国已经能够完全掌握和成熟运用电磁弹射这种复杂的飞机起飞系统，也说明中国海军官兵驾驭高科技装备的能力和水平在不断提高。

　　实现电磁弹射起飞是福建舰具备战斗力的关键环节。电磁弹射提高了舰载机的出动效率，让航母“出拳”更快；而日前完成弹射起飞的三型舰载机能够构成空中作战体系，更好地执行进攻和防御任务，让福建舰的“拳头”更硬。

　　分析认为，歼-35飞机能与歼-15T飞机实现高效协同出击，大大提升航母编队隐身突防与饱和打击双重能力，最大程度发挥航母舰载机的作战能力。凭借电磁弹射技术，起飞速度较慢的空警-600也能作为舰载机出征远海，擦亮航母的“千里眼”。

　　张军社指出，随着这三型舰载机弹射起飞和着舰训练成功，中国海军航母具备制空、制海、预警、电子对抗、反潜能力的核心舰载机体系，即“航母五件套”已经基本成型。“拳头”更硬、“出拳”更快，福建舰入列即具备战斗力，综合作战能力有了显著增强。

　　有报道称，从2024年5月启动首次海试，到2025年9月宣布完成关键弹射试验，福建舰在一年多时间内顺利开展多次海试，进度远超预期。

　　作为新质作战力量的代表，福建舰建设发展之迅速得益于辽宁舰、山东舰“蹚出来”的成功经验。张军社说，辽宁舰、山东舰的经验探索为后续航母的操作、训练、运用提供了极大的借鉴与帮助，“我国航母体系作战能力因此有了很大提升”。

　　福建舰入列服役，中国进入“三航母时代”，这对中国海军的发展意味着什么？

　　“中国海军的远海防御作战能力，特别是在远海独立作战和生存的能力将进一步增强。”张军社指出，可以预见，未来中国三航母编队作战，舰载机出动数量多、防空覆盖范围大、后勤保障和接续掩护能力强，都将进一步提高中国海军在远海的攻防和生存能力。

　　这位专家也表示，航母是国之重器、大国标配。福建舰的入列体现了中国综合国力的增强和科技水平的提高。“三航母时代”的到来将进一步扩大中国防御作战的纵深，增强中国军队“御敌于外”的能力。(完)
    """
    
    # 示例1：限制字数模式 (异步非流式)
    print("=== 限制字数模式 (非流式) ===")
    result1 = await agent.run(
        text=test_text,
        target_words=200,  # 指定目标字数
        summary_type="要点摘要",
        ratio=1.33,
        request_id="test-summary-001"
    )
    
    # 从裁剪后的结果中获取信息
    output_data = result1.get("output", {})
    confidence = result1.get("confidence", 0.0)

    print(f"模式: {'限制字数' if output_data['word_limit_mode'] else '自由长度'}")
    print(f"原文长度: {len(test_text)} 字")
    print(f"摘要长度: {output_data['actual_words']} 字 (目标: {output_data['target_words']} 字)")
    print(f"质量评分: {output_data['quality_score']:.2f}/10")
    print(f"分块数量: {output_data['chunk_count']} 块")
    print(f"置信度: {confidence:.2f}")
    print(f"\n摘要内容:\n{output_data['summarized_text']}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例2：不限制字数模式 (异步流式)
    print("=== 不限制字数模式 (流式) ===")
    full_stream_summary = ""
    async for chunk in agent.run_stream(
        text=test_text,
        target_words=None,  # None表示不限制字数
        summary_type="新闻摘要", 
        request_id="test-summary-002"
    ):
        if chunk.type == "thinking":
            print(f"🤔 {chunk.content}")
        elif chunk.type == "processing":
            print(f"🔄 {chunk.content}")
        elif chunk.type == "summary":
            print(f"{chunk.content}", end="", flush=True)
            full_stream_summary += chunk.content
        elif chunk.type == "final":
            result = json.loads(chunk.content)
            print("\n") # 确保换行
            status = "成功" if result["success"] else "失败"
            print(f"✅ 流式摘要完成: {status}")
            print(f"模式: {'限制字数' if result['word_limit_mode'] else '自由长度'}")
            print(f"摘要类型: {result['summary_type']}")
            print(f"原文长度: {len(test_text)} 字")
            print(f"实际摘要长度: {result['actual_words']} 字 (流式接收到: {len(full_stream_summary)} 字)")
            print(f"质量评分: {result['quality_score']:.2f}/10")
            print(f"分块数量: {result['chunk_count']} 块")
            print(f"置信度: {result['confidence']:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())