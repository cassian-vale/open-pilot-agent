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

from applications.text_classification.text_classification_prompt import (
    TEXT_CLASSIFICATION_SYSTEM_MESSAGE,
    TEXT_CLASSIFICATION_PROMPT
)
from base_agent import BaseAgent
from llm_api.llm_client_chat_model import LLMClientChatModel
from utils.time_count import timer
from utils.stream_chunk import StreamChunk  # 引入标准 StreamChunk


# ===== 输出结构定义 (对应 final_output["output"] 的内容) =====
class TextClassificationOutputContent(BaseModel):
    success: bool = Field(description="分类是否成功")
    predicted_label: str = Field(description="预测的标签")
    predicted_token: str = Field(description="预测的汉字token")
    all_scores: Dict[str, float] = Field(description="所有标签的得分")
    label_mapping: Dict[str, str] = Field(description="标签到汉字的映射关系")
    validation_errors: List[str] = Field(default_factory=list, description="验证错误信息")
    text_length: int = Field(description="输入文本长度")
    label_count: int = Field(description="候选标签数量")


# ===== 状态定义 =====
class TextClassificationState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], "消息历史"]
    text: str
    candidate_labels: List[str]
    label_to_token: Dict[str, str]
    token_to_label: Dict[str, str]
    predicted_token: str
    predicted_label: str
    confidence: float
    all_scores: Dict[str, float]
    validation_errors: List[str]
    # 新增/修改字段以适配统一输出结构
    final_output: Optional[dict]  # 包含 output, reasoning_content, metadata, confidence
    content: str
    reasoning_content: str
    metadata: Dict[str, Any]


# ===== 文本分类Agent主类 =====
class TextClassificationAgent(BaseAgent):
    def __init__(
            self,
            name: str = "text-classification-agent",
            # openai client init config
            base_url: str = "https://api.deepseek.com/v1",
            api_key: Optional[str] = None,
            timeout: float = 60.0,
            max_retries: int = 3,
            # openai client run config
            model: str = "deepseek-chat",
            max_tokens: Optional[int] = 1,  # 只输出一个token
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

        # 构建工作流图
        self.graph = self._build_graph()

    def _get_chinese_tokens(self, num_labels: int) -> List[str]:
        """获取用于映射的字符列表"""
        return [chr(num + 97) for num in range(num_labels)]

    def _build_mapping_prompt(self, candidate_labels: List[str]) -> str:
        """构建标签映射提示词"""
        chinese_tokens = self._get_chinese_tokens(len(candidate_labels))
        label_descriptions = []
        for i, label in enumerate(candidate_labels):
            label_descriptions.append(f"【{label}】对应输出：{chinese_tokens[i]}")
        
        return "\n".join(label_descriptions)

    def _build_classification_prompt(self, text: str, candidate_labels: List[str]) -> str:
        """构建分类提示词"""
        mapping_prompt = self._build_mapping_prompt(candidate_labels)
        return TEXT_CLASSIFICATION_PROMPT.format(mapping_prompt=mapping_prompt, text=text)

    def _parse_model_output(self, content: str, token_to_label: Dict[str, str], candidate_labels: List[str]) -> Dict[str, Any]:
        """解析模型输出"""
        predicted_token = content.strip()
        if predicted_token:
            predicted_token = predicted_token[0]  # 只取第一个字符
        
        predicted_label = token_to_label.get(predicted_token, candidate_labels[0])
        
        return {
            "predicted_token": predicted_token,
            "predicted_label": predicted_label
        }

    def _calculate_confidence(self, chat_completion: Dict[str, Any], label_to_token: Dict[str, str]) -> Dict[str, Any]:
        """基于logprobs计算置信度"""
        all_scores = {}
        confidence = 0.0
        
        try:
            choices = chat_completion.get("choices", [])
            if not choices:
                raise ValueError("choices为空")
            
            choice = choices[0]
            logprobs = choice.get("logprobs", {})
            
            if logprobs and "content" in logprobs:
                content_logprobs = logprobs["content"]
                
                if content_logprobs and len(content_logprobs) > 0:
                    first_token_logprobs = content_logprobs[0]
                    top_logprobs = first_token_logprobs.get("top_logprobs", [])
                    
                    # 计算每个候选token的概率
                    token_probs = {}
                    for token_info in top_logprobs:
                        token = token_info.get("token", "").strip()
                        if token:
                            token = token[0]  # 只取第一个字符
                        logprob = token_info.get("logprob", 0.0)
                        probability = 2 ** logprob
                        token_probs[token] = probability
                    
                    # 计算每个标签的得分
                    total_prob = 0.0
                    for label, token in label_to_token.items():
                        prob = token_probs.get(token, 0.0)
                        all_scores[label] = prob
                        total_prob += prob
                    
                    # 标准化得分并保留两位小数
                    if total_prob > 0:
                        for label in all_scores:
                            normalized_score = all_scores[label] / total_prob
                            all_scores[label] = round(normalized_score, 2)
                    
                    # 获取预测标签的置信度
                    predicted_token = choice.get("message", {}).get("content", "").strip()
                    if predicted_token:
                        predicted_token = predicted_token[0]
                    predicted_label = next((label for label, token in label_to_token.items() if token == predicted_token), list(label_to_token.keys())[0])
                    confidence = all_scores.get(predicted_label, 0.0)
                    
                    self.logger.debug(f"置信度计算成功: predicted_label={predicted_label}, confidence={confidence:.2f}")
                else:
                    raise ValueError("content_logprobs为空或格式不正确")
            else:
                raise ValueError("logprobs数据不存在")
                
        except Exception as e:
            self.logger.warning(f"置信度计算失败: {e}")
            # 使用均匀分布作为回退，并保留两位小数
            even_score = round(1.0 / len(label_to_token), 2)
            for label in label_to_token.keys():
                all_scores[label] = even_score
            confidence = even_score
            self.logger.info("使用均匀分布作为置信度回退方案")
        
        return {
            "all_scores": all_scores,
            "confidence": confidence
        }

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        graph = StateGraph(TextClassificationState)

        async def initialize_node(state: TextClassificationState, config: RunnableConfig) -> TextClassificationState:
            """初始化节点：准备标签映射"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 初始化文本分类任务"):
                self.logger.info(f"request_id: {request_id}, 开始文本分类, 文本长度: {len(state['text'])}, 标签数: {len(state['candidate_labels'])}")
                
                # 创建标签到汉字的映射
                chinese_tokens = self._get_chinese_tokens(len(state['candidate_labels']))
                label_to_token = {}
                token_to_label = {}
                
                for i, label in enumerate(state['candidate_labels']):
                    token = chinese_tokens[i]
                    
                    label_to_token[label] = token
                    token_to_label[token] = label
                
                self.logger.debug(f"request_id: {request_id}, 标签映射: {label_to_token}")
                
                return {
                    **state,
                    "label_to_token": label_to_token,
                    "token_to_label": token_to_label
                }

        async def classify_node(state: TextClassificationState, config: RunnableConfig) -> TextClassificationState:
            """分类节点：执行文本分类"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 执行文本分类"):
                # 构建提示词
                prompt_text = self._build_classification_prompt(
                    state["text"], 
                    state["candidate_labels"]
                )

                self.logger.info(f"request_id: {request_id}, 调用LLM进行文本分类")

                messages = [
                    SystemMessage(content=TEXT_CLASSIFICATION_SYSTEM_MESSAGE), 
                    HumanMessage(content=prompt_text)
                ]

                # 调用LLM
                llm_client = self.get_llm_client(run_config)
                chat_model = LLMClientChatModel(llm_client=llm_client)
                try:
                    response = await chat_model.ainvoke(messages, config=config, logprobs=True, top_logprobs=5)

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

                        self.logger.debug(f"request_id: {request_id}, LLM响应: {content}")

                        # 解析分类结果
                        classification_result = self._parse_model_output(
                            content, 
                            state["token_to_label"], 
                            state["candidate_labels"]
                        )
                        
                        # 计算置信度
                        confidence_result = self._calculate_confidence(chat_completion, state["label_to_token"])
                        
                        # 更新消息历史
                        new_messages = state["messages"] + messages + [AIMessage(content=content)]
                        # output_metadata["messages"] = convert_to_openai_messages(new_messages)
                        
                        return {
                            **state,
                            "messages": new_messages,
                            "predicted_token": classification_result["predicted_token"],
                            "predicted_label": classification_result["predicted_label"],
                            "confidence": confidence_result["confidence"],
                            "all_scores": confidence_result["all_scores"],
                            "content": classification_result["predicted_label"],
                            "reasoning_content": reasoning_content,
                            "metadata": output_metadata
                        }
                    else:
                        raise ValueError("LLM api输出错误, choices为空")
                except asyncio.CancelledError:
                    self.logger.warning(f"⛔ request_id: {request_id}, 任务被中断，已停止 LLM 请求")
                    raise

        def validate_node(state: TextClassificationState, config: RunnableConfig) -> TextClassificationState:
            """验证节点：验证分类结果"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 验证分类结果"):
                validation_errors = []
                
                # 验证预测的token是否在映射中
                if state["predicted_token"] not in state["token_to_label"]:
                    validation_errors.append(f"预测的token '{state['predicted_token']}' 不在有效映射中")
                
                # 验证置信度是否合理
                if state["confidence"] < 0 or state["confidence"] > 1:
                    validation_errors.append(f"置信度 {state['confidence']} 超出合理范围")
                
                # 验证所有标签得分之和约为1
                total_score = sum(state["all_scores"].values())
                if abs(total_score - 1.0) > 0.01 and len(state["all_scores"]) > 0:
                    validation_errors.append(f"标签得分总和 {total_score:.4f} 不等于1")

                if not validation_errors:
                    self.logger.info(f"request_id: {request_id}, 分类结果验证通过")
                else:
                    self.logger.warning(f"request_id: {request_id}, 发现 {len(validation_errors)} 个验证问题")

                state["validation_errors"] = validation_errors
                return state

        def finalize_node(state: TextClassificationState, config: RunnableConfig) -> TextClassificationState:
            """最终处理节点：汇总结果"""
            run_config = config.get("configurable", {})
            request_id = run_config.get("request_id")

            with timer(self.logger, f"request_id: {request_id}, 结果汇总"):
                success = len(state["validation_errors"]) == 0
                
                # 构建 output 字典（业务数据）
                output_data = {
                    "predicted_label": state["predicted_label"],
                    # "predicted_token": state["predicted_token"],
                    "all_scores": state["all_scores"],
                    # "label_mapping": state["label_to_token"],
                    # "validation_errors": state["validation_errors"],
                    # "text_length": len(state["text"]),
                    # "label_count": len(state["candidate_labels"])
                }

                # 构建包含四个固定元素的 final_output
                final_output_structure = {
                    "output": output_data,
                    "content": state["predicted_label"],
                    "reasoning_content": state.get("reasoning_content", ""),
                    "metadata": state.get("metadata", {}),
                    "confidence": state.get("confidence", 0.0)
                }

                status_msg = "成功" if success else f"有{len(state['validation_errors'])}个警告"
                self.logger.success(f"request_id: {request_id}, 文本分类完成, 状态: {status_msg}, 置信度: {state['confidence']:.2f}")

                return {
                    **state,
                    "final_output": final_output_structure
                }

        # 添加节点
        graph.add_node("initialize", initialize_node)
        graph.add_node("classify", classify_node)
        graph.add_node("validate", validate_node)
        graph.add_node("finalize", finalize_node)

        # 设置工作流
        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "classify")
        graph.add_edge("classify", "validate")
        graph.add_edge("validate", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    async def run(self, text: str, candidate_labels: List[str], **kwargs) -> Dict[str, Any]:
        """
        执行文本分类流程

        :param text: 待分类的文本
        :param candidate_labels: 候选标签列表
        :return: 结构化输出字典 {output, reasoning_content, metadata, confidence}
        """
        if not text.strip():
            raise ValueError("文本不能为空")
        if not candidate_labels:
            raise ValueError("候选标签列表不能为空")
        if len(candidate_labels) > 20:
            raise ValueError("候选标签最多不能超过20个")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔧 request_id: {request_id}, 开始处理文本分类请求, text_length: {len(text)}, labels: {candidate_labels}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        inputs = {
            "messages": [],
            "text": text,
            "candidate_labels": candidate_labels,
            "label_to_token": {},
            "token_to_label": {},
            "predicted_token": "",
            "predicted_label": "",
            "confidence": 0.0,
            "all_scores": {},
            "validation_errors": [],
            "final_output": None,
            "reasoning_content": "",
            "metadata": {}
        }

        with timer(self.logger, f"request_id: {request_id}, 完整文本分类流程"):
            # 传递运行时配置
            config = {"configurable": run_config} if run_config else {}
            final_state = await self.graph.ainvoke(inputs, config=config)
            output = final_state.get("final_output", {})
            self.logger.success(f"🎉 request_id: {request_id}, 文本分类完成")

        return output
    

    async def run_stream(self, text: str, candidate_labels: List[str], **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """
        流式执行文本分类流程

        :param text: 待分类的文本
        :param candidate_labels: 候选标签列表
        :return: StreamChunk 流式输出生成器
        """
        if not text.strip():
            raise ValueError("文本不能为空")
        if not candidate_labels:
            raise ValueError("候选标签列表不能为空")
        if len(candidate_labels) > 20:
            raise ValueError("候选标签最多不能超过20个")
        
        request_id = kwargs.get("request_id")

        self.logger.info(f"🔧 request_id: {request_id}, 开始流式处理文本分类请求, text_length: {len(text)}, labels: {candidate_labels}")

        # 构建运行时配置
        run_config = {k: v for k, v in kwargs.items() if k in self.init_config or k == "request_id"}

        # 提前构建标签映射（与initialize节点相同的逻辑）
        chinese_tokens = self._get_chinese_tokens(len(candidate_labels))
        label_to_token = {}
        token_to_label = {}
        
        for i, label in enumerate(candidate_labels):
            token = chinese_tokens[i]
            
            label_to_token[label] = token
            token_to_label[token] = label

        inputs = {
            "messages": [],
            "text": text,
            "candidate_labels": candidate_labels,
            "label_to_token": label_to_token,
            "token_to_label": token_to_label,
            "predicted_token": "",
            "predicted_label": "",
            "confidence": 0.0,
            "all_scores": {},
            "validation_errors": [],
            "final_output": None,
            "reasoning_content": "",
            "metadata": {}
        }

        # 用于跟踪流式输出的状态
        accumulated_token = ""  # 累积的token字符
        label_emitted = False   # 标记是否已经输出了标签

        with timer(self.logger, f"request_id: {request_id}, 完整流式文本分类流程"):
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
                            
                            # 处理分类token输出 - 只取第一个有效字符
                            if content and not label_emitted:
                                # 累积token字符，但只取第一个非空字符
                                if content.strip() and not accumulated_token:
                                    accumulated_token = content.strip()[0]  # 只取第一个字符
                                    
                                    # 尝试映射到标签
                                    predicted_label = token_to_label.get(accumulated_token, candidate_labels[0])
                                    
                                    # 输出映射后的标签
                                    yield StreamChunk(
                                        type="content",
                                        content=predicted_label  # 直接输出标签而不是token
                                    )
                                    
                                    label_emitted = True
                                    self.logger.debug(f"request_id: {request_id}, 流式映射: token='{accumulated_token}' -> label='{predicted_label}'")
                
                # 处理节点开始事件
                elif event_type == "on_chain_start":
                    name = event.get("name", "")
                    if name == "initialize":
                        yield StreamChunk(
                            type="processing",
                            content="初始化..."
                        )
                    elif name == "classify":
                        yield StreamChunk(
                            type="processing",
                            content="分析：正在执行分类推断..."
                        )
                    elif name == "validate":
                        yield StreamChunk(
                            type="processing", 
                            content="验证：检查分类置信度..."
                        )
                    elif name == "finalize":
                        yield StreamChunk(
                            type="processing",
                            content="汇总：正在生成最终结果..."
                        )
                
                # 处理节点结束事件
                elif event_type == "on_chain_end":
                    name = event.get("name", "")
                    # 处理图结束事件, 输出最终结果
                    if name == "LangGraph":
                        output = event.get("data", {}).get("output", {})
                        final_output_data = output.get("final_output", {})
                        
                        # 确保是完整的 final_output 结构
                        if final_output_data:
                            yield StreamChunk(
                                type="final",
                                content="",
                                metadata=final_output_data
                            )

            self.logger.success(f"🎉 request_id: {request_id}, 流式文本分类完成")


# ===== 使用示例 =====
async def main():
    """使用示例"""
    
    # 初始化Agent
    agent = TextClassificationAgent(
        name="test-classification-agent",
        base_url="https://api.deepseek.com/v1",
        api_key="YOUR_API_KEY",  # 替换为实际API密钥
        temperature=0.1
    )
    
    # 测试用例
    test_text = "这家餐厅的食物非常美味, 服务也很周到, 强烈推荐！"
    test_labels = ["正面评价", "负面评价", "中性评价"]
    
    print(f"\n=== 测试文本分类 ===")
    print(f"文本: {test_text}")
    print(f"标签: {test_labels}")
    
    # 示例1: 非流式调用
    print("\n--- 1. 非流式调用 ---")
    result = await agent.run(
        text=test_text,
        candidate_labels=test_labels,
        request_id="test-classify-001"
    )
    
    output_content = result.get("output", {})
    confidence = result.get("confidence", 0.0)
    print(f"✅ 分类结果: {output_content.get('predicted_label')}")
    print(f"📊 置信度: {confidence:.4f}")
    print(f"📝 完整输出keys: {list(result.keys())}") # 验证是否只有4个key

    # 示例2: 流式调用
    print("\n--- 2. 流式调用 ---")
    async for chunk in agent.run_stream(
        text=test_text,
        candidate_labels=test_labels,
        request_id="test-classify-002"
    ):
        if chunk.type == "thinking":
            print(f"🤔 {chunk.content}")
        elif chunk.type == "processing":
            print(f"🔄 {chunk.content}")
        elif chunk.type == "content":
            print(f"📝 生成内容: {chunk.content}")
        elif chunk.type == "final":
            final_data = chunk.metadata
            output = final_data.get("output", {})
            print(f"\n✅ 流式完成. 最终标签: {output.get('predicted_label')}")
            print(f"📈 所有得分: {output.get('all_scores')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())