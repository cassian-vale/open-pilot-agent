import time
from typing import Iterator, Optional

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from langchain_core.messages import BaseMessage, BaseMessageChunk


class ChatMessage(BaseMessage):
    """思考过程的消息块"""
    chat_completion: Optional[ChatCompletion] = None
    type: str = "chat_completion"
    
    @classmethod  
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain", "schema", "messages"]
    

class ChatMessageChunk(BaseMessageChunk):
    """思考过程的消息块"""
    chat_completion_chunk: Optional[ChatCompletionChunk] = None
    type: str = "chat_completion_chunk"
    
    @classmethod  
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain", "schema", "messages"]


# def merge_chunks_to_completion(chunks: Iterator[ChatCompletionChunk]) -> ChatCompletion:
#     """
#     将流式的 ChatCompletionChunk 合并为一个 ChatCompletion，支持思考模式
    
#     Args:
#         chunks: ChatCompletionChunk 的迭代器
        
#     Returns:
#         合并后的 ChatCompletion 对象
#     """
#     # 初始化合并后的数据
#     merged_id = None
#     merged_created = None
#     merged_model = None
#     merged_choices = []
#     merged_usage = None
#     merged_system_fingerprint = None
    
#     # 用于跟踪每个 choice 的合并状态
#     choice_states = {}
    
#     for chunk in chunks:
#         # 处理通用字段（使用第一个非空 chunk 的值）
#         if merged_id is None and chunk.id:
#             merged_id = chunk.id
#         if merged_created is None and chunk.created:
#             merged_created = chunk.created
#         if merged_model is None and chunk.model:
#             merged_model = chunk.model
#         if merged_system_fingerprint is None and chunk.system_fingerprint:
#             merged_system_fingerprint = chunk.system_fingerprint
        
#         # 处理 usage（累加）
#         if chunk.usage:
#             if merged_usage is None:
#                 merged_usage = CompletionUsage(
#                     prompt_tokens=chunk.usage.prompt_tokens or 0,
#                     completion_tokens=chunk.usage.completion_tokens or 0,
#                     total_tokens=chunk.usage.total_tokens or 0
#                 )
#             else:
#                 merged_usage.prompt_tokens = (merged_usage.prompt_tokens or 0) + (chunk.usage.prompt_tokens or 0)
#                 merged_usage.completion_tokens = (merged_usage.completion_tokens or 0) + (chunk.usage.completion_tokens or 0)
#                 merged_usage.total_tokens = (merged_usage.total_tokens or 0) + (chunk.usage.total_tokens or 0)
        
#         # 处理 choices
#         for choice in chunk.choices:
#             choice_index = choice.index
            
#             if choice_index not in choice_states:
#                 # 初始化这个 choice 的状态
#                 choice_states[choice_index] = {
#                     'message': ChatCompletionMessage(role="assistant", content=""),
#                     'finish_reason': None,
#                     'logprobs': None,
#                     'reasoning_content': ""  # 专门存储思考内容
#                 }
            
#             state = choice_states[choice_index]
            
#             # 合并 delta 内容
#             delta = choice.delta
#             if delta:
#                 # 处理 role
#                 if delta.role and not state['message'].role:
#                     state['message'].role = delta.role
                
#                 # 处理 content - 普通回复内容
#                 if delta.content:
#                     state['message'].content = (state['message'].content or "") + delta.content
                
#                 # 处理思考内容 (reasoning_content)
#                 if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
#                     state['reasoning_content'] = (state['reasoning_content'] or "") + delta.reasoning_content
                
#                 # 处理 function_call
#                 if delta.function_call:
#                     if not state['message'].function_call:
#                         state['message'].function_call = {
#                             'name': '',
#                             'arguments': ''
#                         }
#                     if delta.function_call.name:
#                         state['message'].function_call['name'] = (state['message'].function_call['name'] or "") + delta.function_call.name
#                     if delta.function_call.arguments:
#                         state['message'].function_call['arguments'] = (state['message'].function_call['arguments'] or "") + delta.function_call.arguments
                
#                 # 处理 tool_calls
#                 if delta.tool_calls:
#                     if not state['message'].tool_calls:
#                         state['message'].tool_calls = []
                    
#                     for tool_call_delta in delta.tool_calls:
#                         tool_call_index = tool_call_delta.index
#                         if tool_call_index >= len(state['message'].tool_calls):
#                             # 添加新的 tool_call
#                             state['message'].tool_calls.append({
#                                 'id': '',
#                                 'type': 'function',
#                                 'function': {'name': '', 'arguments': ''}
#                             })
                        
#                         tool_call = state['message'].tool_calls[tool_call_index]
#                         if tool_call_delta.id:
#                             tool_call['id'] = (tool_call['id'] or "") + tool_call_delta.id
#                         if tool_call_delta.function:
#                             if tool_call_delta.function.name:
#                                 tool_call['function']['name'] = (tool_call['function']['name'] or "") + tool_call_delta.function.name
#                             if tool_call_delta.function.arguments:
#                                 tool_call['function']['arguments'] = (tool_call['function']['arguments'] or "") + tool_call_delta.function.arguments
            
#             # 更新 finish_reason
#             if choice.finish_reason:
#                 state['finish_reason'] = choice.finish_reason
            
#             # 更新 logprobs（这里简单使用最后一个非空值）
#             if choice.logprobs:
#                 state['logprobs'] = choice.logprobs
    
#     # 构建最终的 choices 列表
#     for index in sorted(choice_states.keys()):
#         state = choice_states[index]
        
#         # 创建最终的 message，包含思考内容
#         final_message = state['message']
        
#         # 如果有思考内容，将其添加到 message 的额外字段中
#         if state['reasoning_content']:
#             # 由于 ChatCompletionMessage 可能没有 reasoning_content 字段，
#             # 我们可以将其作为自定义属性或使用其他方式存储
#             # 这里我们创建一个包含思考内容的扩展消息
#             final_message_dict = final_message.model_dump()
#             final_message_dict['reasoning_content'] = state['reasoning_content']
            
#             # 重新创建消息对象（如果需要保持类型安全，可能需要自定义处理）
#             merged_choices.append({
#                 'index': index,
#                 'message': final_message_dict,  # 这里使用字典而不是严格类型
#                 'finish_reason': state['finish_reason'],
#                 'logprobs': state['logprobs']
#             })
#         else:
#             merged_choices.append({
#                 'index': index,
#                 'message': final_message,
#                 'finish_reason': state['finish_reason'],
#                 'logprobs': state['logprobs']
#             })
    
#     # 设置默认值（如果所有 chunk 都没有某些字段）
#     if merged_id is None:
#         merged_id = f"chatcmpl-{int(time.time())}"
#     if merged_created is None:
#         merged_created = int(time.time())
#     if merged_model is None:
#         merged_model = "default"
    
#     # 创建并返回 ChatCompletion 对象
#     return ChatCompletion(
#         id=merged_id,
#         choices=merged_choices,
#         created=merged_created,
#         model=merged_model,
#         object="chat.completion",
#         usage=merged_usage,
#         system_fingerprint=merged_system_fingerprint
#     )


def merge_chunks_to_completion(chunks: Iterator[ChatCompletionChunk]) -> ChatCompletion:
    """
    将流式的 ChatCompletionChunk 合并为一个 ChatCompletion，支持思考模式
    
    Args:
        chunks: ChatCompletionChunk 的迭代器
        
    Returns:
        合并后的 ChatCompletion 对象
    """
    import time
    from typing import Dict, Any
    
    # 初始化合并后的数据
    merged_id = None
    merged_created = None
    merged_model = None
    merged_choices = []
    merged_usage = None
    merged_system_fingerprint = None
    
    # 用于跟踪每个 choice 的合并状态
    choice_states: Dict[int, Dict[str, Any]] = {}
    
    for chunk in chunks:
        # 处理通用字段（使用第一个非空 chunk 的值）
        if merged_id is None and chunk.id:
            merged_id = chunk.id
        if merged_created is None and chunk.created:
            merged_created = chunk.created
        if merged_model is None and chunk.model:
            merged_model = chunk.model
        if merged_system_fingerprint is None and chunk.system_fingerprint:
            merged_system_fingerprint = chunk.system_fingerprint
        
        # 处理 usage（累加）
        if chunk.usage:
            if merged_usage is None:
                merged_usage = CompletionUsage(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0
                )
            else:
                merged_usage.prompt_tokens = (merged_usage.prompt_tokens or 0) + (chunk.usage.prompt_tokens or 0)
                merged_usage.completion_tokens = (merged_usage.completion_tokens or 0) + (chunk.usage.completion_tokens or 0)
                merged_usage.total_tokens = (merged_usage.total_tokens or 0) + (chunk.usage.total_tokens or 0)
        
        # 处理 choices
        for choice in chunk.choices:
            choice_index = choice.index
            
            if choice_index not in choice_states:
                # 初始化这个 choice 的状态
                choice_states[choice_index] = {
                    'message': ChatCompletionMessage(role="assistant", content=""),
                    'finish_reason': None,
                    'logprobs': None,
                    'reasoning_content': ""  # 专门存储思考内容
                }
            
            state = choice_states[choice_index]
            
            # 合并 delta 内容
            delta = choice.delta
            if delta:
                # 处理 role
                if delta.role and not state['message'].role:
                    state['message'].role = delta.role
                
                # 处理 content - 普通回复内容
                if delta.content:
                    state['message'].content = (state['message'].content or "") + delta.content
                
                # 处理思考内容 (reasoning_content)
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    state['reasoning_content'] = (state['reasoning_content'] or "") + delta.reasoning_content
            
            # 更新 finish_reason
            if choice.finish_reason:
                state['finish_reason'] = choice.finish_reason
            
            # 关键修正：正确处理 logprobs
            if choice.logprobs:
                # 如果 logprobs 是对象，转换为字典
                if hasattr(choice.logprobs, 'model_dump'):
                    state['logprobs'] = choice.logprobs.model_dump()
                elif hasattr(choice.logprobs, 'dict'):
                    state['logprobs'] = choice.logprobs.dict()
                else:
                    # 如果已经是字典，直接使用
                    state['logprobs'] = choice.logprobs
    
    # 构建最终的 choices 列表
    for index in sorted(choice_states.keys()):
        state = choice_states[index]
        
        # 创建最终的 choice 字典
        choice_dict = {
            'index': index,
            'message': state['message'],
            'finish_reason': state['finish_reason'],
            'logprobs': state['logprobs']  # 这里已经是字典格式
        }
        
        # 如果有思考内容，将其添加到 message 的额外字段中
        if state['reasoning_content']:
            # 将思考内容作为 message 的自定义属性
            message_dict = state['message'].model_dump()
            message_dict['reasoning_content'] = state['reasoning_content']
            choice_dict['message'] = message_dict
        
        merged_choices.append(choice_dict)
    
    # 设置默认值（如果所有 chunk 都没有某些字段）
    if merged_id is None:
        merged_id = f"chatcmpl-{int(time.time())}"
    if merged_created is None:
        merged_created = int(time.time())
    if merged_model is None:
        merged_model = "default"
    
    # 创建并返回 ChatCompletion 对象
    return ChatCompletion(
        id=merged_id,
        choices=merged_choices,
        created=merged_created,
        model=merged_model,
        object="chat.completion",
        usage=merged_usage,
        system_fingerprint=merged_system_fingerprint
    )
