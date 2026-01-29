# llm_client.py
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from utils.http_factory import GlobalHTTPFactory

dir_name = Path(__file__).resolve().parent.parent
sys.path.append(str(dir_name))

from llm_api.thinking_config import ThinkingConfig


class LLMClient:
    """LLM客户端，封装OpenAI API调用和思维链配置"""

    def __init__(
            self,
            model: str = "deepseek-chat",
            base_url: str = "https://api.deepseek.com/v1",
            api_key: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            top_p: float = 1.0,
            timeout: float = 10.0,
            max_retries: int = 3,
            stream: bool = False,
            enable_thinking: bool = False
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries
        self.stream = stream
        self.enable_thinking = enable_thinking

        # 初始化思维链配置
        self.thinking_config = ThinkingConfig()

        # 初始化客户端
        self._sync_client: Optional[OpenAI] = None
    
    @property
    def sync_client(self) -> OpenAI:
        """懒加载同步客户端"""
        if self._sync_client is None:
            self._sync_client = self._setup_sync_client()
            if self._sync_client is None:
                raise ValueError("同步客户端初始化失败，请检查配置")
        return self._sync_client


    @staticmethod
    def _resolve_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """解析API密钥"""
        if api_key:
            return api_key

        env_vars = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ZHIPU_API_KEY", "API_KEY"]
        for env_var in env_vars:
            env_key = os.getenv(env_var)
            if env_key:
                return env_key
        return None

    def _setup_sync_client(self) -> Optional[OpenAI]:
        """初始化同步客户端"""
        resolved_api_key = self._resolve_api_key(self.api_key)
        if not resolved_api_key:
            return None

        client_config = {
            "base_url": self.base_url.strip(),
            "api_key": resolved_api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        return OpenAI(**client_config)

    async def _get_async_client(self) -> AsyncOpenAI:
        """初始化异步客户端"""
        resolved_api_key = self._resolve_api_key(self.api_key)
        if not resolved_api_key:
            return None
        
        shared_http = await GlobalHTTPFactory.get_async_http_client()

        client_config = {
            "base_url": self.base_url.strip(),
            "api_key": resolved_api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        return AsyncOpenAI(**client_config, http_client=shared_http)
    

    def _build_call_params(self, messages: List[ChatCompletionMessageParam], **kwargs) -> Dict[str, Any]:
        """构建调用参数，包含思维链配置"""
        call_params = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": kwargs.get("stream", self.stream),
        }

        # invoke可传递参数
        valid_invoke_param_names = {
            "top_logprobs": int,
            "logprobs": bool
        }
        invoke_params = {k: v for k, v in kwargs.items() if k in valid_invoke_param_names and isinstance(v, valid_invoke_param_names[k])}
        call_params.update(invoke_params)

        # 移除None值
        call_params = {k: v for k, v in call_params.items() if v is not None}

        # 添加思维链配置
        thinking_params = self.thinking_config.get_thinking_params(
            call_params.get("model"),
            self.enable_thinking
        )
        if thinking_params:
            call_params["extra_body"] = thinking_params

        if call_params.get("stream", False):
            call_params["stream_options"] = {"include_usage": True}

        return call_params

    def call(
            self,
            messages: List[ChatCompletionMessageParam],
            **kwargs
    ) -> Any:
        """同步调用"""
        if not self.sync_client:
            raise ValueError("同步客户端未初始化，请检查API配置")

        call_params = self._build_call_params(messages, **kwargs)

        response = self.sync_client.chat.completions.create(**call_params)
        return response

    async def acall(
            self,
            messages: List[ChatCompletionMessageParam],
            **kwargs
    ) -> Any:
        """异步调用"""
        async_client = await self._get_async_client()
        call_params = self._build_call_params(messages, **kwargs)
        response = await async_client.chat.completions.create(**call_params)
        return response

    def stream(
            self,
            messages: List[ChatCompletionMessageParam],
            **kwargs
    ):
        """同步流式调用"""
        if not self.sync_client:
            raise ValueError("同步客户端未初始化，请检查API配置")

        call_params = self._build_call_params(messages, **kwargs)

        call_params["stream"] = True

        response = self.sync_client.chat.completions.create(**call_params)
        return response

    async def astream(
            self,
            messages: List[ChatCompletionMessageParam],
            **kwargs
    ):
        """异步流式调用"""
        async_client = await self._get_async_client()

        call_params = self._build_call_params(messages, **kwargs)
        call_params["stream"] = True

        response = await async_client.chat.completions.create(**call_params)
        return response

    def is_configured(self) -> bool:
        """检查客户端是否已配置"""
        return self.sync_client is not None

    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 重新初始化客户端
        if any(key in kwargs for key in ['api_key', 'base_url', 'timeout', 'max_retries']):
            self.sync_client = self._setup_sync_client()

    def enable_thinking_mode(self, enable: bool = True):
        """启用或禁用思考模式"""
        self.enable_thinking = enable

    def get_thinking_status(self) -> Dict[str, Any]:
        """获取思考模式状态"""
        return {
            "enable_thinking": self.enable_thinking,
            "thinking_params": self.thinking_config.get_thinking_params(self.model, self.enable_thinking),
            "model_type": self.thinking_config.get_model_type(self.model)
        }
