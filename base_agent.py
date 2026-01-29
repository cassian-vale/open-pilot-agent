import os
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from langgraph.graph import StateGraph

dir_name = Path(__file__).resolve().parent
sys.path.append(str(dir_name))

from llm_api.llm_client import LLMClient
from utils.log_util import logger_pool


class BaseAgent(ABC):
    def __init__(
            self,
            name: str = "base-agent",
            model: str = "deepseek-chat",
            base_url: str = "https://api.deepseek.com/v1",
            api_key: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: float = 0.0,
            top_p: float = 1.0,
            timeout: float = 60.0,
            max_retries: int = 3,
            stream: bool = False,
            enable_thinking: bool = False
    ):
        self.name = name

        # 保存初始化配置状态
        self.init_config = {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "timeout": timeout,
            "max_retries": max_retries,
            "stream": stream,
            "enable_thinking": enable_thinking
        }

        # 配置日志
        self.logger = logger_pool.get_logger(name)

        # 检查LLM配置完整性
        self._llm_configured = self._check_llm_config_completeness()

        # 初始化 LLM客户端
        self.llm_client = self._setup_llm_client()

        self.logger.info(f"✅ {self.name} 初始化完成")

    def _check_llm_config_completeness(self) -> bool:
        """检查LLM配置是否完整"""
        required_configs = ["model", "base_url", "api_key"]
        missing_configs = []

        for config in required_configs:
            if not self.init_config[config]:
                missing_configs.append(config)

        if missing_configs:
            self.logger.warning(
                f"LLM配置不完整，缺失: {missing_configs}。"
                f"运行时必须通过 run() 方法的参数提供完整配置。"
            )
            return False
        return True

    def _setup_llm_client(self) -> Optional[LLMClient]:
        """初始化LLM客户端"""
        if not self._llm_configured:
            self.logger.info("LLM配置不完整，等待运行时配置")
            return None

        try:
            client = LLMClient(
                model=self.init_config["model"],
                base_url=self.init_config["base_url"],
                api_key=self.init_config["api_key"],
                temperature=self.init_config["temperature"],
                max_tokens=self.init_config["max_tokens"],
                top_p=self.init_config["top_p"],
                timeout=self.init_config["timeout"],
                max_retries=self.init_config["max_retries"],
                stream=self.init_config["stream"],
                enable_thinking=self.init_config["enable_thinking"]
            )

            if client.is_configured():
                self.logger.info("LLM客户端初始化成功")
                return client
            else:
                self.logger.warning("LLM客户端初始化失败")
                return None

        except Exception as e:
            self.logger.error("LLM客户端初始化失败", exception=e)
            return None

    def _resolve_api_key(self, runtime_api_key: Optional[str] = None) -> str:
        """解析API密钥"""
        if runtime_api_key:
            return runtime_api_key

        if self.init_config["api_key"]:
            return self.init_config["api_key"]

        env_vars = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ZHIPU_API_KEY", "API_KEY"]
        for env_var in env_vars:
            env_key = os.getenv(env_var)
            if env_key:
                self.logger.info(f"从环境变量 {env_var} 获取API密钥")
                return env_key

        raise ValueError("未找到API密钥")

    def create_llm_client_with_config(self, runtime_config: Dict[str, Any]) -> LLMClient:
        """根据运行时配置创建LLM客户端"""
        merged_config = {**self.init_config, **runtime_config}

        api_key = self._resolve_api_key(runtime_config.get("api_key"))
        merged_config["api_key"] = api_key

        required_configs = ["model", "base_url", "api_key"]
        missing_configs = []

        for config in required_configs:
            if not merged_config[config]:
                missing_configs.append(config)

        if missing_configs:
            raise ValueError(f"LLM配置不完整，缺失: {missing_configs}")

        try:
            client = LLMClient(
                model=merged_config["model"],
                base_url=merged_config["base_url"],
                api_key=merged_config["api_key"],
                temperature=merged_config["temperature"],
                max_tokens=merged_config["max_tokens"],
                top_p=merged_config["top_p"],
                timeout=merged_config["timeout"],
                max_retries=merged_config["max_retries"],
                stream=merged_config["stream"],
                enable_thinking=merged_config["enable_thinking"],
            )

            if not client.is_configured():
                raise ValueError("LLM客户端配置失败")

            return client

        except Exception as e:
            self.logger.error(f"创建LLM客户端失败", exception=e)
            raise

    def get_llm_client(self, runtime_config: Optional[Dict[str, Any]] = None) -> LLMClient:
        """获取LLM客户端"""
        if runtime_config:
            return self.create_llm_client_with_config(runtime_config)

        if self.llm_client is not None:
            return self.llm_client

        raise ValueError("LLM配置不完整且未提供运行时配置")

    def call_llm(self, messages, runtime_config=None, **kwargs):
        """同步调用"""
        client = self.get_llm_client(runtime_config)
        return client.call(messages, **kwargs)

    async def acall_llm(self, messages, runtime_config=None, **kwargs):
        """异步调用"""
        client = self.get_llm_client(runtime_config)
        return await client.acall(messages, **kwargs)

    def stream_llm(self, messages, runtime_config=None, **kwargs):
        """同步流式调用"""
        client = self.get_llm_client(runtime_config)
        return client.stream(messages, **kwargs)

    async def astream_llm(self, messages, runtime_config=None, **kwargs):
        """异步流式调用"""
        client = self.get_llm_client(runtime_config)
        async for chunk in await client.astream(messages, **kwargs):
            yield chunk

    def enable_thinking_mode(self, enable: bool = True):
        """启用或禁用思考模式"""
        if self.llm_client:
            self.llm_client.enable_thinking_mode(enable)

    def get_thinking_status(self) -> Dict[str, Any]:
        """获取思考模式状态"""
        if self.llm_client:
            return self.llm_client.get_thinking_status()
        return {"enable_thinking": self.init_config.get("enable_thinking", False), "thinking_params": {}, "model_type": ""}

    

    @property
    def is_llm_configured(self) -> bool:
        return self._llm_configured

    def get_config_status(self) -> Dict[str, Any]:
        return {
            "llm_configured": self._llm_configured,
            "init_config": {k: "***" if k == "api_key" and v else v
                            for k, v in self.init_config.items()}
        }

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _build_graph(self) -> StateGraph:
        pass
