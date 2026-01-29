import asyncio
import httpx
from typing import Optional


class GlobalHTTPFactory:
    _async_client: Optional[httpx.AsyncClient] = None
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_async_http_client(cls) -> httpx.AsyncClient:
        """获取全局共享的 HTTP 客户端 (实现连接池复用)"""
        if cls._async_client is None:
            async with cls._lock:
                if cls._async_client is None:
                    # 初始化一个高并发能力的 httpx 客户端
                    cls._async_client = httpx.AsyncClient(
                        limits=httpx.Limits(
                            max_keepalive_connections=10, # 保持较多长连接
                            max_connections=50,          # 物理最大连接数
                            keepalive_expiry=120.0         # 保持时间
                        ),
                        timeout=httpx.Timeout(60.0, connect=5.0),
                        http2=True
                    )
        return cls._async_client

    @classmethod
    async def close(cls):
        """系统关闭时调用"""
        if cls._async_client:
            await cls._async_client.aclose()
            cls._async_client = None
