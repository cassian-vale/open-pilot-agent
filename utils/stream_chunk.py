# coding: utf-8
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


# ===== 流式输出块 =====
class StreamChunk(BaseModel):
    type: str = Field(description="块类型: start|processing|thinking|content|final|end|error")
    content: str = Field(description="内容")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")
