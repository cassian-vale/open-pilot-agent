# coding=utf-8
import asyncio
import os
import sys
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union
from contextlib import asynccontextmanager

import uvicorn
from loguru import logger
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 添加项目根目录到系统路径
dir_name = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(dir_name))


from applications.translation.translation_agent import TranslationAgent
from utils.log_util import logger_pool
from utils.http_factory import GlobalHTTPFactory


# ===== 请求/响应模型 =====
class TranslationRequest(BaseModel):
    request_id: str = Field(..., description="请求ID，用于追踪")
    text: str = Field(..., description="需要翻译的原始文本")
    
    # --- 必填参数 (修改处) ---
    model: str = Field(..., description="模型名称 (必填)")
    base_url: str = Field(..., description="API基础URL (必填)")
    api_key: str = Field(..., description="API密钥 (必填)")
    # -----------------------

    # 翻译特定参数
    translation_direction: str = Field(..., description="翻译方向 (例如: '中译英', '英译中')")
    translation_style: str = Field(default="普通", description="翻译风格 (例如: '普通', '政务', '小红书', '学术', '科幻', '贴吧', '剧本')")

    # 流式控制参数
    stream: bool = Field(default=False, description="是否启用流式输出")
    
    # LLM 配置参数
    max_tokens: Optional[int] = Field(default=None, description="最大token数")
    temperature: float = Field(default=0.3, description="温度参数")
    top_p: float = Field(default=1.0, description="Top-p参数")
    timeout: float = Field(default=60.0, description="超时时间")
    max_retries: int = Field(default=3, description="最大重试次数")
    enable_thinking: bool = Field(default=False, description="是否启用思考过程")


class TranslationResponse(BaseModel):
    output: Dict[str, Any] = Field(description="业务结果，包含 translated_text, quality_score 等")
    content: str = Field(default="", description="模型最终输出")
    reasoning_content: str = Field(default="", description="思考过程")
    metadata: Dict[str, Any] = Field(default=None, description="元数据")
    confidence: float = Field(default=0.0, description="整体置信度", ge=0, le=1)


# ===== 生命周期管理 =====
agent_instance: Optional[TranslationAgent] = None
app_logger = logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    print("🔧 正在初始化 TranslationAgent...")
    try:
        app_name = "translation"
        # 初始化日志配置
        logger_pool.set_logger(
            name=app_name,
            log_level=os.getenv("TR_LOG_LEVEL", "INFO"),
            log_dir=os.getenv("TR_LOG_DIR", ""),
            retention=os.getenv("TR_LOG_RETENTION", ""),
            rotation=os.getenv("TR_LOG_ROTATION", ""),
        )
        app_logger = logger_pool.get_logger(app_name)
        
        # 2. 初始化 Agent (保留默认参数配置)
        agent_instance = TranslationAgent(
            name="translation",
            model=os.getenv("TR_MODEL", "deepseek-chat"),
            base_url=os.getenv("TR_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("TR_API_KEY", ""), # 保留默认读取
            timeout=float(os.getenv("TR_TIMEOUT", "60.0")),
            max_retries=int(os.getenv("TR_MAX_RETRIES", "3")),
            max_tokens=int(os.getenv("TR_MAX_TOKENS", "0")) or None,
            temperature=float(os.getenv("TR_TEMPERATURE", "0.3")),
            top_p=float(os.getenv("TR_TOP_P", "1.0")),
            stream=bool(os.getenv("TR_STREAM", "False")),
            enable_thinking=bool(os.getenv("TR_ENABLE_THINKING", "False")),
        )
        app_logger.info("🚀 TranslationAgent 初始化完成")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        raise

    yield

    # 关闭时清理
    print("🧹 清理资源...")
    await GlobalHTTPFactory.close()
    agent_instance = None


# ===== FastAPI App =====
app = FastAPI(
    title="文本翻译服务 API",
    description="基于 LangGraph + LLM 的文本翻译服务，支持多风格和质量评估",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/text_translation/v1"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 健康检查接口 =====
@app.get("/health", summary="健康检查")
async def health_check():
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    return {"status": "OK", "agent": "initialized"}


# ===== 统一翻译接口 (合并流式与非流式) =====
@app.post("/chat", response_model=Union[TranslationResponse, str], summary="文本翻译（自动识别流式/非流式）")
async def chat_endpoint(request_body: TranslationRequest, raw_request: Request):
    """
    统一文本翻译接口：
    - 如果 request_body.stream == True: 返回 SSE 流 (text/event-stream)
    - 如果 request_body.stream == False: 返回 JSON (application/json)
    均支持客户端断开连接时自动中断后端推理。
    
    根据指定的方向和风格翻译文本，并返回质量评分。
    """
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="服务未就绪，请稍后再试")

    # 构建运行时参数
    run_config = {
        "request_id": request_body.request_id,
        # 必填项
        "model": request_body.model,
        "base_url": request_body.base_url,
        "api_key": request_body.api_key, 

        # 可选项
        "max_tokens": request_body.max_tokens,
        "temperature": request_body.temperature,
        "top_p": request_body.top_p,
        "timeout": request_body.timeout,
        "max_retries": request_body.max_retries,
        "stream": True,  # 注意：强制开启底层流式
        "enable_thinking": request_body.enable_thinking,
    }
    
    # 过滤掉None值
    run_config = {k: v for k, v in run_config.items() if v is not None}

    # === 分支 1：流式响应 (SSE) ===
    if request_body.stream:
        async def generate_sse():
            try:
                # 1. 发送开始事件
                start_event = {
                    "type": "start",
                    "content": "",
                    "metadata": {"request_id": request_body.request_id, "status": "started"}
                }
                yield f"data: {json.dumps(start_event, ensure_ascii=False)}\n\n"
                
                # 2. 循环生成内容
                async for chunk in agent_instance.run_stream(
                    text=request_body.text,
                    translation_direction=request_body.translation_direction,
                    translation_style=request_body.translation_style,
                    **run_config
                ):
                    # 💡 实时检测中断
                    if await raw_request.is_disconnected():
                        app_logger.warning(f"🚫 request_id: {request_body.request_id} [Stream] 客户端断开连接")
                        break
                    
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    
                # 3. 发送结束事件
                end_event = {
                    "type": "end", 
                    "content": "",
                    "metadata": {"request_id": request_body.request_id, "status": "completed"}
                }
                yield f"data: {json.dumps(end_event, ensure_ascii=False)}\n\n"
                
            except asyncio.CancelledError:
                app_logger.warning(f"🚫 request_id: {request_body.request_id} [Stream] 任务被系统取消")
                raise  # 重新抛出以确保资源清理
            except Exception as e:
                app_logger.error(f"流式处理错误: {traceback.format_exc()}")
                error_event = {"type": "error", "content": f"处理错误: {str(e)}"}
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    # === 分支 2：非流式响应 (JSON) ===
    else:
        try:
            final_response = dict()
            
            # 同样调用 run_stream，但在后端消费掉中间过程
            async for chunk in agent_instance.run_stream(
                text=request_body.text,
                translation_direction=request_body.translation_direction,
                translation_style=request_body.translation_style,
                **run_config
            ):
                # 💡 实时检测中断
                if await raw_request.is_disconnected():
                    app_logger.warning(f"🚫 request_id: {request_body.request_id} [Non-Stream] 客户端断开连接")
                    raise HTTPException(status_code=499, detail="Client Closed Request")
                
                # 只捕捉 final 类型的块
                if chunk.type == "final":
                    final_response = chunk.metadata
            
            return TranslationResponse(**final_response)

        except HTTPException:
            raise
        except asyncio.CancelledError:
            app_logger.warning(f"🚫 request_id: {request_body.request_id} [Non-Stream] 任务被取消")
            raise HTTPException(status_code=499, detail="Request Cancelled")
        except Exception as e:
            app_logger.error(f"非流式处理错误: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")


# ===== 启动命令 =====
if __name__ == "__main__":
    # 使用 8108 端口
    uvicorn.run(
        "translation_app:app", 
        host="0.0.0.0", 
        port=8108,  
        log_level="info"
    )