# coding=utf-8
import asyncio
import os
import sys
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from contextlib import asynccontextmanager

import uvicorn
from loguru import logger
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

# 添加项目根目录到系统路径
dir_name = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(dir_name))


from applications.text_classification.text_classification_agent import TextClassificationAgent
from utils.log_util import logger_pool
from utils.http_factory import GlobalHTTPFactory


# ===== 请求/响应模型 =====
class TextClassificationRequest(BaseModel):
    request_id: str = Field(..., description="请求ID，用于追踪")
    text: str = Field(..., description="需要分类的原始文本")
    candidate_labels: List[str] = Field(..., description="候选标签列表", min_length=2, max_length=20)
    
    # --- 必填参数 (修改后) ---
    model: str = Field(..., description="模型名称 (必填)")
    base_url: str = Field(..., description="API基础URL (必填)")
    api_key: str = Field(..., description="API密钥 (必填)")
    # -----------------------

    # 流式控制参数
    stream: bool = Field(default=False, description="是否启用流式输出")
    
    # LLM 配置参数
    max_tokens: Optional[int] = Field(default=None, description="最大token数 (分类任务通常只需很少的token)")
    temperature: float = Field(default=0.1, description="温度参数 (分类任务建议低温)")
    top_p: float = Field(default=1.0, description="Top-p参数")
    timeout: float = Field(default=60.0, description="超时时间")
    max_retries: int = Field(default=3, description="最大重试次数")
    enable_thinking: bool = Field(default=False, description="是否启用思考过程")

    @field_validator('candidate_labels')
    def validate_labels(cls, v):
        if len(v) < 2:
            raise ValueError('至少需要提供两个候选标签')
        return v


class TextClassificationResponse(BaseModel):
    output: Dict[str, Any] = Field(description="业务结果，包含predicted_label, scores等")
    content: str = Field(default="", description="模型最终输出")
    reasoning_content: str = Field(default="", description="思考过程")
    metadata: Dict[str, Any] = Field(default=None, description="元数据，如token消耗")
    confidence: float = Field(default=0.0, description="整体置信度", ge=0, le=1)


# ===== 生命周期管理 =====
agent_instance: Optional[TextClassificationAgent] = None
app_logger = logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    print("🔧 正在初始化 TextClassificationAgent...")
    try:
        app_name = "text_classification"
        # 初始化日志配置
        logger_pool.set_logger(
            name=app_name,
            log_level=os.getenv("TC_LOG_LEVEL", "INFO"),
            log_dir=os.getenv("TC_LOG_DIR", ""),
            retention=os.getenv("TC_LOG_RETENTION", ""),
            rotation=os.getenv("TC_LOG_ROTATION", ""),
        )
        app_logger = logger_pool.get_logger(app_name)
        
        # 2. 初始化 Agent (保留默认参数配置)
        agent_instance = TextClassificationAgent(
            name="textClassification",
            model=os.getenv("TC_MODEL", "deepseek-chat"),
            base_url=os.getenv("TC_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("TC_API_KEY", ""), # 保留默认读取
            timeout=float(os.getenv("TC_TIMEOUT", "60.0")),
            max_retries=int(os.getenv("TC_MAX_RETRIES", "3")),
            # 分类任务通常max_tokens很小，如果没有设置环境变量，默认为10够用了
            max_tokens=int(os.getenv("TC_MAX_TOKENS", "0")) or 10, 
            temperature=float(os.getenv("TC_TEMPERATURE", "0.1")),
            top_p=float(os.getenv("TC_TOP_P", "1.0")),
            stream=bool(os.getenv("TC_STREAM", "False")),
            enable_thinking=bool(os.getenv("TC_ENABLE_THINKING", "False")),
        )
        app_logger.info("✅ TextClassificationAgent 初始化完成")
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
    title="文本分类服务 API",
    description="基于 LangGraph + LLM 的文本分类服务，支持自定义标签和置信度输出",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/text_classification/v1"
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


# ===== 统一分类接口 (合并流式与非流式) =====
@app.post("/chat", response_model=Union[TextClassificationResponse, str], summary="文本分类（自动识别流式/非流式）")
async def chat_endpoint(request_body: TextClassificationRequest, raw_request: Request):
    """
    统一文本分类接口：
    - 如果 request_body.stream == True: 返回 SSE 流 (text/event-stream)
    - 如果 request_body.stream == False: 返回 JSON (application/json)
    均支持客户端断开连接时自动中断后端推理。
    
    根据用户提供的文本和候选标签，进行分类预测。
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
        "stream": True,  # ⚠️ 强制开启底层流式
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
                    candidate_labels=request_body.candidate_labels,
                    **run_config
                ):
                    # 🔍 实时检测中断
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
                candidate_labels=request_body.candidate_labels,
                **run_config
            ):
                # 🔍 实时检测中断
                if await raw_request.is_disconnected():
                    app_logger.warning(f"🚫 request_id: {request_body.request_id} [Non-Stream] 客户端断开连接")
                    raise HTTPException(status_code=499, detail="Client Closed Request")
                
                # 只捕获 final 类型的块
                if chunk.type == "final":
                    final_response = chunk.metadata
            
            return TextClassificationResponse(**final_response)

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
    uvicorn.run(
        "text_classification_app:app", 
        host="0.0.0.0", 
        port=8106, 
        log_level="info"
    )
