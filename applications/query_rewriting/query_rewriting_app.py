# query_rewriting_app.py
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
from pydantic import BaseModel, Field

# 添加项目根目录到路径
dir_name = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(dir_name))


from applications.query_rewriting.query_rewriting_agent import QueryRewriteAgent
from utils.log_util import logger_pool
from utils.http_factory import GlobalHTTPFactory


# ===== 请求/响应模型 =====
class QueryRewriteRequest(BaseModel):
    request_id: str
    query: str = Field(..., description="需要改写的查询")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default=None, description="对话历史")
    domain_context: Optional[str] = Field(default=None, description="领域上下文信�?)
    max_rewrites: Optional[int] = Field(default=None, description="最大改写数�?)
    preserve_system: bool = Field(default=True, description="是否保留系统消息")
    
    # --- 必填参数 (修改�? ---
    model: str = Field(..., description="模型名称 (必填)")
    base_url: str = Field(..., description="API基础URL (必填)")
    api_key: str = Field(..., description="API密钥 (必填)")
    # -----------------------

    # 流式控制参数
    stream: bool = Field(default=False, description="是否启用流式输出")
    # LLM 配置参数
    max_tokens: Optional[int] = Field(default=None, description="最大token�?)
    temperature: float = Field(default=0.3, description="温度参数")
    top_p: float = Field(default=1.0, description="Top-p参数")
    timeout: float = Field(default=60.0, description="超时时间")
    max_retries: int = Field(default=3, description="最大重试次�?)
    enable_thinking: bool = Field(default=False, description="是否启用思考过�?)


class QueryRewriteResponse(BaseModel):
    output: Dict[str, Any]
    content: str = Field(default="", description="模型最终输�?)
    reasoning_content: str = Field(default="", description="思考过�?)
    metadata: Dict[str, Any] = Field(default=None, description="元数�?)
    confidence: float = Field(default=1.0, description="整体置信�?, ge=0, le=1)


# ===== 生命周期管理 =====
agent_instance: Optional[QueryRewriteAgent] = None
app_logger = logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    print("🔧 正在初始�?QueryRewriteAgent...")
    try:
        # 使用默认配置初始�?
        app_name = "query_rewriting"
        logger_pool.set_logger(
            name=app_name,
            log_level=os.getenv("QR_LOG_LEVEL", "INFO"),
            log_dir=os.getenv("QR_LOG_DIR", ""),
            retention=os.getenv("QR_LOG_RETENTION", ""),
            rotation=os.getenv("QR_LOG_ROTATION", ""),
        )
        app_logger = logger_pool.get_logger(app_name)

        agent_instance = QueryRewriteAgent(
            name="queryRewrite",
            model=os.getenv("QR_MODEL", "deepseek-chat"),
            base_url=os.getenv("QR_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("QR_API_KEY", ""),
            timeout=float(os.getenv("QR_TIMEOUT", "60.0")),
            max_retries=int(os.getenv("QR_MAX_RETRIES", "3")),
            max_tokens=int(os.getenv("QR_MAX_TOKENS", "0")) or None,
            temperature=float(os.getenv("QR_TEMPERATURE", "0.3")),
            top_p=float(os.getenv("QR_TOP_P", "1.0")),
            stream=bool(os.getenv("QR_STREAM", "False")),
            enable_thinking=bool(os.getenv("QR_ENABLE_THINKING", "False")),
            default_max_rewrites=int(os.getenv("QR_DEFAULT_MAX_REWRITES", "5")),
        )
        app_logger.info("�?QueryRewriteAgent 初始化完�?)
    except Exception as e:
        print(f"�?初始化失�? {e}")
        raise

    yield

    # 关闭时清�?
    print("🧹 清理资源...")
    await GlobalHTTPFactory.close()
    agent_instance = None


# ===== FastAPI App =====
app = FastAPI(
    title="查询改写服务 API",
    description="基于 LangGraph + LLM 的查询改写服务，支持指代消歧、查询扩写、语义增强等策略，支持流式和非流式输�?,
    version="1.0.0",
    lifespan=lifespan,
    root_path="/query_rewriting/v1"
)

# 添加 CORS 中间�?
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 健康检查接�?=====
@app.get("/health", summary="健康检�?)
async def health_check():
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    return {"status": "OK", "agent": "initialized"}


# ===== 统一改写接口 (合并流式与非流式) =====
@app.post("/chat", response_model=Union[QueryRewriteResponse, str], summary="查询改写（自动识别流�?非流式）")
async def chat_endpoint(request_body: QueryRewriteRequest, raw_request: Request):
    """
    统一查询改写接口�?
    - 如果 request_body.stream == True: 返回 SSE �?(text/event-stream)
    - 如果 request_body.stream == False: 返回 JSON (application/json)
    均支持客户端断开连接时自动中断后端推理�?
    
    支持多种改写策略�?
    - 指代消歧：解析并替换代词，明确指代实�?
    - 查询扩写：添加同义词和相关术�?
    - 查询改写：调整语法结构和表达视角
    - 语义增强：明确隐含上下文信息
    """
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="服务未就绪，请稍后再�?)

    # 构建运行时参�?
    # 注意：这里我们强�?enable stream=True 传给底层 Agent�?
    # 这样底层会按 Token 生成，我们才能在非流式模式下也进行细粒度的中断检测�?
    run_config = {

        "request_id": request_body.request_id,
        # 必填项：使用请求中的参数
        "model": request_body.model,
        "base_url": request_body.base_url,
        "api_key": request_body.api_key,
        # 可选项
        "max_tokens": request_body.max_tokens,
        "temperature": request_body.temperature,
        "top_p": request_body.top_p,
        "timeout": request_body.timeout,
        "max_retries": request_body.max_retries,
        "stream": True,  # �?强制开启底层流式，以便于细粒度控制中断
        "enable_thinking": request_body.enable_thinking,
    }
    
    # 过滤掉None�?
    run_config = {k: v for k, v in run_config.items() if v is not None}

    # === 分支 1：流式响�?(SSE) ===
    if request_body.stream:
        async def generate_sse():
            try:
                # 1. 发送开始事�?
                start_event = {
                    "type": "start",
                    "content": "",
                    "metadata": {"request_id": request_body.request_id, "status": "started"}
                }
                yield f"data: {json.dumps(start_event, ensure_ascii=False)}\n\n"
                
                # 2. 循环生成内容
                async for chunk in agent_instance.run_stream(
                    query=request_body.query,
                    conversation_history=request_body.conversation_history,
                    domain_context=request_body.domain_context,
                    max_rewrites=request_body.max_rewrites,
                    preserve_system=request_body.preserve_system,
                    **run_config
                ):
                    # �?实时检测中�?
                    if await raw_request.is_disconnected():
                        app_logger.warning(f"🚫 request_id: {request_body.request_id} [Stream] 客户端断开连接")
                        break
                    
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    
                # 3. 发送结束事�?
                end_event = {
                    "type": "end", 
                    "content": "",
                    "metadata": {"request_id": request_body.request_id, "status": "completed"}
                }
                yield f"data: {json.dumps(end_event, ensure_ascii=False)}\n\n"
                
            except asyncio.CancelledError:
                app_logger.warning(f"🚫 request_id: {request_body.request_id} [Stream] 任务被系统取�?)
                raise  # 重新抛出以确保资源清�?
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
                query=request_body.query,
                conversation_history=request_body.conversation_history,
                domain_context=request_body.domain_context,
                max_rewrites=request_body.max_rewrites,
                preserve_system=request_body.preserve_system,
                **run_config
            ):
                # �?实时检测中断：即使是非流式，也能在生成过程中被掐断
                if await raw_request.is_disconnected():
                    app_logger.warning(f"🚫 request_id: {request_body.request_id} [Non-Stream] 客户端断开连接")
                    # 这里抛出异常会停�?run_stream 的执�?
                    raise HTTPException(status_code=499, detail="Client Closed Request")
                
                # 只捕�?final 类型的块
                if chunk.type == "final":
                    # metadata 中包含了完整�?QueryRewriteResponse 所需字段
                    final_response = chunk.metadata
            
            return QueryRewriteResponse(**final_response)

        except HTTPException:
            raise
        except asyncio.CancelledError:
            app_logger.warning(f"🚫 request_id: {request_body.request_id} [Non-Stream] 任务被取�?)
            raise HTTPException(status_code=499, detail="Request Cancelled")
        except Exception as e:
            app_logger.error(f"非流式处理错�? {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")


# ===== 启动命令 =====
if __name__ == "__main__":
    uvicorn.run(
        "query_rewriting_app:app", 
        host="0.0.0.0", 
        port=8103,  # 使用不同端口避免冲突
        log_level="info"
    )