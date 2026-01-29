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

# 娣诲姞椤圭洰鏍圭洰褰曞埌璺緞
dir_name = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(dir_name))


from applications.summarization.summarization_agent import TextSummarizationAgent
from utils.log_util import logger_pool
from utils.http_factory import GlobalHTTPFactory


# ===== 璇锋眰/鍝嶅簲妯″瀷 =====
class SummarizationRequest(BaseModel):
    request_id: str = Field(..., description="璇锋眰ID锛岀敤浜庤拷韪?)
    text: str = Field(..., description="闇€瑕佹憳瑕佺殑鍘熷鏂囨湰")
    
    # --- 蹇呭～鍙傛暟 (淇敼鐐? ---
    model: str = Field(..., description="妯″瀷鍚嶇О (蹇呭～)")
    base_url: str = Field(..., description="API鍩虹URL (蹇呭～)")
    api_key: str = Field(..., description="API瀵嗛挜 (蹇呭～)")
    # -----------------------

    target_words: Optional[int] = Field(default=None, description="鐩爣瀛楁暟锛孨one琛ㄧず涓嶉檺鍒跺瓧鏁?)
    summary_type: str = Field(default="瑕佺偣鎽樿", description="鎽樿绫诲瀷 ('瑕佺偣鎽樿', '娈佃惤鎽樿', '鏂伴椈鎽樿', '鎶€鏈憳瑕?, '浼氳鎽樿', '瀛︽湳鎽樿', '鏁呬簨鎽樿')")
    ratio: float = Field(default=1.5, description="瀛楁暟璋冩暣姣斾緥锛屼粎鍦ㄩ檺鍒跺瓧鏁版ā寮忎笅鏈夋晥锛岀敤浜庡唴閮ㄨ皟鏁碙LM杈撳嚭瀛楁暟")
    
    # 娴佸紡鎺у埗鍙傛暟
    stream: bool = Field(default=False, description="鏄惁鍚敤娴佸紡杈撳嚭")
    # LLM 閰嶇疆鍙傛暟
    max_tokens: Optional[int] = Field(default=None, description="鏈€澶oken鏁?)
    temperature: float = Field(default=0.3, description="娓╁害鍙傛暟")
    top_p: float = Field(default=1.0, description="Top-p鍙傛暟")
    timeout: float = Field(default=60.0, description="瓒呮椂鏃堕棿")
    max_retries: int = Field(default=3, description="鏈€澶ч噸璇曟鏁?)
    enable_thinking: bool = Field(default=False, description="鏄惁鍚敤鎬濊€冭繃绋?)


class SummarizationResponse(BaseModel):
    output: Dict[str, Any]
    content: str = Field(default="", description="妯″瀷鏈€缁堣緭鍑?)
    reasoning_content: str = Field(default="", description="鎬濊€冭繃绋?)
    metadata: Dict[str, Any] = Field(default=None, description="鍏冩暟鎹?)
    confidence: float = Field(default=1.0, description="鏁翠綋缃俊搴?, ge=0, le=1)


# ===== 鐢熷懡鍛ㄦ湡绠＄悊 =====
agent_instance: Optional[TextSummarizationAgent] = None
app_logger = logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    print("馃敡 姝ｅ湪鍒濆鍖?TextSummarizationAgent...")
    try:
        app_name = "text_summarization"
        logger_pool.set_logger(
            name=app_name,
            log_level=os.getenv("TS_LOG_LEVEL", "INFO"),
            log_dir=os.getenv("TS_LOG_DIR", ""),
            retention=os.getenv("TS_LOG_RETENTION", ""),
            rotation=os.getenv("TS_LOG_ROTATION", ""),
        )
        app_logger = logger_pool.get_logger(app_name)

        # 2. 鍒濆鍖?Agent (淇濈暀榛樿鍙傛暟閰嶇疆)
        agent_instance = TextSummarizationAgent(
            name="textSummarization",
            model=os.getenv("TS_MODEL", "deepseek-chat"),
            base_url=os.getenv("TS_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("TS_API_KEY", ""), # 淇濈暀榛樿璇诲彇
            timeout=float(os.getenv("TS_TIMEOUT", "60.0")),
            max_retries=int(os.getenv("TS_MAX_RETRIES", "3")),
            max_tokens=int(os.getenv("TS_MAX_TOKENS", "0")) or None,
            temperature=float(os.getenv("TS_TEMPERATURE", "0.3")),
            top_p=float(os.getenv("TS_TOP_P", "1.0")),
            stream=bool(os.getenv("TS_STREAM", "False")),
            enable_thinking=bool(os.getenv("TS_ENABLE_THINKING", "False")),
            max_chunk_length=int(os.getenv("TS_MAX_CHUNK_LENGTH", "1000")),
        )
        app_logger.info("鉁?TextSummarizationAgent 鍒濆鍖栧畬鎴?)
    except Exception as e:
        print(f"鉂?鍒濆鍖栧け璐? {e}")
        raise

    yield

    # 鍏抽棴鏃舵竻鐞?
    print("馃Ч 娓呯悊璧勬簮...")
    await GlobalHTTPFactory.close()
    agent_instance = None


# ===== FastAPI App =====
app = FastAPI(
    title="鏂囨湰鎽樿鏈嶅姟 API",
    description="鍩轰簬 LangGraph + LLM 鐨勬枃鏈憳瑕佹湇鍔★紝鏀寔澶氱鎽樿绫诲瀷鍜屽瓧鏁伴檺鍒?,
    version="1.0.0",
    lifespan=lifespan,
    root_path="/text_summarization/v1"
)

# 娣诲姞 CORS 涓棿浠?
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 鍋ュ悍妫€鏌ユ帴鍙?=====
@app.get("/health", summary="鍋ュ悍妫€鏌?)
async def health_check():
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="Agent 鏈垵濮嬪寲")
    return {"status": "OK", "agent": "initialized"}


# ===== 缁熶竴鎽樿鎺ュ彛 (鍚堝苟娴佸紡涓庨潪娴佸紡) =====
@app.post("/chat", response_model=Union[SummarizationResponse, str], summary="鏂囨湰鎽樿锛堣嚜鍔ㄨ瘑鍒祦寮?闈炴祦寮忥級")
async def chat_endpoint(request_body: SummarizationRequest, raw_request: Request):
    """
    缁熶竴鏂囨湰鎽樿鎺ュ彛锛?
    - 濡傛灉 request_body.stream == True: 杩斿洖 SSE 娴?(text/event-stream)
    - 濡傛灉 request_body.stream == False: 杩斿洖 JSON (application/json)
    鍧囨敮鎸佸鎴风鏂紑杩炴帴鏃惰嚜鍔ㄤ腑鏂悗绔帹鐞嗐€?
    
    鏍规嵁鐢ㄦ埛鎻愪緵鐨勬枃鏈拰闇€姹傦紝鐢熸垚鎸囧畾绫诲瀷鍜屽瓧鏁伴檺鍒剁殑鎽樿銆?
    """
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="鏈嶅姟鏈氨缁紝璇风◢鍚庡啀璇?)

    # 鏋勫缓杩愯鏃跺弬鏁?
    run_config = {
        "request_id": request_body.request_id,
        # 蹇呭～椤?
        "model": request_body.model,
        "base_url": request_body.base_url,
        "api_key": request_body.api_key,

        # 鍙€夐」
        "max_tokens": request_body.max_tokens,
        "temperature": request_body.temperature,
        "top_p": request_body.top_p,
        "timeout": request_body.timeout,
        "max_retries": request_body.max_retries,
        "stream": True,  # 鈽?寮哄埗寮€鍚簳灞傛祦寮?
        "enable_thinking": request_body.enable_thinking,
    }
    
    # 杩囨护鎺塏one鍊?
    run_config = {k: v for k, v in run_config.items() if v is not None}

    # === 鍒嗘敮 1锛氭祦寮忓搷搴?(SSE) ===
    if request_body.stream:
        async def generate_sse():
            try:
                # 1. 鍙戦€佸紑濮嬩簨浠?
                start_event = {
                    "type": "start",
                    "content": "",
                    "metadata": {"request_id": request_body.request_id, "status": "started"}
                }
                yield f"data: {json.dumps(start_event, ensure_ascii=False)}\n\n"
                
                # 2. 寰幆鐢熸垚鍐呭
                async for chunk in agent_instance.run_stream(
                    text=request_body.text,
                    target_words=request_body.target_words,
                    summary_type=request_body.summary_type,
                    ratio=request_body.ratio,
                    **run_config
                ):
                    # 鈽?瀹炴椂妫€娴嬩腑鏂?
                    if await raw_request.is_disconnected():
                        app_logger.warning(f"馃毇 request_id: {request_body.request_id} [Stream] 瀹㈡埛绔柇寮€杩炴帴")
                        break
                    
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    
                # 3. 鍙戦€佺粨鏉熶簨浠?
                end_event = {
                    "type": "end", 
                    "content": "",
                    "metadata": {"request_id": request_body.request_id, "status": "completed"}
                }
                yield f"data: {json.dumps(end_event, ensure_ascii=False)}\n\n"
                
            except asyncio.CancelledError:
                app_logger.warning(f"馃毇 request_id: {request_body.request_id} [Stream] 浠诲姟琚郴缁熷彇娑?)
                raise  # 閲嶆柊鎶涘嚭浠ョ‘淇濊祫婧愭竻鐞?
            except Exception as e:
                app_logger.error(f"娴佸紡澶勭悊閿欒: {traceback.format_exc()}")
                error_event = {"type": "error", "content": f"澶勭悊閿欒: {str(e)}"}
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

    # === 鍒嗘敮 2锛氶潪娴佸紡鍝嶅簲 (JSON) ===
    else:
        try:
            final_response = dict()
            
            # 鍚屾牱璋冪敤 run_stream锛屼絾鍦ㄥ悗绔秷璐规帀涓棿杩囩▼
            async for chunk in agent_instance.run_stream(
                text=request_body.text,
                target_words=request_body.target_words,
                summary_type=request_body.summary_type,
                ratio=request_body.ratio,
                **run_config
            ):
                # 鈽?瀹炴椂妫€娴嬩腑鏂?
                if await raw_request.is_disconnected():
                    app_logger.warning(f"馃毇 request_id: {request_body.request_id} [Non-Stream] 瀹㈡埛绔柇寮€杩炴帴")
                    raise HTTPException(status_code=499, detail="Client Closed Request")
                
                # 鍙崟鑾?final 绫诲瀷鐨勫潡
                if chunk.type == "final":
                    final_response = chunk.metadata
            
            return SummarizationResponse(**final_response)

        except HTTPException:
            raise
        except asyncio.CancelledError:
            app_logger.warning(f"馃毇 request_id: {request_body.request_id} [Non-Stream] 浠诲姟琚彇娑?)
            raise HTTPException(status_code=499, detail="Request Cancelled")
        except Exception as e:
            app_logger.error(f"闈炴祦寮忓鐞嗛敊璇? {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"鍐呴儴閿欒: {str(e)}")


# ===== 鍚姩鍛戒护 =====
if __name__ == "__main__":
    uvicorn.run(
        "summarization_app:app", 
        host="0.0.0.0", 
        port=8105, 
        log_level="info"
    )
