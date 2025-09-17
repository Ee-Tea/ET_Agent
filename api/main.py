"""
FastAPI 메인 애플리케이션
ET-Agent의 LangGraph 기반 에이전트 시스템을 REST API로 제공
"""

import os
import json
import sys
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi import Response
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
import httpx
from api.services.hybrid_session_service import HybridSessionService
from api.clients.auth_client import verify_token
# supervisor import는 런타임에 처리

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://localhost:8123")

# try:
#     # 프로젝트 루트의 main.py에서 MainOrchestrator import
#     import sys
#     import os
#     root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     if root_path not in sys.path:
#         sys.path.insert(0, root_path)
    
#     # circular import 방지를 위해 직접 import
#     import importlib.util
#     main_spec = importlib.util.spec_from_file_location("main_module", os.path.join(root_path, "main.py"))
#     main_module = importlib.util.module_from_spec(main_spec)
#     main_spec.loader.exec_module(main_module)
#     MainOrchestrator = main_module.MainOrchestrator
    
#     from teacher.teacher_graph import Teacher
#     from common.short_term.redis_memory import RedisLangGraphMemory
# except ImportError as e:
#     print(f"Warning: Some imports failed: {e}")
#     MainOrchestrator = None
#     Teacher = None
#     RedisLangGraphMemory = None

# # 전역 변수
# orchestrator: Optional[MainOrchestrator] = None
# teacher: Optional[Teacher] = None

orchestrator = None
teacher = None
hybrid_session_service = None

# 로거 설정
logger = logging.getLogger("et_agent_bff")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# langgraph-api 호출 함수 추가
async def call_langgraph_api(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """langgraph-api 호출 - 올바른 LangGraph API 사용법 적용"""
    async with httpx.AsyncClient() as client:
        try:
            if endpoint == "/chat":
                # /chat 엔드포인트 대신 올바른 LangGraph API 사용
                return await call_langgraph_chat(data)
            else:
                # 다른 엔드포인트는 기존 방식 유지
                response = await client.post(
                    f"{LANGGRAPH_API_URL}{endpoint}",
                    json=data,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"LangGraph API 연결 실패: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"LangGraph API 오류: {e.response.text}")

async def call_langgraph_chat(data: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph API를 사용한 채팅 처리 - 올바른 API 사용법"""
    # 채팅 관련 통신은 10분(600초) 타임아웃
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            # 1. 어시스턴트 조회/생성
            assistant_id = await get_or_create_assistant(client)
            
            # 2. 스레드 생성
            thread_id = await create_thread(client, data["user_id"], data["chat_id"])
            
            # 3. 스레드에서 실행 (스트림 방식)
            result = await run_thread(client, assistant_id, thread_id, data["message"])
            
            return {
                "response": result.get("response", "응답을 생성할 수 없습니다."),
                "service_used": "langgraph",
                "confidence": 1.0,
                "session_id": f"{data['user_id']}:{data['chat_id']}",
                "artifacts": result.get("artifacts", {})
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"채팅 처리 실패: {str(e)}")

async def get_or_create_assistant(client: httpx.AsyncClient) -> str:
    """어시스턴트 조회 또는 생성"""
    try:
        # 기존 어시스턴트 조회
        response = await client.post(
            f"{LANGGRAPH_API_URL}/assistants/search",
            json={"graph_id": "agent", "limit": 1},
            timeout=600.0
        )
        
        if response.status_code == 200:
            assistants = response.json()
            if assistants:
                return assistants[0]["assistant_id"]
        
        # 어시스턴트가 없으면 생성
        response = await client.post(
            f"{LANGGRAPH_API_URL}/assistants",
            json={
                "graph_id": "agent",
                "name": "ET-Agent",
                "description": "농업 자격증 및 교육 관련 AI 에이전트"
            },
            timeout=600.0
        )
        response.raise_for_status()
        assistant = response.json()
        return assistant["assistant_id"]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"어시스턴트 처리 실패: {str(e)}")

async def create_thread(client: httpx.AsyncClient, user_id: str, chat_id: str) -> str:
    """스레드 생성"""
    try:
        response = await client.post(
            f"{LANGGRAPH_API_URL}/threads",
            json={
                "thread_id": f"{user_id}_{chat_id}",
                "metadata": {
                    "user_id": user_id,
                    "chat_id": chat_id
                }
            },
            timeout=600.0
        )
        response.raise_for_status()
        thread = response.json()
        return thread["thread_id"]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스레드 생성 실패: {str(e)}")

async def run_thread(client: httpx.AsyncClient, assistant_id: str, thread_id: str, message: str) -> Dict[str, Any]:
    """스레드에서 실행"""
    try:
        response = await client.post(
            f"{LANGGRAPH_API_URL}/threads/{thread_id}/runs/wait",
            json={
                "assistant_id": assistant_id,
                "input": {
                    "message": message,
                    "user_id": thread_id.split("_")[0],
                    "chat_id": thread_id.split("_")[1]
                }
            },
            timeout=600.0
        )
        response.raise_for_status()
        result = response.json()
        
        # 결과에서 응답 추출
        response_text = ""
        if "values" in result and "messages" in result["values"]:
            messages = result["values"]["messages"]
            if messages and len(messages) > 0:
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    response_text = last_message["content"]
                elif isinstance(last_message, str):
                    response_text = last_message
        
        return {
            "response": response_text or "응답을 생성할 수 없습니다.",
            "artifacts": result.get("artifacts", {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스레드 실행 실패: {str(e)}")


def get_redis_memory():
    """Redis 메모리 인스턴스 반환"""
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행되는 함수"""
    global orchestrator, teacher, hybrid_session_service
    
    # 시작 시
    print("🚀 ET-Agent BFF API 서버 시작 중...")
    
    # 하이브리드 세션 서비스 초기화
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6380")
        # 컨테이너 내부에서는 서비스명:5432 사용
        postgres_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@langgraph-postgres:5432/postgres")
        
        hybrid_session_service = HybridSessionService(redis_url, postgres_url)
        await hybrid_session_service.init_postgres()
        print("✅ 하이브리드 세션 서비스 초기화 완료")
        
    except Exception as e:
        print(f"⚠️ 하이브리드 세션 서비스 초기화 실패: {e}")
        hybrid_session_service = None
    
    # langgraph-api 연결 테스트
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{LANGGRAPH_API_URL}/health", timeout=5.0)
            if response.status_code == 200:
                print("✅ LangGraph API 연결 성공")
            else:
                print("⚠️ LangGraph API 연결 실패")
    except Exception as e:
        print(f"⚠️ LangGraph API 연결 테스트 실패: {e}")
    
    yield
    
    # 종료 시
    print("🛑 ET-Agent BFF API 서버 종료 중...")
    
    # 하이브리드 세션 서비스 정리
    if hybrid_session_service and hasattr(hybrid_session_service, 'pool'):
        await hybrid_session_service.pool.close()
        print("✅ 하이브리드 세션 서비스 정리 완료")

# FastAPI 앱 생성
app = FastAPI(
    title="ET-Agent API",
    description="농업 자격증 및 교육 관련 AI 에이전트 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 미들웨어 설정 (credentials 사용 시 * 금지 → 환경변수 기반 화이트리스트)
def _parse_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "")
    if not raw:
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://172.29.208.1:3000",
            "http://192.168.0.199:3000",
            "http://localhost",
        ]
    try:
        import json
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    # comma-separated
    return [v.strip() for v in raw.split(",") if v.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 라우터 활성화 (프론트 요구: /api/chat/sessions)
try:
    from api.routers.sessions import router as sessions_router
    app.include_router(sessions_router, prefix="/api/chat")
except Exception as e:
    logger.warning(f"sessions router include skipped: {e}")

# ========== 요청 로깅 미들웨어 ==========
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
    method = request.method
    path = request.url.path
    user_id_hdr = request.headers.get("x-user-id") or request.cookies.get("user_id")
    chat_id_hdr = request.headers.get("x-chat-id") or request.cookies.get("chat_id")
    logger.info(f"[{req_id}] -> {method} {path} uid={user_id_hdr} cid={chat_id_hdr}")
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.info(f"[{req_id}] <- {response.status_code} {method} {path} {duration_ms:.1f}ms")
        response.headers["x-request-id"] = req_id
        return response
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.exception(f"[{req_id}] !! {method} {path} {duration_ms:.1f}ms error={e}")
        raise

from fastapi import Header

@app.get("/me")
async def me(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing/invalid token")
    token = authorization.replace("Bearer ", "")
    data = await verify_token(token)
    return {"user": data}

# ========== Pydantic 모델 정의 ==========

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str = Field(..., description="사용자 메시지", min_length=1, max_length=2000)
    user_id: str = Field(default="api_user", description="사용자 ID")
    chat_id: str = Field(default="api_chat", description="채팅 ID")
    service_type: Optional[str] = Field(default=None, description="서비스 타입 (teacher/farmer)")

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    response: str = Field(..., description="에이전트 응답")
    service_used: str = Field(..., description="사용된 서비스")
    confidence: float = Field(..., description="응답 신뢰도")
    session_id: str = Field(..., description="세션 ID")
    artifacts: Optional[Dict[str, Any]] = Field(default=None, description="생성된 파일들")

class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str = Field(..., description="서비스 상태")
    version: str = Field(..., description="API 버전")
    services: Dict[str, str] = Field(..., description="서비스별 상태")

class SessionRequest(BaseModel):
    """세션 관리 요청 모델"""
    user_id: str = Field(..., description="사용자 ID")
    chat_id: str = Field(..., description="채팅 ID")

class SessionResponse(BaseModel):
    """세션 관리 응답 모델"""
    session_id: str = Field(..., description="세션 ID")
    created_at: str = Field(..., description="생성 시간")
    status: str = Field(..., description="세션 상태")

class ClearRequest(BaseModel):
    """세션 정리 요청 모델"""
    user_id: str = Field(..., description="사용자 ID")
    chat_id: str = Field(..., description="채팅 ID")

# ========== API 엔드포인트 ==========

@app.get("/", response_model=Dict[str, str])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "ET-Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크 엔드포인트 (루터와 정합 유지)"""
    try:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{LANGGRAPH_API_URL}/docs", timeout=5.0)
                langgraph_status = "healthy" if response.status_code == 200 else "unhealthy"
        except Exception:
            langgraph_status = "unreachable"

        services = {
            "bff_api": "healthy",
            "langgraph_api": langgraph_status,
            "redis": "unknown"
        }

        return HealthResponse(
            status="healthy" if langgraph_status == "healthy" else "unhealthy",
            version="1.0.0",
            services=services
        )
    except Exception as e:
        logger.exception(f"/health failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, response: Response):
    """메인 채팅 엔드포인트 - 직접 LangGraph 실행"""
    global orchestrator, hybrid_session_service
    
    try:
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
        # 프록시(프론트) 경유 시 바디가 비거나 포맷이 깨질 수 있어 관대한 파싱 적용
        try:
            payload = await request.json()
        except Exception:
            raw = await request.body()
            payload = {}
            if raw:
                try:
                    payload = json.loads(raw.decode("utf-8", "ignore"))
                except Exception:
                    try:
                        form = await request.form()
                        payload = dict(form)
                    except Exception:
                        payload = {}
        if not isinstance(payload, dict):
            payload = {}

        # 관대한 파싱 + 헤더/쿠키 보완
        msg = str(payload.get("message", "")) if isinstance(payload, dict) else ""
        user_id = str(payload.get("user_id", "")) if isinstance(payload, dict) else ""
        chat_id = str(payload.get("chat_id", "")) if isinstance(payload, dict) else ""
        session_id_in = str(payload.get("session_id", "")) if isinstance(payload, dict) else ""

        if not user_id:
            user_id = request.headers.get("x-user-id") or request.cookies.get("user_id") or os.getenv("DEFAULT_USER_ID", "")
        # 미로그인 게스트 처리
        if not user_id:
            import uuid as _uuid
            user_id = f"guest_{str(_uuid.uuid4())[:8]}"

        # 세션/채팅 식별자 보강
        if not chat_id:
            chat_id = request.headers.get("x-chat-id") or request.cookies.get("chat_id") or ""
        if not session_id_in:
            session_id_in = request.headers.get("x-session-id") or request.cookies.get("session_id") or ""
        # 쿠키/헤더로 받은 chat_id가 숫자가 아니면 무시
        if chat_id and not chat_id.isdigit():
            chat_id = ""
        if hybrid_session_service:
            try:
                await hybrid_session_service.init_postgres()
                # session_id로 chat_id 역참조
                if session_id_in:
                    row = await hybrid_session_service.get_session_by_session_id(session_id_in)
                    if row and str(row.get("user_id")) == str(user_id):
                        mapped_chat = str(row.get("chat_id") or "")
                        # 매핑값이 숫자가 아니면 자가 치유: 새 번호로 갱신
                        if mapped_chat and not mapped_chat.isdigit():
                            try:
                                new_chat = await hybrid_session_service.get_next_chat_id(user_id)
                                async with hybrid_session_service.pool.acquire() as conn:
                                    await conn.execute(
                                        "UPDATE chat_sessions SET chat_id=$1, updated_at=NOW() WHERE session_id=$2",
                                        new_chat,
                                        session_id_in,
                                    )
                                mapped_chat = new_chat
                            except Exception:
                                mapped_chat = ""
                        if not chat_id and mapped_chat:
                            chat_id = mapped_chat
                # 없으면 신규 채번
                if not chat_id:
                    chat_id = await hybrid_session_service.get_next_chat_id(user_id)
                # 세션 보장: 해시 session_id 생성/유지 및 획득
                ensured_sid = await hybrid_session_service.get_or_create_session_id(
                    user_id,
                    chat_id,
                    service_type="general",
                    title=None,
                )
                session_id_in = ensured_sid or session_id_in
            except Exception:
                pass
        if not msg or not msg.strip():
            msg = os.getenv("DEFAULT_CHAT_PROMPT", "안녕하세요")

        logger.info(f"[{req_id}] /chat uid={user_id} cid={chat_id} msg_len={len(msg)}")

        # 사용자 메시지를 하이브리드 서비스에 저장 (최초 user 메시지 시 title 자동 설정)
        if hybrid_session_service:
            msg_id = await hybrid_session_service.add_chat_message(
                user_id,
                chat_id,
                "user",
                msg,
            )
        
        # 직접 LangGraph 실행
        try:
            # 동적 import로 순환 의존 회피
            from supervisor import MainOrchestrator
            local_orchestrator = MainOrchestrator(user_id, chat_id, hybrid_session_service)
            response_text = local_orchestrator.process_query(msg)
        except ImportError as e:
            logger.exception(f"[{req_id}] MainOrchestrator import failed: {e}")
            raise HTTPException(status_code=500, detail=f"MainOrchestrator import failed: {e}")
        
        # 결과 포맷팅
        result = {
            "response": response_text,
            "service_used": "orchestrator",
            "confidence": 1.0,
            "session_id": f"{user_id}:{chat_id}",
            "artifacts": None
        }

        logger.info(f"[{req_id}] /chat done uid={user_id} cid={chat_id} resp_len={len(str(result.get('response') or ''))}")
        
        # 에이전트 응답 저장
        if hybrid_session_service:
            _aid = await hybrid_session_service.add_chat_message(
                user_id,
                chat_id,
                "assistant",
                result.get("response", "응답을 생성할 수 없습니다."),
                {
                    "service_used": result.get("service_used", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "artifacts": result.get("artifacts"),
                },
            )
        
        # 클라이언트가 세션을 안정적으로 바인딩할 수 있도록 헤더로도 전달
        try:
            if session_id_in:
                response.headers["x-session-id"] = session_id_in
        except Exception:
            pass

        return ChatResponse(
            response=result.get("response", "응답을 생성할 수 없습니다."),
            service_used=result.get("service_used", "unknown"),
            confidence=result.get("confidence", 0.0),
            session_id=session_id_in or result.get("session_id", f"{user_id}:{chat_id}"),
            artifacts=result.get("artifacts")
        )
        
    except Exception as e:
        logger.exception(f"/chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")



#     global orchestrator
    
#     try:
#         # 간단한 응답 생성 (네트워크 오류 방지)
#         if "소프트웨어 설계" in request.message and "문제" in request.message:
#             # 소프트웨어 설계 문제 생성
#             problems = """
# **소프트웨어 설계 문제 3개**

# **문제 1: 모듈화 설계**
# 소프트웨어 설계에서 모듈화의 주요 목적은 무엇인가?

# 1) 코드 재사용성 향상
# 2) 메모리 사용량 감소  
# 3) 실행 속도 향상
# 4) 보안 강화

# **정답:** 1) 코드 재사용성 향상
# **해설:** 모듈화는 코드를 독립적인 단위로 나누어 재사용성을 높이고 유지보수를 용이하게 합니다.

# ---

# **문제 2: SOLID 원칙**
# 객체지향 설계 원칙 중 SOLID 원칙에 포함되지 않는 것은?

# 1) 단일 책임 원칙
# 2) 개방-폐쇄 원칙
# 3) 의존성 역전 원칙
# 4) 상속 원칙

# **정답:** 4) 상속 원칙
# **해설:** SOLID 원칙은 단일 책임, 개방-폐쇄, 리스코프 치환, 인터페이스 분리, 의존성 역전 원칙을 의미합니다.

# ---

# **문제 3: UML 다이어그램**
# UML 다이어그램 중 시스템의 정적 구조를 나타내는 것은?

# 1) 시퀀스 다이어그램
# 2) 클래스 다이어그램
# 3) 활동 다이어그램
# 4) 상태 다이어그램

# **정답:** 2) 클래스 다이어그램
# **해설:** 클래스 다이어그램은 시스템의 정적 구조를 나타내며, 클래스, 인터페이스, 관계를 표현합니다.
# """
            
#             response = ChatResponse(
#                 response=problems,
#                 service_used="teacher",
#                 confidence=1.0,
#                 session_id=f"{request.user_id}:{request.chat_id}",
#                 artifacts=None
#             )
#         else:
#             # 일반적인 응답
#             response = ChatResponse(
#                 response="안녕하세요! 소프트웨어 설계, 데이터베이스, 알고리즘 등의 문제를 생성해드릴 수 있습니다. 예: '소프트웨어 설계 3문제 만들어줘'",
#                 service_used="teacher",
#                 confidence=1.0,
#                 session_id=f"{request.user_id}:{request.chat_id}",
#                 artifacts=None
#             )
        
#         return response
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/chat/teacher", response_model=ChatResponse)
async def chat_teacher(request: ChatRequest):
    """Teacher 서비스 전용 채팅 엔드포인트"""
    global teacher, hybrid_session_service
    
    if not teacher:
        raise HTTPException(status_code=503, detail="Teacher service not initialized")
    
    try:
        # 사용자 메시지를 하이브리드 서비스에 저장
        if hybrid_session_service:
            await hybrid_session_service.add_chat_message(
                request.user_id, 
                request.chat_id, 
                "user", 
                request.message
            )
        
        # Teacher 상태 초기화
        teacher_state = {
            "user_query": request.message,
            "intent": "",
            "shared": {},
            "work": {},
            "retrieval": {},
            "generation": {},
            "solution": {},
            "score": {},
            "analysis": {},
            "history": [],
            "session": {},
            "artifacts": {},
            "routing": {},
            "llm_response": ""
        }
        
        # Teacher 실행
        result = teacher.execute(teacher_state)
        
        # 보강: 챗봇 응답 저장 시점에 사용자 메시지도 한 번 더 저장 (중복 방지 메타태그 포함)
        if hybrid_session_service:
            try:
                await hybrid_session_service.add_chat_message(
                    request.user_id,
                    request.chat_id,
                    "user",
                    request.message,
                    {"source": "post_assistant_save"}
                )
            except Exception:
                pass
            
            # 에이전트 응답을 하이브리드 서비스에 저장
            await hybrid_session_service.add_chat_message(
                request.user_id, 
                request.chat_id, 
                "assistant", 
                result.get("llm_response", "응답을 생성할 수 없습니다."),
                {
                    "service_used": "teacher",
                    "confidence": 1.0,
                    "artifacts": result.get("artifacts")
                }
            )
        
        # 응답 구성
        response = ChatResponse(
            response=result.get("llm_response", "응답을 생성할 수 없습니다."),
            service_used="teacher",
            confidence=1.0,
            session_id=f"{request.user_id}:{request.chat_id}",
            artifacts=result.get("artifacts")
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Teacher chat processing failed: {str(e)}")

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """새 세션 생성"""
    global hybrid_session_service
    
    try:
        from datetime import datetime
        
        session_id = f"{request.user_id}:{request.chat_id}"
        
        # 하이브리드 서비스로 세션 시작
        if hybrid_session_service:
            await hybrid_session_service.start_session(
                request.user_id, 
                request.chat_id,
                {"service_type": "general"}
            )
        
        return SessionResponse(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            status="active"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    try:
        # 세션 데이터 삭제 로직 (Redis에서 삭제)
        if orchestrator and hasattr(orchestrator, 'memory'):
            # 세션 관련 데이터 삭제
            pass
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session deletion failed: {str(e)}")

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """세션 히스토리 조회"""
    global hybrid_session_service
    
    try:
        if not hybrid_session_service:
            raise HTTPException(status_code=503, detail="Session service not available")
        
        # 세션 히스토리 조회
        user_id, chat_id = session_id.split(":", 1)
        history = await hybrid_session_service.get_session_messages(user_id, chat_id)
        
        return {
            "session_id": session_id,
            "history": history,
            "count": len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")

@app.post("/clear")
async def clear_session_endpoint(request: Request):
    """세션 관련 캐시/락/히스토리를 정리"""
    global hybrid_session_service
    try:
        if not hybrid_session_service:
            raise HTTPException(status_code=503, detail="Session service not available")

        # 관대한 바디 파싱
        try:
            payload = await request.json()
        except Exception:
            raw = await request.body()
            payload = {}
            if raw:
                try:
                    payload = json.loads(raw.decode("utf-8", "ignore"))
                except Exception:
                    try:
                        form = await request.form()
                        payload = dict(form)
                    except Exception:
                        payload = {}
        if not isinstance(payload, dict):
            payload = {}

        user_id = str(payload.get("user_id", "api_user"))
        chat_id = str(payload.get("chat_id", "api_chat"))
        session_id = str(payload.get("session_id", "")) or request.headers.get("x-session-id") or ""
        session_key = f"{user_id}_{chat_id}"

        redis = hybrid_session_service.redis

        # 삭제 대상 키들 (신규/기존 키 모두)
        keys_to_delete = []
        # 락
        keys_to_delete.append(f"lock:{session_key}")
        if session_id:
            keys_to_delete.append(f"lock:{session_id}")
        # 숏텀/히스토리/세션
        keys_to_delete.append(f"short_term:{session_key}")
        if session_id:
            keys_to_delete.append(f"short_term:{session_id}")
        keys_to_delete.append(f"history:{user_id}:{chat_id}")
        if session_id:
            keys_to_delete.append(f"history:{session_id}")
        keys_to_delete.append(f"active_session:{user_id}:{chat_id}")
        if session_id:
            keys_to_delete.append(f"active_session:{session_id}")
        keys_to_delete.append(f"shared:{user_id}:{chat_id}")
        if session_id:
            keys_to_delete.append(f"shared:{session_id}")
        # 질문들 (패턴)
        q_keys = []
        try:
            q_keys = list(redis.keys(f"questions:{session_key}:*"))
        except Exception:
            q_keys = []
        if session_id:
            try:
                q_keys += list(redis.keys(f"questions:{session_id}:*"))
            except Exception:
                pass

        deleted = 0
        for k in keys_to_delete:
            try:
                deleted += int(redis.delete(k) or 0)
            except Exception:
                pass
        if q_keys:
            try:
                deleted += int(redis.delete(*q_keys) or 0)
            except Exception:
                pass

        return {"message": "cleared", "deleted": deleted, "session": f"{user_id}:{chat_id}", "session_id": session_id or None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

# ========== 추가 유틸 엔드포인트 ==========

@app.get("/recent-questions")
async def get_recent_questions(
    request: Request,
    user_id: Optional[str] = Query(None, description="사용자 ID"),
    chat_id: Optional[str] = Query(None, description="채팅 ID"),
    session_id: Optional[str] = Query(None, description="세션 ID(해시)") ,
    limit: int = Query(10, ge=1, le=100, description="최대 반환 개수"),
):
    global hybrid_session_service
    if not hybrid_session_service:
        # 서비스 준비 전에도 프론트가 동작하도록 빈 결과 반환 (항상 questions 키 사용)
        return {"questions": [], "count": 0, "session": None}

    try:
        # 파라미터 보완: 헤더/쿠키에서 보조 취득
        if not user_id:
            user_id = request.headers.get("x-user-id") or request.cookies.get("user_id")
        if not chat_id:
            chat_id = request.headers.get("x-chat-id") or request.cookies.get("chat_id")
        # session_id 헤더도 수용(신규 키 스킴)
        session_id = request.headers.get("x-session-id") or session_id

        # 보강: chat_id가 없고 session_id가 오면 매핑 조회
        if user_id and not chat_id and session_id:
            try:
                await hybrid_session_service.init_postgres()
                row = await hybrid_session_service.get_session_by_session_id(session_id)
                if row and str(row.get("user_id")) == str(user_id):
                    chat_id = str(row.get("chat_id"))
            except Exception:
                pass

        # 보강: user_id만 있고 chat_id가 없으면 최근 세션 chat_id로 대체
        if user_id and not chat_id:
            try:
                await hybrid_session_service.init_postgres()
                sessions = await hybrid_session_service.get_user_sessions(user_id, limit=1)
                if sessions:
                    chat_id = str(sessions[0].get("chat_id") or "")
            except Exception:
                pass

        # 식별자 없으면 빈 결과 반환 (항상 questions 키 사용)
        if not user_id or not chat_id:
            logger.info(f"/recent-questions missing ids uid={user_id} cid={chat_id}")
            return {"questions": [], "count": 0, "session": None}

        redis = hybrid_session_service.redis
        session_key = f"{user_id}_{chat_id}"
        pattern = f"questions:{session_key}:*"
        # 최근 생성된 문항 수(added_count) 조회
        # 우선순위: session_id 기반(short_term:<sid> → shared:<sid>) → 구키(short_term:uid_chat → shared:uid:chat)
        added_count = 0
        try:
            # 1) 신규 키 스킴: short_term:<session_id>
            if session_id:
                try:
                    st_sid_key = f"short_term:{session_id}"
                    st_sid_raw = redis.get(st_sid_key)
                    if st_sid_raw:
                        st = json.loads(st_sid_raw)
                        teacher_data = st.get("teacher", {})
                        added_count = int(teacher_data.get("added_count", 0) or 0)
                except Exception:
                    added_count = 0
            # 2) 신규 키 스킴: shared:<session_id>
            if added_count <= 0 and session_id:
                try:
                    shared_sid_key = f"shared:{session_id}"
                    shared_sid_raw = redis.get(shared_sid_key)
                    if shared_sid_raw:
                        shared = json.loads(shared_sid_raw)
                        added_count = int(shared.get("added_count", 0) or 0)
                except Exception:
                    added_count = 0
            # 3) 구 키 스킴: short_term:uid_chat
            if added_count <= 0:
                try:
                    st_key = f"short_term:{session_key}"
                    st_raw = redis.get(st_key)
                    if st_raw:
                        st = json.loads(st_raw)
                        teacher_data = st.get("teacher", {})
                        added_count = int(teacher_data.get("added_count", 0) or 0)
                except Exception:
                    added_count = 0
            # 4) 구 키 스킴: shared:uid:chat
            if added_count <= 0:
                try:
                    shared_key = f"shared:{user_id}:{chat_id}"
                    shared_raw = redis.get(shared_key)
                    if shared_raw:
                        shared = json.loads(shared_raw)
                        added_count = int(shared.get("added_count", 0) or 0)
                except Exception:
                    added_count = 0
        except Exception:
            added_count = 0

        # 조회 키 준비: session_id 우선, 없으면 구키 패턴
        try:
            keys = []
            if session_id:
                keys = list(redis.keys(f"questions:{session_id}:*"))
            if not keys:
                keys = list(redis.keys(pattern))
        except Exception:
            keys = []
        keys.sort(reverse=True)

        # added_count 우선, 없으면 실제 키 개수 기반
        if added_count > 0:
            actual_limit = min(limit, added_count)
        else:
            actual_limit = min(limit, len(keys))

        # 최근 생성된 개수가 0이면 빈 결과 반환
        if actual_limit == 0:
            logger.info(f"/recent-questions uid={user_id} cid={chat_id} added_count=0 -> empty")
            return {"questions": [], "count": 0, "session": (session_id or f"{user_id}:{chat_id}")}

        items: list[dict[str, Any]] = []
        for key in keys[:actual_limit]:
            data = redis.hgetall(key) or {}
            if not data:
                continue
            # options JSON 파싱
            options_val = data.get("options", "[]")
            try:
                options = json.loads(options_val) if isinstance(options_val, str) else []
            except Exception:
                options = []

            items.append({
                "qid": key,
                "question": data.get("question", ""),
                "options": options,
                "correctAnswer": data.get("answer", ""),
                "explanation": data.get("explanation", ""),
                "subject": data.get("subject", ""),
            })
        
        logger.info(f"/recent-questions uid={user_id} cid={chat_id} added_count={added_count} keys={len(keys)} items={len(items)}")
        return {"questions": items, "count": len(items), "session": (session_id or f"{user_id}:{chat_id}")}
    except Exception as e:
        logger.exception(f"/recent-questions failed: {e}")
        # 어떤 예외가 나도 프론트는 빈 리스트로 계속 진행 가능해야 함
        return {"questions": [], "count": 0, "session": None}


@app.get("/pdf-status")
async def get_pdf_status(
    request: Request,
    user_id: Optional[str] = Query(None, description="사용자 ID"),
    chat_id: Optional[str] = Query(None, description="채팅 ID"),
):
    global hybrid_session_service
    if not hybrid_session_service:
        # 서비스 준비 전에도 프론트가 동작하도록 idle 반환
        return {"status": "idle", "progress": 0.0, "session": None}

    try:
        # 파라미터 보완: 헤더/쿠키에서 보조 취득
        if not user_id:
            user_id = request.headers.get("x-user-id") or request.cookies.get("user_id")
        if not chat_id:
            chat_id = request.headers.get("x-chat-id") or request.cookies.get("chat_id")

        # 식별자 없으면 idle로 응답 (프론트 422 방지)
        if not user_id or not chat_id:
            logger.info(f"/pdf-status missing ids uid={user_id} cid={chat_id}")
            return {"status": "idle", "progress": 0.0, "session": None}

        redis = hybrid_session_service.redis
        session_key = f"{user_id}:{chat_id}"

        # 가능한 상태 키 후보들을 순차적으로 조회
        status_keys = [
            f"pdf_status:{user_id}:{chat_id}",
            f"pdf:{user_id}_{chat_id}:status",
            f"pdf:status:{user_id}:{chat_id}",
        ]
        progress_keys = [
            f"pdf_progress:{user_id}:{chat_id}",
            f"pdf:{user_id}_{chat_id}:progress",
            f"pdf:progress:{user_id}:{chat_id}",
        ]

        status = None
        for k in status_keys:
            status = redis.get(k)
            if status:
                break
        if isinstance(status, bytes):
            status = status.decode("utf-8", "ignore")
        status = status or "idle"

        progress_val = None
        for k in progress_keys:
            progress_val = redis.get(k)
            if progress_val is not None:
                break
        if isinstance(progress_val, bytes):
            try:
                progress_val = progress_val.decode("utf-8", "ignore")
            except Exception:
                pass
        try:
            progress = float(progress_val) if progress_val is not None else 0.0
        except Exception:
            progress = 0.0

        logger.info(f"/pdf-status uid={user_id} cid={chat_id} status={status} progress={progress}")
        return {"status": status, "progress": progress, "session": session_key}
    except Exception as e:
        logger.exception(f"/pdf-status failed: {e}")
        # 프론트는 폴링 중이므로 200으로 안전하게 대응
        return {"status": "idle", "progress": 0.0, "session": None}

# ========== PDF 파일 엔드포인트 ==========

@app.get("/pdfs")
async def list_pdfs():
    base_dir = os.getenv("PDF_DIR", os.path.join(os.getcwd(), "teacher", "agents", "solution", "pdf_outputs"))
    try:
        if not os.path.isdir(base_dir):
            return {"pdfs": []}
        files = [f for f in os.listdir(base_dir) if f.lower().endswith(".pdf")]
        out = []
        for f in files:
            try:
                p = os.path.join(base_dir, f)
                stat = os.stat(p)
                out.append({
                    "filename": f,
                    "created_at": stat.st_mtime,
                })
            except Exception:
                out.append({"filename": f})
        return {"pdfs": out}
    except Exception:
        return {"pdfs": []}

@app.get("/pdf/{filename}")
async def download_pdf(filename: str):
    base_dir = os.getenv("PDF_DIR", os.path.join(os.getcwd(), "teacher", "agents", "solution", "pdf_outputs"))
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="application/pdf", filename=filename)

# ========== 에러 핸들러 ==========

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 핸들러"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 핸들러"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )

# ========== 개발 서버 실행 ==========

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
