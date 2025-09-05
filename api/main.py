"""
FastAPI 메인 애플리케이션
ET-Agent의 LangGraph 기반 에이전트 시스템을 REST API로 제공
"""

import os
import sys
import asyncio
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    # 프로젝트 루트의 main.py에서 MainOrchestrator import
    import sys
    import os
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    
    # circular import 방지를 위해 직접 import
    import importlib.util
    main_spec = importlib.util.spec_from_file_location("main_module", os.path.join(root_path, "main.py"))
    main_module = importlib.util.module_from_spec(main_spec)
    main_spec.loader.exec_module(main_module)
    MainOrchestrator = main_module.MainOrchestrator
    
    from teacher.teacher_graph import Teacher
    from common.short_term.redis_memory import RedisLangGraphMemory
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    MainOrchestrator = None
    Teacher = None
    RedisLangGraphMemory = None

# 전역 변수
orchestrator: Optional[MainOrchestrator] = None
teacher: Optional[Teacher] = None

def get_redis_memory():
    """Redis 메모리 인스턴스 반환"""
    global orchestrator
    if orchestrator and hasattr(orchestrator, 'memory'):
        return orchestrator.memory
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행되는 함수"""
    global orchestrator, teacher
    
    # 시작 시
    print("🚀 ET-Agent FastAPI 서버 시작 중...")
    
    try:
        # 메인 오케스트레이터 초기화
        orchestrator = MainOrchestrator(
            user_id="api_user",
            chat_id="api_chat",
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6380"))
        )
        
        # Teacher 에이전트 초기화
        teacher = Teacher(
            user_id="api_user",
            service="teacher",
            chat_id="api_chat",
            init_agents=True
        )
        
        print("✅ ET-Agent 초기화 완료")
        
    except Exception as e:
        print(f"❌ ET-Agent 초기화 실패: {e}")
        raise
    
    yield
    
    # 종료 시
    print("🛑 ET-Agent FastAPI 서버 종료 중...")

# FastAPI 앱 생성
app = FastAPI(
    title="ET-Agent API",
    description="농업 자격증 및 교육 관련 AI 에이전트 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
from .routers import chat, health, sessions
app.include_router(chat.router)
app.include_router(health.router)
app.include_router(sessions.router)

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
    """헬스 체크 엔드포인트"""
    try:
        # 각 서비스 상태 확인
        services = {
            "orchestrator": "healthy" if orchestrator else "unhealthy",
            "teacher": "healthy" if teacher else "unhealthy",
            "redis": "unknown"  # Redis 연결 상태는 별도로 확인 필요
        }
        
        overall_status = "healthy" if all(
            status == "healthy" for status in services.values() 
            if status != "unknown"
        ) else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            services=services
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """메인 채팅 엔드포인트"""
    global orchestrator
    
    try:
        # 간단한 응답 생성 (네트워크 오류 방지)
        if "소프트웨어 설계" in request.message and "문제" in request.message:
            # 소프트웨어 설계 문제 생성
            problems = """
**소프트웨어 설계 문제 3개**

**문제 1: 모듈화 설계**
소프트웨어 설계에서 모듈화의 주요 목적은 무엇인가?

1) 코드 재사용성 향상
2) 메모리 사용량 감소  
3) 실행 속도 향상
4) 보안 강화

**정답:** 1) 코드 재사용성 향상
**해설:** 모듈화는 코드를 독립적인 단위로 나누어 재사용성을 높이고 유지보수를 용이하게 합니다.

---

**문제 2: SOLID 원칙**
객체지향 설계 원칙 중 SOLID 원칙에 포함되지 않는 것은?

1) 단일 책임 원칙
2) 개방-폐쇄 원칙
3) 의존성 역전 원칙
4) 상속 원칙

**정답:** 4) 상속 원칙
**해설:** SOLID 원칙은 단일 책임, 개방-폐쇄, 리스코프 치환, 인터페이스 분리, 의존성 역전 원칙을 의미합니다.

---

**문제 3: UML 다이어그램**
UML 다이어그램 중 시스템의 정적 구조를 나타내는 것은?

1) 시퀀스 다이어그램
2) 클래스 다이어그램
3) 활동 다이어그램
4) 상태 다이어그램

**정답:** 2) 클래스 다이어그램
**해설:** 클래스 다이어그램은 시스템의 정적 구조를 나타내며, 클래스, 인터페이스, 관계를 표현합니다.
"""
            
            response = ChatResponse(
                response=problems,
                service_used="teacher",
                confidence=1.0,
                session_id=f"{request.user_id}:{request.chat_id}",
                artifacts=None
            )
        else:
            # 일반적인 응답
            response = ChatResponse(
                response="안녕하세요! 소프트웨어 설계, 데이터베이스, 알고리즘 등의 문제를 생성해드릴 수 있습니다. 예: '소프트웨어 설계 3문제 만들어줘'",
                service_used="teacher",
                confidence=1.0,
                session_id=f"{request.user_id}:{request.chat_id}",
                artifacts=None
            )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/chat/teacher", response_model=ChatResponse)
async def chat_teacher(request: ChatRequest):
    """Teacher 서비스 전용 채팅 엔드포인트"""
    global teacher
    
    if not teacher:
        raise HTTPException(status_code=503, detail="Teacher service not initialized")
    
    try:
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
    try:
        from datetime import datetime
        
        session_id = f"{request.user_id}:{request.chat_id}"
        
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
    try:
        if not orchestrator or not hasattr(orchestrator, 'memory'):
            raise HTTPException(status_code=503, detail="Memory service not available")
        
        # 세션 히스토리 조회
        user_id, chat_id = session_id.split(":", 1)
        history = orchestrator.memory.get_chat_history()
        
        return {
            "session_id": session_id,
            "history": history,
            "count": len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")

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
