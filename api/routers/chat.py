"""
채팅 관련 API 라우터
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
from ..services.hybrid_session_service import HybridSessionService

from ..models import (
    ChatRequest, ChatResponse, 
    TeacherRequest, TeacherResponse,
    FarmerRequest, FarmerResponse
)

router = APIRouter(prefix="/chat", tags=["chat"])

# 전역 변수 (실제로는 의존성 주입으로 처리해야 함)
orchestrator = None
teacher = None
hybrid_session_service = None

def set_services(orch, teach, session_service=None):
    """서비스 인스턴스 설정"""
    global orchestrator, teacher, hybrid_session_service
    orchestrator = orch
    teacher = teach
    hybrid_session_service = session_service

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """메인 채팅 엔드포인트 - 자동 서비스 분류"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        # 오케스트레이터 실행
        result = orchestrator.run(
            user_query=request.message,
            config={
                "configurable": {
                    "thread_id": f"supervisor:{request.user_id}:{request.chat_id}"
                }
            }
        )
        
        # 응답 구성
        response = ChatResponse(
            response=result.get("final_response", "응답을 생성할 수 없습니다."),
            service_used=result.get("service_classification", "unknown"),
            confidence=result.get("classification_confidence", 0.0),
            session_id=f"{request.user_id}:{request.chat_id}",
            artifacts=result.get("artifacts")
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@router.post("/teacher", response_model=TeacherResponse)
async def chat_teacher(request: TeacherRequest):
    """Teacher 서비스 전용 채팅 엔드포인트"""
    global teacher
    
    if not teacher:
        raise HTTPException(status_code=503, detail="Teacher service not initialized")
    
    try:
        # Teacher 상태 초기화
        teacher_state = {
            "user_query": request.message,
            "intent": request.intent or "",
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
        response = TeacherResponse(
            response=result.get("llm_response", "응답을 생성할 수 없습니다."),
            intent=result.get("intent", "unknown"),
            artifacts=result.get("artifacts"),
            session_id=f"{request.user_id}:{request.chat_id}",
            confidence=1.0
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Teacher chat processing failed: {str(e)}")

@router.post("/farmer", response_model=FarmerResponse)
async def chat_farmer(request: FarmerRequest):
    """Farmer 서비스 전용 채팅 엔드포인트"""
    try:
        # Farmer 서비스는 현재 기본 구현
        # 실제로는 farmer 모듈을 import해서 사용해야 함
        
        response_text = f"""
🌱 Farmer 서비스 응답

질문: {request.message}
작물 종류: {request.crop_type or "미지정"}
지역: {request.region or "미지정"}

현재 Farmer 서비스는 개발 중입니다.
농업 관련 질문에 대한 기본 응답을 제공합니다.

더 자세한 농업 정보가 필요하시면 구체적인 질문을 해주세요.
        """.strip()
        
        response = FarmerResponse(
            response=response_text,
            recommendations=[
                "적절한 작물 선택",
                "시기별 재배 관리",
                "병해충 방제",
                "수확 및 저장 방법"
            ],
            session_id=f"{request.user_id}:{request.chat_id}",
            confidence=0.8
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Farmer chat processing failed: {str(e)}")

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """스트리밍 채팅 엔드포인트 (Server-Sent Events)"""
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate_response():
        """스트리밍 응답 생성"""
        try:
            if not orchestrator:
                yield f"data: {json.dumps({'error': 'Orchestrator not initialized'})}\n\n"
                return
            
            # 오케스트레이터 실행
            result = orchestrator.run(
                user_query=request.message,
                config={
                    "configurable": {
                        "thread_id": f"supervisor:{request.user_id}_{request.chat_id}",
                    }
                }
            )
            
            # 스트리밍 응답 전송
            response_data = {
                "response": result.get("final_response", "응답을 생성할 수 없습니다."),
                "service_used": result.get("service_classification", "unknown"),
                "confidence": result.get("classification_confidence", 0.0),
                "session_id": f"{request.user_id}:{request.chat_id}",
                "artifacts": result.get("artifacts"),
                "timestamp": datetime.now().isoformat()
            }
            
            yield f"data: {json.dumps(response_data)}\n\n"
            
        except Exception as e:
            error_data = {
                "error": f"Streaming failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@router.post("/clear")
async def clear_session(request: ChatRequest):
    """세션과 서비스 락 초기화"""
    try:
        from ..main import get_redis_memory
        from common.short_term.redis_memory import RedisLangGraphMemory
        
        # Redis 메모리 인스턴스 가져오기
        redis_memory = get_redis_memory()
        
        if not redis_memory:
            # 직접 Redis 메모리 인스턴스 생성
            redis_memory = RedisLangGraphMemory(
                user_id=request.user_id,
                service="teacher",
                chat_id=request.chat_id,
                redis_host="localhost",
                redis_port=6380
            )
        
        # 완전한 세션 초기화 실행
        redis_memory.clear_all_session_data()
        
        return {
            "success": True,
            "message": "✅ 세션과 서비스 락이 성공적으로 초기화되었습니다.",
            "cleared_items": [
                "세션 메모리 (shared, history)",
                "문제 데이터",
                "숏텀 메모리",
                "채팅 히스토리",
                "서비스 락",
                "세션 관련 모든 키"
            ],
            "session_id": f"{request.user_id}:{request.chat_id}"
        }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"❌ 초기화 중 오류가 발생했습니다: {str(e)}"
        }


@router.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    """사용자의 모든 채팅 세션 조회"""
    global hybrid_session_service
    
    if not hybrid_session_service:
        raise HTTPException(status_code=503, detail="Session service not initialized")
    
    sessions = await hybrid_session_service.get_user_sessions(user_id)
    return {"sessions": sessions}

@router.post("/sessions/{user_id}/{chat_id}/save")
async def save_session(user_id: str, chat_id: str, session_data: dict):
    """채팅 세션 저장"""
    global hybrid_session_service
    
    if not hybrid_session_service:
        raise HTTPException(status_code=503, detail="Session service not initialized")
    
    await hybrid_session_service.save_session_metadata(user_id, chat_id, session_data)
    return {"status": "saved"}

@router.post("/sessions/{user_id}/{chat_id}/start")
async def start_session(user_id: str, chat_id: str, initial_data: dict = None):
    """새 세션 시작"""
    global hybrid_session_service
    
    if not hybrid_session_service:
        raise HTTPException(status_code=503, detail="Session service not initialized")
    
    await hybrid_session_service.start_session(user_id, chat_id, initial_data)
    return {"status": "started", "session_id": f"{user_id}:{chat_id}"}

@router.get("/sessions/{user_id}/{chat_id}/messages")
async def get_session_messages(user_id: str, chat_id: str, limit: int = 100):
    """특정 세션의 메시지 조회"""
    global hybrid_session_service
    
    if not hybrid_session_service:
        raise HTTPException(status_code=503, detail="Session service not initialized")
    
    messages = await hybrid_session_service.get_session_messages(user_id, chat_id, limit)
    return {"messages": messages}

@router.post("/sessions/{user_id}/{chat_id}/archive")
async def archive_session(user_id: str, chat_id: str):
    """세션 아카이브 (Redis → PostgreSQL)"""
    global hybrid_session_service
    
    if not hybrid_session_service:
        raise HTTPException(status_code=503, detail="Session service not initialized")
    
    await hybrid_session_service.archive_session(user_id, chat_id)
    return {"status": "archived", "session_id": f"{user_id}:{chat_id}"}