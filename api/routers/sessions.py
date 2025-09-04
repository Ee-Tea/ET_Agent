"""
세션 관리 API 라우터
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime

from ..models import (
    SessionRequest, SessionResponse, SessionListResponse,
    ChatHistoryResponse, ChatHistoryItem
)

router = APIRouter(prefix="/sessions", tags=["sessions"])

# 임시 세션 저장소 (실제로는 Redis나 DB 사용)
sessions_db: Dict[str, Dict[str, Any]] = {}

@router.post("/", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """새 세션 생성"""
    try:
        # 채팅 ID가 없으면 자동 생성
        if not request.chat_id:
            import uuid
            request.chat_id = str(uuid.uuid4())[:8]
        
        session_id = f"{request.user_id}:{request.chat_id}"
        
        # 세션 정보 저장
        session_data = {
            "session_id": session_id,
            "user_id": request.user_id,
            "chat_id": request.chat_id,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "service_type": request.service_type,
            "message_count": 0
        }
        
        sessions_db[session_id] = session_data
        
        return SessionResponse(**session_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@router.get("/", response_model=SessionListResponse)
async def list_sessions(
    user_id: str = Query(..., description="사용자 ID"),
    limit: int = Query(10, description="최대 결과 수"),
    offset: int = Query(0, description="오프셋")
):
    """사용자의 세션 목록 조회"""
    try:
        # 사용자 세션 필터링
        user_sessions = [
            session for session in sessions_db.values()
            if session["user_id"] == user_id
        ]
        
        # 정렬 (최신순)
        user_sessions.sort(key=lambda x: x["created_at"], reverse=True)
        
        # 페이징
        total = len(user_sessions)
        sessions = user_sessions[offset:offset + limit]
        
        return SessionListResponse(
            sessions=[SessionResponse(**session) for session in sessions],
            total=total
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session listing failed: {str(e)}")

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """특정 세션 조회"""
    try:
        if session_id not in sessions_db:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = sessions_db[session_id]
        return SessionResponse(**session_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session retrieval failed: {str(e)}")

@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    try:
        if session_id not in sessions_db:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 세션 삭제
        del sessions_db[session_id]
        
        # 실제로는 Redis나 DB에서도 삭제해야 함
        # if orchestrator and hasattr(orchestrator, 'memory'):
        #     orchestrator.memory.delete_session(session_id)
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session deletion failed: {str(e)}")

@router.get("/{session_id}/history", response_model=ChatHistoryResponse)
async def get_session_history(
    session_id: str,
    limit: int = Query(50, description="최대 메시지 수"),
    offset: int = Query(0, description="오프셋")
):
    """세션 히스토리 조회"""
    try:
        if session_id not in sessions_db:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 실제로는 orchestrator의 memory에서 히스토리를 가져와야 함
        # 임시로 빈 히스토리 반환
        history = []
        
        # 예시 히스토리 데이터
        if session_id in sessions_db:
            session_data = sessions_db[session_id]
            if session_data.get("message_count", 0) > 0:
                history = [
                    ChatHistoryItem(
                        timestamp=session_data["created_at"],
                        user_message="안녕하세요",
                        bot_response="안녕하세요! 무엇을 도와드릴까요?",
                        service_used="teacher",
                        confidence=1.0
                    )
                ]
        
        return ChatHistoryResponse(
            session_id=session_id,
            history=history[offset:offset + limit],
            total_count=len(history)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")

@router.put("/{session_id}/status")
async def update_session_status(
    session_id: str,
    status: str = Query(..., description="새로운 상태 (active/inactive/archived)")
):
    """세션 상태 업데이트"""
    try:
        if session_id not in sessions_db:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if status not in ["active", "inactive", "archived"]:
            raise HTTPException(status_code=400, detail="Invalid status")
        
        sessions_db[session_id]["status"] = status
        sessions_db[session_id]["updated_at"] = datetime.now().isoformat()
        
        return {"message": f"Session {session_id} status updated to {status}"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status update failed: {str(e)}")

@router.get("/{session_id}/stats")
async def get_session_stats(session_id: str):
    """세션 통계 조회"""
    try:
        if session_id not in sessions_db:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = sessions_db[session_id]
        
        # 실제로는 더 상세한 통계를 계산해야 함
        stats = {
            "session_id": session_id,
            "message_count": session_data.get("message_count", 0),
            "created_at": session_data["created_at"],
            "last_activity": session_data.get("updated_at", session_data["created_at"]),
            "status": session_data["status"],
            "service_type": session_data["service_type"]
        }
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

