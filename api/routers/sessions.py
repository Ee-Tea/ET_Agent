"""
세션 관리 API 라우터
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Body, Request
import logging
from datetime import datetime
import hashlib, uuid

from ..models import (
    SessionRequest, SessionResponse, SessionListResponse,
    ChatHistoryResponse, ChatHistoryItem
)
from api.services.hybrid_session_service import HybridSessionService
import os

router = APIRouter(prefix="/sessions", tags=["sessions"])
logger = logging.getLogger("sessions_router")

# 전역 서비스 (api/main.py에서 쓰는 것과 분리 사용시에는 별도로 초기화 필요)
_redis_url = os.getenv("REDIS_URL", "redis://localhost:6380")
_postgres_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@langgraph-postgres:5432/postgres")
_hybrid = HybridSessionService(_redis_url, _postgres_url)


# 임시 세션 저장소 (실제로는 Redis나 DB 사용)
sessions_db: Dict[str, Dict[str, Any]] = {}

@router.post("/", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """새 세션 생성: 요구사항에 맞게 chat_id 증가, title은 최초 메시지 저장 시 설정"""
    try:
        # DB 준비
        await _hybrid.init_postgres()

        user_id = request.user_id or "guest_anon"
        # chat_id 자동 증가 규칙
        chat_id = request.chat_id or await _hybrid.get_next_chat_id(user_id)

        # 숏텀(Shared/History) 포함하여 세션 초기화 + 메타데이터 업서트
        await _hybrid.start_session(
            user_id,
            chat_id,
            {"service_type": request.service_type or "teacher"},
            title=None,
        )

        # 생성된 session_id/제목/시간을 DB에서 확인 후 응답 구성
        async with _hybrid.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT session_id, title, created_at,
                       COALESCE((session_data->>'service_type'),'teacher') AS service_type
                FROM chat_sessions
                WHERE user_id = $1 AND chat_id = $2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                user_id,
                chat_id,
            )
        session_id = row["session_id"] if row else await _hybrid.get_or_create_session_id(user_id, chat_id, request.service_type)
        title = row["title"] if row else None
        created_at = (row["created_at"].isoformat() if hasattr(row["created_at"], 'isoformat') else str(row["created_at"])) if row else datetime.now().isoformat()
        service_type = row["service_type"] if row else (request.service_type or "teacher")

        # 인메모리 캐시(선택)
        sessions_db[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "title": title,
            "created_at": created_at,
            "status": "active",
            "service_type": service_type,
            "message_count": 0,
        }

        return SessionResponse(
            session_id=session_id,
            user_id=user_id,
            chat_id=str(chat_id),
            title=title,
            created_at=created_at,
            status="active",
            service_type=service_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@router.get("/", response_model=SessionListResponse)
async def list_sessions(
    request: Request,
    user_id: str | None = Query(None, description="사용자 ID (미지정 시 헤더/쿠키에서 추출)"),
    limit: int = Query(50, description="최대 결과 수"),
    offset: int = Query(0, description="오프셋")
):
    """사용자의 세션 목록 조회 (오래된 것 하단 노출 요구에 맞게 created_at 오름차순 반환)"""
    try:
        # 보조 추출: 헤더/쿠키
        if not user_id:
            user_id = request.headers.get("x-user-id") or request.cookies.get("user_id")
        if not user_id:
            return SessionListResponse(sessions=[], total=0)
        await _hybrid.init_postgres()
        async with _hybrid.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, user_id, chat_id, title, created_at, 'active' AS status,
                       COALESCE((session_data->>'service_type'),'teacher') AS service_type
                FROM chat_sessions
                WHERE user_id = $1
                ORDER BY created_at ASC
                LIMIT $2 OFFSET $3
                """,
                user_id, limit, offset,
            )
            try:
                logger.info("list_sessions uid=%s limit=%s offset=%s -> %d rows", user_id, limit, offset, len(rows))
            except Exception:
                pass
            # 세션ID 보정: 빈 값/콜론(:) 포함/24-hex 아님 → 해시 session_id로 마이그레이션
            def _is_valid_sid(s: str | None) -> bool:
                if not s or not isinstance(s, str):
                    return False
                if ":" in s:
                    return False
                if len(s) != 24:
                    return False
                try:
                    int(s, 16)
                    return True
                except Exception:
                    return False

            normalized: list[SessionResponse] = []
            invalid: list[Dict[str, Any]] = []
            for r in rows:
                sid = r["session_id"]
                if not _is_valid_sid(sid):
                    invalid.append({
                        "user_id": r["user_id"],
                        "chat_id": str(r["chat_id"]),
                        "session_id": sid,
                    })
                    continue
                # 메시지 존재 여부 조회(경량화)
                has_messages = False
                try:
                    cnt = await conn.fetchval(
                        "SELECT 1 FROM chat_messages WHERE session_id=$1 LIMIT 1",
                        sid,
                    )
                    has_messages = bool(cnt is not None)
                except Exception:
                    has_messages = False
                normalized.append(
                    SessionResponse(
                        session_id=sid,
                        user_id=r["user_id"],
                        chat_id=str(r["chat_id"]),
                        title=r["title"],
                        created_at=r["created_at"].isoformat() if hasattr(r["created_at"], 'isoformat') else str(r["created_at"]),
                        status=r["status"],
                        service_type=r["service_type"],
                        has_messages=has_messages,
                    )
                )

            if invalid:
                try:
                    logger.warning("list_sessions: filtered invalid sessions count=%d examples=%s", len(invalid), invalid[:3])
                except Exception:
                    pass

            sessions = normalized
            total_row = await conn.fetchrow("SELECT COUNT(*) AS c FROM chat_sessions WHERE user_id=$1", user_id)
            return SessionListResponse(sessions=sessions, total=int(total_row["c"]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session listing failed: {str(e)}")

@router.post("/migrate")
async def migrate_user_sessions(user_id: str = Query(...)):
    """한 번만 호출해도 되는 유지보수용: 해당 사용자의 구형 session_id를 해시로 일괄 교체"""
    try:
        await _hybrid.init_postgres()
        async with _hybrid.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, chat_id
                FROM chat_sessions
                WHERE user_id = $1
                """,
                user_id,
            )
            def _is_valid_sid(s: str | None) -> bool:
                if not s or not isinstance(s, str):
                    return False
                if ":" in s:
                    return False
                if len(s) != 24:
                    return False
                try:
                    int(s, 16)
                    return True
                except Exception:
                    return False
            migrated = 0
            for r in rows:
                sid = r["session_id"]
                if _is_valid_sid(sid):
                    continue
                raw = f"{user_id}:{r['chat_id']}:{datetime.utcnow().isoformat()}:{uuid.uuid4()}"
                new_sid = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
                await conn.execute(
                    """
                    UPDATE chat_sessions
                    SET session_id = $3, updated_at = NOW(), is_active = TRUE
                    WHERE user_id = $1 AND chat_id = $2 AND session_id = $4
                    """,
                    user_id,
                    str(r["chat_id"]),
                    new_sid,
                    sid,
                )
                migrated += 1
            return {"user_id": user_id, "migrated": migrated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")

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

@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = Query(1000)):
    """세션 메시지 조회 (오름차순)"""
    try:
        await _hybrid.init_postgres()
        messages = await _hybrid.get_messages_by_session_id(session_id, limit=limit)
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Messages retrieval failed: {str(e)}")

@router.post("/{session_id}/messages")
async def add_session_message(
    session_id: str,
    content: str = Body(..., embed=True),
    role: str = Body(..., embed=True),
):
    """세션에 메시지 저장: role=user|assistant"""
    try:
        await _hybrid.init_postgres()
        await _hybrid.add_message_by_session_id(session_id, role, content)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Message save failed: {str(e)}")

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


