# api/services/hybrid_session_service.py
import redis
import asyncpg
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

class HybridSessionService:
    def __init__(self, redis_url: str, postgres_url: str):
        # Redis: 실시간 세션 관리
        self.redis = redis.from_url(redis_url, decode_responses=True)
        
        # PostgreSQL: 영구 저장소
        self.postgres_url = postgres_url
        self.pool = None
    
    async def init_postgres(self):
        """PostgreSQL 연결 풀 초기화"""
        self.pool = await asyncpg.create_pool(self.postgres_url)
        await self.create_tables_if_not_exists()
    
    async def create_tables_if_not_exists(self):
        """PostgreSQL 테이블 생성"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    session_data JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(user_id, chat_id)
                );
                
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    message_type TEXT NOT NULL, -- 'user', 'assistant', 'system'
                    content TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_chat_messages_user_chat ON chat_messages(user_id, chat_id);
            """)
    
    # ==================== Redis (실시간 세션) ====================
    
    def get_active_session_key(self, user_id: str, chat_id: str) -> str:
        """활성 세션 Redis 키"""
        return f"active_session:{user_id}:{chat_id}"
    
    def get_shared_key(self, user_id: str, chat_id: str) -> str:
        """공유 메모리 Redis 키"""
        return f"shared:{user_id}:{chat_id}"
    
    def get_history_key(self, user_id: str, chat_id: str) -> str:
        """채팅 히스토리 Redis 키"""
        return f"history:{user_id}:{chat_id}"
    
    async def start_session(self, user_id: str, chat_id: str, initial_data: Dict[str, Any] = None):
        """새 세션 시작 (Redis에 저장)"""
        session_key = self.get_active_session_key(user_id, chat_id)
        shared_key = self.get_shared_key(user_id, chat_id)
        history_key = self.get_history_key(user_id, chat_id)
        
        # Redis에 실시간 세션 데이터 저장
        session_data = {
            "user_id": user_id,
            "chat_id": chat_id,
            "started_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "status": "active"
        }
        
        if initial_data:
            session_data.update(initial_data)
        
        # Redis 저장 (TTL: 24시간)
        self.redis.setex(session_key, 86400, json.dumps(session_data))
        
        # 공유 메모리 초기화
        self.redis.setex(shared_key, 86400, json.dumps({
            "question": [],
            "options": [],
            "answer": [],
            "explanation": [],
            "subject": []
        }))
        
        # 히스토리 초기화
        self.redis.setex(history_key, 86400, json.dumps([]))
        
        # PostgreSQL에도 메타데이터 저장
        await self.save_session_metadata(user_id, chat_id, session_data)
    
    async def update_active_session(self, user_id: str, chat_id: str, updates: Dict[str, Any]):
        """활성 세션 업데이트 (Redis)"""
        session_key = self.get_active_session_key(user_id, chat_id)
        
        # 기존 데이터 로드
        existing = self.redis.get(session_key)
        if existing:
            session_data = json.loads(existing)
        else:
            session_data = {"user_id": user_id, "chat_id": chat_id}
        
        # 업데이트
        session_data.update(updates)
        session_data["last_activity"] = datetime.now().isoformat()
        
        # Redis에 저장
        self.redis.setex(session_key, 86400, json.dumps(session_data))
    
    async def get_active_session(self, user_id: str, chat_id: str) -> Optional[Dict[str, Any]]:
        """활성 세션 조회 (Redis)"""
        session_key = self.get_active_session_key(user_id, chat_id)
        data = self.redis.get(session_key)
        return json.loads(data) if data else None
    
    async def save_shared_memory(self, user_id: str, chat_id: str, shared_data: Dict[str, Any]):
        """공유 메모리 저장 (Redis)"""
        shared_key = self.get_shared_key(user_id, chat_id)
        self.redis.setex(shared_key, 86400, json.dumps(shared_data))
    
    async def get_shared_memory(self, user_id: str, chat_id: str) -> Dict[str, Any]:
        """공유 메모리 조회 (Redis)"""
        shared_key = self.get_shared_key(user_id, chat_id)
        data = self.redis.get(shared_key)
        return json.loads(data) if data else {}
    
    async def add_chat_message(self, user_id: str, chat_id: str, message_type: str, content: str, metadata: Dict[str, Any] = None):
        """채팅 메시지 추가 (Redis + PostgreSQL)"""
        message = {
            "type": message_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Redis에 실시간 히스토리 추가 (최근 50개만)
        history_key = self.get_history_key(user_id, chat_id)
        history = self.redis.get(history_key)
        if history:
            history_list = json.loads(history)
        else:
            history_list = []
        
        history_list.append(message)
        # 최근 50개만 유지
        if len(history_list) > 50:
            history_list = history_list[-50:]
        
        self.redis.setex(history_key, 86400, json.dumps(history_list))
        
        # PostgreSQL에 영구 저장
        await self.save_message_to_postgres(user_id, chat_id, message_type, content, metadata)
    
    # ==================== PostgreSQL (영구 저장) ====================
    
    async def save_session_metadata(self, user_id: str, chat_id: str, session_data: Dict[str, Any]):
        """세션 메타데이터 저장 (PostgreSQL)"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO chat_sessions (user_id, chat_id, session_data, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, chat_id) DO UPDATE SET
                    session_data = $3,
                    metadata = $4,
                    updated_at = NOW(),
                    is_active = TRUE
            """, user_id, chat_id, json.dumps(session_data), json.dumps({"last_sync": datetime.now().isoformat()}))
    
    async def save_message_to_postgres(self, user_id: str, chat_id: str, message_type: str, content: str, metadata: Dict[str, Any] = None):
        """메시지를 PostgreSQL에 영구 저장"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO chat_messages (user_id, chat_id, message_type, content, metadata)
                VALUES ($1, $2, $3, $4, $5)
            """, user_id, chat_id, message_type, content, json.dumps(metadata or {}))
    
    async def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """사용자의 모든 세션 조회 (PostgreSQL)"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT chat_id, session_data, metadata, created_at, updated_at, is_active
                FROM chat_sessions 
                WHERE user_id = $1 
                ORDER BY updated_at DESC 
                LIMIT $2
            """, user_id, limit)
            
            return [dict(row) for row in rows]
    
    async def get_session_messages(self, user_id: str, chat_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """특정 세션의 메시지 조회 (PostgreSQL)"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT message_type, content, metadata, created_at
                FROM chat_messages 
                WHERE user_id = $1 AND chat_id = $2
                ORDER BY created_at ASC 
                LIMIT $3
            """, user_id, chat_id, limit)
            
            return [dict(row) for row in rows]
    
    async def archive_session(self, user_id: str, chat_id: str):
        """세션 아카이브 (Redis → PostgreSQL)"""
        # Redis에서 데이터 수집
        session_data = await self.get_active_session(user_id, chat_id)
        shared_data = await self.get_shared_memory(user_id, chat_id)
        history_data = self.redis.get(self.get_history_key(user_id, chat_id))
        
        if history_data:
            history_list = json.loads(history_data)
        else:
            history_list = []
        
        # PostgreSQL에 통합 저장
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE chat_sessions 
                SET session_data = $3, 
                    metadata = $4,
                    updated_at = NOW(),
                    is_active = FALSE
                WHERE user_id = $1 AND chat_id = $2
            """, user_id, chat_id, 
                json.dumps({
                    "session": session_data,
                    "shared": shared_data,
                    "history": history_list
                }),
                json.dumps({"archived_at": datetime.now().isoformat()})
            )
        
        # Redis에서 정리
        self.redis.delete(
            self.get_active_session_key(user_id, chat_id),
            self.get_shared_key(user_id, chat_id),
            self.get_history_key(user_id, chat_id)
        )
    
    # ==================== 통합 메서드 ====================
    
    async def get_session_data(self, user_id: str, chat_id: str) -> Dict[str, Any]:
        """세션 데이터 조회 (Redis 우선, 없으면 PostgreSQL)"""
        # 먼저 Redis에서 활성 세션 확인
        active_session = await self.get_active_session(user_id, chat_id)
        if active_session:
            return {
                "source": "redis",
                "session": active_session,
                "shared": await self.get_shared_memory(user_id, chat_id),
                "history": json.loads(self.redis.get(self.get_history_key(user_id, chat_id)) or "[]")
            }
        
        # Redis에 없으면 PostgreSQL에서 조회
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT session_data FROM chat_sessions 
                WHERE user_id = $1 AND chat_id = $2
                ORDER BY updated_at DESC LIMIT 1
            """, user_id, chat_id)
            
            if row:
                session_data = json.loads(row['session_data'])
                return {
                    "source": "postgres",
                    "session": session_data.get("session", {}),
                    "shared": session_data.get("shared", {}),
                    "history": session_data.get("history", [])
                }
        
        return {"source": "none", "session": {}, "shared": {}, "history": []}