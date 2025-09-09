# api/services/chat_session_service.py
import asyncpg
from typing import List, Dict, Any, Optional
import json

class ChatSessionService:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
    
    async def init_pool(self):
        """PostgreSQL 연결 풀 초기화"""
        self.pool = await asyncpg.create_pool(self.db_url)
    
    async def create_tables_if_not_exists(self):
        """checkpoints 테이블이 없으면 생성"""
        async with self.pool.acquire() as conn:
            # checkpoints 테이블 생성
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    checkpoint_ns TEXT,
                    checkpoint JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # checkpoint_blobs 테이블 생성
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                    thread_id TEXT,
                    checkpoint_id TEXT,
                    channel TEXT,
                    version TEXT,
                    type TEXT,
                    blob BYTEA,
                    created_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (thread_id, checkpoint_id, channel, version)
                )
            """)
    
    async def save_chat_session(self, user_id: str, chat_id: str, session_data: Dict[str, Any]):
        """채팅 세션 저장"""
        thread_id = f"{user_id}:{chat_id}"
        checkpoint_id = f"checkpoint_{int(time.time())}"
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (thread_id) DO UPDATE SET
                    checkpoint_id = $2,
                    checkpoint = $3,
                    metadata = $4,
                    created_at = NOW()
            """, thread_id, checkpoint_id, json.dumps(session_data), json.dumps({"user_id": user_id, "chat_id": chat_id}))
    
    async def load_chat_session(self, user_id: str, chat_id: str) -> Optional[Dict[str, Any]]:
        """채팅 세션 로드"""
        thread_id = f"{user_id}:{chat_id}"
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT checkpoint FROM checkpoints 
                WHERE thread_id = $1 
                ORDER BY created_at DESC LIMIT 1
            """, thread_id)
            
            if row:
                return json.loads(row['checkpoint'])
            return None
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """사용자의 모든 채팅 세션 조회"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT thread_id, checkpoint_id, checkpoint, metadata, created_at
                FROM checkpoints 
                WHERE metadata->>'user_id' = $1
                ORDER BY created_at DESC
            """, user_id)
            
            return [dict(row) for row in rows]