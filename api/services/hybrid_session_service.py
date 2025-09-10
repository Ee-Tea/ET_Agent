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
    
    async def _ensure_pool(self):
        """연결 풀 보장: None/closed 상태이면 재생성"""
        import asyncio
        # 풀이 없으면 생성
        if getattr(self, "pool", None) is None:
            self.pool = await asyncpg.create_pool(self.postgres_url)
            return
        # 닫힘 또는 이벤트 루프 불일치 시 재생성
        pool_closed = getattr(self.pool, "_closed", False)
        pool_loop = getattr(self.pool, "_loop", None)
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        loop_mismatch = pool_loop is not None and current_loop is not None and pool_loop is not current_loop
        if pool_closed or loop_mismatch:
            try:
                await self.pool.close()
            except Exception:
                pass
            self.pool = await asyncpg.create_pool(self.postgres_url)
    
    async def create_tables_if_not_exists(self):
        """PostgreSQL 테이블 생성"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            await conn.execute("""
                -- Users
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    user_name TEXT,
                    token TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );

                -- Chat Sessions (augment existing table if present)
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    session_id TEXT,
                    title TEXT,
                    session_data JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(user_id, chat_id)
                );

                -- Ensure new columns exist
                ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS session_id TEXT;
                ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS title TEXT;
                
                -- FK to users
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.table_constraints 
                        WHERE constraint_name = 'fk_chat_sessions_user'
                    ) THEN
                        ALTER TABLE chat_sessions 
                        ADD CONSTRAINT fk_chat_sessions_user 
                        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;
                    END IF;
                END$$;

                -- Chat Messages (augment existing table if present)
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    chat_id TEXT NOT NULL,
                    session_id TEXT,
                    speaker TEXT NOT NULL, -- 'user', 'assistant', 'system'
                    content TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );

                -- Ensure new columns exist
                ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS session_id TEXT;
                ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS speaker TEXT;

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_chat_messages_user_chat ON chat_messages(user_id, chat_id);

                -- Ensure UNIQUE constraint on chat_sessions(session_id) for FK target
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.table_constraints 
                        WHERE table_name='chat_sessions' AND constraint_name='uq_chat_sessions_session_id'
                    ) THEN
                        ALTER TABLE chat_sessions ADD CONSTRAINT uq_chat_sessions_session_id UNIQUE (session_id);
                    END IF;
                END$$;

                -- FK from chat_messages to chat_sessions(session_id)
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.table_constraints 
                        WHERE constraint_name = 'fk_chat_messages_session'
                    ) THEN
                        ALTER TABLE chat_messages 
                        ADD CONSTRAINT fk_chat_messages_session 
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE;
                    END IF;
                END$$;
            """)

    # ===== Helpers =====
    @staticmethod
    def _generate_session_id(user_id: str, chat_id: str) -> str:
        return f"{user_id}:{chat_id}"

    @staticmethod
    def _normalize_speaker(message_type: str) -> str:
        """스피커 표준화: 다양한 입력 값을 'user' | 'chatbot' | 'system' 으로 매핑"""
        mt = str(message_type or "").strip().lower()
        if mt in {"user", "human", "human_message", "user_query", "input", "request"}:
            return "user"
        if mt in {"assistant", "bot", "chatbot", "model", "assistant_message", "ai", "response", "reply"}:
            return "chatbot"
        if mt in {"system", "server"}:
            return "system"
        # 부분 일치 보정
        if "user" in mt or "human" in mt:
            return "user"
        if "assistant" in mt or "bot" in mt or "model" in mt or "ai" in mt:
            return "chatbot"
        return mt or "system"

    async def _ensure_user(self, user_id: str, user_name: Optional[str] = None, token: Optional[str] = None):
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, user_name, token)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id) DO UPDATE SET
                    user_name = COALESCE(EXCLUDED.user_name, users.user_name),
                    token = COALESCE(EXCLUDED.token, users.token),
                    updated_at = NOW()
                """,
                user_id, user_name, token
            )

    async def ensure_user(self, user_id: str, user_name: Optional[str] = None, token: Optional[str] = None):
        """공개 API: 사용자 업서트"""
        await self._ensure_user(user_id, user_name, token)
    
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
    
    async def start_session(self, user_id: str, chat_id: str, initial_data: Dict[str, Any] = None, title: Optional[str] = None):
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
        await self.save_session_metadata(user_id, chat_id, session_data, title=title)
    
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
        """채팅 메시지 추가 (Redis + PostgreSQL)
        세션 FK 제약 보장을 위해 메시지 저장 전에 세션 레코드를 업서트합니다.
        """
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
        
        # PostgreSQL: 세션 레코드 보장 후 메시지 저장 (FK 보호)
        try:
            # 세션 메타데이터 업서트 (last_activity 갱신, 첫 user 메시지는 title 후보 전달)
            session_snapshot = {
                "last_activity": message["timestamp"],
                "status": "active"
            }
            title_hint = content if message_type == "user" else None
            await self.save_session_metadata(user_id, chat_id, session_snapshot, title=title_hint)
        except Exception:
            # 세션 보강 실패해도 메시지 저장 시 FK로 다시 오류 나므로 그대로 전파
            pass

        await self.save_message_to_postgres(user_id, chat_id, message_type, content, metadata)

        # 세션 타이틀이 없으면 첫 사용자 메시지로 설정
        if message_type == "user":
            await self._ensure_session_title(user_id, chat_id, content)
    
    # ==================== PostgreSQL (영구 저장) ====================
    
    async def save_session_metadata(self, user_id: str, chat_id: str, session_data: Dict[str, Any], title: Optional[str] = None):
        """세션 메타데이터 저장 (PostgreSQL)"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            await self._ensure_user(user_id)
            session_id = self._generate_session_id(user_id, chat_id)
            await conn.execute(
                """
                INSERT INTO chat_sessions (user_id, chat_id, session_id, title, session_data, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, chat_id) DO UPDATE SET
                    session_id = $3,
                    title = COALESCE($4, chat_sessions.title),
                    session_data = $5,
                    metadata = $6,
                    updated_at = NOW(),
                    is_active = TRUE
                """,
                user_id,
                chat_id,
                session_id,
                title,
                json.dumps(session_data, ensure_ascii=False, default=str),
                json.dumps({"last_sync": datetime.now().isoformat()}, ensure_ascii=False, default=str)
            )
    
    async def save_message_to_postgres(self, user_id: str, chat_id: str, message_type: str, content: str, metadata: Dict[str, Any] = None):
        """메시지를 PostgreSQL에 영구 저장"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            session_id = self._generate_session_id(user_id, chat_id)
            # speaker 표준화
            normalized_speaker = self._normalize_speaker(message_type)
            try:
                print(
                    f"[HybridSessionService] insert chat_message: session_id={session_id}, "
                    f"speaker={normalized_speaker}, user_id={user_id}, chat_id={chat_id}"
                )
            except Exception:
                pass
            try:
                await conn.execute(
                    """
                    INSERT INTO chat_messages (user_id, chat_id, session_id, speaker, content, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    user_id,
                    chat_id,
                    session_id,
                    normalized_speaker,
                    content,
                    json.dumps(metadata or {})
                )
            except Exception as err:
                try:
                    print(f"[HybridSessionService][ERROR] insert failed: {err}")
                except Exception:
                    pass
                raise
    
    async def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """사용자의 모든 세션 조회 (PostgreSQL)"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT chat_id, session_id, title, session_data, metadata, created_at, updated_at, is_active
                FROM chat_sessions 
                WHERE user_id = $1 
                ORDER BY updated_at DESC 
                LIMIT $2
            """, user_id, limit)
            
            return [dict(row) for row in rows]
    
    async def get_session_messages(self, user_id: str, chat_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """특정 세션의 메시지 조회 (PostgreSQL)"""
        async with self.pool.acquire() as conn:
            session_id = self._generate_session_id(user_id, chat_id)
            rows = await conn.fetch(
                """
                SELECT speaker, content, metadata, created_at
                FROM chat_messages 
                WHERE session_id = $1
                ORDER BY created_at ASC 
                LIMIT $2
                """,
                session_id,
                limit
            )
            
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
                SELECT session_id, session_data, title FROM chat_sessions 
                WHERE user_id = $1 AND chat_id = $2
                ORDER BY updated_at DESC LIMIT 1
            """, user_id, chat_id)
            
            if row:
                session_data = json.loads(row['session_data']) if row['session_data'] else {}
                return {
                    "source": "postgres",
                    "session": session_data.get("session", {}),
                    "shared": session_data.get("shared", {}),
                    "history": session_data.get("history", [])
                }
        
        return {"source": "none", "session": {}, "shared": {}, "history": []}

    async def _ensure_session_title(self, user_id: str, chat_id: str, first_message: str):
        """세션 타이틀이 없을 경우 첫 사용자 메시지 기준으로 설정"""
        session_id = self._generate_session_id(user_id, chat_id)
        title = (first_message or "").strip()
        if not title:
            return
        # 간단 요약: 앞 50자
        title = title[:50]
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE chat_sessions
                SET title = COALESCE(title, $3), updated_at = NOW()
                WHERE user_id = $1 AND chat_id = $2
                """,
                user_id,
                chat_id,
                title
            )