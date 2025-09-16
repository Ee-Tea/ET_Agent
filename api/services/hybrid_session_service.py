# api/services/hybrid_session_service.py
import redis
import asyncpg
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import hashlib
import uuid

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

                -- Add OAuth columns to users table
                ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS provider TEXT;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS provider_account_id TEXT;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS access_token TEXT;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS refresh_token TEXT;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS expires_at TEXT;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS scope TEXT;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

                -- Add unique constraint for provider + provider_account_id
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.table_constraints 
                        WHERE table_name='users' AND constraint_name='uq_users_provider_account'
                    ) THEN
                        ALTER TABLE users ADD CONSTRAINT uq_users_provider_account UNIQUE (provider, provider_account_id);
                    END IF;
                END$$;

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
                ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS chat_id TEXT;
                ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS session_id TEXT;
                ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS title TEXT;

                -- Ensure UNIQUE(user_id, chat_id) exists for ON CONFLICT
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.table_constraints 
                        WHERE table_name='chat_sessions' AND constraint_name='uq_chat_sessions_user_chat'
                    ) THEN
                        ALTER TABLE chat_sessions ADD CONSTRAINT uq_chat_sessions_user_chat UNIQUE (user_id, chat_id);
                    END IF;
                END$$;
                
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
                ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS chat_id TEXT;
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
                -- 항상 ON UPDATE CASCADE 보장: 기존 제약은 드롭 후 재생성
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.table_constraints 
                        WHERE constraint_name = 'fk_chat_messages_session'
                    ) THEN
                        ALTER TABLE chat_messages DROP CONSTRAINT fk_chat_messages_session;
                    END IF;
                END$$;
                ALTER TABLE chat_messages 
                    ADD CONSTRAINT fk_chat_messages_session 
                    FOREIGN KEY (session_id) 
                    REFERENCES chat_sessions(session_id)
                    ON UPDATE CASCADE
                    ON DELETE CASCADE;

                -- 보조 인덱스 (session_id, created_at)
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created ON chat_messages(session_id, created_at);
            """)

    # ===== Helpers =====
    @staticmethod
    def _generate_session_id(user_id: str, chat_id: str) -> str:
        """Deprecated: kept for backward compat. Not used for new sessions."""
        return f"{user_id}:{chat_id}"

    async def get_or_create_session_id(
        self,
        user_id: str,
        chat_id: str,
        service_type: Optional[str] = None,
        title: Optional[str] = None,
    ) -> str:
        """(user_id, chat_id)로 세션을 조회하고 없으면 시간기반 해시 session_id로 생성 후 반환"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT session_id FROM chat_sessions
                WHERE user_id = $1 AND chat_id = $2
                """,
                user_id,
                chat_id,
            )
            if row and row.get("session_id"):
                existing_sid = row["session_id"]
                # 기존 포맷(user:chat) 발견 시 해시로 마이그레이션
                if isinstance(existing_sid, str) and ":" in existing_sid:
                    raw = f"{user_id}:{chat_id}:{datetime.utcnow().isoformat()}:{uuid.uuid4()}"
                    new_sid = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
                    # chat_sessions.session_id를 먼저 새 해시로 변경하면
                    # FK(ON UPDATE CASCADE)에 의해 chat_messages.session_id도 자동 업데이트됩니다.
                    await conn.execute(
                        """
                        UPDATE chat_sessions
                        SET session_id = $3, updated_at = NOW(), is_active = TRUE
                        WHERE user_id = $1 AND chat_id = $2 AND session_id = $4
                        """,
                        user_id,
                        chat_id,
                        new_sid,
                        existing_sid,
                    )
                    return new_sid
                return existing_sid

            # 새 session_id 생성: 시간 + uuid 기반 해시(짧게 24자)
            raw = f"{user_id}:{chat_id}:{datetime.utcnow().isoformat()}:{uuid.uuid4()}"
            sid = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]

            await conn.execute(
                """
                INSERT INTO chat_sessions (user_id, chat_id, session_id, title, session_data, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, chat_id) DO UPDATE SET
                    session_id = COALESCE(chat_sessions.session_id, EXCLUDED.session_id),
                    title = COALESCE(EXCLUDED.title, chat_sessions.title),
                    session_data = COALESCE(EXCLUDED.session_data, chat_sessions.session_data),
                    metadata = COALESCE(EXCLUDED.metadata, chat_sessions.metadata),
                    updated_at = NOW(),
                    is_active = TRUE
                """,
                user_id,
                chat_id,
                sid,
                title,
                json.dumps({"service_type": service_type} if service_type else {}, ensure_ascii=False, default=str),
                json.dumps({"created_via": "get_or_create_session_id", "ts": datetime.utcnow().isoformat()}, ensure_ascii=False, default=str),
            )
            return sid

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
    
    async def add_chat_message(self, user_id: str, chat_id: str, message_type: str, content: str, metadata: Dict[str, Any] = None) -> Optional[int]:
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

        msg_id = await self.save_message_to_postgres(user_id, chat_id, message_type, content, metadata)

        # 세션 타이틀이 없으면 첫 사용자 메시지로 설정
        if message_type == "user":
            await self._ensure_session_title(user_id, chat_id, content)
        return msg_id
    # ==================== PostgreSQL (영구 저장) ====================
    
    async def save_session_metadata(self, user_id: str, chat_id: str, session_data: Dict[str, Any], title: Optional[str] = None):
        """세션 메타데이터 저장 (PostgreSQL)"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            await self._ensure_user(user_id)
            # 해시 session_id 사용
            # 먼저 기존 존재 여부 확인 후 없으면 생성
            row = await conn.fetchrow(
                """
                SELECT session_id FROM chat_sessions WHERE user_id=$1 AND chat_id=$2
                """,
                user_id,
                chat_id,
            )
            if row and row.get("session_id"):
                session_id = row["session_id"]
            else:
                raw = f"{user_id}:{chat_id}:{datetime.utcnow().isoformat()}:{uuid.uuid4()}"
                session_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
            # session_id는 UNIQUE라 충돌 가능성이 있어 기존 값 유지 우선
            try:
                await conn.execute(
                    """
                    INSERT INTO chat_sessions (user_id, chat_id, session_id, title, session_data, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (user_id, chat_id) DO UPDATE SET
                        -- 기존 session_id가 있으면 유지, 없을 때만 갱신
                        session_id = COALESCE(chat_sessions.session_id, EXCLUDED.session_id),
                        title = COALESCE(EXCLUDED.title, chat_sessions.title),
                        session_data = EXCLUDED.session_data,
                        metadata = EXCLUDED.metadata,
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
            except Exception as err:
                # session_id 유니크 제약 관련 충돌은 무시하고 메타만 갱신 시도
                try:
                    await conn.execute(
                        """
                        UPDATE chat_sessions
                        SET title = COALESCE($3::text, title),
                            session_data = $4,
                            metadata = $5,
                            updated_at = NOW(),
                            is_active = TRUE
                        WHERE user_id = $1 AND chat_id = $2
                        """,
                        user_id,
                        chat_id,
                        title,
                        json.dumps(session_data, ensure_ascii=False, default=str),
                        json.dumps({"last_sync": datetime.now().isoformat()}, ensure_ascii=False, default=str)
                    )
                except Exception:
                    # 최종 실패 시에는 상위에서 처리하도록 전파
                    raise
    
    async def save_message_to_postgres(self, user_id: str, chat_id: str, message_type: str, content: str, metadata: Dict[str, Any] = None) -> Optional[int]:
        """메시지를 PostgreSQL에 영구 저장"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            # ensure existing session_id (hashed)
            session_id = await self.get_or_create_session_id(user_id, chat_id)
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
                row = await conn.fetchrow(
                    """
                    INSERT INTO chat_messages (user_id, chat_id, session_id, speaker, content, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                    """,
                    user_id,
                    chat_id,
                    session_id,
                    normalized_speaker,
                    content,
                    json.dumps(metadata or {})
                )
                # 세션 타이틀 자동 설정(최초 user 메시지일 때만)
                if normalized_speaker == "user":
                    await conn.execute(
                        """
                        UPDATE chat_sessions
                        SET title = COALESCE(title, $3), updated_at = NOW()
                        WHERE user_id = $1 AND chat_id = $2 AND (title IS NULL OR title = '')
                        """,
                        user_id,
                        chat_id,
                        content[:80],
                    )
                return int(row["id"]) if row and "id" in row else None
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

    # ==================== Helpers for chat sessions/messages ====================

    async def get_next_chat_id(self, user_id: str) -> str:
        """해당 사용자에 대한 다음 chat_id(1부터 증가)를 문자열로 반환"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COALESCE(MAX(CASE WHEN chat_id ~ '^[0-9]+$' THEN chat_id::int ELSE 0 END), 0) AS max_id
                FROM chat_sessions
                WHERE user_id = $1
                """,
                user_id,
            )
            max_id = int(row["max_id"] or 0)
            return str(max_id + 1)

    async def get_session_by_session_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """session_id로 세션 조회"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT user_id, chat_id, session_id, title, created_at, updated_at, is_active
                FROM chat_sessions
                WHERE session_id = $1
                """,
                session_id,
            )
            return dict(row) if row else None

    async def get_messages_by_session_id(self, session_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """session_id 기준 메시지 조회 (오름차순)"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, speaker, content, metadata, created_at
                FROM chat_messages
                WHERE session_id = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                session_id,
                limit,
            )
            return [dict(r) for r in rows]

    async def add_message_by_session_id(
        self,
        session_id: str,
        message_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """session_id로 user_id/chat_id를 찾아 메시지 저장"""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT user_id, chat_id
                FROM chat_sessions
                WHERE session_id = $1
                """,
                session_id,
            )
            if not row:
                raise ValueError("Session not found")
            user_id = row["user_id"]
            chat_id = row["chat_id"]
        await self.add_chat_message(user_id, chat_id, message_type, content, metadata or {})
    
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
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE chat_sessions
                SET title = COALESCE($3::text, title), updated_at = NOW()
                WHERE user_id = $1 AND chat_id = $2
                """,
                user_id,
                chat_id,
                title
            )