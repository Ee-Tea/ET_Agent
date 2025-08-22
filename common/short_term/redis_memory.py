# redis_memory.py  (et_agent/common/short_term/redis_memory.py)
import json
import redis
from typing import Any, Dict, List, Optional

# 길이 제한/TTL 설정 (필요에 맞게 조정)
MAX_QAS = 200          # shared의 리스트 항목 최대 길이(최근 N개 유지)
HISTORY_LIMIT = 200    # history 최근 N개만 유지
DEFAULT_TTL = 72 * 3600  # 72시간

SHARED_DEFAULTS = {
    "question": [],
    "options": [],
    "answer": [],
    "explanation": [],
    "subject": [],
    "wrong_question": [],
    "weak_type": [],
    "notes": [],
    "user_answer": [],
    "retrieve_answer": "",
}

def ensure_shared(state: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(state or {})
    state.setdefault("shared", {})
    for k, v in SHARED_DEFAULTS.items():
        cur = state["shared"].get(k)
        if cur is None or not isinstance(cur, type(v)):
            state["shared"][k] = json.loads(json.dumps(v)) if isinstance(v, (dict, list)) else v
    return state


class RedisLangGraphMemory:
    """
    단기 메모리:
    - shared 전체를 JSON으로 {user_id}:{service}:{chat_id}:shared 키에 저장/로드
    - 채팅 히스토리는 리스트 {…}:history 에 rpush / lrange
    - append-only 병합 + TTL + 길이 제한
    """
    def __init__(
        self,
        user_id: str,
        service: str,
        chat_id: str,
        redis_host: str = "localhost",
        redis_port: int = 6380,
        ttl_seconds: Optional[int] = DEFAULT_TTL,
    ) -> None:
        self.user_id = user_id
        self.service = service
        self.chat_id = chat_id
        import os
        # 환경 변수/인자 우선순위로 Redis 설정
        host = redis_host if redis_host is not None else os.getenv("REDIS_HOST", "langgraph-redis")
        port = redis_port if redis_port is not None else int(os.getenv("REDIS_PORT", "6379"))
        self.ttl_seconds = ttl_seconds
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)

    def _k(self, suffix: str) -> str:
        return f"{self.user_id}:{self.service}:{self.chat_id}:{suffix}"

    @property
    def k_shared(self) -> str:
        return self._k("shared")

    @property
    def k_history(self) -> str:
        return self._k("history")

    # ---------- 내부 IO ----------
    def _load_shared(self) -> Dict[str, Any]:
        # redis-py sync client returns str|None
        raw = self.redis.get(self.k_shared)  # type: ignore[assignment]
        if not raw:
            return json.loads(json.dumps(SHARED_DEFAULTS))  # deepcopy
        try:
            data = json.loads(raw)  # type: ignore[arg-type]
            tmp = {"shared": data}
            tmp = ensure_shared(tmp)
            return tmp["shared"]
        except Exception:
            return json.loads(json.dumps(SHARED_DEFAULTS))

    def _enforce_limits(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        for k in ("question","options","answer","explanation","subject","wrong_question","notes","user_answer","weak_type"):
            if isinstance(shared.get(k), list) and len(shared[k]) > MAX_QAS:
                shared[k] = shared[k][-MAX_QAS:]
        return shared

    def _save_shared(self, shared: Dict[str, Any]) -> None:
        shared = self._enforce_limits(shared)
        payload = json.dumps(shared, ensure_ascii=False)
        with self.redis.pipeline() as pipe:
            pipe.set(self.k_shared, payload)
            if self.ttl_seconds:
                pipe.expire(self.k_shared, self.ttl_seconds)
            pipe.execute()

    def _append_history_entries(self, entries: List[Dict[str, Any]]) -> None:
        if not entries:
            return
        with self.redis.pipeline() as pipe:
            for e in entries:
                role = e.get("role", "user")
                content = e.get("content", "")
                # trace_id/node/ts 같은 메타를 함께 넣으시면 디버깅이 쉬워집니다.
                pipe.rpush(self.k_history, json.dumps({"role": role, "content": content}, ensure_ascii=False))
            # 최근 HISTORY_LIMIT만 유지
            pipe.ltrim(self.k_history, -HISTORY_LIMIT, -1)
            if self.ttl_seconds:
                pipe.expire(self.k_history, self.ttl_seconds)
            pipe.execute()

    def _load_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        def _force_int(x):
            try:
                return int(x)  # type: ignore
            except Exception:
                return 0
        def _force_list(x):
            try:
                return list(x)  # type: ignore
            except Exception:
                return []
        if limit is None:
            _entries = self.redis.lrange(self.k_history, 0, -1)  # type: ignore
        else:
            _len_val = self.redis.llen(self.k_history)  # type: ignore
            length = _force_int(_len_val)
            start = max(0, length - int(limit))
            _entries = self.redis.lrange(self.k_history, start, -1)  # type: ignore
        entries: List[str] = _force_list(_entries)
        out: List[Dict[str, Any]] = []
        for e in entries:
            try:
                out.append(json.loads(e))
            except Exception:
                pass
        return out

    # ---------- append-only 병합 로직 ----------
    @staticmethod
    def _merge_shared_append_only(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        """
        리스트 필드: 기존 길이 이후로만 append (순서/중복 보존)
        문자열/스칼라: incoming이 있으면 덮어쓰기, 없으면 기존 유지
        """
        result: Dict[str, Any] = {}
        for key, default in SHARED_DEFAULTS.items():
            old_val = existing.get(key, default)
            new_val = incoming.get(key, default)

            if isinstance(default, list):
                old_list = old_val if isinstance(old_val, list) else []
                new_list = new_val if isinstance(new_val, list) else []
                if len(new_list) > len(old_list):
                    tail = new_list[len(old_list):]
                    merged = old_list + tail
                else:
                    merged = old_list  # 짧아졌거나 같으면 기존 유지
                result[key] = merged
            else:
                result[key] = new_val if (new_val is not None and new_val != "") else old_val
        return result

    # ---------- LangGraph 인터페이스 ----------
    def load(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = ensure_shared(state or {})
        stored_shared = self._load_shared()

        # 현재 state.shared가 비어있다면 저장된 것을 채워 넣음 (현재 값 우선)
        merged_shared = {}
        for k, default in SHARED_DEFAULTS.items():
            cur = state["shared"].get(k)
            merged_shared[k] = cur if (cur not in (None, []) and cur != "") else stored_shared.get(k, default)

        state["shared"] = merged_shared
        state["history"] = self._load_history()
        return state

    def save(self, state: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(output or {})
        incoming_shared = (out.get("shared") or state.get("shared") or {})
        incoming_shared = ensure_shared({"shared": incoming_shared})["shared"]

        # 기존 shared 불러와서 append-only 병합 후 저장
        existing_shared = self._load_shared()
        merged_shared = self._merge_shared_append_only(existing_shared, incoming_shared)
        self._save_shared(merged_shared)

        # history append
        new_history = out.get("history")
        if isinstance(new_history, list) and new_history:
            self._append_history_entries(new_history)

        out["shared"] = merged_shared
        return out

    # 편의
    def clear(self) -> None:
        with self.redis.pipeline() as pipe:
            pipe.delete(self.k_shared)
            pipe.delete(self.k_history)
            pipe.execute()
