import redis
import json

class RedisMemory:
    def __init__(self, session_id: str,chat_id : str, redis_host="localhost", redis_port=6379):
        self.client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.session_id = session_id
        self.chat_id = chat_id

    def _key(self, step: str):
        return f"{self.user_id}:{self.chat_id}:{step}"

    # 전체 저장 (리스트 단위)
    def save_list(self, step: str, value_list: list):
        self.client.set(self._key(step), json.dumps(value_list))

    def load_list(self, step: str) -> list:
        data = self.client.get(self._key(step))
        return json.loads(data) if data else []

    # 히스토리 누적
    def append_history(self, role: str, content: str):
        entry = {"role": role, "content": content}
        self.client.rpush(self._key("history"), json.dumps(entry))

    def get_history(self, limit: int = None):
        entries = self.client.lrange(self._key("history"), 0, -1 if limit is None else limit - 1)
        return [json.loads(e) for e in entries]

    # def clear_all(self):
    #     for step in ["question", "solution", "answer", "score", "note", "history"]:
    #         self.client.delete(self._key(step))

class RedisLangGraphMemory:
    def __init__(self, user_id: str, service: str, chat_id: str, redis_host="localhost", redis_port=6379):
        """
        :param user_id: 사용자 ID (ex: user123)
        :param service: 서비스 종류 (ex: exam, farming)
        :param chat_id: 채팅 세션 ID (ex: chat1)
        """
        self.user_id = user_id
        self.service = service
        self.chat_id = chat_id
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    def _key(self, field: str) -> str:
        """
        Redis 키 구성: {user_id}:{service}:{chat_id}:{field}
        """
        return f"{self.user_id}:{self.service}:{self.chat_id}:{field}"

    def _load_list(self, field: str):
        """
        리스트 형태로 저장된 JSON 불러오기
        """
        data = self.redis.get(self._key(field))
        return json.loads(data) if data else []

    def _save_list(self, field: str, value: list):
        """
        리스트 형태로 JSON 저장
        """
        self.redis.set(self._key(field), json.dumps(value))

    def _append_history(self, entry: dict):
        """
        채팅 히스토리 한 항목 추가 (role, content 포함)
        """
        self.redis.rpush(self._key("history"), json.dumps(entry))

    def _load_history(self):
        """
        전체 채팅 히스토리 로드
        """
        entries = self.redis.lrange(self._key("history"), 0, -1)
        return [json.loads(e) for e in entries]

    def load(self, state: dict) -> dict:
        """
        LangGraph가 노드 실행 전에 호출. Redis에서 상태 불러오기.
        """
        return {
            **state,
            "questions": self._load_list("question"),
            "solutions": self._load_list("solution"),
            "answers": self._load_list("answer"),
            "score": self._load_list("score"),
            "notes": self._load_list("note"),
            "history": self._load_history()
        }

    def save(self, state: dict, output: dict) -> dict:
        """
        LangGraph가 노드 실행 후 호출. Redis에 필요한 항목 저장.
        """
        for key in ["questions", "solutions", "answers", "score", "notes"]:
            if key in output:
                self._save_list(key[:-1] if key.endswith("s") else key, output[key])

        if "history" in output:
            for h in output["history"]:
                self._append_history(h)

        return output