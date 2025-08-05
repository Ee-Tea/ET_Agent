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
