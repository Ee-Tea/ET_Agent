# farmer_redis.py - Farmer 서비스용 Redis 메모리 시스템
import json
import redis
import time
import hashlib
import os
import sys
from typing import Any, Dict, List, Optional

# 길이 제한/TTL 설정
MAX_QUERIES = 200          # 최근 질문 최대 개수
MAX_AGENT_RESULTS = 200    # 에이전트 결과 최대 개수
MAX_CROP_INFO = 100        # 작물 정보 최대 개수
DEFAULT_TTL = 72 * 3600    # 72시간

# Farmer 서비스용 기본 상태 구조
FARMER_DEFAULTS = {
    "query": [],                    # 사용자 질문들
    "selected_agents": [],          # 선택된 에이전트들
    "question_parts": {},           # 질문 분할 정보
    "execution_order": [],          # 실행 순서
    "crop_info": [],               # 작물 추천 정보
    "selected_crop": [],           # 선택된 작물
    "agent_results": {},           # 에이전트별 결과
    "output": [],                  # 최종 출력
    "session": {},                 # 세션 정보
    "artifacts": {},               # 아티팩트
    "routing": {},                 # 라우팅 정보
    "error_info": {},              # 에러 정보
    "conversation_history": [],    # 대화 히스토리
    "user_preferences": {},        # 사용자 선호도
    "crop_recommendations": [],    # 작물 추천 히스토리
    "weather_history": [],         # 날씨 조회 히스토리
    "disaster_alerts": [],         # 재해 알림 히스토리
    "market_info": [],             # 시장 정보 히스토리
}

def ensure_farmer_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Farmer 상태 구조 보장"""
    state = dict(state or {})
    for k, v in FARMER_DEFAULTS.items():
        if k not in state or not isinstance(state[k], type(v)):
            state[k] = json.loads(json.dumps(v)) if isinstance(v, (dict, list)) else v
    return state


class FarmerRedisMemory:
    """
    Farmer 서비스 전용 Redis 기반 메모리 시스템
    - 농업 관련 대화 맥락 유지
    - 작물 추천 히스토리 관리
    - 에이전트 실행 결과 캐싱
    - 사용자 선호도 및 설정 저장
    """
    
    def __init__(
        self,
        user_id: str,
        service: str = "farmer",
        chat_id: str = "default_chat",
        redis_host: str = "localhost",
        redis_port: int = 6380,
        ttl_seconds: Optional[int] = DEFAULT_TTL,
    ):
        self.user_id = user_id
        self.service = service
        self.chat_id = chat_id
        self.ttl_seconds = ttl_seconds
        
        # Redis 연결
        try:
            self.redis = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # 연결 테스트
            self.redis.ping()
            self.connected = True
            print(f"✅ Redis 연결 성공: {redis_host}:{redis_port}")
        except Exception as e:
            print(f"❌ Redis 연결 실패: {e}")
            self.connected = False
            self.redis = None
    
    def _k(self, suffix: str) -> str:
        """Redis 키 생성"""
        return f"{self.user_id}:{self.service}:{self.chat_id}:{suffix}"
    
    @property
    def k_state(self) -> str:
        """메인 상태 키"""
        return self._k("state")
    
    @property
    def k_history(self) -> str:
        """대화 히스토리 키"""
        return self._k("history")
    
    @property
    def k_crop_recommendations(self) -> str:
        """작물 추천 히스토리 키"""
        return self._k("crop_recommendations")
    
    @property
    def k_user_preferences(self) -> str:
        """사용자 선호도 키"""
        return self._k("preferences")
    
    def _now_ts(self) -> int:
        """현재 타임스탬프"""
        return int(time.time())
    
    def _normalize_text(self, text: Any) -> str:
        """텍스트 정규화"""
        try:
            return " ".join(str(text or "").split()).strip().lower()
        except Exception:
            return str(text or "").strip().lower()
    
    def _enforce_limits(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """리스트 길이 제한 적용"""
        limits = {
            "query": MAX_QUERIES,
            "crop_info": MAX_CROP_INFO,
            "conversation_history": MAX_QUERIES,
            "crop_recommendations": MAX_CROP_INFO,
            "weather_history": 100,
            "disaster_alerts": 100,
            "market_info": 100,
        }
        
        for key, limit in limits.items():
            if key in state and isinstance(state[key], list) and len(state[key]) > limit:
                state[key] = state[key][-limit:]
        
        return state
    
    def _save_state(self, state: Dict[str, Any]) -> None:
        """상태 저장"""
        if not self.connected:
            return
        
        state = self._enforce_limits(state)
        payload = json.dumps(state, ensure_ascii=False)
        
        try:
            with self.redis.pipeline() as pipe:
                pipe.set(self.k_state, payload)
                if self.ttl_seconds:
                    pipe.expire(self.k_state, self.ttl_seconds)
                pipe.execute()
        except Exception as e:
            print(f"⚠️ 상태 저장 실패: {e}")
    
    def _load_state(self) -> Dict[str, Any]:
        """상태 불러오기"""
        if not self.connected:
            return ensure_farmer_state({})
        
        try:
            raw = self.redis.get(self.k_state)
            if not raw:
                return ensure_farmer_state({})
            
            data = json.loads(raw)
            return ensure_farmer_state(data)
        except Exception as e:
            print(f"⚠️ 상태 불러오기 실패: {e}")
            return ensure_farmer_state({})
    
    def _append_history_entry(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """대화 히스토리 추가"""
        if not self.connected:
            return
        
        entry = {
            "role": role,
            "content": content,
            "timestamp": self._now_ts(),
            "metadata": metadata or {}
        }
        
        try:
            with self.redis.pipeline() as pipe:
                pipe.rpush(self.k_history, json.dumps(entry, ensure_ascii=False))
                pipe.ltrim(self.k_history, -MAX_QUERIES, -1)  # 최근 N개만 유지
                if self.ttl_seconds:
                    pipe.expire(self.k_history, self.ttl_seconds)
                pipe.execute()
        except Exception as e:
            print(f"⚠️ 히스토리 저장 실패: {e}")
    
    def _load_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """대화 히스토리 불러오기"""
        if not self.connected:
            return []
        
        try:
            if limit is None:
                entries = self.redis.lrange(self.k_history, 0, -1)
            else:
                length = self.redis.llen(self.k_history)
                start = max(0, length - limit)
                entries = self.redis.lrange(self.k_history, start, -1)
            
            out = []
            for e in entries:
                try:
                    out.append(json.loads(e))
                except Exception:
                    pass
            return out
        except Exception as e:
            print(f"⚠️ 히스토리 불러오기 실패: {e}")
            return []
    
    def _save_crop_recommendation(self, crop_info: str, selected_crop: str, user_query: str) -> None:
        """작물 추천 결과 저장"""
        if not self.connected:
            return
        
        recommendation = {
            "crop_info": crop_info,
            "selected_crop": selected_crop,
            "user_query": user_query,
            "timestamp": self._now_ts()
        }
        
        try:
            with self.redis.pipeline() as pipe:
                pipe.rpush(self.k_crop_recommendations, json.dumps(recommendation, ensure_ascii=False))
                pipe.ltrim(self.k_crop_recommendations, -MAX_CROP_INFO, -1)
                if self.ttl_seconds:
                    pipe.expire(self.k_crop_recommendations, self.ttl_seconds)
                pipe.execute()
        except Exception as e:
            print(f"⚠️ 작물 추천 저장 실패: {e}")
    
    def _load_crop_recommendations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """작물 추천 히스토리 불러오기"""
        if not self.connected:
            return []
        
        try:
            entries = self.redis.lrange(self.k_crop_recommendations, -limit, -1)
            out = []
            for e in entries:
                try:
                    out.append(json.loads(e))
                except Exception:
                    pass
            return out
        except Exception as e:
            print(f"⚠️ 작물 추천 불러오기 실패: {e}")
            return []
    
    def _save_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """사용자 선호도 저장"""
        if not self.connected:
            return
        
        try:
            payload = json.dumps(preferences, ensure_ascii=False)
            with self.redis.pipeline() as pipe:
                pipe.set(self.k_user_preferences, payload)
                if self.ttl_seconds:
                    pipe.expire(self.k_user_preferences, self.ttl_seconds)
                pipe.execute()
        except Exception as e:
            print(f"⚠️ 사용자 선호도 저장 실패: {e}")
    
    def _load_user_preferences(self) -> Dict[str, Any]:
        """사용자 선호도 불러오기"""
        if not self.connected:
            return {}
        
        try:
            raw = self.redis.get(self.k_user_preferences)
            if not raw:
                return {}
            return json.loads(raw)
        except Exception as e:
            print(f"⚠️ 사용자 선호도 불러오기 실패: {e}")
            return {}
    
    # ==================== LangGraph 인터페이스 ====================
    
    def load(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph에서 호출되는 상태 불러오기"""
        if not self.connected:
            return ensure_farmer_state(state or {})
        
        # 저장된 상태 불러오기
        stored_state = self._load_state()
        
        # 현재 state와 병합 (현재 값 우선)
        merged_state = ensure_farmer_state(state or {})
        for key, default_value in FARMER_DEFAULTS.items():
            current_value = merged_state.get(key)
            stored_value = stored_state.get(key, default_value)
            
            # 현재 값이 있으면 사용, 없으면 저장된 값 사용
            if current_value is not None and current_value != default_value:
                merged_state[key] = current_value
            else:
                merged_state[key] = stored_value
        
        # 대화 히스토리 추가
        merged_state["conversation_history"] = self._load_history()
        
        # 사용자 선호도 추가
        merged_state["user_preferences"] = self._load_user_preferences()
        
        # 작물 추천 히스토리 추가
        merged_state["crop_recommendations"] = self._load_crop_recommendations()
        
        return merged_state
    
    def save(self, state: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph에서 호출되는 상태 저장"""
        if not self.connected:
            return output or {}
        
        # 현재 상태와 출력 병합
        current_state = ensure_farmer_state(state or {})
        output_state = ensure_farmer_state(output or {})
        
        # 상태 병합 (append-only 방식)
        merged_state = {}
        for key, default_value in FARMER_DEFAULTS.items():
            current_value = current_state.get(key, default_value)
            output_value = output_state.get(key, default_value)
            
            if isinstance(default_value, list):
                # 리스트는 append-only
                if isinstance(current_value, list) and isinstance(output_value, list):
                    if len(output_value) > len(current_value):
                        merged_state[key] = output_value
                    else:
                        merged_state[key] = current_value
                else:
                    merged_state[key] = current_value
            elif isinstance(default_value, dict):
                # 딕셔너리는 업데이트
                merged = dict(current_value)
                merged.update(output_value)
                merged_state[key] = merged
            else:
                # 스칼라는 출력 값 우선
                merged_state[key] = output_value if output_value is not None else current_value
        
        # 상태 저장
        self._save_state(merged_state)
        
        # 대화 히스토리 저장
        if "query" in merged_state and merged_state["query"]:
            latest_query = merged_state["query"][-1] if merged_state["query"] else ""
            if latest_query:
                self._append_history_entry("user", latest_query)
        
        if "output" in merged_state and merged_state["output"]:
            latest_output = merged_state["output"][-1] if merged_state["output"] else ""
            if latest_output:
                self._append_history_entry("assistant", latest_output)
        
        # 작물 추천 결과 저장
        if "crop_info" in merged_state and "selected_crop" in merged_state:
            crop_info = merged_state["crop_info"][-1] if merged_state["crop_info"] else ""
            selected_crop = merged_state["selected_crop"][-1] if merged_state["selected_crop"] else ""
            user_query = merged_state["query"][-1] if merged_state["query"] else ""
            
            if crop_info and selected_crop:
                self._save_crop_recommendation(crop_info, selected_crop, user_query)
        
        # 사용자 선호도 저장
        if "user_preferences" in merged_state:
            self._save_user_preferences(merged_state["user_preferences"])
        
        return merged_state
    
    # ==================== Farmer 전용 메서드들 ====================
    
    def get_conversation_context(self, limit: int = 5) -> str:
        """대화 맥락을 문자열로 반환"""
        history = self._load_history(limit)
        if not history:
            return ""
        
        context_parts = []
        for entry in history:
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role and content:
                context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def get_crop_recommendation_context(self, limit: int = 3) -> str:
        """작물 추천 맥락을 문자열로 반환"""
        recommendations = self._load_crop_recommendations(limit)
        if not recommendations:
            return ""
        
        context_parts = []
        for rec in recommendations:
            query = rec.get("user_query", "")
            selected = rec.get("selected_crop", "")
            if query and selected:
                context_parts.append(f"이전 질문: {query} → 선택된 작물: {selected}")
        
        return "\n".join(context_parts)
    
    def update_user_preference(self, key: str, value: Any) -> None:
        """사용자 선호도 업데이트"""
        preferences = self._load_user_preferences()
        preferences[key] = value
        self._save_user_preferences(preferences)
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """사용자 선호도 조회"""
        preferences = self._load_user_preferences()
        return preferences.get(key, default)
    
    def clear_memory(self) -> None:
        """메모리 전체 삭제"""
        if not self.connected:
            return
        
        try:
            keys_to_delete = [
                self.k_state,
                self.k_history,
                self.k_crop_recommendations,
                self.k_user_preferences
            ]
            
            with self.redis.pipeline() as pipe:
                for key in keys_to_delete:
                    pipe.delete(key)
                pipe.execute()
            
            print("✅ Farmer 메모리 삭제 완료")
        except Exception as e:
            print(f"⚠️ 메모리 삭제 실패: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 사용 통계"""
        if not self.connected:
            return {"connected": False}
        
        try:
            stats = {
                "connected": True,
                "state_exists": bool(self.redis.exists(self.k_state)),
                "history_count": self.redis.llen(self.k_history),
                "crop_recommendations_count": self.redis.llen(self.k_crop_recommendations),
                "preferences_exists": bool(self.redis.exists(self.k_user_preferences)),
                "ttl_seconds": self.ttl_seconds
            }
            return stats
        except Exception as e:
            return {"connected": False, "error": str(e)}


# 편의 함수들
def create_farmer_memory(user_id: str, service: str = "farmer", chat_id: str = "default_chat") -> FarmerRedisMemory:
    """Farmer 메모리 인스턴스 생성"""
    return FarmerRedisMemory(user_id=user_id, service=service, chat_id=chat_id)


def test_farmer_memory():
    """Farmer 메모리 시스템 테스트"""
    print("=== Farmer Redis 메모리 시스템 테스트 ===")
    
    # 메모리 인스턴스 생성
    memory = create_farmer_memory("test_user", "farmer", "test_chat")
    
    if not memory.connected:
        print("❌ Redis 연결 실패 - 테스트 중단")
        return
    
    # 테스트 상태
    test_state = {
        "query": ["토마토 재배 방법을 알려주세요"],
        "selected_agents": ["작물재배_agent"],
        "crop_info": ["토마토는 따뜻한 기후에서 잘 자랍니다"],
        "selected_crop": ["토마토"]
    }
    
    # 저장 테스트
    print("1. 상태 저장 테스트...")
    memory.save(test_state, test_state)
    
    # 불러오기 테스트
    print("2. 상태 불러오기 테스트...")
    loaded_state = memory.load({})
    print(f"   불러온 상태: {loaded_state.get('query', [])}")
    
    # 통계 조회
    print("3. 메모리 통계...")
    stats = memory.get_memory_stats()
    print(f"   통계: {stats}")
    
    # 맥락 조회
    print("4. 대화 맥락 조회...")
    context = memory.get_conversation_context()
    print(f"   맥락: {context[:100]}...")
    
    # 정리
    print("5. 테스트 데이터 정리...")
    memory.clear_memory()
    
    print("✅ 테스트 완료")


if __name__ == "__main__":
    test_farmer_memory()
