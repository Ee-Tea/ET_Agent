# redis_memory.py  (et_agent/common/short_term/redis_memory.py)
import json
import redis
import time
import hashlib
from typing import Any, Dict, List, Optional

# 길이 제한/TTL 설정 (필요에 맞게 조정)
MAX_QAS = 200          # shared의 리스트 항목 최대 길이(최근 N개 유지)
HISTORY_LIMIT = 200    # history 최근 N개만 유지
DEFAULT_TTL = 72 * 3600  # 72시간

# teacher_graph의 규격 재사용
try:
    from ...teacher.teacher_graph import ensure_shared, SHARED_DEFAULTS
except Exception:
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
            if k not in state["shared"] or not isinstance(state["shared"][k], type(v)):
                state["shared"][k] = json.loads(json.dumps(v)) if isinstance(v, (dict, list)) else v
        return state


class RedisLangGraphMemory:
    """
    통합된 Redis 기반 숏텀 메모리
    - 기존 LangGraph 메모리 기능 (shared, history)
    - 문제 중심 스키마 (질문/풀이 분리 저장, 중복 검사)
    - 부분 선택 실행, 취약점 분석, 맞춤형 문제 생성 지원
    """
    
    def __init__(
        self,
        user_id: str,
        service: str,
        chat_id: str,
        redis_host: str = "localhost",
        redis_port: int = 6380,
        ttl_seconds: Optional[int] = DEFAULT_TTL,
    ):
        self.user_id = user_id
        self.service = service
        self.chat_id = chat_id
        self.ttl_seconds = ttl_seconds
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # 문제 중심 스키마를 위한 네임스페이스
        self.question_ns = f"{user_id}:{service}:{chat_id}:questions"

    # ==================== 기존 LangGraph 메모리 기능 ====================
    
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
        raw = self.redis.get(self.k_shared)
        if not raw:
            return json.loads(json.dumps(SHARED_DEFAULTS))  # deepcopy
        try:
            data = json.loads(raw)
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
    def clear(self, include_questions: bool = True) -> None:
        """
        메모리 데이터 삭제
        
        Args:
            include_questions: 문제 데이터도 함께 삭제할지 여부
        """
        with self.redis.pipeline() as pipe:
            pipe.delete(self.k_shared)
            pipe.delete(self.k_history)
            if include_questions:
                pipe.delete(self._k_all())
                pipe.delete(self._k_dedupe())
                pipe.delete(self._k_wrong())
                # 과목별 인덱스도 삭제
                pattern = f"{self.question_ns}:q:by_subject:*"
                for key in self.redis.scan_iter(match=pattern):
                    pipe.delete(key)
            pipe.execute()

    # ==================== 문제 중심 스키마 기능 ====================
    
    def _now_ts(self) -> int:
        """현재 타임스탬프"""
        return int(time.time())
    
    def _norm_text(self, text: str) -> str:
        """텍스트 정규화 (공백 정리, 소문자 변환)"""
        return " ".join((text or "").split()).strip().lower()
    
    def _question_hash(self, question_text: str, options: List[str]) -> str:
        """질문과 보기를 기반으로 한 중복 검사용 해시 생성"""
        base = self._norm_text(question_text) + "||" + "||".join([self._norm_text(o) for o in options or []])
        return hashlib.sha256(base.encode("utf-8")).hexdigest()
    
    # --------------------------
    # 키 헬퍼 메서드들 (문제 중심 스키마용)
    def _k_q(self, qid: str) -> str:
        """질문 키"""
        return f"{self.question_ns}:q:{qid}"
    
    def _k_sol(self, qid: str) -> str:
        """풀이 키"""
        return f"{self.question_ns}:sol:{qid}"
    
    def _k_all(self) -> str:
        """전체 질문 인덱스 (ZSET)"""
        return f"{self.question_ns}:q:all"
    
    def _k_by_subject(self, subject: str) -> str:
        """과목별 질문 인덱스 (ZSET)"""
        return f"{self.question_ns}:q:by_subject:{subject}"
    
    def _k_dedupe(self) -> str:
        """중복 방지용 해시 세트 (SET)"""
        return f"{self.question_ns}:q:dedupe"
    
    def _k_wrong(self) -> str:
        """오답 인덱스 (SET)"""
        return f"{self.question_ns}:q:wrong"
    
    def _k_events(self) -> str:
        """이벤트 로그 (Stream)"""
        return f"{self.question_ns}:events"
    
    # --------------------------
    # 1) 문제 중복 검사 + 삽입
    def add_questions(self, questions: List[Dict[str, Any]]) -> List[str]:
        """
        문제들을 추가 (중복 자동 필터링)
        
        Args:
            questions: [{"question": "질문", "options": ["보기1", "보기2"], "subject": "과목", "qid": "선택적"}]
        
        Returns:
            신규로 추가된 qid 리스트 (중복 제외)
        """
        pipe = self.redis.pipeline()
        new_qids = []
        ts = self._now_ts()
        
        for q in questions:
            qtext = q.get("question") or ""
            options = q.get("options") or []
            subject = q.get("subject") or "unknown"
            qid = q.get("qid") or hashlib.md5((qtext + str(ts)).encode()).hexdigest()[:12]
            
            # 중복 검사
            ch = self._question_hash(qtext, options)
            if self.redis.sismember(self._k_dedupe(), ch):
                continue  # 이미 존재 → skip
            
            # 질문 저장
            pipe.hset(self._k_q(qid), mapping={
                "question_text": qtext,
                "options_json": json.dumps(options, ensure_ascii=False),
                "subject": subject,
                "content_hash": ch,
                "created_at": ts,
                "updated_at": ts,
            })
            
            # 인덱스 & dedupe 세트
            pipe.zadd(self._k_all(), {qid: ts})
            pipe.sadd(self._k_by_subject(subject), qid)
            pipe.sadd(self._k_dedupe(), ch)
            
            new_qids.append(qid)
        
        pipe.execute()
        
        # TTL 설정 (기본값 사용)
        if new_qids and self.ttl_seconds:
            self.set_question_ttl()
        
        return new_qids
    
    # --------------------------
    # 2) 풀이 수정(업데이트, 부분 필드 upsert)
    def upsert_solution(self, qid: str, **kwargs) -> None:
        """
        풀이 정보를 업데이트 (부분 필드만 업데이트 가능)
        
        Args:
            qid: 질문 ID
            **kwargs: 업데이트할 필드들 (user_answer, model_answer, explanation, score, is_correct, meta 등)
        """
        if not self.redis.exists(self._k_q(qid)):
            raise ValueError(f"질문 {qid}가 존재하지 않습니다.")
        
        update = {"updated_at": self._now_ts()}
        
        # 업데이트할 필드들 처리
        if "user_answer" in kwargs:
            update["user_answer"] = kwargs["user_answer"]
        if "model_answer" in kwargs:
            update["model_answer"] = kwargs["model_answer"]
        if "explanation" in kwargs:
            update["explanation"] = kwargs["explanation"]
        if "score" in kwargs:
            update["score"] = kwargs["score"]
        if "is_correct" in kwargs:
            update["is_correct"] = int(bool(kwargs["is_correct"]))
        if "meta" in kwargs:
            update["meta_json"] = json.dumps(kwargs["meta"], ensure_ascii=False)
        
        pipe = self.redis.pipeline()
        pipe.hincrby(self._k_sol(qid), "attempts", 1)
        pipe.hset(self._k_sol(qid), mapping=update)
        
        # 오답 인덱스 관리
        if "is_correct" in kwargs:
            if kwargs["is_correct"]:
                pipe.srem(self._k_wrong(), qid)
            else:
                pipe.sadd(self._k_wrong(), qid)
        
        pipe.execute()
    
    # --------------------------
    # 3) 특정 문제 선택(여러 기준)
    def select_qids(self, limit: int = 10, subject: Optional[str] = None,
                   only_wrong: bool = False, recent_first: bool = True) -> List[str]:
        """
        조건에 맞는 문제 ID들을 선택
        
        Args:
            limit: 최대 개수
            subject: 특정 과목만
            only_wrong: 오답만 선택
            recent_first: 최근 순으로 정렬
        
        Returns:
            선택된 qid 리스트
        """
        if only_wrong:
            # 오답만 선택
            wrong = list(self.redis.smembers(self._k_wrong()))
            if not wrong:
                return []
            if recent_first:
                # all zset 점수(=ts) 참조해서 정렬
                scores = self.redis.zmscore(self._k_all(), wrong)
                pairs = sorted(zip(wrong, scores), key=lambda x: (x[1] or 0), reverse=True)
                return [qid for qid, _ in pairs[:limit]]
            return wrong[:limit]
        
        if subject:
            # 과목별 선택
            qids = list(self.redis.smembers(self._k_by_subject(subject)))
            if recent_first:
                scores = self.redis.zmscore(self._k_all(), qids)
                pairs = sorted(zip(qids, scores), key=lambda x: (x[1] or 0), reverse=True)
                return [qid for qid, _ in pairs[:limit]]
            return qids[:limit]
        
        # 기본: 전체 중 최근 N개
        if recent_first:
            return self.redis.zrevrange(self._k_all(), 0, limit - 1)
        else:
            return self.redis.zrange(self._k_all(), 0, limit - 1)
    
    # --------------------------
    # 4) 기존 문제 → 풀이/채점 저장
    def get_question(self, qid: str) -> Dict[str, Any]:
        """질문 정보 조회"""
        q = self.redis.hgetall(self._k_q(qid))
        if not q:
            raise ValueError("질문이 없습니다.")
        
        # JSON 필드 파싱
        if "options_json" in q:
            try:
                q["options"] = json.loads(q["options_json"])
            except:
                q["options"] = []
        
        return q
    
    def get_solution(self, qid: str) -> Dict[str, Any]:
        """풀이 정보 조회"""
        s = self.redis.hgetall(self._k_sol(qid))
        if not s:
            return {}
        
        # JSON 필드 파싱
        if "meta_json" in s:
            try:
                s["meta"] = json.loads(s["meta_json"])
            except:
                pass
        
        # 타입 변환
        if "is_correct" in s:
            s["is_correct"] = bool(int(s["is_correct"]))
        if "score" in s:
            try:
                s["score"] = float(s["score"])
            except:
                pass
        
        return s
    
    # --------------------------
    # 5) 취약점 분석(간단 집계)
    def weakness_summary(self, top_k: int = 3) -> Dict[str, Any]:
        """
        과목별 정답률 단순 집계
        
        Args:
            top_k: 취약 과목 상위 개수
        
        Returns:
            취약점 분석 결과
        """
        qids = self.redis.zrevrange(self._k_all(), 0, -1)
        by_subject = {}
        
        for qid in qids:
            q = self.redis.hget(self._k_q(qid), "subject") or "unknown"
            sol = self.redis.hgetall(self._k_sol(qid))
            if not sol:
                continue
            
            attempts = int(sol.get("attempts", 0) or 0)
            score = float(sol.get("score", 0) or 0)
            if attempts == 0:
                continue
            
            d = by_subject.setdefault(q, {"attempts": 0, "score_sum": 0.0})
            d["attempts"] += attempts
            d["score_sum"] += score
        
        # 정답률(혹은 평균 점수) 낮은 순
        items = []
        for subj, v in by_subject.items():
            avg = v["score_sum"] / max(1, v["attempts"])
            items.append((subj, round(avg, 4), v["attempts"]))
        
        items.sort(key=lambda x: x[1])  # 낮은 점수 우선
        
        return {
            "weak_subjects": items[:top_k],
            "basis": "avg_score_per_attempt(lowest_first)",
            "total_questions": len(qids)
        }
    
    # --------------------------
    # 6) 기존 문제 기반 풀이 생성 및 채점
    def generate_solution_for_question(self, qid: str, model_answer: str, 
                                     explanation: str, score: float, is_correct: bool) -> None:
        """
        기존 문제에 대한 풀이 생성 및 채점 결과 저장
        
        Args:
            qid: 질문 ID
            model_answer: 모델이 생성한 답
            explanation: 해설
            score: 점수 (0~1 또는 0~100)
            is_correct: 정답 여부
        """
        self.upsert_solution(
            qid=qid,
            model_answer=model_answer,
            explanation=explanation,
            score=score,
            is_correct=is_correct
        )
    
    # --------------------------
    # 7) 맞춤형 문제 생성 (취약점 기반)
    def get_weakness_based_prompt(self, top_k: int = 3) -> str:
        """
        취약점 분석 결과를 기반으로 한 맞춤형 문제 생성 프롬프트 생성
        
        Args:
            top_k: 취약 과목 상위 개수
        
        Returns:
            맞춤형 문제 생성용 프롬프트
        """
        weakness = self.weakness_summary(top_k)
        
        prompt = "다음 취약 과목들을 기반으로 맞춤형 문제를 생성해주세요:\n\n"
        
        for i, (subject, avg_score, attempts) in enumerate(weakness["weak_subjects"], 1):
            prompt += f"{i}. {subject}: 평균 점수 {avg_score:.2f} (시도 횟수: {attempts})\n"
        
        prompt += f"\n총 문제 수: {weakness['total_questions']}\n"
        prompt += "위 취약 과목들을 보완할 수 있는 문제를 생성해주세요."
        
        return prompt
    
    # --------------------------
    # 8) 유틸리티 메서드들
    def get_question_count(self) -> int:
        """전체 질문 수 조회"""
        return self.redis.zcard(self._k_all())
    
    def get_subject_list(self) -> List[str]:
        """등록된 과목 목록 조회"""
        # 과목별 인덱스 키들을 스캔하여 과목명 추출
        subjects = set()
        pattern = f"{self.question_ns}:q:by_subject:*"
        
        for key in self.redis.scan_iter(match=pattern):
            subject = key.split(":", -1)[-1]
            subjects.add(subject)
        
        return list(subjects)
    
    def clear_questions(self) -> None:
        """현재 세션의 모든 문제 데이터 삭제"""
        pattern = f"{self.question_ns}:*"
        keys = self.redis.scan_iter(match=pattern)
        
        if keys:
            self.redis.delete(*keys)
    
    def set_question_ttl(self, ttl_seconds: Optional[int] = None) -> None:
        """
        현재 세션의 모든 문제 키에 TTL 설정
        
        Args:
            ttl_seconds: TTL 초 단위, None이면 기본값 사용
        """
        if ttl_seconds is None:
            ttl_seconds = self.ttl_seconds or DEFAULT_TTL
            
        pattern = f"{self.question_ns}:*"
        keys = self.redis.scan_iter(match=pattern)
        
        pipe = self.redis.pipeline()
        for key in keys:
            pipe.expire(key, ttl_seconds)
        pipe.execute()
    
    # --------------------------
    # 9) 기존 shared와 연동하는 편의 메서드들
    def sync_shared_to_questions(self) -> None:
        """
        기존 shared의 question 데이터를 문제 중심 스키마로 동기화
        """
        shared = self._load_shared()
        questions = shared.get("question", [])
        options = shared.get("options", [])
        subjects = shared.get("subject", [])
        
        if not questions:
            return
        
        # shared 데이터를 문제 형태로 변환
        question_data = []
        for i, q in enumerate(questions):
            if i < len(options) and i < len(subjects):
                question_data.append({
                    "question": q,
                    "options": options[i] if isinstance(options[i], list) else [],
                    "subject": subjects[i] if i < len(subjects) else "unknown"
                })
        
        if question_data:
            self.add_questions(question_data)
    
    def get_questions_for_shared(self) -> Dict[str, Any]:
        """
        문제 중심 스키마의 데이터를 shared 형태로 변환하여 반환
        """
        qids = self.redis.zrevrange(self._k_all(), 0, -1)
        
        questions = []
        options = []
        subjects = []
        
        for qid in qids:
            q = self.get_question(qid)
            questions.append(q.get("question_text", ""))
            options.append(q.get("options", []))
            subjects.append(q.get("subject", "unknown"))
        
        return {
            "question": questions,
            "options": options,
            "subject": subjects
        }
