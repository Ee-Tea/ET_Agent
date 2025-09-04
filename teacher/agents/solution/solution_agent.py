import os
from typing import TypedDict, List, Dict, Literal, Optional, Tuple, Any, Union
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import connections
import time
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
import json, re
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ..base_agent import BaseAgent
from datetime import datetime
# from teacher.agents.milvus_utils import connect_milvus_fallback


load_dotenv()
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ✅ 상태 정의
class SolutionState(TypedDict):
    # 사용자 입력
    user_input_txt: str

    # 문제리스트, 문제, 보기
    user_problem: str
    user_problem_options: List[str]
    
<<<<<<< HEAD
    vectorstore: Optional[Milvus]
=======
    vectorstore_p: Optional[Milvus]
    vectorstore_c: Optional[Milvus]
    vectorstore_config: Dict[str, str]
>>>>>>> origin

    retrieved_docs: List[Document]
    similar_questions_text : str

    # 문제 해답/풀이/과목 생성
    generated_answer: str         # 해답
    generated_explanation: str   # 풀이
    generated_subject: str

    results: List[Dict]
    validated: bool
    retry_count: int             # 검증 실패 시 재시도 횟수

    chat_history: List[str]
    
class SolutionAgent(BaseAgent):
    """문제 해답/풀이 생성 에이전트"""

    def __init__(self):
        self.graph = self._create_graph()
        
    @property
    def name(self) -> str:
        return "SolutionAgent"

    @property
    def description(self) -> str:
        return "시험문제를 인식하여 답과 풀이, 해설을 제공하는 에이전트입니다."

    def _llm(self, temperature: float = 0):
        return ChatOpenAI(
            api_key=SecretStr(OPENAI_API_KEY=REDACTED OPENAI_API_KEY=REDACTED None,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            model=OPENAI_LLM_MODEL,
            temperature=temperature,
        )

<<<<<<< HEAD
    def _create_graph(self):
=======
    # def _ensure_vectorstores(
    #     self,
    #     host: str = "localhost",
    #     port: str = "19530",
    #     coll_p: str = "problems",
    #     coll_c: str = "concepts",
    #     model_name: str = "jhgan/ko-sroberta-multitask",
    # ):
    def _ensure_vectorstores(
        self,
        coll: str,
        host: str = "localhost",
        port: str = "19530",
        *, text_field: str | None = None,
        vector_field: str | None = None,
        metric_type: str | None = None) -> Milvus:
        
        # 싱글톤 임베딩 모델 사용
        if not hasattr(self, '_cached_embedding_model'):
            try:
                self._cached_embedding_model = HuggingFaceEmbeddings(
                    model_name="jhgan/ko-sroberta-multitask",
                    model_kwargs={
                        "device": "cpu"
                    },
                    encode_kwargs={"normalize_embeddings": True}
                )
                print("✅ 임베딩 모델 캐시 생성 완료")
            except Exception as e:
                print(f"❌ 임베딩 모델 생성 실패: {e}")
                raise
        
        emb = self._cached_embedding_model
        if "default" not in connections.list_connections():
            connections.connect(alias="default", host=host, port=port)

        actual_metric = metric_type
        try:
            col = Collection(coll)
            if col.indexes:
                params = col.indexes[0].params or {}
                actual_metric = params.get("metric_type") or params.get("METRIC_TYPE") or actual_metric
        except Exception:
            pass
        if not actual_metric:
            actual_metric = "L2"

        kwargs = {
            "embedding_function": emb,
            "collection_name": coll,
            "connection_args": {"host": host, "port": port},
            "search_params": {"metric_type": actual_metric, "params": {"nprobe": 10}},
        }
        if text_field is not None:
            kwargs["text_field"] = text_field
        if vector_field is not None:
            kwargs["vector_field"] = vector_field
        return Milvus(**kwargs)


    def _reset_tunables(self):
        self.RETRIEVAL_FETCH_K = self.BASE_RETRIEVAL_FETCH_K
        self.HYBRID_TOPK       = self.BASE_HYBRID_TOPK
        self.RERANK_TOPK       = self.BASE_RERANK_TOPK
        self.HYBRID_ALPHA      = self.BASE_HYBRID_ALPHA

    def _build_concept_query(self, problem: str, options: List[str]) -> str:
        opts = "\n".join([f"{i+1}) {o}" for i, o in enumerate(options or [])])
        return f"{(problem or '').strip()}\n{opts}"
    
    def _split_blocks(self, text: str) -> list[str]:
        # 빈 블록 제거, 순서 유지
        if not isinstance(text, str) or not text.strip():
            return []
        return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    def _strip_md(self, s: str) -> str:
        if not s: return s
        # 코드펜스 제거
        s = re.sub(r"```(?:\w+)?\n([\s\S]*?)```", r"\1", s)
        # 인라인 코드 제거
        s = re.sub(r"`([^`]+)`", r"\1", s)
        # 굵게/밑줄 제거
        s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
        s = re.sub(r"__(.*?)__", r"\1", s)
        # 인용부호 제거
        s = re.sub(r"^>+\s?", "", s, flags=re.MULTILINE)
        return s

    def _normalize_digits(self, s: str) -> str:
        if not s: return s
        trans = str.maketrans({"①":"1","②":"2","③":"3","④":"4"})
        return s.translate(trans)
    
    def _extract_triplet(self, text: str, options: List[str]) -> tuple[Optional[int], str, str]:
        """
        LLM 응답에서 정답(번호), 풀이(텍스트), 과목(정규화) 3종을 뽑아냄.
        - 정답: '정답:' 줄의 숫자/원형 숫자/한글표현/텍스트(보기 내용) 모두 허용
        - 풀이: '풀이:'부터 다음 라벨(과목:) 전까지
        - 과목: 5개 셋으로 정규화
        """
        t = self._normalize_digits(self._strip_md(text))

        # 1) '정답:' 라인 파싱
        ans_num = None
        ans_line = None
        m_ans_line = re.search(r"정\s*답\s*[:：\-]\s*(.+)", t)
        if m_ans_line:
            ans_line = m_ans_line.group(1).strip()

        # (a) 숫자 직접 추출: 1~4 또는 '2번'
        if ans_line:
            m_num = re.search(r"\b([1-4])\b|([1-4])\s*번", ans_line)
            if m_num:
                ans_num = int(next(g for g in m_num.groups() if g))
            else:
                # (b) '①②③④'는 _normalize_digits가 처리함
                m_num2 = re.search(r"[1-4]", ans_line)
                if m_num2:
                    ans_num = int(m_num2.group(0))

        # (c) 텍스트로만 적힌 경우 → 보기와 유사도 매칭
        def _best_match_option(txt: str, opts: List[str]) -> Optional[int]:
            txt = (txt or "").strip()
            if not txt or not opts: return None
            best_i, best_s = None, 0.0
            for i, o in enumerate(opts, start=1):
                s = self._sim_ratio(txt, str(o))
                if s > best_s:
                    best_i, best_s = i, s
            return best_i if best_s >= 0.6 else None  # 임계값 0.6 정도

        if ans_num is None and ans_line and options:
            # '정답: 우선순위 스케줄링' 같은 패턴일 때
            # 괄호/따옴표 제거 후 매칭
            txt = re.sub(r"^[\(\[\{\"\']|[\)\]\}\"\']$", "", ans_line).strip()
            ans_num = _best_match_option(txt, options)

        # 2) 풀이: '풀이:' ~ '과목:' (또는 끝)
        expl = ""
        m_sol = re.search(r"풀이\s*[:：\-]\s*", t)
        if m_sol:
            start = m_sol.end()
            # 과목 라벨 찾기
            m_sub = re.search(r"\n?\s*과\s*목\s*[:：\-]\s*", t[start:])
            if m_sub:
                expl = t[start:start + m_sub.start()].strip()
            else:
                expl = t[start:].strip()
        else:
            # 라벨이 없으면 전체에서 '과목:' 이전을 풀이로 가정
            m_sub = re.search(r"과\s*목\s*[:：\-]\s*", t)
            expl = (t[:m_sub.start()] if m_sub else t).strip()

        # 3) 과목: 5개로 정규화
        subject_raw = ""
        m_subject = re.search(r"과\s*목\s*[:：\-]\s*([^\n\r]+)", t)
        if m_subject:
            subject_raw = m_subject.group(1).strip()

        SUBJECT_SET = {
            "소프트웨어설계": {"소프트웨어 설계", "소프트웨어-설계", "설계"},
            "소프트웨어개발": {"소프트웨어 개발", "개발"},
            "데이터베이스구축": {"데이터베이스 구축", "데이터베이스", "DB", "DB구축"},
            "프로그래밍언어활용": {"프로그래밍 언어 활용", "언어활용", "프언활"},
            "정보시스템구축관리": {"정보시스템 구축 관리", "정보시스템", "구축관리", "정시관"},
        }

        def _normalize_subject(s: str) -> str:
            s = re.sub(r"\s+", "", s)
            for canon, aliases in SUBJECT_SET.items():
                if s == canon or any(s == re.sub(r"\s+", "", a) for a in aliases):
                    return canon
            # 키워드 포함 매칭 (느슨)
            for canon, aliases in SUBJECT_SET.items():
                if any(a.replace(" ", "") in s for a in aliases | {canon}):
                    return canon
            return ""

        subject = _normalize_subject(subject_raw)

        # 최종 안전장치: 보기 범위 체크
        if ans_num is not None:
            if not (1 <= ans_num <= max(1, len(options or []))):
                ans_num = None

        return ans_num, expl, subject
    
    def _format_ctx_for_prompt(self, blocks, p_blocks, c_blocks, max_chars=900):
        """프롬프트용 컨텍스트 포맷터: [CTX i | 출처] 헤더 + per-block 트리밍"""
        formatted = []
        for i, b in enumerate(blocks, 1):
            if not b or not b.strip():
                continue
            src = "유사문제" if b in (p_blocks or []) else ("개념컨텍스트" if b in (c_blocks or []) else "기타")
            piece = b.strip()[:max_chars]
            formatted.append(f"[CTX {i} | {src}] {piece}")
        return "\n\n".join(formatted)

    # === RAGAS trace parser 안전 패치 ===
    @staticmethod
    def _patch_ragas_trace_parsing():
        try:
            import ragas.callbacks as _rcb
            _orig = getattr(_rcb, "parse_run_traces", None)
            if not callable(_orig) or getattr(_orig, "_patched_safe", False):
                return

            def _safe_parse_run_traces(*args, **kwargs):
                try:
                    run_traces = kwargs.get("run_traces", None)
                    if run_traces is None and args:
                        run_traces = args[0]
                    # 빈/None이면 바로 빈 딕셔너리 반환 → IndexError 근본 차단
                    if not run_traces:
                        return {}
                    if isinstance(run_traces, (list, tuple)) and len(run_traces) == 0:
                        return {}
                    return _orig(*args, **kwargs)
                except Exception as e:
                    print(f"[RAGAS] parse_run_traces bypass: {e}")
                    return {}

            _safe_parse_run_traces._patched_safe = True
            _rcb.parse_run_traces = _safe_parse_run_traces
            print("[RAGAS] Patched callbacks.parse_run_traces (safe on empty traces)")
        except Exception as e:
            print(f"[RAGAS] patch failed: {e}")

    # 패치 즉시 적용
    _patch_ragas_trace_parsing()

    def _route_after_validate(self, s: SolutionState) -> str:
        ra = (s.get("eval", {}) or {}).get("ragas", {}) or {}
        f = float(ra.get("faithfulness", 0) or 0.0)
        r = float(ra.get("answer_relevancy", 0) or 0.0)
        thr = (ra.get("thresholds", {}) or {})
        thr_f = float(thr.get("faithfulness", float(os.getenv("RAGAS_THR_FAITH", "0.6"))))
        thr_r = float(thr.get("answer_relevancy", float(os.getenv("RAGAS_THR_RELEVANCY", "0.6"))))

        if (f >= thr_f) and (r >= thr_r):
            return "ok"

        # 둘 다 실패하면 검색(retrieve) 우선
        if f < thr_f and int(s.get("retry_retrieve", 0) or 0) < self.MAX_RETRIEVE_RETRIES:
            return "requery"   # → retrieve_parallel

        if r < thr_r and int(s.get("retry_gen", 0) or 0) < self.MAX_GEN_RETRIES:
            s["pass_f"] = True         # ← 문자열 키로 저장
            return "regen"             # → generate_solution

        return "force_store"
    
    #----------------------------------------create graph------------------------------------------------------

    def _create_graph(self) -> StateGraph:
>>>>>>> origin
        """워크플로우 그래프 생성"""

        # ✅ LangGraph 구성
        print("📚 LangGraph 흐름 구성 중...")
        
        graph = StateGraph(SolutionState)

        # 공통 처리
        graph.add_node("search_similarity", self._search_similar_questions)
        graph.add_node("generate_solution", self._generate_solution)
        graph.add_node("validate", self._validate_solution)
        graph.add_node("store", self._store_to_vector_db)

        graph.set_entry_point("search_similarity")
        graph.add_edge("search_similarity", "generate_solution")
        graph.add_edge("generate_solution", "validate")
        graph.add_edge("store", END)

        graph.add_conditional_edges(
            "validate", 
            lambda s: "ok" if s["validated"] else ("back" if s.get("retry_count", 0) < 5 else END),
            {"ok": "store", "back": "generate_solution"}
        )
        return graph.compile()
    
    #----------------------------------------nodes------------------------------------------------------

    def _search_similar_questions(self, state: SolutionState) -> SolutionState:
        print("\n🔍 [1단계] 유사 문제 검색 시작")
        print(state["user_problem"], state["user_problem_options"])
<<<<<<< HEAD
        
        vectorstore = state.get("vectorstore")
        if vectorstore is None:
            print("⚠️ 벡터스토어가 없어 유사 문제 검색을 건너뜁니다.")
            state["retrieved_docs"] = []
            state["similar_questions_text"] = ""
            print("🔍 [1단계] 유사 문제 검색 함수 종료 (건너뜀)")
            return state
        
=======
            
        vectorstore_p = state.get("vectorstore_p")

        if vectorstore_p is None:
            # vectorstore_config를 사용하여 동적으로 생성
            config = state.get("vectorstore_config", {})
            if config:
                try:
                    vectorstore_p = self._ensure_vectorstores(
                        config.get("problems_coll", "problems"),
                        config.get("milvus_host", "localhost"),
                        config.get("milvus_port", "19530")
                    )
                    print("✅ vectorstore_p 동적 생성 완료")
                except Exception as e:
                    print(f"❌ vectorstore_p 생성 실패: {e}")
                    state["problems_contexts"] = []
                    state["problems_contexts_text"] = ""
                    return state
            else:
                print("⚠️ vectorstore_p없음 → 유사 문제 검색 건너뜀")
                state["problems_contexts"] = []
                state["problems_contexts_text"] = ""
                return state

        q = self._build_concept_query(state.get("user_problem",""), state.get("user_problem_options", []))

        # ---------- (1) Dense 후보 넉넉히 수집 ----------
>>>>>>> origin
        try:
            results = vectorstore.similarity_search(state["user_problem"], k=3)
        except Exception as e:
            print(f"⚠️ 유사 문제 검색 실패: {e}")
            results = []
        
        similar_questions = []
        for i, doc in enumerate(results):
            metadata = doc.metadata
            options = json.loads(metadata.get("options", "[]"))
            answer = metadata.get("answer", "")
            explanation = metadata.get("explanation", "")
            subject = metadata.get("subject", "기타")

            formatted = f"""[유사문제 {i+1}]
                문제: {doc.page_content}
                보기:
                """ + "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)]) + f"""
                정답: {answer}
                풀이: {explanation}
                과목: {subject}
                """
            similar_questions.append(formatted)
        
        state["retrieved_docs"] = results
        state["similar_questions_text"] = "\n\n".join(similar_questions) 

        print(f"유사 문제 {len(results)}개 검색 완료.")
        print("🔍 [1단계] 유사 문제 검색 함수 종료")
        return state

<<<<<<< HEAD
=======
    
    def _search_concepts_summary(self, state: SolutionState) -> SolutionState:
        print("\n📚 [1-확장] 개념 요약 컨텍스트 검색 시작")

        vectorstore_c = state.get("vectorstore_c")
        if vectorstore_c is None:
            # vectorstore_config를 사용하여 동적으로 생성
            config = state.get("vectorstore_config", {})
            if config:
                try:
                    vectorstore_c = self._ensure_vectorstores(
                        config.get("concept_coll", "concepts"),
                        config.get("milvus_host", "localhost"),
                        config.get("milvus_port", "19530"),
                        text_field="content",
                        vector_field="embedding"
                    )
                    print("✅ vectorstore_c 동적 생성 완료")
                except Exception as e:
                    print(f"❌ vectorstore_c 생성 실패: {e}")
                    state["concept_contexts"], state["concept_contexts_text"] = [], ""
                    return state
            else:
                print("⚠️ vectorstore_c 없음 → 개념 검색 건너뜀")
                state["concept_contexts"], state["concept_contexts_text"] = [], ""
                return state

        q = self._build_concept_query(state.get("user_problem",""), state.get("user_problem_options", []))


        # ---------- (1) Dense 후보 넉넉히 수집 ----------
        try:
            dense_scored = vectorstore_c.similarity_search_with_score(q, k=self.RETRIEVAL_FETCH_K)
            dense_docs   = [d for d, _ in dense_scored]
            dense_scores = {id(d): float(s) for d, s in dense_scored}
            print(f"[Dense(concepts)] fetched: {len(dense_docs)}")
        except Exception as e:
            print(f"[Dense(concepts)] similarity_search_with_score 실패 → {e} → score 없이 fallback")
            dense_docs   = vectorstore_c.similarity_search(q, k=self.RETRIEVAL_FETCH_K)
            dense_scores = {id(d): 1.0/(r+1) for r, d in enumerate(dense_docs)}

        # ---------- (2) Sparse 후보 결합 (BM25-lite over dense pool) ----------
        # 개념 코퍼스용 별도 BM25 인덱스가 없다면, dense 후보군 위에서만 BM25 점수 근사
        sparse_scores = {}
        try:
            if dense_docs and HAS_RANK_BM25:
                def tok(s: str) -> List[str]:
                    return re.findall(r"[가-힣A-Za-z0-9_]+", (s or "").lower())
                corpus_toks = [tok(d.page_content) for d in dense_docs]
                bm25 = BM25Okapi(corpus_toks)
                q_scores = bm25.get_scores(tok(q))
                # 0~1 정규화
                if q_scores is not None and len(q_scores) == len(dense_docs):
                    min_s, max_s = float(min(q_scores)), float(max(q_scores))
                    rng = (max_s - min_s) or 1.0
                    for d, s in zip(dense_docs, q_scores):
                        sparse_scores[id(d)] = (float(s) - min_s) / rng
                print(f"[BM25-lite(concepts)] computed over dense pool: {len(dense_docs)}")
            else:
                print("[BM25-lite(concepts)] 건너뜀 (dense_docs 비었거나 rank_bm25 미설치)")
        except Exception as e:
            print(f"[BM25-lite(concepts)] 실패 → {e}")

        # ---------- (3) Dense + Sparse 앙상블 ----------
        def _safe_meta_str(md: Dict[str, Any]) -> str:
            try:
                norm = {
                    str(k): (
                        v.item() if hasattr(v, "item") else (
                            str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                        )
                    )
                    for k, v in (md or {}).items()
                }
                return json.dumps(norm, ensure_ascii=False, sort_keys=True)
            except Exception:
                try:
                    return str({k: str(v) for k, v in (md or {}).items()})
                except Exception:
                    return ""

        def key_of(doc: Document) -> Tuple[str, str]:
            return ((doc.page_content or "")[:150], _safe_meta_str(doc.metadata)[:150])

        pool: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r, d in enumerate(dense_docs):
            k = key_of(d)
            pool.setdefault(k, {"doc": d, "dense": 0.0, "sparse": 0.0})
            wd = max(1.0/(r+1), dense_scores.get(id(d), 0.0))
            pool[k]["dense"] = max(pool[k]["dense"], wd)

        # BM25-lite 점수만 존재 (별도 sparse_docs 없음)
        for r, d in enumerate(dense_docs):
            k = key_of(d)
            if k not in pool:
                pool[k] = {"doc": d, "dense": 0.0, "sparse": 0.0}
            ws = max(1.0/(r+1), sparse_scores.get(id(d), 0.0))
            pool[k]["sparse"] = max(pool[k]["sparse"], ws)

        alpha = self.HYBRID_ALPHA
        scored = []
        for k, v in pool.items():
            score = alpha * v["dense"] + (1.0 - alpha) * v["sparse"]
            scored.append((v["doc"], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        hybrid_top = [d for d, _ in scored[:self.HYBRID_TOPK]]
        print(f"[Hybrid(concepts)] pool={len(pool)} → top{self.HYBRID_TOPK} 선정")

        # ---------- (4) Cross-Encoder rerank ----------
        try:
            if self.reranker is not None and len(hybrid_top) > 0:
                pairs  = [[q, d.page_content] for d in hybrid_top]
                scores = self.reranker.predict(pairs)
                order  = sorted(range(len(hybrid_top)), key=lambda i: float(scores[i]), reverse=True)
                reranked = [hybrid_top[i] for i in order[:self.RERANK_TOPK]]
                print(f"[Rerank(concepts)] 최종 top{self.RERANK_TOPK} (CrossEncoder)")
            else:
                reranked = hybrid_top[:self.RERANK_TOPK]
                print("[Rerank(concepts)] 사용 안 함 → hybrid_top 그대로 사용")
        except Exception as e:
            print(f"[Rerank(concepts)] 실패 → hybrid_top 그대로 사용: {e}")
            reranked = hybrid_top[:self.RERANK_TOPK]

        # ---------- (4.5) Near-duplicate 제거 & 상한 3개 ----------
        #   - 거의 같은 문단이 중복되는 현상 방지
        #   - 토큰 Jaccard 와 5-gram Jaccard 의 평균 유사도가 threshold 이상이면 중복으로 간주
        MAX_KEEP = getattr(self, "CONCEPT_MAXK", 3)  # 기본 3개로 제한
        SIM_THRESHOLD = getattr(self, "CONCEPT_SIM_THRESHOLD", 0.82)

        def _norm_text(s: str) -> str:
            s = (s or "").lower().strip()
            s = re.sub(r"\s+", " ", s)
            return s

        def _tokens(s: str) -> set:
            return set(re.findall(r"[가-힣a-z0-9]{2,}", _norm_text(s)))

        def _char_ngrams(s: str, n: int = 5) -> set:
            t = _norm_text(s)
            return set(t[i:i+n] for i in range(max(0, len(t) - n + 1)))

        def _jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            inter = len(a & b)
            if inter == 0:
                return 0.0
            return inter / float(len(a | b))

        def _similar(a_txt: str, b_txt: str) -> float:
            A_tok, B_tok = _tokens(a_txt), _tokens(b_txt)
            A_ng,  B_ng  = _char_ngrams(a_txt, 5), _char_ngrams(b_txt, 5)
            j_tok = _jaccard(A_tok, B_tok)
            j_ng  = _jaccard(A_ng,  B_ng)
            return 0.5 * (j_tok + j_ng)

        deduped = []
        for d in reranked:
            txt = (d.metadata.get("content") if d.metadata else None) or d.page_content or ""
            if not txt.strip():
                continue
            is_dup = False
            for kept in deduped:
                kept_txt = (kept.metadata.get("content") if kept.metadata else None) or kept.page_content or ""
                if _similar(txt, kept_txt) >= SIM_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(d)
            if len(deduped) >= MAX_KEEP:
                break

        print(f"[Dedup(concepts)] reranked={len(reranked)} → deduped={len(deduped)} (max={MAX_KEEP}, thr={SIM_THRESHOLD})")

        final_docs = deduped  # 최종 사용 문서(최대 3개)

        # ---------- (5) LLM 프롬프트용 정리 ----------
        chunks, cleaned_docs = [], []
        for idx, d in enumerate(final_docs, start=1):
            md = d.metadata or {}
            content = (md.get("content") or d.page_content or "").strip()
            subject = (md.get("subject") or "").strip()
            if not content and d.page_content:
                content = d.page_content.strip()

            cleaned = Document(page_content=content, metadata={"subject": subject})
            cleaned_docs.append(cleaned)

            chunks.append(f"과목: {subject} 내용: {content}")
            print(f" - [{idx}] subject='{subject}' content={content[:30]}...")

        state["concept_contexts"] = cleaned_docs
        state["concept_contexts_text"] = "\n\n".join(chunks)
        print(f"📚 개념 컨텍스트 {len(cleaned_docs)}개 수집")
        return state

    
    def _retrieve_parallel(self, state: SolutionState) -> SolutionState:

        tries = int(state.get("retry_retrieve", 0) or 0)
        # 예: fetch 폭 점증
        self.RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "30")) + tries*10
        self.HYBRID_TOPK       = int(os.getenv("HYBRID_TOPK", "12")) + min(tries*2, 8)
        # 필요하면 BM25 비중/알파도 약간 조정
        self.HYBRID_ALPHA      = max(0.3, min(0.7, float(os.getenv("HYBRID_ALPHA", "0.5")) - 0.05*tries))

        # state를 복사해서 각 작업이 독립적으로 수정하도록 함
        s1 = copy.deepcopy(state)
        s2 = copy.deepcopy(state)

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_sim = ex.submit(self._search_similar_problems, s1)
            f_con = ex.submit(self._search_concepts_summary, s2)
            r_sim = f_sim.result()
            r_con = f_con.result()

        # 결과 합치기
        state["problems_contexts"]        = r_sim.get("problems_contexts", [])
        state["problems_contexts_text"]= r_sim.get("problems_contexts_text", "")
        state["concept_contexts"]      = r_con.get("concept_contexts", [])
        state["concept_contexts_text"] = r_con.get("concept_contexts_text", "")

        # ✅ 각 소스에서 2개씩(환경변수로 조정 가능) 선별해 미리 저장
        p_blocks = self._split_blocks(state["problems_contexts_text"])
        c_blocks = self._split_blocks(state["concept_contexts_text"])
        p_sel = (p_blocks or [])[: self.USE_P_BLOCKS]
        c_sel = (c_blocks or [])[: self.USE_C_BLOCKS]

        selected_blocks = p_sel + c_sel
        state["ctx_blocks_used"] = selected_blocks

        # 디버그 로그
        print(f"[Parallel] similar_problems={len(state['problems_contexts'])}, "
              f"similar_concepts={len(state['concept_contexts'])}")
        print(f"[Parallel] selected for LLM -> problems:{len(p_sel)} + concepts:{len(c_sel)} = total:{len(selected_blocks)}")

        return state



>>>>>>> origin
    def _generate_solution(self, state: SolutionState) -> SolutionState:

        print("\n✏️ [2단계] 해답 및 풀이 생성 시작")

        llm_gen = self._llm(0.5)  

        similar_problems = state.get("similar_questions_text", "")
        print("유사 문제들:\n", similar_problems[:100])

        prompt = f"""
            사용자가 입력한 질문:
            {state['user_input_txt']}

            다음은 사용자가 입력한 문제:
            {state['user_problem']}
            {state['user_problem_options']}

            아래는 이 문제와 유사한 문제들:
            {similar_problems}

            1. 사용가자 입력한 문제의 **정답**의 보기 번호를 정답으로 작성해 주세요.
            2. 이어서 그 정답인 근거를 담은 **풀이 과정**을 상세히 설명해 주세요.
            3. 이 문제의 과목을 정보처리기사 과목 5개 중에서 가장 적합한 것으로 지정해 주세요. 유사 문제들의 과목을 참고해도 좋습니다. [소프트웨어설계, 소프트웨어개발, 데이터베이스구축, 프로그래밍언어활용, 정보시스템구축관리]

            출력 형식:
            정답: ...
            풀이: ...
            과목: ...
        """

        response = llm_gen.invoke(prompt)
        # response.content 가 list / str 둘 다 가능성
        raw_content: Union[str, List[Any]] = response.content  # type: ignore
        if isinstance(raw_content, list):
            # 메시지 조각 결합
            result = "\n".join([c if isinstance(c, str) else json.dumps(c, ensure_ascii=False) for c in raw_content])
        else:
            result = raw_content or ""
        result = result.strip()
        print("🧠 LLM 응답 완료")

        answer_match = re.search(r"정답:\s*(.+)", result)
        explanation_match = re.search(r"풀이:\s*(.+)", result, re.DOTALL)
        subject_match = re.search(r"과목:\s*(.+)", result)
        state["generated_answer"] = answer_match.group(1).strip() if answer_match else ""
        state["generated_explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        state["generated_subject"] = subject_match.group(1).strip() if subject_match else "기타"

        state["chat_history"].append(f"Q: {state['user_input_txt']}\nP: {state['user_problem']}\nA: {state['generated_answer']}\nE: {state['generated_explanation']}")

        return state

    # ✅ 정합성 검증 (간단히 길이 기준 사용)

    def _validate_solution(self, state: SolutionState) -> SolutionState:
        print("\n🧐 [3단계] 정합성 검증 시작")
        
        llm = self._llm(0)

        validation_prompt = f"""
        사용자 요구사항: {state['user_input_txt']}

        문제 질문: {state['user_problem']}
        문제 보기: {state['user_problem_options']}

        생성된 정답: {state['generated_answer']}
        생성된 풀이: {state['generated_explanation']}
        생성된 과목: {state['generated_subject']}

        생성된 해답과 풀이, 과목이 문제와 사용자 요구사항에 맞고, 논리적 오류나 잘못된 정보가 없습니까?
        적절하다면 '네', 그렇지 않다면 '아니오'로만 답변하세요.
        """

<<<<<<< HEAD
        validation_response = llm.invoke(validation_prompt)
        vr = validation_response.content  # type: ignore
        if isinstance(vr, list):
            vr_text = "\n".join([v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for v in vr])
=======
        def _norm(s: str) -> str:
            return (s or "").strip()

        # 0) 안전장치: 다시 한 번 트레이스 파서 패치 보장
        self._patch_ragas_trace_parsing()

        # 1) question 구성
        parts = []
        if _norm(state.get("user_input_txt")):
            parts.append(_norm(state.get("user_input_txt")))
        parts.append(_norm(state.get("user_problem")))
        opts = state.get("user_problem_options", []) or []
        if opts:
            parts.append("[보기]")
            parts.extend([f"{i+1}) {str(o)}" for i, o in enumerate(opts)])
        question_text = "\n".join([p for p in parts if p])

        # 2) answer 구성(원문 그대로)
        ans_num = _norm(state.get("generated_answer"))
        expl    = _norm(state.get("generated_explanation"))
        subj    = _norm(state.get("generated_subject"))
        answer_text = "\n".join([x for x in [
            f"정답: {ans_num}" if ans_num else "",
            expl,
            f"과목: {subj}" if subj else "",
        ] if x])

        # 3) contexts 구성(원문 그대로) + 폴백
        p_txt = _norm(state.get("problems_contexts_text", ""))
        c_txt = _norm(state.get("concept_contexts_text", ""))
        ctx_list: List[str] = [t for t in (p_txt, c_txt) if t]
        if not ctx_list:
            print("[RAGAS] contexts 텍스트 비어있음 → problems_contexts + concept_contexts 원문 결합 시도")
            buf = []
            for d in (state.get("problems_contexts") or []):
                if _norm(getattr(d, "page_content", "")):
                    buf.append(_norm(d.page_content))
                md = getattr(d, "metadata", {}) or {}
                if _norm(md.get("explanation")):
                    buf.append(_norm(md.get("explanation")))
            for d in (state.get("concept_contexts") or []):
                if _norm(getattr(d, "page_content", "")):
                    buf.append(_norm(d.page_content))
            ctx_list = ["\n\n".join([s for s in buf if _norm(s)]) or ""]
        # 최소 1개 보장
        if not ctx_list:
            ctx_list = [""]

        print(f"[RAGAS] contexts 원문 사용: {len(ctx_list)}개")

        # 4) SingleTurnSample + EvaluationDataset
        sample = SingleTurnSample(
            user_input=question_text,
            response=answer_text,
            retrieved_contexts=ctx_list,
            reference=None,   # ground_truth 없음
        )
        dataset = EvaluationDataset(samples=[sample])

        # 5) Wrappers
        eval_llm = ChatOpenAI(
            model=OPENAI_LLM_MODEL,
            base_url=OPENAI_BASE_URL,
            temperature=0.1,
            api_key=OPENAI_API_KEY=REDACTED = RagasLLMWrapper(eval_llm)

        emb = None
        for vs in (state.get("vectorstore_c"), state.get("vectorstore_p"),
                getattr(self, "vectorstore_c", None), getattr(self, "vectorstore_p", None)):
            try:
                ef = getattr(vs, "embedding_function", None)
                if ef is not None:
                    emb = ef
                    break
            except Exception:
                pass
        if emb is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            emb = HuggingFaceEmbeddings(
                model_name=os.getenv("RAGAS_EMBED_MODEL", "jhgan/ko-sroberta-multitask"),
                model_kwargs={
                    "device": os.getenv("RAGAS_EMBED_DEVICE", "cpu")
                },
                encode_kwargs={"normalize_embeddings": True}
            )
        emb_wrapped = RagasEmbWrapper(emb)

        # 6) 평가 함수 (콜백 완전 차단 → trace 파서 경로 차단)
        def _run(ds):
            return ragas_evaluate(
                ds,
                metrics=[faithfulness, answer_relevancy],
                llm=llm_wrapped,
                embeddings=emb_wrapped,
            )

        # 7) 실행 + 견고한 폴백
        try:
            res = _run(dataset)
            f_sc, r_sc = self._extract_scores(res)
        except Exception as e:
            print(f"[RAGAS] 1st pass failed: {e} -> retry with 1 ctx")
            subset = ctx_list[:1] or [""]
            dataset2 = EvaluationDataset(samples=[
                SingleTurnSample(
                    user_input=question_text,
                    response=answer_text,
                    retrieved_contexts=subset,
                    reference=None,
                )
            ])
            res = _run(dataset2)
            f_sc, r_sc = self._extract_scores(res)

        # ✅ 임계값: 먼저 정의
        thr_f = float(os.getenv("RAGAS_THR_FAITH", "0.6"))
        thr_r = float(os.getenv("RAGAS_THR_RELEVANCY", "0.6"))

        pass_f = (f_sc >= thr_f)
        pass_r = (r_sc >= thr_r)
        # ▶ 라우터가 남긴 'pass_f' 힌트가 있으면 한 번만 강제 통과로 처리
        if state.get("pass_f"):
            pass_f = True
            state.pop("pass_f", None)  # 재사용 방지 (중요)

        state["validated"] = bool(pass_f and pass_r)

        # ✅ 최신 점수/메타를 항상 기록 (validated 여부 무관)
        ts = datetime.now().isoformat(timespec="seconds")
        state.setdefault("eval", {})
        eval_record = {
            "faithfulness": f_sc,
            "answer_relevancy": r_sc,
            "thresholds": {"faithfulness": thr_f, "answer_relevancy": thr_r},
            "n_contexts": len(ctx_list),
            "timestamp": ts,
            "pass_faith": pass_f,
            "pass_ansrel": pass_r,
        }
        state["eval"]["ragas"] = eval_record

        # (선택) 히스토리 적재
        state.setdefault("ragas_history", [])
        # 히스토리에 넣을 땐 복사본 권장(참조 공유 방지)
        state["ragas_history"].append(eval_record.copy())

        # ✅ 실패 유형별로 카운터 분리 증가 (둘 다 실패면 검색 우선)
        if not pass_f:
            state["retry_retrieve"] = int(state.get("retry_retrieve", 0)) + 1
            print(f"✅ [RAGAS] faith={f_sc:.3f}(thr {thr_f}) | "
                f"ans_rel={r_sc:.3f}(thr {thr_r}) | "
                f"pass_f={pass_f} pass_r={pass_r} | "
                f"retry(retrieve/gen)={state['retry_retrieve']}/{state['retry_gen']}")
        elif not pass_r:
            state["retry_gen"] = int(state.get("retry_gen", 0)) + 1
            print(f"✅ [RAGAS] faith={f_sc:.3f}(thr {thr_f}) | "
                f"ans_rel={r_sc:.3f}(thr {thr_r}) | "
                f"pass_f={pass_f} pass_r={pass_r} | "
                f"retry(retrieve/gen)={state['retry_retrieve']}/{state['retry_gen']}")
>>>>>>> origin
        else:
            vr_text = (vr or "")
        result_text = vr_text.strip().lower()

        # ✅ '네'가 포함된 응답일 경우에만 유효한 풀이로 판단
        print("📌 검증 응답:", result_text)
        state["validated"] = "네" in result_text
        
        if not state["validated"]:
            state["retry_count"] = state.get("retry_count", 0) + 1
            print(f"⚠️ 검증 실패 (재시도 {state['retry_count']}/5)")
        else:
            print("✅ 검증 결과: 통과")
            
        return state


    # ✅ 임베딩 후 벡터 DB 저장
    def _store_to_vector_db(self, state: SolutionState) -> SolutionState:  
        print("\n🧩 [4단계] 임베딩 및 벡터 DB 저장 시작")

        vectorstore = state.get("vectorstore")
        if not vectorstore:
            print("⚠️ 벡터스토어 없음 – 저장 단계 건너뜀")
            return state

        try:
            similar = vectorstore.similarity_search(state["user_problem"], k=1)
        except Exception as e:
            print(f"⚠️ 중복 확인 실패 (유사 검색 오류): {e}")
            similar = []

        if similar and state["user_problem"].strip() in (similar[0].page_content or ""):
            print("⚠️ 동일한 문제가 존재하여 저장 생략")
        else:
            try:
                doc = Document(
                    page_content=state["user_problem"],
                    metadata={
                        "options": json.dumps(state.get("user_problem_options", [])),
                        "answer": state.get("generated_answer", ""),
                        "explanation": state.get("generated_explanation", ""),
                        "subject": state.get("generated_subject", ""),
                    }
                )
                vectorstore.add_documents([doc])
                print("✅ 문제+해답+풀이 저장 완료")
            except Exception as e:
                print(f"🚫 문서 저장 실패: {e}")

        print(f"\n📝 결과 저장 시작:")
        print(f"   - 현재 문제: {state['user_problem'][:50]}...")
        print(f"   - 생성된 정답: {state.get('generated_answer','')[:30]}...")
        print(f"   - 검증 상태: {state.get('validated')}")

        item = {
            "user_problem": state.get("user_problem", ""),
            "user_problem_options": state.get("user_problem_options", []),
            "generated_answer": state.get("generated_answer", ""),
            "generated_explanation": state.get("generated_explanation", ""),
            "generated_subject": state.get("generated_subject", ""),
            "validated": state.get("validated", False),
            "chat_history": state.get("chat_history", []),
        }
        state.setdefault("results", []).append(item)
        print(f"✅ 결과 저장 완료: {len(state['results'])}개")
        return state

    def invoke(
        self,
        user_input_txt: str,
        user_problem: str,
        user_problem_options: List[str],
        vectorstore: Optional[Milvus] = None,
        recursion_limit: int = 1000,
    ) -> Dict:
        # ✅ Milvus 연결 및 벡터스토어 생성
        if vectorstore is None:
            embedding_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={"device": "cpu"}
            )
            port = os.getenv("MILVUS_PORT", "19530")
            collection = os.getenv("MILVUS_COLLECTION", "problems")
            used_host = connect_milvus_fallback(port=port)
            if used_host:
                vectorstore = Milvus(
                    embedding_function=embedding_model,
                    collection_name=collection,
                    connection_args={"host": used_host, "port": port}
                )
            else:
                print("🚫 Milvus 연결 실패 → 벡터스토어 없이 진행")
                vectorstore = None

<<<<<<< HEAD
=======
        # # 1) 벡터스토어 설정 정보 준비 (Milvus 객체는 상태에 저장하지 않음)
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        problems_coll = os.getenv("PROBLEMS_COLL", "problems")
        concept_coll = os.getenv("CONCEPT_COLL", "concepts")
        
        # 벡터스토어 객체는 필요할 때마다 생성하도록 설정
        vectorstore_config = {
            "milvus_host": milvus_host,
            "milvus_port": milvus_port,
            "problems_coll": problems_coll,
            "concept_coll": concept_coll
        }

        self._reset_tunables()
        
>>>>>>> origin
        initial_state: SolutionState = {
            "user_input_txt": user_input_txt,
            "user_problem": user_problem,
            "user_problem_options": user_problem_options,
<<<<<<< HEAD
            "vectorstore": vectorstore,  # Optional
            "retrieved_docs": [],
            "similar_questions_text": "",
=======

            "vectorstore_p": None,  # Milvus 객체는 직렬화할 수 없으므로 None으로 설정
            "vectorstore_c": None,  # 필요할 때 vectorstore_config를 사용하여 생성
            "vectorstore_config": vectorstore_config,

            "problems_contexts": [],
            "problems_contexts_text": "",
            "concept_contexts": [],
            "concept_contexts_text": "",

>>>>>>> origin
            "generated_answer": "",
            "generated_explanation": "",
            "generated_subject": "",
            "validated": False,
            "retry_count": 0,
            "results": [],
            "chat_history": []
        }

        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})  # type: ignore

        results = final_state.get("results", [])
        print(f"   - 총 결과 수: {len(results)}")
        if not results:
            print("   ⚠️ results가 비어있습니다!")
            print(f"   - final_state 내용: {final_state}")
        return final_state


if __name__ == "__main__":

<<<<<<< HEAD
    # ✅ Milvus 연결 및 벡터스토어 생성
    embedding_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"}
    )
=======
    # --- app.py 참고한 벡터 연결 함수 ---
    def init_vectorstore(host: str, port: str, coll: str,
                         *, text_field: str | None = None,
                         vector_field: str | None = None,
                         metric_type: str | None = None) -> Milvus:
        emb = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={"normalize_embeddings": True}
        )
        if "default" not in connections.list_connections():
            connections.connect(alias="default", host=host, port=port)
>>>>>>> origin

    if "default" in connections.list_connections():
        connections.disconnect("default")
    connections.connect(alias="default", host="localhost", port="19530")

    vectorstore = Milvus(
        embedding_function=embedding_model,
        collection_name="problems",
        connection_args={"host": "localhost", "port":"19530"}
    )

    agent = SolutionAgent()

    # 그래프 시각화 (선택)
    # try:
    #     graph_image_path = "solution_agent_workflow.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(agent.graph.get_graph().draw_mermaid_png())
    #     print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류 발생: {e}")
    #     print("워크플로우는 정상적으로 작동합니다.")

    user_input_txt = input("\n❓ 사용자 질문: ").strip()
    user_problem = input("\n❓ 사용자 문제: ").strip()
    user_problem_options_raw = input("\n❓ 사용자 보기 (쉼표로 구분): ").strip()
    user_problem_options = [opt.strip() for opt in user_problem_options_raw.split(",") if opt.strip()]

    final_state = agent.invoke(
        user_input_txt=user_input_txt,
        user_problem=user_problem,
        user_problem_options=user_problem_options,
    )

    # # 결과를 JSON 파일로 저장
    # results = final_state.get("results", [])
    # results_data = {
    #     "timestamp": datetime.now().isoformat(),
    #     "user_input_txt": final_state.get("user_input_txt",""),
    #     "total_results": len(results),
    #     "results": results,
    # }

    # results_filename = os.path.join(f"solution_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    # os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    # with open(results_filename, "w", encoding="utf-8") as f:
    #     json.dump(results_data, f, ensure_ascii=False, indent=2)
    # print(f"✅ 해답 결과가 JSON 파일로 저장되었습니다: {results_filename}")
