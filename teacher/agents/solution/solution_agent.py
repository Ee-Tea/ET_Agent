import csv
import os
from typing import TypedDict, List, Dict, Optional, Tuple, Any
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import connections, Collection, DataType
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
import json, re
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ..base_agent import BaseAgent
from langchain_community.retrievers import BM25Retriever
import asyncio, sys
from concurrent.futures import ThreadPoolExecutor
import copy
from datasets import Dataset, Features, Value, Sequence
import os
from collections.abc import Mapping

os.environ.setdefault("RAGAS_DISABLE_TRACING", "1")
os.environ.setdefault("OPENINFERENCE_DISABLED", "1")
os.environ.pop("LANGCHAIN_TRACING_V2", None)
os.environ.pop("LANGSMITH_API_KEY=REDACTED)

def _install_ragas_safe_parse():
    try:
        import ragas.callbacks as _cbs
        import ragas.dataset_schema as _ds

        # 원본 레퍼런스들
        _orig_cbs = getattr(_cbs, "parse_run_traces", None)
        _orig_ds  = getattr(_ds,  "parse_run_traces", None)

        # 이미 패치됐다면 무시
        if getattr(_orig_cbs, "_patched_safe", False) and getattr(_orig_ds, "_patched_safe", False):
            return

        def _safe_parse_run_traces(*args, **kwargs):
            """
            Accepts both positional and keyword calls:
            parse_run_traces(run_traces, run_id=...) / parse_run_traces(run_traces)
            Returns {} on empty/None traces or on any exception.
            """
            try:
                run_traces = kwargs.get("run_traces", None)
                if run_traces is None and args:
                    run_traces = args[0]
                # 빈/None → 안전 반환
                if not run_traces:
                    return {}
                if isinstance(run_traces, (list, tuple)) and len(run_traces) == 0:
                    return {}
                # 원본 중 살아있는 쪽을 호출
                target = _orig_cbs if callable(_orig_cbs) else _orig_ds
                if callable(target):
                    return target(*args, **kwargs)
                return {}
            except Exception as e:
                print(f"[RAGAS] parse_run_traces bypass: {e}")
                return {}

        # 두 모듈 모두 덮어쓰기 (dataset_schema는 import-by-value라 별도 패치 필수)
        _safe_parse_run_traces._patched_safe = True
        _cbs.parse_run_traces = _safe_parse_run_traces
        _ds.parse_run_traces  = _safe_parse_run_traces
        print("[RAGAS] Patched parse_run_traces in both callbacks and dataset_schema")
    except Exception as e:
        print(f"[RAGAS] safe patch not applied: {e}")

_install_ragas_safe_parse()
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
import os, json, glob
from datetime import datetime
from langchain_milvus import Milvus
from pymilvus import connections, Collection
from difflib import SequenceMatcher
# RAGAS 래퍼 & 데이터 스키마
from ragas.llms import LangchainLLMWrapper as RagasLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper as RagasEmbWrapper
from ragas.dataset_schema import SingleTurnSample

# RAGAS 지표
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate as ragas_evaluate

try:
    from ragas import EvaluationDataset  # 신버전 권장 경로
except Exception:
    from ragas.dataset_schema import EvaluationDataset  # 구버전 폴백
try:
    from ragas import EvaluationDataset  # 신버전
except Exception:
    from ragas.dataset_schema import EvaluationDataset  # 구버전 폴백


try:
    from rank_bm25 import BM25Okapi  # optional fallback(bm25 인덱스 없이 후보군 위에서 sparse 스코어링)
    HAS_RANK_BM25 = True
except Exception:
    HAS_RANK_BM25 = False

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except Exception:
    HAS_CROSS_ENCODER = False



# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED = os.getenv("OPENAI_LLM_MODEL", "gpt-o4-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
GROQAI_API_KEY = os.getenv("GROQAI_API_KEY", "")
GROQAI_LLM_MODEL = os.getenv("GROQAI_LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQAI_BASE_URL = os.getenv("GROQAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ✅ 상태 정의
class SolutionState(TypedDict):
    # 사용자 입력
    user_input_txt: str

    # 문제리스트, 문제, 보기
    user_problem: str
    user_problem_options: List[str]
    
    vectorstore_p: Optional[Milvus]
    vectorstore_c: Optional[Milvus]
    vectorstore_config: Dict[str, str]

    problems_contexts: List[Document]
    problems_contexts_text : str

    concept_contexts : List[Document]
    concept_contexts_text: str

    # 문제 해답/풀이/과목 생성
    generated_answer: str         # 해답
    generated_explanation: str   # 풀이
    generated_subject: str

    ctx_blocks_used : List[str]

    results: List[Dict]
    validated: bool
    retry_count: int
    retry_gen: int            # 생성 재시도 횟수
    retry_retrieve: int       # 검색 재시도 횟수

    chat_history: List[str]
    eval: Dict[str, Any]
    
class SolutionAgent(BaseAgent):
    """문제 해답/풀이 생성 에이전트"""

    def __init__(self):
        # --- 하이브리드/리랭크 파라미터 ---
        # 1) 베이스 값 저장
        self.BASE_RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "30"))
        self.BASE_HYBRID_TOPK       = int(os.getenv("HYBRID_TOPK", "12"))
        self.BASE_RERANK_TOPK       = int(os.getenv("RERANK_TOPK", "3"))
        self.BASE_HYBRID_ALPHA      = float(os.getenv("HYBRID_ALPHA", "0.5"))

        # 2) 현재(가변) 값 초기화
        self.RETRIEVAL_FETCH_K = self.BASE_RETRIEVAL_FETCH_K
        self.HYBRID_TOPK       = self.BASE_HYBRID_TOPK
        self.RERANK_TOPK       = self.BASE_RERANK_TOPK
        self.HYBRID_ALPHA      = self.BASE_HYBRID_ALPHA
        # --- 유사 컨텍스트 필터링 ---
        self.USE_P_BLOCKS = int(os.getenv("USE_PROBLEM_CTX", "2"))
        self.USE_C_BLOCKS = int(os.getenv("USE_CONCEPT_CTX", "2"))
        # --- 반드시 기본값으로 생성해 두기 ---
        self.bm25_retriever = None      # ← 없으면 AttributeError
        self.reranker = None            # ← 리랭커도 안전하게 기본값
        self.rerank_model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        # --- retry ---
        self.MAX_RETRIEVE_RETRIES = int(os.getenv("MAX_RETRIEVE_RETRIES", "5"))
        self.MAX_GEN_RETRIES = int(os.getenv("MAX_GEN_RETRIES", "5"))
        
        try:
            self.reranker = CrossEncoder(self.rerank_model_name, device=os.getenv("RERANK_DEVICE","cpu"))
            print(f"[Rerank] CrossEncoder loaded: {self.rerank_model_name}")
        except Exception as e:
            print(f"[Rerank] load skipped: {e}")

        # (선택) BM25 말뭉치가 있다면 로드
        bm25_jsonl = os.getenv("BM25_CORPUS_JSONL")
        if bm25_jsonl and os.path.exists(bm25_jsonl):
            docs = []
            with open(bm25_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        docs.append(Document(page_content=obj.get("page_content",""),
                                             metadata=obj.get("metadata",{})))
                    except Exception:
                        pass
            if docs:
                self.bm25_retriever = BM25Retriever.from_documents(docs)
                print(f"[BM25] 인덱스 문서 수: {len(docs)}")

        self.vectorstore_p = None
        self.vectorstore_c = None
        self.graph = self._create_graph()

    @property
    def name(self) -> str:
        return "SolutionAgent"

    @property
    def description(self) -> str:
        return "시험문제를 인식하여 답과 풀이, 해설을 제공하는 에이전트입니다."

    # def _llm(self, temperature: float = 0):
    #     return ChatOpenAI(
    #         api_key=GROQAI_API_KEY,
    #         base_url=GROQAI_BASE_URL,
    #         model=GROQAI_LLM_MODEL,
    #         temperature=temperature,
    #         max_tokens=min(LLM_MAX_TOKENS, 2048),
    #     )

    def _llm(self, temperature: float = 0):
        return ChatOpenAI(
            api_key=OPENAI_API_KEY=REDACTED=OPENAI_BASE_URL,
            model=OPENAI_LLM_MODEL,
            temperature=temperature,
            max_tokens=min(LLM_MAX_TOKENS, 2048),
        )

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
        """워크플로우 그래프 생성"""

        # ✅ LangGraph 구성
        print("📚 LangGraph 흐름 구성 중...")
        
        graph = StateGraph(SolutionState)

        # 공통 처리
        graph.add_node("search_problems", self._search_similar_problems)
        graph.add_node("search_concepts", self._search_concepts_summary)
        graph.add_node("retrieve_parallel", self._retrieve_parallel)
        graph.add_node("generate_solution", self._generate_solution)
        graph.add_node("validate", self._validate_solution)
        graph.add_node("store", self._store_to_vector_db)

        graph.set_entry_point("retrieve_parallel")
        graph.add_edge("retrieve_parallel", "generate_solution")
        graph.add_edge("generate_solution", "validate")
        graph.add_edge("store", END)

        # ✅ 지표 기반 분기
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {
                "ok": "store",
                "requery": "retrieve_parallel",   # faith 실패 → 재검색
                "regen": "generate_solution",     # ans_rel 실패 → 재생성
                "force_store": "store",
            },
        )
        return graph.compile()
    
    #----------------------------------------nodes------------------------------------------------------

    def _search_similar_problems(self, state: SolutionState) -> SolutionState:
        print("\n🔍 [1단계] 유사 문제 검색 시작")
        print(state["user_problem"], state["user_problem_options"])
            
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
        try:
            dense_scored = vectorstore_p.similarity_search_with_score(q, k=self.RETRIEVAL_FETCH_K)
            dense_docs = [d for d, _ in dense_scored]
            dense_scores = {id(d): float(s) for d, s in dense_scored}
            print(f"[Dense] fetched: {len(dense_docs)}")
        except Exception as e:
            print(f"[Dense] similarity_search_with_score 실패 → {e} → score 없이 fallback")
            dense_docs = vectorstore_p.similarity_search(q, k=self.RETRIEVAL_FETCH_K)
            dense_scores = {id(d): 1.0/(r+1) for r, d in enumerate(dense_docs)}

        # ---------- (2) Sparse 후보(BM25) 결합 ----------
        sparse_docs = []
        sparse_scores = {}

        if self.bm25_retriever is not None:
            try:
                sparse_docs = self.bm25_retriever.get_relevant_documents(q)[:self.RETRIEVAL_FETCH_K]
                for r, d in enumerate(sparse_docs):
                    sparse_scores[id(d)] = 1.0/(r+1)
                print(f"[BM25] fetched: {len(sparse_docs)}")
            except Exception as e:
                print(f"[BM25] 실패 → {e}")

        elif HAS_RANK_BM25 and dense_docs:
            try:
                def tok(s: str) -> List[str]:
                    return re.findall(r"[가-힣A-Za-z0-9_]+", (s or "").lower())
                corpus_toks = [tok(d.page_content) for d in dense_docs]
                bm25 = BM25Okapi(corpus_toks)
                q_scores = bm25.get_scores(tok(q))
                if q_scores is not None and len(q_scores) == len(dense_docs):
                    min_s, max_s = float(min(q_scores)), float(max(q_scores))
                    rng = (max_s - min_s) or 1.0
                    for d, s in zip(dense_docs, q_scores):
                        sparse_scores[id(d)] = (float(s) - min_s) / rng
                print(f"[BM25-lite] computed over dense pool: {len(dense_docs)}")
            except Exception as e:
                print(f"[BM25-lite] 실패 → {e}")

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

        pool: Dict[Tuple[str,str], Dict[str, Any]] = {}
        for r, d in enumerate(dense_docs):
            k = key_of(d)
            pool.setdefault(k, {"doc": d, "dense": 0.0, "sparse": 0.0})
            wd = 1.0/(r+1)
            wd = max(wd, dense_scores.get(id(d), 0.0))
            pool[k]["dense"] = max(pool[k]["dense"], wd)

        for r, d in enumerate(sparse_docs):
            k = key_of(d)
            pool.setdefault(k, {"doc": d, "dense": 0.0, "sparse": 0.0})
            ws = 1.0/(r+1)
            ws = max(ws, sparse_scores.get(id(d), 0.0))
            pool[k]["sparse"] = max(pool[k]["sparse"], ws)

        alpha = self.HYBRID_ALPHA
        scored = []
        for k, v in pool.items():
            score = alpha * v["dense"] + (1.0 - alpha) * v["sparse"]
            scored.append((v["doc"], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        hybrid_top = [d for d, _ in scored[:self.HYBRID_TOPK]]
        print(f"[Hybrid] pool={len(pool)} → top{self.HYBRID_TOPK} 선정")

        # ---------- (4) Cross-Encoder rerank ----------
        try:
            if self.reranker is not None and len(hybrid_top) > 0:
                pairs = [[q, d.page_content] for d in hybrid_top]
                scores = self.reranker.predict(pairs)
                order = sorted(range(len(hybrid_top)), key=lambda i: float(scores[i]), reverse=True)
                reranked = [hybrid_top[i] for i in order[:self.RERANK_TOPK]]
                print(f"[Rerank] 최종 top{self.RERANK_TOPK} (CrossEncoder)")
            else:
                reranked = hybrid_top[:self.RERANK_TOPK]
                print("[Rerank] 사용 안 함 → hybrid_top 그대로 사용")
        except Exception as e:
            print(f"[Rerank] 실패 → hybrid_top 그대로 사용: {e}")
            reranked = hybrid_top[:self.RERANK_TOPK]

        # ---------- (4.5) Near-duplicate 제거 & 상한 3개 ----------
        MAX_KEEP = getattr(self, "PROBLEM_MAXK", 3)              # 최종 유지 개수 (기본 3)
        SIM_THRESHOLD = getattr(self, "PROBLEM_SIM_THRESHOLD", 0.82)  # 유사도 임계값

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

        # 문제 본문(content) 기준으로 중복 제거(보기/메타는 보조)
        deduped = []
        for d in reranked:
            txt = (d.page_content or "").strip()
            if not txt:
                continue
            is_dup = False
            for kept in deduped:
                kept_txt = (kept.page_content or "").strip()
                if _similar(txt, kept_txt) >= SIM_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(d)
            if len(deduped) >= MAX_KEEP:
                break

        print(f"[Dedup(problems)] reranked={len(reranked)} → deduped={len(deduped)} (max={MAX_KEEP}, thr={SIM_THRESHOLD})")
        results = deduped  # 최종 문서들 (최대 3개)

        # ---------- (5) 기존 포맷/저장 ----------
        similar_questions = []
        for i, doc in enumerate(results, start=1):
            metadata = doc.metadata or {}
            options = json.loads(metadata.get("options", "[]" )) if isinstance(metadata.get("options"), str) else (metadata.get("options", []) or [])
            answer = metadata.get("answer", "")
            explanation = metadata.get("explanation", "")
            subject = metadata.get("subject", "기타")

            # 정답 번호 → 텍스트
            answer_text = ""
            try:
                answer_idx = int(answer) - 1
                if 0 <= answer_idx < len(options):
                    answer_text = options[answer_idx]
            except Exception:
                pass

            formatted = f"""[유사문제 {i}] 문제: {doc.page_content}
                보기:
                """ + "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)]) + f"""
                정답: {answer} ({answer_text})
                풀이: {explanation}
                과목: {subject}
                """
            similar_questions.append(formatted)

        state["problems_contexts"] = results
        state["problems_contexts_text"] = "\n\n".join(similar_questions)

        print(f"유사 문제 {len(results)}개 (dense fetch={len(dense_docs)}, hybrid_pool={len(pool)})")
        print("🔍 [1단계] 유사 문제 검색 함수 종료")
        return state

    
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



    def _generate_solution(self, state: SolutionState) -> SolutionState:

        print("\n✏️ [2단계] 해답 및 풀이 생성 시작")

        tries = int(state.get("retry_gen", 0) or 0)
        # 예: 살짝 다양성 부여
        llm = self._llm(temperature=min(0.5, 0.2 + 0.1*tries))

        # 1) 최종 블록 확정 (현행 그대로)
        preselected = state.get("ctx_blocks_used")
        if isinstance(preselected, list) and preselected:
            final_ctx_blocks = preselected
        else:
            p_blocks_all = self._split_blocks(state.get("problems_contexts_text", ""))
            c_blocks_all = self._split_blocks(state.get("concept_contexts_text", ""))
            final_ctx_blocks = (p_blocks_all[: self.USE_P_BLOCKS]) + (c_blocks_all[: self.USE_C_BLOCKS])

        # 2) 소스별 전체 블록(라벨링용) 준비
        problems_ctx_text = state.get("problems_contexts_text", "")
        concept_ctx_text  = state.get("concept_contexts_text", "")
        p_blocks_all = self._split_blocks(problems_ctx_text)
        c_blocks_all = self._split_blocks(concept_ctx_text)

        # 3) 프롬프트용 구조화 텍스트 만들기 (헤더 + per-block 트리밍)
        max_chars = int(os.getenv("PROMPT_CTX_MAX_CHARS", "900"))
        ctx_structured = self._format_ctx_for_prompt(final_ctx_blocks, p_blocks_all, c_blocks_all, max_chars=max_chars)

        # 4) 상태 저장(검증은 블록 리스트로 사용, 프롬프트는 구조화 텍스트)
        state["ctx_blocks_used"] = final_ctx_blocks

        def preview_context(ctx, label: str, head_chars: int = 500):
            """
            ctx: 문자열(\n\n로 블록 분리) 또는 블록 리스트 둘 다 지원
            """
            if isinstance(ctx, list):
                blocks = [str(b).strip() for b in ctx if b and str(b).strip()]
                total_len = sum(len(b) for b in blocks)
            else:
                ctx_text = (ctx or "")
                total_len = len(ctx_text)
                blocks = [b.strip() for b in ctx_text.split("\n\n") if b and b.strip()]

            print(f"{label} 전체 길이: {total_len}, 블록 수: {len(blocks)}")
            if not blocks:
                print(f"{label}: (비어 있음)")
                return

            for i, b in enumerate(blocks, 1):
                lines = b.splitlines()
                first = lines[0] if lines else b
                print(f" - {label} {i}: {first[:head_chars]}...")

        preview_context(problems_ctx_text, "유사문제")
        preview_context(concept_ctx_text, "개념컨텍스트")
        preview_context(ctx_structured, "최종컨텍스트(구조화)")

        opts_lines = "\n".join(f"{i+1}) {o}" for i, o in enumerate(state['user_problem_options'] or []))


        prompt = f"""
            사용자가 입력한 질문:
            {state['user_input_txt']}

            다음은 사용자가 입력한 문제:
            {state['user_problem']}
            [보기]
            {opts_lines}

            아래는 이 문제 풀이에 사용할 컨텍스트 블록들입니다.
            각 블록은 [CTX i | 출처] 머리글로 구분됩니다. 서로 다른 출처의 정보를 섞지 말고,
            문제 해결에 직접적으로 필요한 블록만 근거로 사용하세요.

            {ctx_structured}


            1. 사용자가 입력한 문제의 정답을 의 보기 번호를 정답으로 작성해 주세요.
            2. 이어서 그 정답인 근거를 담은 풀이 과정을 상세히 설명해 주세요.
            3. 이 문제의 과목을 정보처리기사 과목 5개 중에서 가장 적합한 것으로 지정해 주세요.
                [소프트웨어설계, 소프트웨어개발, 데이터베이스구축, 프로그래밍언어활용, 정보시스템구축관리]
                (유사문제와 개념 요약 컨텍스트의 과목을 참고해도 좋습니다.)
            4. 절대 마크다운(굵게, 기울임, 코드블록 등)을 사용하지 말고, 숫자와 텍스트는 평문으로만 작성하라.

            출력 형식:
            정답: ...
            풀이: ...
            과목: ...
        """

        # llm = self._llm()
        response = llm.invoke(prompt)
        result = response.content.strip()
        clean = self._strip_md(result)

        # NEW: 정답/풀이/과목 3종 한 번에 추출
        ans_idx, explanation, subject = self._extract_triplet(
            clean,
            state.get('user_problem_options') or []
        )

        state["generated_answer"] = str(ans_idx) if ans_idx is not None else ""
        state["generated_explanation"] = explanation
        state["generated_subject"] = subject
        state["chat_history"].append(f"Q: {state['user_input_txt']}\nP: {state['user_problem']}\nA: {state['generated_answer']}\nE: {state['generated_explanation']}")

        print("🧠 LLM 응답 완료")
        
        print(f" - 예측 정답 번호: {state['generated_answer']}")
        print(f" - 예측 과목: {state['generated_subject']}")
        print(f" - 풀이(앞 100자): {state['generated_explanation'][:100]}...")

        p_blocks = self._split_blocks(problems_ctx_text)
        c_blocks = self._split_blocks(concept_ctx_text)
        p_count = sum(1 for b in final_ctx_blocks if b in p_blocks)
        c_count = sum(1 for b in final_ctx_blocks if b in c_blocks)
        print(f"[PromptCtx] 최종 컨텍스트: 총 {len(final_ctx_blocks)}개 (유사문제 {p_count}, 개념 {c_count})")

        return state


    @staticmethod
    def _extract_scores(res) -> tuple[float, float]:
        def _as_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        def _from_mapping(m: Mapping) -> tuple[float, float] | None:
            if not isinstance(m, Mapping):
                return None
            f = _as_float(m.get("faithfulness", 0.0))
            r = _as_float(m.get("answer_relevancy", 0.0))
            return (f, r)

        f_sc = r_sc = 0.0
        try:
            # 0) 결과 자체가 dict/매핑인 경우
            got = _from_mapping(res) if isinstance(res, Mapping) else None
            if got is not None:
                print("[RAGAS] score dict: direct mapping", res)
                return got
                

            # 1) 신버전: res.scores 가 dict
            if hasattr(res, "scores") and isinstance(getattr(res, "scores"), dict):
                print("[RAGAS] score dict: new", res.scores)
                got = _from_mapping(res.scores)
                if got is not None:
                    return got

            # 2) 사전 변환기가 있으면 먼저 시도
            if hasattr(res, "to_dict"):
                try:
                    d = res.to_dict()
                    got = _from_mapping(d)
                    if got is not None:
                        print("[RAGAS] score dict: to_dict", d)
                        return got
                except Exception:
                    pass

            # 3) 구버전: pandas DataFrame
            if hasattr(res, "to_pandas"):
                print("[RAGAS] score dict: legacy via pandas", res)
                df = res.to_pandas()
                # 여러 샘플일 때 평균
                if "faithfulness" in df.columns:
                    f_sc = _as_float(df["faithfulness"].astype(float).mean())
                if "answer_relevancy" in df.columns:
                    r_sc = _as_float(df["answer_relevancy"].astype(float).mean())
                return f_sc, r_sc

        except Exception as e:
            print(f"[RAGAS] score parse fallback: {e}")

        return f_sc, r_sc  # 기본 0.0, 0.0


    def _validate_solution(self, state: SolutionState) -> SolutionState:
        """
        RAGAS 검증 (SingleTurnSample + Wrappers, ground_truth 없음)
        - question : user_input_txt + user_problem + user_problem_options (원문)
        - answer   : generated_answer / generated_explanation / generated_subject (원문 결합)
        - contexts : problems_contexts_text, concept_contexts_text (원문 그대로)
        - metrics  : faithfulness, answer_relevancy
        """
        print("\n🧪 [3단계] RAGAS 검증 시작")

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
        else:
            print(f"✅ [RAGAS] faith={f_sc:.3f}(thr {thr_f}) | "
                f"ans_rel={r_sc:.3f}(thr {thr_r}) | "
                f"pass_f={pass_f} pass_r={pass_r} | "
                f"retry(retrieve/gen)={state['retry_retrieve']}/{state['retry_gen']}")
            state["retry_retrieve"] = 0
            state["retry_gen"] = 0

        return state



    # ✅ 임베딩 후 벡터 DB 저장
    def _store_to_vector_db(self, state: SolutionState) -> SolutionState:

        # if not state.get("validated", False):
        #     print("⚠️ 검증 실패 상태 → 벡터DB 저장을 건너뛰고 종료합니다.")
        #     # (선택) 결과 로그는 남기고 싶으면 아래 유지, 완전 스킵하려면 이 블록을 지워도 됨
        #     state.setdefault("results", []).append({
        #         "user_problem": state.get("user_problem", "") or "",
        #         "user_problem_options": state.get("user_problem_options", []) or [],
        #         "generated_answer": state.get("generated_answer", ""),
        #         "generated_explanation": state.get("generated_explanation", ""),
        #         "generated_subject": state.get("generated_subject", ""),
        #         "validated": False,
        #         "chat_history": state.get("chat_history", []),
        #     })
        #     return state
        
        # vectorstore_p = state.get("vectorstore_p")
        # q    = state.get("user_problem", "") or ""
        # opts = state.get("user_problem_options", []) or []

        # from langchain_core.documents import Document
        # import json, hashlib

        # # ---------- helpers ----------
        # norm = lambda s: " ".join((s or "").split()).strip()

        # def parse_opts(v):
        #     if isinstance(v, str):
        #         try: v = json.loads(v)
        #         except: v = [v]
        #     return [norm(str(x)) for x in (v or [])]

        # def doc_id_of(q, opts):
        #     base = norm(q) + "||" + "||".join(parse_opts(opts))
        #     return hashlib.sha1(base.encode()).hexdigest()

        # def _clean_str(v):
        #     if isinstance(v, (bytes, bytearray)):
        #         try: v = v.decode("utf-8", "ignore")
        #         except Exception: v = str(v)
        #     if isinstance(v, str):
        #         s = v.strip()
        #         if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        #             try:
        #                 u = json.loads(s)
        #                 if isinstance(u, str): s = u.strip()
        #             except Exception:
        #                 pass
        #         if s.lower() in ("", "null", "none"):
        #             return ""
        #         return s
        #     return v

        # def _is_blank(v):
        #     v = _clean_str(v)
        #     if v is None: return True
        #     if isinstance(v, str): return v == ""
        #     try: return len(v) == 0
        #     except Exception: return False

        # def _escape(s: str) -> str:
        #     # Milvus expr용 이스케이프
        #     return s.replace("\\", "\\\\").replace('"', r"\"")

        # did = doc_id_of(q, opts)

        # # ---------- 완전 일치 1개만: problems_contexts[0] ----------
        # docs = state.get("problems_contexts", []) or []
        # exact = None
        # if docs:
        #     d = docs[0]
        #     same_q = norm(d.page_content) == norm(q)
        #     same_o = parse_opts(d.metadata.get("options", "[]")) == parse_opts(opts)
        #     print("[DEBUG] exact-match?", same_q and same_o,
        #         "| Q:", repr(norm(d.page_content)), "==", repr(norm(q)),
        #         "| OPTS:", parse_opts(d.metadata.get("options", "[]")), "==", parse_opts(opts))
        #     if same_q and same_o:
        #         exact = d

        # # ---------- 삭제→추가 (upsert) ----------
        # def upsert(meta, pk_to_delete=None, text_to_delete=None):
        #     if not vectorstore_p:
        #         print("⚠️ vectorstore_p 없음 → 저장 스킵(결과만 기록)")
        #         return
        #     # 1) PK로 삭제 (가장 안전)
        #     if pk_to_delete is not None:
        #         try:
        #             vectorstore_p.delete([pk_to_delete])
        #             print(f"[DEBUG] delete by PK ok: {pk_to_delete}")
        #         except Exception as e:
        #             print(f"[DEBUG] delete(ids=[{pk_to_delete}]) 실패: {e}")

        #     # 2) expr 삭제: 필드명 후보 순회 (컬렉션 스키마마다 다름)
        #     elif text_to_delete:
        #         expr_fields = ["text", "page_content", "content", "question"]
        #         esc = _escape(text_to_delete)
        #         for f in expr_fields:
        #             try:
        #                 vectorstore_p.delete(expr=f'{f} == "{esc}"')
        #                 print(f"[DEBUG] delete by expr ok: {f} == \"{esc}\"")
        #                 break
        #             except Exception as e:
        #                 print(f"[DEBUG] delete by expr 실패({f}): {e}")

        #     # 새 문서 추가
        #     vectorstore_p.add_documents([Document(
        #         page_content=q,
        #         metadata={
        #             # 참고용 fingerprint(스키마 필드는 아님)
        #             "doc_id": did,
        #             "options": json.dumps(opts, ensure_ascii=False),
        #             "answer":      meta.get("answer", "") or "",
        #             "explanation": meta.get("explanation", "") or "",
        #             "subject":     meta.get("subject", "") or "",
        #         },
        #     )])

        # if exact:
        #     meta = exact.metadata.copy()
        #     updated = False

        #     new_answer      = _clean_str(state.get("generated_answer"))
        #     new_explanation = _clean_str(state.get("generated_explanation"))
        #     new_subject     = _clean_str(state.get("generated_subject"))

        #     for k, new_val in [("answer", new_answer),
        #                     ("explanation", new_explanation),
        #                     ("subject", new_subject)]:
        #         cur_val   = meta.get(k)
        #         cur_blank = _is_blank(cur_val)
        #         new_blank = _is_blank(new_val)
        #         print(f"[DEBUG] {k}: current={repr(cur_val)} (blank={cur_blank}) "
        #             f"new={repr(new_val)} (blank={new_blank})")
        #         if cur_blank and not new_blank:
        #             meta[k] = new_val
        #             updated = True

        #     if updated:
        #         # PK 추출 시도 (환경에 따라 'pk'/'id'/'_id' 등일 수 있음)
        #         pk = None
        #         for k in ("pk", "id", "_id", "pk_id", "milvus_id"):
        #             if k in exact.metadata:
        #                 pk = exact.metadata[k]; break
        #         if pk is not None:
        #             upsert(meta, pk_to_delete=pk)
        #         else:
        #             # PK 못 찾으면 텍스트로 expr 삭제 시도
        #             upsert(meta, text_to_delete=q)
        #         print("✅ 동일 문항(완전 일치) 빈 컬럼만 채워 갱신")
        #     else:
        #         print("⚠️ 동일 문항(완전 일치) 존재, 저장 생략")
        # else:
        #     upsert({
        #         "answer": _clean_str(state.get("generated_answer")),
        #         "explanation": _clean_str(state.get("generated_explanation")),
        #         "subject": _clean_str(state.get("generated_subject")),
        #     }, text_to_delete=q)
        #     print("🆕 신규 문항 저장")

        # ---------- 결과 기록 ----------
        eval_info = (state.get("eval", {}) or {}).get("ragas", {}) or {}
        if not eval_info:
            # 폴백: 히스토리가 있으면 가장 최근 값 사용
            hist = state.get("ragas_history") or []
            if hist:
                last = hist[-1]
                eval_info = {
                    "faithfulness": last.get("faithfulness", 0.0),
                    "answer_relevancy": last.get("answer_relevancy", 0.0),
                    "thresholds": {
                        "faithfulness": float(os.getenv("RAGAS_THR_FAITH", "0.6")),
                        "answer_relevancy": float(os.getenv("RAGAS_THR_RELEVANCY", "0.6")),
                    },
                    "n_contexts": last.get("n_contexts", 0),
                    "timestamp": last.get("timestamp", ""),
                }

        state.setdefault("results", []).append({
            "user_problem":state.get("user_problem", "") or "",
            "user_problem_options": state.get("user_problem_options", []) or [],
            "generated_answer": state.get("generated_answer", ""),
            "generated_explanation": state.get("generated_explanation", ""),
            "generated_subject": state.get("generated_subject", ""),
            "retry_gen": int(state.get("retry_gen", 0) or 0),
            "retry_retrieve": int(state.get("retry_retrieve", 0) or 0),
            "validated": state.get("validated", False),
            "chat_history": state.get("chat_history", []),
            # RAGAS 메타(→ CSV로 직행)
            "ragas_faithfulness": float(eval_info.get("faithfulness", 0.0) or 0.0),
            "ragas_answer_relevancy": float(eval_info.get("answer_relevancy", 0.0) or 0.0),
            "ragas_thr_f": float(((eval_info.get("thresholds", {}) or {}).get("faithfulness", 0.0)) or 0.0),
            "ragas_thr_r": float(((eval_info.get("thresholds", {}) or {}).get("answer_relevancy", 0.0)) or 0.0),
            "ragas_n_contexts": int(eval_info.get("n_contexts", 0) or 0),
            "ragas_timestamp": eval_info.get("timestamp", ""),
            })
        return state

    def invoke(
            self, 
            user_input_txt: str,
            user_problem: str,
            user_problem_options: List[str],
            vectorstore_p: Optional[Milvus] = None,
            vectorstore_c: Optional[Milvus] = None,
            recursion_limit: int = 1000,
        ) -> Dict:  

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
        
        initial_state: SolutionState = {
            "user_input_txt": user_input_txt,
            "user_problem": user_problem,
            "user_problem_options": user_problem_options,

            "vectorstore_p": None,  # Milvus 객체는 직렬화할 수 없으므로 None으로 설정
            "vectorstore_c": None,  # 필요할 때 vectorstore_config를 사용하여 생성
            "vectorstore_config": vectorstore_config,

            "problems_contexts": [],
            "problems_contexts_text": "",
            "concept_contexts": [],
            "concept_contexts_text": "",

            "generated_answer": "",
            "generated_explanation": "",
            "generated_subject": "",
            "validated": False,

            "retry_count": 0,
            "retry_gen": 0,
            "retry_retrieve": 0,

            "results": [],
            
            "chat_history": [],

            "eval": {},   
            "ragas_history": [], 
        }
        
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
        
        # # 그래프 시각화
        # try:
        #     graph_image_path = "solution_agent_workflow.png"
        #     with open(graph_image_path, "wb") as f:
        #         f.write(self.graph.get_graph().draw_mermaid_png())
        #     print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
        # except Exception as e:
        #     print(f"그래프 시각화 중 오류 발생: {e}")
        #     print("워크플로우는 정상적으로 작동합니다.")

        # 결과 확인 및 디버깅
        results = final_state.get("results", [])
        print(f"   - 총 결과 수: {len(results)}")
        
        if results:
            for i, result in enumerate(results):
                print(f"   - 결과 {i+1}: {result.get('user_problem', '')[:30]}...")
        else:
            print("   ⚠️ results가 비어있습니다!")
            print(f"   - final_state 내용: {final_state}")
        return final_state


# ====== replace the entire __main__ block in solution_agent.py ======
if __name__ == "__main__":
    # ----------------------------
    # 고정 실행 파라미터 (원하면 여기만 수정)
    # ----------------------------
    JSON_DIR        = os.getenv("PROBLEMS_JSON_DIR", "./teacher/exam/test_parsed_exam_json")  # 폴더 경로
    MILVUS_HOST     = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT     = os.getenv("MILVUS_PORT", "19530")
    PROBLEMS_COLL   = os.getenv("PROBLEMS_COLL", "problems")
    CONCEPT_COLL    = os.getenv("CONCEPT_COLL", "concepts")
    INSTRUCTION     = os.getenv("AGENT_INSTRUCTION", "이 문제의 정답 번호와 풀이, 그리고 과목을 알려줘.")  # ← input() 제거
    RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "200"))
    ONLY_INDEX      = int(os.getenv("AGENT_ONLY_INDEX", "10"))  # 0이면 전체, 1 이상이면 해당 문제(1-based)

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

    # --- JSON 폴더 내 파일 목록 ---
    if not os.path.isdir(JSON_DIR):
        raise FileNotFoundError(f"문제 JSON 폴더를 찾을 수 없습니다: {JSON_DIR}")
    json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
    if not json_files:
        raise ValueError(f"{JSON_DIR} 안에 .json 파일이 없습니다.")

    # --- Milvus 벡터스토어 초기화 ---
    vectorstore_p = init_vectorstore(MILVUS_HOST, MILVUS_PORT, PROBLEMS_COLL)
    vectorstore_c = init_vectorstore(
        MILVUS_HOST, MILVUS_PORT, CONCEPT_COLL,
        text_field="content",
        vector_field="embedding",
    )

    agent = SolutionAgent()

    def run_one(p: dict) -> dict:
        return agent.invoke(
            user_input_txt=INSTRUCTION,
            user_problem=p.get("question", "") or "",
            user_problem_options=p.get("options", []) or [],
            vectorstore_p=vectorstore_p,
            vectorstore_c=vectorstore_c,
            recursion_limit=RECURSION_LIMIT,
        )

    # --- 각 파일 순회 실행 ---
    for jf in json_files:
        print(f"\n=== JSON 파일 처리 시작: {jf} ===")
        with open(jf, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 1) 파일 구조: dict에 "questions"가 있으면 그걸 사용, 아니면 list 그대로 사용
        if isinstance(raw, dict) and isinstance(raw.get("questions"), list):
            items = raw["questions"]
        elif isinstance(raw, list):
            items = raw
        else:
            raise ValueError(f"{jf}: 지원하지 않는 JSON 구조 (list 또는 {{'questions':[...]}} )")

        # 2) ✅ 인덱싱만: question / options 두 필드만 뽑아서 전달
        #    - options가 list가 아니거나 없는 항목은 건너뜀 (불필요한 정규화는 하지 않음)
        problems = []
        for it in items:
            if not isinstance(it, dict):
                continue
            q = it.get("question")
            opts = it.get("options")
            if isinstance(q, str) and isinstance(opts, list):
                problems.append({"question": q, "options": opts})

        if not problems:
            print(f"[WARN] {jf}: question/options 형식의 문제를 찾지 못했습니다. 건너뜀.")
            continue

        print(f"[LOAD] {jf}: {len(problems)}문항 (question/options만 사용)")

        outputs = []
        if ONLY_INDEX and ONLY_INDEX > 0:
            idx = ONLY_INDEX
            if not (1 <= idx <= len(problems)):
                raise IndexError(f"--index={idx} 범위 벗어남 (1..{len(problems)}) in {jf}")
            res_state = run_one(problems[idx - 1])
            outputs.append((idx, (res_state.get("results") or [{}])[-1]))
        else:
            for i, p in enumerate(problems, 1):
                res_state = run_one(p)
                outputs.append((i, (res_state.get("results") or [{}])[-1]))

         # --- CSV: 입력 JSON 파일별로 별도 생성/누적 저장 ---
        CSV_DIR = os.getenv("RAGAS_CSV_DIR", "./teacher/agents/solution/eval_results")
        os.makedirs(CSV_DIR, exist_ok=True)

        # 입력 파일명 그대로 사용하되 확장자는 .csv 로
        base_name = os.path.splitext(os.path.basename(jf))[0]
        CSV_PATH = os.path.join(CSV_DIR, f"{base_name}.csv")

        # 필요 컬럼 정의 (answer_explanation을 CSV에 포함하려면 필드에도 추가)
        CSV_FIELDS = [
            "timestamp", "file", "index",
            "question", "options",
            "answer_pred", "subject_pred", "validated", 
            "retry_count", "retry_retrieve", "retry_gen",
            "faithfulness", "answer_relevancy", "thr_f", "thr_r", "n_contexts",
            "answer_explanation",  # ← 설명 컬럼도 저장하고 싶다면 유지, 아니면 제거
        ]

        def _to_json_str(x):
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(x)

        def append_rows(rows, path=CSV_PATH):
            write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
            # Excel로 바로 여는 경우 utf-8-sig 권장 (일반 뷰어/파이프라인이면 utf-8 그대로 사용해도 무방)
            with open(path, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if write_header:
                    w.writeheader()
                for row in rows:
                    w.writerow(row)

        now = datetime.now().isoformat(timespec='seconds')
        rows = []
        for i, r in outputs:
            rows.append({
                "timestamp": now,
                "file": os.path.basename(jf),
                "index": i,
                "question": (r.get("user_problem") or "")[:500],
                "options": _to_json_str(r.get("user_problem_options", [])),
                "answer_pred": r.get("generated_answer", ""),
                "subject_pred": r.get("generated_subject", ""),
                "validated": r.get("validated", False),
                "retry_gen": int(r.get("retry_gen", 0) or 0),
                "retry_retrieve": int(r.get("retry_retrieve", 0) or 0),
                "retry_count": int(r.get("retry_gen", 0) or 0) + int(r.get("retry_retrieve", 0) or 0),
                # RAGAS 메타
                "faithfulness": r.get("ragas_faithfulness", ""),
                "answer_relevancy": r.get("ragas_answer_relevancy", ""),
                "thr_f": r.get("ragas_thr_f", ""),
                "thr_r": r.get("ragas_thr_r", ""),
                "n_contexts": r.get("ragas_n_contexts", ""),
                "answer_explanation": (r.get("generated_explanation") or ""),  # CSV_FIELDS에 없애면 이 줄도 제거
            })

        append_rows(rows)
        print(f"[CSV] {len(rows)}개 행을 '{CSV_PATH}'에 추가했습니다.")

        # --- 콘솔 출력 ---
        print("\n================= 결과 =================")
        print(f"- 실행시각: {datetime.now().isoformat(timespec='seconds')}")
        print(f"- 입력파일: {jf}")
        for i, r in outputs:
            print(f"   - 결과 {i+1}: {(r.get('user_problem', '') or '')[:30]}...")
            print(f"- 정답(번호): {r.get('generated_answer','-')}")
            print(f"- 과목:{r.get('generated_subject','-')}")
            print(f"- 풀이:{r.get('generated_explanation','-')}")
            print(f"- faith={r.get('ragas_faithfulness')}, ans_rel={r.get('ragas_answer_relevancy')}, valid={r.get('validated')}")
        print("========================================\n")


        
