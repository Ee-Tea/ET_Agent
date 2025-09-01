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
from ..base_agent import BaseAgent
from langchain_community.retrievers import BM25Retriever
import asyncio, sys
from concurrent.futures import ThreadPoolExecutor
import copy
from datasets import Dataset, Features, Value, Sequence
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
import json
import os, json, glob
from datetime import datetime
from langchain_milvus import Milvus
from pymilvus import connections, Collection
from difflib import SequenceMatcher

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
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED = os.getenv("GROQAI_API_KEY", "")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

ragas_llm = ChatOpenAI(
    api_key=GROQAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model=OPENAI_LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    max_tokens=min(LLM_MAX_TOKENS, 2048),
)

ragas_emb = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"}
)

# ✅ 상태 정의
class SolutionState(TypedDict):
    # 사용자 입력
    user_input_txt: str

    # 문제리스트, 문제, 보기
    user_problem: str
    user_problem_options: List[str]
    
    vectorstore_p: Milvus
    vectorstore_c: Milvus

    retrieved_docs: List[Document]
    problems_contexts_text : str
    concept_contexts: List[Document]
    concept_contexts_text: str

    # 문제 해답/풀이/과목 생성
    generated_answer: str         # 해답
    generated_explanation: str   # 풀이
    generated_subject: str

    extra_context_text: str
    ctx_blocks_used : List[str]

    results: List[Dict]
    validated: bool
    retry_count: int             # 검증 실패 시 재시도 횟수

    chat_history: List[str]
    
class SolutionAgent(BaseAgent):
    """문제 해답/풀이 생성 에이전트"""

    def __init__(self):
        # --- 하이브리드/리랭크 파라미터 ---
        self.RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "30"))
        self.HYBRID_TOPK       = int(os.getenv("HYBRID_TOPK", "12"))
        self.RERANK_TOPK       = int(os.getenv("RERANK_TOPK", "5"))
        self.HYBRID_ALPHA      = float(os.getenv("HYBRID_ALPHA", "0.5"))

        # --- 반드시 기본값으로 생성해 두기 ---
        self.bm25_retriever = None      # ← 없으면 AttributeError
        self.reranker = None            # ← 리랭커도 안전하게 기본값
        self.rerank_model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

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

    def _llm(self, temperature: float = 0):
        return ChatOpenAI(
            api_key=GROQAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=OPENAI_LLM_MODEL,
            temperature=temperature,
            max_tokens=min(LLM_MAX_TOKENS, 2048),
        )

    def _ensure_vectorstores(
        self,
        host: str = "localhost",
        port: str = "19530",
        coll_p: str = "problems",
        coll_c: str = "concepts",
        model_name: str = "jhgan/ko-sroberta-multitask",
    ):
        # 1) 이벤트 루프 보장
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if sys.platform.startswith("win"):
                try:
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                except Exception:
                    pass
            asyncio.set_event_loop(asyncio.new_event_loop())

        # 2) 동기 http 스킴으로 연결
        if "default" not in connections.list_connections():
            connections.connect(alias="default", uri=f"http://{host}:{port}")

        def infer_fields(coll_name: str):
            c = Collection(coll_name)
            vec_field, text_field, dim = None, None, None

            # 벡터 필드(FLOAT_VECTOR) 찾기
            for f in c.schema.fields:
                if f.dtype == DataType.FLOAT_VECTOR and vec_field is None:
                    vec_field = f.name
                    try:
                        dim = int(f.params.get("dim") or 0)
                    except Exception:
                        dim = None

            # 텍스트 필드는 후보군에서 우선 선택, 없으면 첫 VARCHAR 사용
            candidates = ("text", "page_content", "content", "question", "item_title", "title")
            varchar_fields = [f.name for f in c.schema.fields if f.dtype == DataType.VARCHAR]
            for name in candidates:
                if name in varchar_fields:
                    text_field = name
                    break
            if text_field is None and varchar_fields:
                text_field = varchar_fields[0]

            if vec_field is None:
                raise RuntimeError(f"[Milvus] '{coll_name}'에 FLOAT_VECTOR 필드가 없습니다.")
            print(f"[Milvus] '{coll_name}' → text_field='{text_field}', vector_field='{vec_field}', dim={dim}")
            return text_field, vec_field, dim

        emb = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})

        if self.vectorstore_p is None:
            txt_p, vec_p, _ = infer_fields(coll_p)
            self.vectorstore_p = Milvus(
                embedding_function=emb,
                collection_name=coll_p,
                connection_args={"uri": f"http://{host}:{port}"},
                text_field=txt_p,
                vector_field=vec_p,   # ← 자동 감지된 이름 사용 (대개 'vector')
            )
            print(f"✅ Milvus '{coll_p}' 연결 OK (text_field={txt_p}, vector_field={vec_p})")

        if self.vectorstore_c is None:
            txt_c, vec_c, _ = infer_fields(coll_c)
            self.vectorstore_c = Milvus(
                embedding_function=emb,
                collection_name=coll_c,
                connection_args={"uri": f"http://{host}:{port}"},
                text_field=txt_c,     # concepts라면 보통 'content'
                vector_field=vec_c,   # ← 여기서 반드시 'embedding'으로 잡힐 것
            )
            print(f"✅ Milvus '{coll_c}' 연결 OK (text_field={txt_c}, vector_field={vec_c})")

        
    
    def _build_concept_query(self, problem: str, options: List[str]) -> str:
        opts = "\n".join([f"{i+1}) {o}" for i, o in enumerate(options or [])])
        return f"{(problem or '').strip()}\n{opts}"
    
    @staticmethod
    def _safe_eval_metric(ds, metric, llm_obj, emb_obj, name: str, *,
                        features=None, question=None, answer=None, ctxs=None) -> float:
        from ragas import evaluate
        def _parse_score(res):
            score = 0.0
            if hasattr(res, "scores"):
                sc = res.scores.get(name, 0.0)
                if isinstance(sc, (list, tuple)):
                    score = float(sc[0]) if len(sc) else 0.0
                else:
                    try:
                        import numpy as _np, pandas as _pd
                        if isinstance(sc, _np.ndarray):
                            score = float(sc.item() if sc.size == 1 else sc[0])
                        elif isinstance(sc, _pd.Series):
                            score = float(sc.iloc[0])
                        else:
                            score = float(sc)
                    except Exception:
                        score = float(sc) if sc is not None else 0.0
            elif hasattr(res, "to_pandas"):
                df_res = res.to_pandas()
                if name in df_res.columns and len(df_res) > 0:
                    score = float(df_res[name].iloc[0])
            return score

        try:
            res = evaluate(ds, metrics=[metric], llm=llm_obj, embeddings=emb_obj)
            score = _parse_score(res)
            print(f"[RAGAS:{name}] {score:.3f}")
            return score
        except Exception as e:
            msg = repr(e)
            print(f"[RAGAS:{name}] 실패 → 1차 0.0 ({msg})")

            # 📌 IndexError는 멀티-컨텍스트 처리 중 생기는 알려진 이슈라 단일 블록으로 재시도
            if "IndexError" in msg and all(v is not None for v in (features, question, answer, ctxs)):
                try:
                    joined = "\n\n".join(ctxs)
                    ds2 = Dataset.from_dict(
                        {"question":[question], "answer":[answer], "contexts":[[joined]]},
                        features=features
                    )
                    print(f"[RAGAS:{name}] 단일 블록 재시도")
                    res2 = evaluate(ds2, metrics=[metric], llm=llm_obj, embeddings=emb_obj)
                    score2 = _parse_score(res2)
                    print(f"[RAGAS:{name}] (단일블록) {score2:.3f}")
                    return score2
                except Exception as e2:
                    print(f"[RAGAS:{name}] 단일 블록 재시도 실패 → 0.0 처리 ({repr(e2)})")
                    return 0.0
            return 0.0

        
    def _split_blocks(self, text: str) -> list[str]:
        # 빈 블록 제거, 순서 유지
        if not isinstance(text, str) or not text.strip():
            return []
        return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    def _sim_ratio(self, a: str, b: str) -> float:
        # 간단·빠른 유사도(토치/임베딩 불필요)
        return SequenceMatcher(a=a, b=b).ratio()

    def _sanitize_context_blocks(
        self,
        problems_ctx_text: str,
        concept_ctx_text: str,
        *,
        max_blocks: int = None,
        max_chars_per_block: int = None,
        sim_thresh: float = None,
    ) -> list[str]:
        """
        - 두 소스(유사문제/개념)에서 블록 나누기
        - 중복/유사 블록 제거(SequenceMatcher)
        - 블록 수 상한, 블록 길이 상한 적용
        """
        import os

        max_blocks = max_blocks or int(os.getenv("CTX_MAX_BLOCKS", "3"))          # 기본 3개
        max_chars_per_block = max_chars_per_block or int(os.getenv("CTX_MAX_CHARS_PER_BLOCK", "6000"))
        sim_thresh = sim_thresh or float(os.getenv("CTX_SIM_THRESH", "0.90"))     # 0~1, 높을수록 더 많이 제거

        blocks = []
        blocks += self._split_blocks(problems_ctx_text)
        blocks += self._split_blocks(concept_ctx_text)

        # 유사 중복 제거(앞선 블록을 우선)
        kept = []
        kept_norm = []
        for b in blocks:
            nb = re.sub(r"\s+", " ", b).strip().lower()
            if any(self._sim_ratio(nb, kb) >= sim_thresh for kb in kept_norm):
                continue
            kept.append(b)
            kept_norm.append(nb)

        # 블록 길이 상한 적용 (내용만 자르기, 의미 변화 없음)
        trimmed = [b[:max_chars_per_block] for b in kept]

        # 개수 제한
        return trimmed[:max_blocks]


    
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

        graph.add_conditional_edges(
            "validate", 
            # 검증 실패 → retry<5이면 back, 아니면 그냥 store로 진행
            lambda s: "ok" if s["validated"] else ("back" if s.get("retry_count", 0) < 5 else "force_store"),
            {"ok": "store", "back": "generate_solution", "force_store": "store"}
        )

        return graph.compile()
    
    #----------------------------------------nodes------------------------------------------------------

    def _search_similar_problems(self, state: SolutionState) -> SolutionState:
        print("\n🔍 [1단계] 유사 문제 검색 시작")
        print(state["user_problem"], state["user_problem_options"])
            
        vectorstore_p = state.get("vectorstore_p")

        if vectorstore_p is None:
            print("⚠️ vectorstore_p없음 → 유사 문제 검색 건너뜀")
            state["retrieved_docs"] = []
            state["problems_contexts_text"] = ""
            return state

        # q = state["user_problem"]
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

            formatted = f"""[유사문제 {i}]
                문제: {doc.page_content}
                보기:
                """ + "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)]) + f"""
                정답: {answer} ({answer_text})
                풀이: {explanation}
                과목: {subject}
                """
            similar_questions.append(formatted)

        state["retrieved_docs"] = results
        state["problems_contexts_text"] = "\n\n".join(similar_questions)

        print(f"유사 문제 {len(results)}개 (dense fetch={len(dense_docs)}, hybrid_pool={len(pool)})")
        print("🔍 [1단계] 유사 문제 검색 함수 종료")
        return state

    
    def _search_concepts_summary(self, state: SolutionState) -> SolutionState:
        print("\n📚 [1-확장] 개념 요약 컨텍스트 검색 시작")

        vectorstore_c = state.get("vectorstore_c")
        if vectorstore_c is None:
            print("⚠️ vectorstore_c 없음 → 개념 검색 건너뜀")
            state["concept_contexts"], state["concept_contexts_text"] = [], ""
            return state

        q = self._build_concept_query(
            state.get("user_problem", ""),
            state.get("user_problem_options", []),
        )

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

            chunks.append(f"과목: {subject}\n내용: {content}")
            print(f" - [{idx}] subject='{subject}' content={content[:30]}...")

        state["concept_contexts"] = cleaned_docs
        state["concept_contexts_text"] = "\n\n".join(chunks)
        print(f"📚 개념 컨텍스트 {len(cleaned_docs)}개 수집")
        return state

    
    def _retrieve_parallel(self, state: SolutionState) -> SolutionState:
        # state를 복사해서 각 작업이 독립적으로 수정하도록 함
        s1 = copy.deepcopy(state)
        s2 = copy.deepcopy(state)

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_sim = ex.submit(self._search_similar_problems, s1)
            f_con = ex.submit(self._search_concepts_summary, s2)
            r_sim = f_sim.result()
            r_con = f_con.result()

        # 결과 합치기
        state["retrieved_docs"]        = r_sim.get("retrieved_docs", [])
        state["problems_contexts_text"]= r_sim.get("problems_contexts_text", "")
        state["concept_contexts"]      = r_con.get("concept_contexts", [])
        state["concept_contexts_text"] = r_con.get("concept_contexts_text", "")

        # 디버그 로그
        print(f"[Parallel] similar_problems={len(state['retrieved_docs'])}, "
            f"similar_concepts={len(state['concept_contexts'])}")
        return state



    def _generate_solution(self, state: SolutionState) -> SolutionState:

        print("\n✏️ [2단계] 해답 및 풀이 생성 시작")

        llm_gen = self._llm(0.5)  

        problems_ctx_text = state.get("problems_contexts_text", "")
        concept_ctx_text  = state.get("concept_contexts_text", "")

        final_ctx_blocks = self._sanitize_context_blocks(
            problems_ctx_text, concept_ctx_text,
            # 필요하면 ENV로 조절. 기본 max_blocks=3, max_chars_per_block=6000, sim_thresh=0.90
        )

        # LLM 프롬프트에 넣는 ‘최종 컨텍스트’
        final_ctx_text = "\n\n".join(final_ctx_blocks)

        # ✅ 생성 단계에서 실제 사용한 컨텍스트를 저장 (검증에서 그대로 재사용)
        state["ctx_blocks_used"] = final_ctx_blocks
        state["ctx_used_text"]   = final_ctx_text

        def preview_context(ctx_text: str, label: str):
            print(f"{label} 전체 길이: {len(ctx_text)}")
            if not ctx_text.strip():
                print(f"{label}: (비어 있음)")
                return
            # 블록 단위로 분리 (두 줄 개행 기준)
            blocks = [b.strip() for b in ctx_text.split("\n\n") if b.strip()]
            for i, b in enumerate(blocks[:3], 1):   # 최대 3개까지만
                first_line = b.splitlines()[0]
                print(f" - {label} {i}: {first_line[:500]}...")  # 앞 120자만
            if len(blocks) > 3:
                print(f" ... (총 {len(blocks)}개 중 상위 3개만 표시)")

        # 출력
        preview_context(problems_ctx_text, "유사문제")
        preview_context(concept_ctx_text, "개념컨텍스트")
        preview_context(final_ctx_text, "최종컨텍스트")
        opts_lines = "\n".join(f"{i+1}) {o}" for i, o in enumerate(state['user_problem_options'] or []))


        prompt = f"""
            사용자가 입력한 질문:
            {state['user_input_txt']}

            다음은 사용자가 입력한 문제:
            {state['user_problem']}
            [보기]
            {opts_lines}

            아래는 이 문제 풀이에 사용할 최종 컨텍스트 블록(최대 3개)입니다:
            {final_ctx_text}


            1. 사용자가 입력한 문제의 **정답**을 의 보기 번호를 정답으로 작성해 주세요.
            2. 이어서 그 정답인 근거를 담은 **풀이 과정**을 상세히 설명해 주세요.
            3. 이 문제의 과목을 정보처리기사 과목 5개 중에서 가장 적합한 것으로 지정해 주세요.
                [소프트웨어설계, 소프트웨어개발, 데이터베이스구축, 프로그래밍언어활용, 정보시스템구축관리]
                (유사문제와 개념 요약 컨텍스트의 과목을 참고해도 좋습니다.)

            출력 형식:
            정답: ...
            풀이: ...
            과목: ...
        """

        response = llm_gen.invoke(prompt)
        result = response.content.strip()
        print("🧠 LLM 응답 완료")

        answer_match = re.search(r"정답:\s*(.+)", result)
        explanation_match = re.search(r"풀이:\s*(.+)", result, re.DOTALL)
        subject_match = re.search(r"과목:\s*(.+)", result)
        state["generated_answer"] = answer_match.group(1).strip() if answer_match else ""
        state["generated_explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        state["generated_subject"] = subject_match.group(1).strip() if subject_match else "기타"
        state["chat_history"].append(f"Q: {state['user_input_txt']}\nP: {state['user_problem']}\nA: {state['generated_answer']}\nE: {state['generated_explanation']}")

        p_blocks = self._split_blocks(problems_ctx_text)
        c_blocks = self._split_blocks(concept_ctx_text)
        p_count = sum(1 for b in final_ctx_blocks if b in p_blocks)
        c_count = sum(1 for b in final_ctx_blocks if b in c_blocks)
        print(f"[PromptCtx] 최종 컨텍스트: 총 {len(final_ctx_blocks)}개 (유사문제 {p_count}, 개념 {c_count})")

        return state


    def _validate_solution(self, state: SolutionState) -> SolutionState:
        """
        RAGAS 검증 (컨텍스트 무가공):
        - faithfulness / answer_relevancy
        - 검증용 LLM은 넉넉한 max_tokens/timeout
        - 스키마 명시 + 지표 분리 평가
        """
        print("\n🧐 [3단계] RAGAS 기반 정합성 검증 시작 (컨텍스트 무가공)")

        # 임계값
        f_min = float(os.getenv("VALIDATE_FAITHFULNESS_MIN", "0.65"))
        a_min = float(os.getenv("VALIDATE_ANS_REL_MIN", "0.55"))

        # 질의/답변 텍스트
        question = (state.get("user_input_txt") or "").strip()
        prob = (state.get("user_problem") or "").strip()
        opts = state.get("user_problem_options") or []
        if opts:
            opts_blob = "\n".join(f"{i+1}) {o}" for i, o in enumerate(opts))
            question = f"{question}\n\n[문제]\n{prob}\n\n[보기]\n{opts_blob}"

        answer = (
            f"정답: {state.get('generated_answer','')}\n"
            f"풀이: {state.get('generated_explanation','')}\n"
            f"과목: {state.get('generated_subject','')}"
        )

        # 컨텍스트(무가공)
        raw_context_candidates = [
            state.get("generation_context_text"),
            state.get("problems_contexts_text"),
            state.get("concept_contexts_text"),
            state.get("extra_context_text"),
        ]
        # 컨텍스트(무가공) → 이제는 ‘생성 시 사용한 정제본’을 그대로 사용
        ctx_used_text = state.get("ctx_used_text", "")
        if isinstance(ctx_used_text, str) and ctx_used_text.strip():
            # 생성 때 LLM에 들어간 단일 문자열 그대로
            ctxs = [ctx_used_text]
        else:
            # 혹시 누락됐으면 같은 정제 규칙으로 만든 뒤 '단일 문자열'로 합치기
            blocks = self._sanitize_context_blocks(
                state.get("problems_contexts_text", ""),
                state.get("concept_contexts_text", "")
            )
            ctxs = ["\n\n".join(blocks)] if blocks else []

        if not ctxs:
            print("⚠️ RAGAS 검증 스킵: 생성 시 전달된 컨텍스트 텍스트가 비었습니다.")
            state["validated"] = False
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

        print(f"[Validate] using {len(ctxs)} raw context block(s) exactly as used for generation")

        # ---------- 평가 입력 (스키마 명시) ----------
        features = Features({
            "question": Value("string"),
            "answer":   Value("string"),
            "contexts": Sequence(Value("string")),
        })
        data = {"question": [question], "answer": [answer], "contexts": [ctxs]}  # ctxs는 길이 1
        ds = Dataset.from_dict(data, features=features)

        # ---------- 검증용 LLM/임베딩 ----------
        VALIDATION_LLM_MODEL = os.getenv("VALIDATION_LLM_MODEL", os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"))
        VALIDATION_BASE_URL  = os.getenv("VALIDATION_BASE_URL",  os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        VALIDATION_API_KEY   = os.getenv("VALIDATION_API_KEY",   os.getenv("OPENAI_API_KEY=REDACTED = int(os.getenv("VALIDATION_LLM_MAX_TOKENS", "4096"))
        VALIDATION_TIMEOUT    = int(os.getenv("VALIDATION_LLM_TIMEOUT", "120"))

        validation_llm = ChatOpenAI(
            api_key=VALIDATION_API_KEY,
            base_url=VALIDATION_BASE_URL,
            model=VALIDATION_LLM_MODEL,
            temperature=0.0,
            max_tokens=VALIDATION_MAX_TOKENS,
            timeout=VALIDATION_TIMEOUT,
        )
        validation_emb = HuggingFaceEmbeddings(
            model_name=os.getenv("VALIDATION_EMB_MODEL", "jhgan/ko-sroberta-multitask"),
            model_kwargs={"device": "cpu"}
        )

        # ---------- ✅ 지표 분리 평가만 실행 (배치 평가 제거) ----------
        print(f"[RAGAS] evaluating 1 sample with thresholds f>={f_min}, a>={a_min} (tokens={VALIDATION_MAX_TOKENS}, timeout={VALIDATION_TIMEOUT}s)")
        print(f"[RAGAS] evaluating 1 sample with thresholds f>={f_min}, a>={a_min} (tokens={VALIDATION_MAX_TOKENS}, timeout={VALIDATION_TIMEOUT}s)")
        f = self._safe_eval_metric(
                ds, faithfulness, validation_llm, validation_emb, "faithfulness",
                features=features, question=question, answer=answer, ctxs=ctxs
        )
        a = self._safe_eval_metric(
                ds, answer_relevancy, validation_llm, validation_emb, "answer_relevancy",
                features=features, question=question, answer=answer, ctxs=ctxs
        )

        # ----- (지표 분리 평가 이미 수행한 뒤) f, a 값이 여기 있음 -----
        print(f"[RAGAS] faithfulness={f:.3f}, answer_relevancy={a:.3f}")

        # === 폴백: RAGAS가 두 지표 모두 실패(0.0)했을 때 임베딩 유사도로 대체 ===
        if (f == 0.0 and a == 0.0):
            print("[Fallback] RAGAS 두 지표가 실패 → 임베딩 유사도 기반 휴리스틱 평가로 대체")

            # 간단한 코사인 유사도
            import math
            def _cosine(u, v):
                if not u or not v: return 0.0
                num = sum(x*y for x, y in zip(u, v))
                den = math.sqrt(sum(x*x for x in u)) * math.sqrt(sum(y*y for y in v))
                return (num / den) if den else 0.0

            # 현재 검증용 임베딩 인스턴스 재사용 (validation_emb)
            try:
                q_vec = validation_emb.embed_query(question)
                a_vec = validation_emb.embed_query(answer)
                ctx_joined = "".join(ctxs)  # 내용 그대로 연결
                c_vec = validation_emb.embed_query(ctx_joined)

                a_fb = _cosine(q_vec, a_vec)   # answer relevancy 대체값
                f_fb = _cosine(c_vec, a_vec)   # faithfulness 대체값

                # 폴백 임계값(환경변수로 조정 가능)
                a_fb_min = float(os.getenv("FALLBACK_ANS_REL_MIN", "0.32"))
                f_fb_min = float(os.getenv("FALLBACK_FAITH_MIN",   "0.28"))

                print(f"[Fallback] emb-based answer_relevancy={a_fb:.3f} (min {a_fb_min})")
                print(f"[Fallback] emb-based faithfulness={f_fb:.3f} (min {f_fb_min})")

                # 폴백 점수는 본래 점수 대신 사용
                a, f = a_fb, f_fb
                a_min_eff = a_fb_min
                f_min_eff = f_fb_min
            except Exception as e:
                print(f"[Fallback] 임베딩 계산 실패: {repr(e)}")
                # 폴백도 실패하면 그대로 0점 유지
                a_min_eff = a_min
                f_min_eff = f_min
        else:
            a_min_eff = a_min
            f_min_eff = f_min

        # --- 최종 판정 ---
        ok = (f >= f_min_eff) and (a >= a_min_eff)
        state["validated"] = bool(ok)
        print(f"[Validate] final f={f:.3f} (min {f_min_eff}), a={a:.3f} (min {a_min_eff}) -> {'PASS' if ok else 'FAIL'}")

        # ---------- 📄 CSV 로깅 ----------
        # 경로: 기본 ./validation_results.csv (ENV로 변경 가능)
        CSV_PATH = os.getenv("VALIDATION_CSV_PATH", "./teacher/agents/solution/eval_results/validation_results.csv")
        os.makedirs(os.path.dirname(CSV_PATH) or ".", exist_ok=True)
        write_header = not os.path.exists(CSV_PATH)

        # 컨텍스트 미니 미리보기(첫 줄만)
        previews = []
        for b in ctxs[:3]:
            first_line = (b.splitlines()[0] if b.splitlines() else b)[:200]
            previews.append(first_line)
        contexts_preview = " || ".join(previews)

        row = {
            "validated": "PASS" if ok else "FAIL",
            "faithfulness": f,
            "answer_relevancy": a,
            "f_min": f_min,
            "a_min": a_min,
            "f_min_eff": f_min_eff,
            "a_min_eff": a_min_eff,
            "question_snippet": prob[:200],
            "options_count": len(opts),
            "generated_answer": state.get("generated_answer",""),
            "generated_subject": state.get("generated_subject",""),
            "explanation_snippet": (state.get("generated_explanation","") or "")[:300],
            "contexts_count": len(ctxs),
            "contexts_preview": contexts_preview,
        }

        try:
            with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(f"[CSV] 검증 결과 저장: {CSV_PATH}")
        except Exception as e:
            print(f"[CSV] 저장 실패: {repr(e)}")
            
        # 재시도 정책
        if not state["validated"]:
            state["retry_count"] = state.get("retry_count", 0) + 1
            if state["retry_count"] >= 5:
                print("⚠️ RAGAS 임계 미달 5회 → 결과를 저장 단계로 강제 진행")
            else:
                print(f"⚠️ RAGAS 임계 미달 → 재생성 시도 ({state['retry_count']}/5)")
        else:
            print("✅ RAGAS 검증 통과")

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

        # # ---------- 완전 일치 1개만: retrieved_docs[0] ----------
        # docs = state.get("retrieved_docs", []) or []
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
        state.setdefault("results", []).append({
            "user_problem":state.get("user_problem", "") or "",
            "user_problem_options": state.get("user_problem_options", []) or [],
            "generated_answer": state.get("generated_answer", ""),
            "generated_explanation": state.get("generated_explanation", ""),
            "generated_subject": state.get("generated_subject", ""),
            "validated": state.get("validated", False),
            "chat_history": state.get("chat_history", []),
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

        # # 1) 외부에서 하나라도 안 넘겼으면 내부 디폴트 준비
        # if vectorstore_p is None or vectorstore_c is None:
        #     self._ensure_vectorstores()

        # # 2) 최종으로 쓸 벡터스토어 결정 (외부 > 내부)
        # vs_p = vectorstore_p or self.vectorstore_p
        # vs_c = vectorstore_c or self.vectorstore_c

        # # (선택) 안전장치: 그래도 None이면 경고만 찍고 계속
        # if vs_p is None:
        #     print("⚠️ vectorstore_p가 없습니다. 유사 문제 검색이 비활성화됩니다.")
        # if vs_c is None:
        #     print("⚠️ vectorstore_c가 없습니다. 개념 검색이 비활성화됩니다.")
        
        initial_state: SolutionState = {
            "user_input_txt": user_input_txt,

            "user_problem": user_problem,
            "user_problem_options": user_problem_options,

            "vectorstore_p": vectorstore_p,
            "vectorstore_c": vectorstore_c,

            "retrieved_docs": [],
            "problems_contexts_text": "",
            "concept_contexts": [],
            "concept_contexts_text": "",

            "generated_answer": "",
            "generated_explanation": "",
            "generated_subject": "",
            "validated": False,
            "retry_count": 0,

            "results": [],
            
            "chat_history": []
        }
        
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
        
        # 그래프 시각화
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
                print(f"   - 결과 {i+1}: {result.get('question', '')[:30]}...")
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
    INSTRUCTION     = os.getenv("AGENT_INSTRUCTION", "정답 번호와 풀이, 과목을 알려줘.")  # ← input() 제거
    RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "200"))
    ONLY_INDEX      = int(os.getenv("AGENT_ONLY_INDEX", "0"))  # 0이면 전체, 1 이상이면 해당 문제(1-based)

    # --- app.py 참고한 벡터 연결 함수 ---
    def init_vectorstore(host: str, port: str, coll: str,
                         *, text_field: str | None = None,
                         vector_field: str | None = None,
                         metric_type: str | None = None) -> Milvus:
        emb = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={"device": "cpu"},
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

        # --- 콘솔 출력 ---
        print("\n================= 결과 =================")
        print(f"- 실행시각: {datetime.now().isoformat(timespec='seconds')}")
        print(f"- 입력파일: {jf}")
        for i, r in outputs:
            print(f"\n# 결과 {i}")
            print(f"- 정답(번호): {r.get('generated_answer','-')}")
            print(f"- 과목: {r.get('generated_subject','-')}")
            print(f"- 풀이:\n{r.get('generated_explanation','-')}")
        print("========================================\n")
