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
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import asyncio, sys
from concurrent.futures import ThreadPoolExecutor
import copy

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import InjectedToolCallId, tool


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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQAI_API_KEY = os.getenv("GROQAI_API_KEY", "")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    
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

    results: List[Dict]
    validated: bool
    retry_count: int             # 검증 실패 시 재시도 횟수

    chat_history: List[str]
    
    # 풀이 평가 및 피드백 관련
    solution_score: float        # 풀이 품질 점수 (0-100)
    feedback_analysis: str      # 사용자 피드백 분석 결과
    needs_improvement: bool     # 풀이 개선 필요 여부
    improved_solution: str      # 개선된 풀이
    search_results: str         # 검색 결과
    
    # 테스트 모드 관련
    test_mode: bool             # 테스트 모드 활성화 여부
    test_score: int             # 테스트용 강제 점수
    test_feedback_type: str     # 테스트용 강제 피드백 타입

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
        
        # checkpointer 추가
        self.checkpointer = InMemorySaver()
        self.graph = self._create_graph()


    def _ensure_vectorstores(
        self,
        host: str = "localhost",
        port: str = "19530",
        coll_p: str = "problems",
        coll_c: str = "concept_summary",
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
                text_field=txt_c,     # concept_summary라면 보통 'content'
                vector_field=vec_c,   # ← 여기서 반드시 'embedding'으로 잡힐 것
            )
            print(f"✅ Milvus '{coll_c}' 연결 OK (text_field={txt_c}, vector_field={vec_c})")

        
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
        )
    
    def _evaluate_solution(self, state: SolutionState) -> SolutionState:
        """LLM을 활용하여 풀이의 정확성과 설명의 충분함을 평가합니다."""
        print("\n📊 [평가] 풀이 품질 평가 시작")
        
        evaluation_prompt = f"""
        다음 문제와 풀이를 평가하여 0-100점 사이의 점수를 매기고 개선이 필요한지 판단해주세요.

        문제: {state.get('user_problem', '')}
        보기: {state.get('user_problem_options', [])}
        정답: {state.get('generated_answer', '')}
        풀이: {state.get('generated_explanation', '')}

        다음 기준으로 평가해주세요:
        1. 정확성 (40점): 정답이 맞는지, 논리가 올바른지
        2. 설명의 충분함 (30점): 단계별 설명이 충분한지, 이해하기 쉬운지
        3. 용어 설명 (20점): 전문 용어나 개념에 대한 설명이 있는지
        4. 전체적인 품질 (10점): 전체적으로 잘 작성되었는지

        반드시 다음 JSON 형태로만 응답해주세요. 다른 텍스트는 포함하지 마세요:
        {{
            "score": 점수,
            "needs_improvement": true/false,
            "improvement_reasons": ["개선이 필요한 이유들"],
            "evaluation_summary": "평가 요약"
        }}

        점수가 70점 미만이면 needs_improvement를 true로 설정해주세요.
        """
        
        try:
            llm_evaluator = self._llm(0.3)
            evaluation_response = llm_evaluator.invoke(evaluation_prompt)
            
            # JSON 파싱 - 더 안전하게 처리
            import json
            import re
            
            response_content = evaluation_response.content.strip()
            print(f"🔍 LLM 평가 응답: {response_content[:200]}...")
            
            # JSON 부분만 추출 (```json ... ``` 형태일 수 있음)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # JSON 객체만 찾기
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_content
            
            try:
                evaluation_result = json.loads(json_str)
                print("✅ JSON 파싱 성공")
            except json.JSONDecodeError as json_err:
                print(f"⚠️ JSON 파싱 실패, 기본값 사용: {json_err}")
                # 기본값으로 평가 결과 생성
                evaluation_result = {
                    "score": 70.0,
                    "needs_improvement": True,
                    "improvement_reasons": ["LLM 응답 파싱 실패"],
                    "evaluation_summary": "평가 중 오류 발생"
                }
            
            state["solution_score"] = evaluation_result.get("score", 70.0)
            # state["needs_improvement"] = evaluation_result.get("needs_improvement", True)
            state["needs_improvement"] = True
            
            print(f"📊 풀이 평가 점수: {state['solution_score']}/100")
            print(f"📊 개선 필요: {state['needs_improvement']}")
            
        except Exception as e:
            print(f"⚠️ 풀이 평가 중 오류 발생: {e}")
            state["solution_score"] = 70.0
            state["needs_improvement"] = True
        
        return state

    def _collect_user_feedback(self, state: SolutionState) -> SolutionState:
        """사용자로부터 피드백을 수집하고 분석합니다."""
        print("\n💬 [피드백] 사용자 피드백 수집 시작")
        
        if not state.get("needs_improvement", False):
            print("💬 개선이 필요하지 않아 피드백 수집을 건너뜁니다.")
            return state
        
        # Command(resume)을 통해 전달된 사용자 피드백 처리
        # 이 노드는 tools 노드에서 interrupt 후 resume 시 실행됨
        # 사용자 피드백은 이미 state에 포함되어 있음
        
        # 피드백 분석을 위한 상태 설정
        state["feedback_analysis"] = "easier_explanation"  # 기본값
        print("💬 피드백 수집을 위한 상태 설정 완료")
        
        return state

    def _improve_solution(self, state: SolutionState) -> SolutionState:
        """사용자 피드백을 바탕으로 풀이를 개선합니다."""
        print("\n🔧 [개선] 풀이 개선 시작")
        
        if not state.get("needs_improvement", False):
            print("🔧 개선이 필요하지 않아 풀이 개선을 건너뜁니다.")
            return state
        
        feedback = state.get("feedback_analysis", "understand")
        original_solution = state.get("generated_explanation", "")
        
        improvement_prompt = f"""
        다음 피드백을 바탕으로 풀이를 개선해주세요.

        원본 풀이: {original_solution}
        사용자 피드백: {feedback}

        피드백 분석 결과에 따라:
        - "easier_explanation": 더 쉬운 단계별 설명으로 개선
        - "term_explanation": 전문 용어와 개념에 대한 자세한 설명 추가
        - "term_easier_explanation": 용어 설명과 쉬운 설명 모두 개선
        - "understand": 원본 풀이 유지

        개선된 풀이를 생성해주세요.
        """
        
        try:
            llm_improver = self._llm(0.7)
            improvement_response = llm_improver.invoke(improvement_prompt)
            
            state["improved_solution"] = improvement_response.content
            print("🔧 풀이 개선 완료")
            
        except Exception as e:
            print(f"⚠️ 풀이 개선 중 오류 발생: {e}")
            state["improved_solution"] = original_solution
        
        return state
    
    #----------------------------------------create graph------------------------------------------------------

    def _create_graph(self) -> StateGraph:
        """워크플로우 그래프 생성"""

        # ✅ LangGraph 구성
        print("📚 LangGraph 흐름 구성 중...")
        
        graph = StateGraph(SolutionState)

        # 공통 처리
        graph.add_node("search_similarity", self._search_similar_problems)
        graph.add_node("search_concepts", self._search_concepts_summary)
        graph.add_node("retrieve_parallel", self._retrieve_parallel)
        graph.add_node("generate_solution", self._generate_solution)
        graph.add_node("validate", self._validate_solution)
        graph.add_node("store", self._store_to_vector_db)
        
        # 새로운 노드들
        graph.add_node("evaluate_solution", self._evaluate_solution)
        graph.add_node("collect_feedback", self._collect_user_feedback)
        graph.add_node("improve_solution", self._improve_solution)
        graph.add_node("search_additional_info", self._search_additional_info)
        graph.add_node("finalize_solution", self._finalize_solution)

        # 도구 노드 추가
        tools = [self.user_feedback]
        tool_node = ToolNode(tools=tools)
        graph.add_node("tools", tool_node)

        # 시작점 설정
        graph.set_entry_point("retrieve_parallel")
        
        # 기본 흐름
        graph.add_edge("retrieve_parallel", "generate_solution")
        graph.add_edge("generate_solution", "validate")
        
        # 검증 후 분기
        graph.add_conditional_edges(
            "validate", 
            # 검증 실패 → retry<5이면 back, 아니면 evaluate로 진행
            lambda s: "ok" if s["validated"] else ("back" if s.get("retry_count", 0) < 5 else "evaluate"),
            {"ok": "evaluate_solution", "back": "generate_solution", "evaluate": "evaluate_solution"}
        )
        
        # 풀이 평가 후 분기
        graph.add_conditional_edges(
            "evaluate_solution",
            lambda s: "needs_improvement" if s.get("needs_improvement", False) else "store",
            {"needs_improvement": "collect_feedback", "store": "store"}
        )
        
        # 개선이 필요한 경우의 흐름
        graph.add_edge("collect_feedback", "tools")  # collect_feedback 후 tools 노드로
        graph.add_conditional_edges(
            "tools",
            tools_condition,
            {"tools": "improve_solution"}
        )
        graph.add_edge("improve_solution", "search_additional_info")
        graph.add_edge("search_additional_info", "finalize_solution")
        graph.add_edge("finalize_solution", "store")
        
        # 최종 저장
        graph.add_edge("store", END)

        return graph.compile(checkpointer=self.checkpointer)
    
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
            # 점수 포함 버전이 가능하면 사용(없으면 아래 except에서 대체)
            dense_scored = vectorstore_p.similarity_search_with_score(q, k=self.RETRIEVAL_FETCH_K)
            dense_docs = [d for d, _ in dense_scored]
            dense_scores = {id(d): float(s) for d, s in dense_scored}
            # 주의: 어떤 백엔드는 "코사인거리/거리" 등 낮을수록 유사함. 랭크 기반으로 치환해 안정화.
            print(f"[Dense] fetched: {len(dense_docs)}")
        except Exception as e:
            print(f"[Dense] similarity_search_with_score 실패 → {e} → score 없이 fallback")
            dense_docs = vectorstore_p.similarity_search(q, k=self.RETRIEVAL_FETCH_K)
            dense_scores = {id(d): 1.0/(r+1) for r, d in enumerate(dense_docs)}  # 랭크 기반 가중치

        # ---------- (2) Sparse 후보(BM25) 결합 ----------
        sparse_docs = []
        sparse_scores = {}  # id(doc) → score (rank 기반 또는 점수 정규화)

        if self.bm25_retriever is not None:
            try:
                sparse_docs = self.bm25_retriever.get_relevant_documents(q)[:self.RETRIEVAL_FETCH_K]
                for r, d in enumerate(sparse_docs):
                    sparse_scores[id(d)] = 1.0/(r+1)  # 랭크 기반
                print(f"[BM25] fetched: {len(sparse_docs)}")
            except Exception as e:
                print(f"[BM25] 실패 → {e}")

        elif HAS_RANK_BM25 and dense_docs:
            # 별도 인덱스가 없다면, dense 후보군 위에서만 BM25 근사 스코어 계산
            try:
                def tok(s: str) -> List[str]:
                    return re.findall(r"[가-힣A-Za-z0-9_]+", (s or "").lower())
                corpus_toks = [tok(d.page_content) for d in dense_docs]
                bm25 = BM25Okapi(corpus_toks)
                q_scores = bm25.get_scores(tok(q))
                # 점수 정규화 (0~1)
                if q_scores is not None and len(q_scores) == len(dense_docs):
                    min_s, max_s = float(min(q_scores)), float(max(q_scores))
                    rng = (max_s - min_s) or 1.0
                    for d, s in zip(dense_docs, q_scores):
                        sparse_scores[id(d)] = (float(s) - min_s) / rng
                print(f"[BM25-lite] computed over dense pool: {len(dense_docs)}")
            except Exception as e:
                print(f"[BM25-lite] 실패 → {e}")

        # ---------- (3) Dense + Sparse 앙상블 ----------
        # 동일 문서가 양쪽에 섞여 들어올 수 있으므로 content+metadata 일부로 키를 만든다
        def _safe_meta_str(md: Dict[str, Any]) -> str:
            try:
                # numpy 타입 등은 str로 변환
                norm = {str(k): (v.item() if isinstance(v, (np.generic,)) else (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v))
                        for k, v in (md or {}).items()}
                return json.dumps(norm, ensure_ascii=False, sort_keys=True)
            except Exception:
                try:
                    return str({k: str(v) for k, v in (md or {}).items()})
                except Exception:
                    return ""

        def key_of(doc: Document) -> Tuple[str, str]:
            return ( (doc.page_content or "")[:150], _safe_meta_str(doc.metadata)[:150] )

        pool: Dict[Tuple[str,str], Dict[str, Any]] = {}
        # dense 쪽부터 적재
        for r, d in enumerate(dense_docs):
            k = key_of(d)
            pool.setdefault(k, {"doc": d, "dense": 0.0, "sparse": 0.0})
            # 랭크 기반 가중치(안전)
            wd = 1.0/(r+1)
            # 점수 있으면 둘 중 큰 것을 사용
            wd = max(wd, dense_scores.get(id(d), 0.0))
            pool[k]["dense"] = max(pool[k]["dense"], wd)

        # sparse 쪽 반영
        for r, d in enumerate(sparse_docs):
            k = key_of(d)
            pool.setdefault(k, {"doc": d, "dense": 0.0, "sparse": 0.0})
            ws = 1.0/(r+1)
            ws = max(ws, sparse_scores.get(id(d), 0.0))
            pool[k]["sparse"] = max(pool[k]["sparse"], ws)

        # 가중합
        alpha = self.HYBRID_ALPHA  # 0~1, 1이면 dense만
        scored = []
        for k, v in pool.items():
            score = alpha * v["dense"] + (1.0 - alpha) * v["sparse"]
            scored.append((v["doc"], score))

        # 상위 K 추림
        scored.sort(key=lambda x: x[1], reverse=True)
        hybrid_top = [d for d, _ in scored[:self.HYBRID_TOPK]]
        print(f"[Hybrid] pool={len(pool)} → top{self.HYBRID_TOPK} 선정")

        # ---------- (4) Cross-Encoder rerank ----------
        try:
            if self.reranker is not None and len(hybrid_top) > 0:
                pairs = [[q, d.page_content] for d in hybrid_top]
                scores = self.reranker.predict(pairs)  # shape: (len(hybrid_top),)
                order = sorted(range(len(hybrid_top)), key=lambda i: float(scores[i]), reverse=True)
                reranked = [hybrid_top[i] for i in order[:self.RERANK_TOPK]]
                print(f"[Rerank] 최종 top{self.RERANK_TOPK} (CrossEncoder)")
            else:
                reranked = hybrid_top[:self.RERANK_TOPK]
                print("[Rerank] 사용 안 함 → hybrid_top 그대로 사용")
        except Exception as e:
            print(f"[Rerank] 실패 → hybrid_top 그대로 사용: {e}")
            reranked = hybrid_top[:self.RERANK_TOPK]


        # ---------- (5) 기존 포맷/저장 ----------
        results = reranked  # 최종 문서들
        similar_questions = []
        for i, doc in enumerate(results, start=1):
            metadata = doc.metadata or {}
            options = json.loads(metadata.get("options", "[]")) if isinstance(metadata.get("options"), str) else metadata.get("options", []) or []
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

        # ---------- (5) LLM 프롬프트용 정리 ----------
        chunks, cleaned_docs = [], []
        for idx, d in enumerate(reranked, start=1):
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
        print(f"[Parallel] similar={len(state['retrieved_docs'])}, "
            f"concepts={len(state['concept_contexts'])}")
        return state



    def _generate_solution(self, state: SolutionState) -> SolutionState:

        print("\n✏️ [2단계] 해답 및 풀이 생성 시작")

        llm_gen = self._llm(0.5)  

        problems_ctx = state.get("problems_contexts_text", "")
        print("유사 문제들 길이:", len(problems_ctx))
        print("유사 문제들: ", problems_ctx[:500])
        concept_ctx = state.get("concept_contexts_text", "")
        print("개념 컨텍스트 길이:", len(concept_ctx))
        print("개념 컨텍스트: ", concept_ctx[:500])

        prompt = f"""
            사용자가 입력한 질문:
            {state['user_input_txt']}

            다음은 사용자가 입력한 문제:
            {state['user_problem']}
            {state['user_problem_options']}

            아래는 이 문제와 유사한 문제들:
            {problems_ctx}

            아래는 이 문제와 관련된 개념 요약:
            {concept_ctx}


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

        return state

    def _search_additional_info(self, state: SolutionState) -> SolutionState:
        """필요한 경우 추가 정보를 검색합니다."""
        print("\n🔍 [검색] 추가 정보 검색 시작")
        
        feedback = state.get("feedback_analysis", "understand")
        
        # 용어 설명이나 추가 정보가 필요한 경우에만 검색
        if "term_explanation" in feedback or "term_easier_explanation" in feedback:
            try:
                # retrieve_agent를 실행하여 관련 정보 검색
                search_query = f"{state.get('user_problem', '')} {state.get('generated_explanation', '')}"
                
                # 벡터스토어에서 관련 개념 검색
                if state.get("vectorstore_c"):
                    concept_results = state["vectorstore_c"].similarity_search(search_query, k=3)
                    concept_texts = [doc.page_content for doc in concept_results]
                    state["search_results"] = "\n\n".join(concept_texts)
                    print(f"🔍 관련 개념 {len(concept_results)}개 검색 완료")
                else:
                    state["search_results"] = "벡터스토어를 사용할 수 없습니다."
                    print("⚠️ 개념 벡터스토어를 사용할 수 없습니다.")
                    
            except Exception as e:
                print(f"⚠️ 검색 중 오류 발생: {e}")
                state["search_results"] = "검색 중 오류가 발생했습니다."
        else:
            state["search_results"] = ""
            print("🔍 추가 검색이 필요하지 않습니다.")
        
        return state

    def _finalize_solution(self, state: SolutionState) -> SolutionState:
        """최종 풀이를 정리합니다."""
        print("\n✨ [정리] 최종 풀이 정리 시작")
        
        if state.get("needs_improvement", False) and state.get("improved_solution"):
            # 개선된 풀이가 있는 경우
            final_solution = state["improved_solution"]
            
            # 검색 결과가 있으면 추가
            if state.get("search_results"):
                final_solution += f"\n\n📚 추가 참고 자료:\n{state['search_results']}"
            
            state["generated_explanation"] = final_solution
            print("✨ 개선된 풀이로 최종 정리 완료")
        else:
            print("✨ 원본 풀이 유지")
        
        return state
    
    def _build_concept_query(self, problem: str, options: List[str]) -> str:
        opts = "\n".join([f"{i+1}) {o}" for i, o in enumerate(options or [])])
        return f"{(problem or '').strip()}\n{opts}"
    
    @tool
    def user_feedback(self, query: str) -> str:
        """사용자로부터 풀이에 대한 피드백을 수집하고 LLM을 통해 분석합니다."""
        print("interrupt 실행!!!!!!!!!!!!!!")
        
        # interrupt를 통해 워크플로우 중단 및 사용자 입력 요청
        interrupt({"query": "풀이에 대한 의견을 자유롭게 입력해주세요."})
        
        # interrupt 후에는 이 코드가 실행되지 않음
        # 실제 사용자 입력은 Command(resume)을 통해 전달됨
        return "pending_feedback"

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

        validation_response = llm.invoke(validation_prompt)
        result_text = validation_response.content.strip().lower()

        # ✅ '네'가 포함된 응답일 경우에만 유효한 풀이로 판단
        print("📌 검증 응답:", result_text)
        state["validated"] = "네" in result_text
        
        if not state["validated"]:
            state["retry_count"] = state.get("retry_count", 0) + 1
            if state["retry_count"] >= 5:
                print("⚠️ 검증 5회 실패 → 그래도 결과를 저장 단계로 진행합니다.")
            else:
                print(f"⚠️ 검증 실패 (재시도 {state['retry_count']}/5)")
        else:
            print("✅ 검증 결과: 통과")

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
            memory_key: Optional[str] = None,  # 숏텀 메모리 키 추가
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
        
        try:
            # thread_id를 포함한 config 생성
            graph_config = {
                "recursion_limit": recursion_limit,
                "configurable": {
                    "thread_id": memory_key or "default"
                }
            }
            final_state = self.graph.invoke(initial_state, config=graph_config)
        except Exception as e:
            print(f"⚠️ 그래프 실행 중 오류 발생: {e}")
            if "interrupt" in str(e).lower():
                print("🔄 interrupt가 발생했습니다. 현재 상태를 반환합니다.")
                
                # memory_key가 있으면 숏텀 메모리에 상태 저장
                if memory_key:
                    try:
                        from common.short_term.redis_memory import RedisMemory
                        redis_memory = RedisMemory()
                        # 현재 상태를 숏텀 메모리에 저장
                        state_data = {
                            "user_input_txt": user_input_txt,
                            "user_problem": user_problem,
                            "user_problem_options": user_problem_options,
                            "generated_answer": initial_state.get("generated_answer", ""),
                            "generated_explanation": initial_state.get("generated_explanation", ""),
                            "generated_subject": initial_state.get("generated_subject", ""),
                            "interrupt_occurred": True,
                            "interrupt_message": str(e),
                            "memory_key": memory_key
                        }
                        redis_memory.set(memory_key, state_data, ttl=3600)  # 1시간 TTL
                        print(f"💾 상태를 숏텀 메모리에 저장: {memory_key}")
                    except Exception as mem_err:
                        print(f"⚠️ 숏텀 메모리 저장 실패: {mem_err}")
                
                # interrupt 발생 시 현재 상태를 결과로 반환
                current_state = {
                    "user_input_txt": user_input_txt,
                    "user_problem": user_problem,
                    "user_problem_options": user_problem_options,
                    "generated_answer": initial_state.get("generated_answer", ""),
                    "generated_explanation": initial_state.get("generated_explanation", ""),
                    "generated_subject": initial_state.get("generated_subject", ""),
                    "interrupt_occurred": True,
                    "interrupt_message": str(e),
                    "memory_key": memory_key
                }
                return current_state
            else:
                raise
        

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


if __name__ == "__main__":

    # ✅ Milvus 연결 및 벡터스토어 생성
    embedding_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"}
    )

    if "default" in connections.list_connections():
        connections.disconnect("default")
    connections.connect(alias="default", host="localhost", port="19530")

    vectorstore_p = Milvus(
        embedding_function=embedding_model,
        collection_name="problems",
        connection_args={"host": "localhost", "port":"19530"}
    )

    vectorstore_c = Milvus(
        embedding_function=embedding_model,
        collection_name="concept_summary",
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
