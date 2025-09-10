import csv

import os

from typing import TypedDict, List, Dict, Optional, Tuple, Any

from langchain_core.documents import Document

# MilvusDB는 common.milvus_helpers를 통해 사용
from langgraph.graph import StateGraph, END

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



# RAGAS 대신 LLM 기반 검증 사용

# RAGAS 관련 코드 제거됨 - LLM 기반 검증 사용


import os, json, glob
from datetime import datetime
# MilvusDB는 common.milvus_helpers를 통해 사용
from difflib import SequenceMatcher

# LLM 기반 검증 함수들
def evaluate_with_llm(question: str, answer: str, contexts: list) -> dict:
    """
    LLM을 사용한 답변 검증
    - faithfulness: 답변이 컨텍스트에 기반한 정확성
    - answer_relevancy: 답변이 질문에 대한 관련성
    """
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    
    llm = ChatOpenAI(
        model=OPENAI_LLM_MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY=REDACTED=0.1
    )
    
    # 컨텍스트 결합
    context_text = "\n\n".join(contexts) if contexts else ""
    
    # 검증 프롬프트
    prompt_template = PromptTemplate(
        input_variables=["question", "answer", "context"],
        template="""다음 질문, 답변, 컨텍스트를 검증해주세요.

질문: {question}

답변: {answer}

컨텍스트: {context}

검증 기준:
1. Faithfulness (정확성): 답변이 제공된 컨텍스트에 기반하여 정확한가? (0.0-1.0)
2. Answer Relevancy (관련성): 답변이 질문에 적절히 관련되어 있는가? (0.0-1.0)

다음 JSON 형식으로 응답해주세요:
{{
    "faithfulness": 0.85,
    "answer_relevancy": 0.90,
    "reasoning": "답변이 컨텍스트에 기반하여 정확하고 질문에 관련성이 있습니다."
}}"""
    )
    
    try:
        response = llm.invoke(prompt_template.format(
            question=question,
            answer=answer,
            context=context_text
        ))
        
        # JSON 파싱
        import json
        result = json.loads(response.content)
        
        return {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"[LLM 검증] 오류 발생: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "reasoning": f"검증 실패: {e}"
        }


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

    

    milvus_data: Optional[Dict[str, Any]]


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



        # MilvusDB는 milvus_data를 통해 사용
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

    

    def _extract_triplet(self, text: str, options: List[str], similar_problems: List = None) -> tuple[Optional[int], str, str]:

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



        # 3) 과목: 유사도 검색 결과를 활용한 정확한 과목 판단

        subject_raw = ""

        m_subject = re.search(r"과\s*목\s*[:：\-]\s*([^\n\r]+)", t)

        if m_subject:

            subject_raw = m_subject.group(1).strip()



        # 유사도 검색 결과에서 과목 정보 추출

        subject_from_similar = self._extract_subject_from_similar_problems(similar_problems or [])

        

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



        # LLM 응답에서 추출한 과목과 유사도 검색 결과를 결합

        llm_subject = _normalize_subject(subject_raw)

        if llm_subject and llm_subject != "":

            subject = llm_subject

        elif subject_from_similar and subject_from_similar != "":

            subject = subject_from_similar

        else:

            subject = "일반"



        # 최종 안전장치: 보기 범위 체크

        if ans_num is not None:

            if not (1 <= ans_num <= max(1, len(options or []))):

                ans_num = None



        return ans_num, expl, subject

    def _extract_subject_from_similar_problems(self, similar_problems: List) -> str:
        """
        유사도 검색 결과에서 과목 정보를 추출하는 함수
        """
        try:
            if not similar_problems:
                return ""
            
            # 과목 키워드 매핑
            subject_keywords = {
                "소프트웨어설계": ["소프트웨어 설계", "소프트웨어-설계", "설계", "UML", "요구사항", "아키텍처"],
                "소프트웨어개발": ["소프트웨어 개발", "개발", "프로그래밍", "코딩", "구현"],
                "데이터베이스구축": ["데이터베이스", "DB", "SQL", "관계형", "정규화", "인덱스", "트랜잭션"],
                "프로그래밍언어활용": ["프로그래밍 언어", "언어활용", "프언활", "Java", "Python", "C++"],
                "정보시스템구축관리": ["정보시스템", "구축관리", "정시관", "시스템", "관리", "운영"]
            }
            
            # 유사 문제들의 과목 정보 수집
            subject_votes = {}
            
            for problem in similar_problems:
                if isinstance(problem, dict):
                    # 메타데이터에서 과목 정보 추출
                    metadata = problem.get("metadata", {})
                    subject = metadata.get("subject", "")
                    
                    if subject and subject != "일반":
                        subject_votes[subject] = subject_votes.get(subject, 0) + 1
                        continue
                    
                    # 문제 내용에서 과목 키워드 추출
                    problem_text = problem.get("page_content", "")
                    for subject, keywords in subject_keywords.items():
                        for keyword in keywords:
                            if keyword.lower() in problem_text.lower():
                                subject_votes[subject] = subject_votes.get(subject, 0) + 1
                                break
            
            # 가장 많이 나온 과목 반환
            if subject_votes:
                best_subject = max(subject_votes.items(), key=lambda x: x[1])[0]
                print(f"🔍 [Subject] 유사 문제에서 과목 추출: {best_subject} (투표: {subject_votes})")
                return best_subject
            
            return ""
            
        except Exception as e:
            print(f"⚠️ [Subject] 유사 문제에서 과목 추출 실패: {e}")
            return ""

    

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

    # RAGAS 관련 패치 코드 제거됨 - LLM 기반 검증 사용


    def _route_after_validate(self, s: SolutionState) -> str:

        llm_eval = (s.get("eval", {}) or {}).get("llm", {}) or {}
        f = float(llm_eval.get("faithfulness", 0) or 0.0)
        r = float(llm_eval.get("answer_relevancy", 0) or 0.0)
        thr = (llm_eval.get("thresholds", {}) or {})
        thr_f = float(thr.get("faithfulness", float(os.getenv("LLM_THR_FAITH", "0.6"))))
        thr_r = float(thr.get("answer_relevancy", float(os.getenv("LLM_THR_RELEVANCY", "0.6"))))

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

        # 공통 처리 - 바인딩된 메서드 사용
        graph.add_node("search_problems", self._search_similar_problems.__get__(self, SolutionAgent))
        graph.add_node("search_concepts", self._search_concepts_summary.__get__(self, SolutionAgent))
        graph.add_node("retrieve_parallel", self._retrieve_parallel.__get__(self, SolutionAgent))
        graph.add_node("generate_solution", self._generate_solution.__get__(self, SolutionAgent))
        graph.add_node("validate", self._validate_solution.__get__(self, SolutionAgent))
        graph.add_node("store", self._store_to_vector_db.__get__(self, SolutionAgent))

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
            
        # MilvusDB 검색 사용
        milvus_data = state.get("milvus_data", {})
        if not milvus_data:
            print("⚠️ milvus_data 없음 → 유사 문제 검색 건너뜀")
            state["problems_contexts"] = []
            state["problems_contexts_text"] = ""
            return state

        try:
            from common.milvus_helpers import search_milvus_documents
            
            # 문제 검색을 위한 쿼리 구성
            query = self._build_concept_query(state.get("user_problem",""), state.get("user_problem_options", []))
            
            # MilvusDB에서 유사 문제 검색
            results = search_milvus_documents(
                milvus_data=milvus_data,
                collection_name="problems",
                query=query,
                k=20
            )
            
            if not results:
                print("⚠️ MilvusDB에서 유사 문제를 찾을 수 없음")
                state["problems_contexts"] = []
                state["problems_contexts_text"] = ""
                return state

                
            print(f"✅ MilvusDB에서 {len(results)}개의 유사 문제 발견")
            
            # MilvusDB 결과를 Document 객체로 변환
            from langchain.schema import Document
            
            problems_contexts = []
            for result in results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("source", ""),
                        "subject": result.get("subject", ""),
                        "score": result.get("score", 0.0)
                    }
                )
                problems_contexts.append(doc)
            
            # 컨텍스트 텍스트 생성
            problems_contexts_text = "\n\n".join([
                f"[문제 {i+1}] {doc.page_content}" 
                for i, doc in enumerate(problems_contexts)
            ])
            
            state["problems_contexts"] = problems_contexts
            state["problems_contexts_text"] = problems_contexts_text
            
            print(f"✅ 유사 문제 {len(problems_contexts)}개 처리 완료")
            
        except Exception as e:
            print(f"❌ MilvusDB 검색 실패: {e}")
            state["problems_contexts"] = []
            state["problems_contexts_text"] = ""
            
        print("🔍 [1단계] 유사 문제 검색 함수 종료")
        return state



    def _search_concepts_summary(self, state: SolutionState) -> SolutionState:
        print("\n📚 [1-확장] 개념 요약 컨텍스트 검색 시작")

        # MilvusDB 검색 사용
        milvus_data = state.get("milvus_data", {})
        if not milvus_data:
            print("⚠️ milvus_data 없음 → 개념 검색 건너뜀")
            state["concept_contexts"] = []
            state["concept_contexts_text"] = ""
            return state

        try:
            from common.milvus_helpers import search_milvus_documents
            
            # 개념 검색을 위한 쿼리 구성
            query = self._build_concept_query(state.get("user_problem",""), state.get("user_problem_options", []))
            
            # MilvusDB에서 개념 검색
            results = search_milvus_documents(
                milvus_data=milvus_data,
                collection_name="concepts",
                query=query,
                k=20
            )
            
            if not results:
                print("⚠️ MilvusDB에서 개념을 찾을 수 없음")
                state["concept_contexts"] = []
                state["concept_contexts_text"] = ""
                return state
                
            print(f"✅ MilvusDB에서 {len(results)}개의 개념 발견")
            
            # MilvusDB 결과를 Document 객체로 변환
            from langchain.schema import Document
            
            concept_contexts = []
            for result in results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("source", ""),
                        "subject": result.get("subject", ""),
                        "score": result.get("score", 0.0)
                    }
                )
                concept_contexts.append(doc)
            
            # 컨텍스트 텍스트 생성
            concept_contexts_text = "\n\n".join([
                f"[개념 {i+1}] {doc.page_content}" 
                for i, doc in enumerate(concept_contexts)
            ])
            
            state["concept_contexts"] = concept_contexts
            state["concept_contexts_text"] = concept_contexts_text
            
            print(f"✅ 개념 {len(concept_contexts)}개 처리 완료")
            
        except Exception as e:
            print(f"❌ MilvusDB 검색 실패: {e}")
            state["concept_contexts"] = []
            state["concept_contexts_text"] = ""
            
        print("📚 [1-확장] 개념 요약 컨텍스트 검색 함수 종료")
        return state

    def _retrieve_parallel(self, state: SolutionState) -> SolutionState:
        print("\n🔄 [1-병렬] 병렬 검색 시작")
        
        # MilvusDB 검색 사용
        milvus_data = state.get("milvus_data", {})
        if not milvus_data:
            print("⚠️ milvus_data 없음 → 병렬 검색 건너뜀")
            return state

        try:
            from common.milvus_helpers import search_milvus_documents
            
            # 문제와 개념을 병렬로 검색
            query = self._build_concept_query(state.get("user_problem",""), state.get("user_problem_options", []))
            
            # 문제 검색
            problems_results = search_milvus_documents(
                milvus_data=milvus_data,
                collection_name="problems",
                query=query,
                k=10
            )
            
            # 개념 검색
            concepts_results = search_milvus_documents(
                milvus_data=milvus_data,
                collection_name="concepts",
                query=query,
                k=10
            )
            
            print(f"✅ 병렬 검색 완료: 문제 {len(problems_results)}개, 개념 {len(concepts_results)}개")
            
            # 검색 결과를 state에 저장
            state["problems_contexts"] = problems_results
            state["concept_contexts"] = concepts_results
            
            # 텍스트 형태로도 저장
            problems_text = "\n\n".join([doc.page_content for doc in problems_results])
            concepts_text = "\n\n".join([doc.page_content for doc in concepts_results])
            
            state["problems_contexts_text"] = problems_text
            state["concept_contexts_text"] = concepts_text
            
            print(f"📚 컨텍스트 저장 완료: 문제 {len(problems_text)}자, 개념 {len(concepts_text)}자")
            
        except Exception as e:

            print(f"❌ 병렬 검색 실패: {e}")
            # 실패 시 빈 상태로 설정
            state["problems_contexts"] = []
            state["concept_contexts"] = []
            state["problems_contexts_text"] = ""
            state["concept_contexts_text"] = ""
            
        print("🔄 [1-병렬] 병렬 검색 함수 종료")
        return state


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

            state.get('user_problem_options') or [],

            state.get('problems_contexts', [])  # 유사도 검색 결과 전달

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

    # RAGAS 관련 점수 추출 함수 제거됨 - LLM 기반 검증 사용




    def _validate_solution(self, state: SolutionState) -> SolutionState:

        """

        LLM 기반 검증
        - question : user_input_txt + user_problem + user_problem_options (원문)

        - answer   : generated_answer / generated_explanation / generated_subject (원문 결합)

        - contexts : problems_contexts_text, concept_contexts_text (원문 그대로)

        - metrics  : faithfulness, answer_relevancy

        """

        print("\n🧪 [3단계] LLM 기반 검증 시작")


        def _norm(s: str) -> str:

            return (s or "").strip()



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



        print(f"[LLM 검증] contexts 원문 사용: {len(ctx_list)}개")

        # 4) LLM 기반 검증 실행
        try:
            result = evaluate_with_llm(question_text, answer_text, ctx_list)
            f_sc = result["faithfulness"]
            r_sc = result["answer_relevancy"]
            reasoning = result["reasoning"]
            print(f"[LLM 검증] faithfulness: {f_sc:.3f}, answer_relevancy: {r_sc:.3f}")
            print(f"[LLM 검증] reasoning: {reasoning}")
        except Exception as e:

            print(f"[LLM 검증] 평가 실패: {e}")
            f_sc, r_sc = 0.0, 0.0
            reasoning = f"검증 실패: {e}"

        # 5) 임계값 설정


        # ✅ 임계값: 먼저 정의

        thr_f = float(os.getenv("LLM_THR_FAITH", "0.6"))
        thr_r = float(os.getenv("LLM_THR_RELEVANCY", "0.6"))


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

            "reasoning": reasoning,
            "thresholds": {"faithfulness": thr_f, "answer_relevancy": thr_r},

            "n_contexts": len(ctx_list),

            "timestamp": ts,

            "pass_faith": pass_f,

            "pass_ansrel": pass_r,

        }

        state["eval"]["llm"] = eval_record


        # (선택) 히스토리 적재

        state.setdefault("llm_history", [])
        # 히스토리에 넣을 땐 복사본 권장(참조 공유 방지)

        state["llm_history"].append(eval_record.copy())


        # ✅ 실패 유형별로 카운터 분리 증가 (둘 다 실패면 검색 우선)

        if not pass_f:

            state["retry_retrieve"] = int(state.get("retry_retrieve", 0)) + 1

            print(f"✅ [LLM 검증] faith={f_sc:.3f}(thr {thr_f}) | "
                f"ans_rel={r_sc:.3f}(thr {thr_r}) | "

                f"pass_f={pass_f} pass_r={pass_r} | "

                f"retry(retrieve/gen)={state['retry_retrieve']}/{state['retry_gen']}")

        elif not pass_r:

            state["retry_gen"] = int(state.get("retry_gen", 0)) + 1

            print(f"✅ [LLM 검증] faith={f_sc:.3f}(thr {thr_f}) | "
                f"ans_rel={r_sc:.3f}(thr {thr_r}) | "

                f"pass_f={pass_f} pass_r={pass_r} | "

                f"retry(retrieve/gen)={state['retry_retrieve']}/{state['retry_gen']}")

        else:

            print(f"✅ [LLM 검증] faith={f_sc:.3f}(thr {thr_f}) | "
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

            milvus_data: Optional[Dict[str, Any]] = None,
            recursion_limit: int = 1000,

        ) -> Dict:  



        # MilvusDB 데이터 설정
        if not milvus_data:
            print("⚠️ milvus_data가 제공되지 않음")
            milvus_data = {}


        self._reset_tunables()

        

        initial_state: SolutionState = {

            "user_input_txt": user_input_txt,

            "user_problem": user_problem,

            "user_problem_options": user_problem_options,



            "milvus_data": milvus_data,


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



    # --- JSON 폴더 내 파일 목록 ---

    if not os.path.isdir(JSON_DIR):

        raise FileNotFoundError(f"문제 JSON 폴더를 찾을 수 없습니다: {JSON_DIR}")

    json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))

    if not json_files:

        raise ValueError(f"{JSON_DIR} 안에 .json 파일이 없습니다.")



    # --- MilvusDB는 이제 common.milvus_helpers를 통해 사용 ---
    agent = SolutionAgent()



    def run_one(p: dict) -> dict:

        return agent.invoke(

            user_input_txt=INSTRUCTION,

            user_problem=p.get("question", "") or "",

            user_problem_options=p.get("options", []) or [],

            milvus_data={},  # MilvusDB 데이터는 외부에서 주입
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