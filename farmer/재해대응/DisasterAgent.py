import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========[ 벡터스토어 / 임베딩 관련 Import (맨 위) ]=========
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# =========[ 표준/외부 라이브러리 ]=========
import os
import re
import json
import time
from typing import TypedDict, Optional, Any, Dict, List
from operator import itemgetter
from argparse import ArgumentParser
from datetime import datetime
from zoneinfo import ZoneInfo
import sys

import numpy as np
import pandas as pd
import math
from dotenv import load_dotenv
from tavily import TavilyClient
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

# =========[ LangChain / LangGraph / LLM ]=========
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END

# ===== (신규) RAGAS 관련 Import (0.3.x 호환) =====
_HAS_RAGAS = False
try:
    from ragas import evaluate, SingleTurnSample
    from ragas.metrics import ResponseRelevancy, Faithfulness
    from ragas.metrics import LLMContextPrecisionWithoutReference
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    # RAGAS 0.3.x에서는 직접 LangChain 객체를 전달
    from datasets import Dataset
    _HAS_RAGAS = True
except ImportError as e:
    print(f"   - ⚠️ RAGAS/의존성 임포트 실패: {e}")

# torch는 선택 사항
try:
    import torch
    print("   - 🚀 GPU 가속 활성화 (RAGAS)" if torch.cuda.is_available() else "   - 💻 CPU 모드 (RAGAS)")
except Exception:
    torch = None
    print("   - ℹ️ torch 미설치: CPU 모드 (RAGAS)")

load_dotenv()

# =========[ 환경설정 ]=========
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agri_disaster_docs")

***REMOVED*** 설정
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

TAVILY_API_KEY=REDACTED("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# =========[ RAGAS 백엔드 설정 ]=========
RAGAS_BACKEND = os.getenv("RAGAS_BACKEND", "openai").lower()
RAGAS_OPENAI_LLM = os.getenv("RAGAS_OPENAI_LLM", "gpt-4o-mini")
RAGAS_OPENAI_EMBED = os.getenv("RAGAS_OPENAI_EMBED", "text-embedding-3-small")

_RAGAS_LLM = None
_RAGAS_EMB = None

def _init_ragas_backend():
    """RAGAS LLM/Embedding 백엔드 초기화. OpenAI LLM + HuggingFace Embeddings 사용."""
    global _RAGAS_LLM, _RAGAS_EMB, RAGAS_BACKEND
    if not _HAS_RAGAS:
        return

    try:
        if not OPENAI_API_KEY=REDACTED("   - ⚠️ OPENAI_API_KEY=REDACTED 비활성화")
            return
        
        # env 세팅
        os.environ["OPENAI_API_KEY=REDACTED
        if OPENAI_BASE_URL:
            os.environ["OPENAI_API_BASE"] = OPENAI_BASE_URL
        
        llm = ChatOpenAI(model=RAGAS_OPENAI_LLM, temperature=0)
        ***REMOVED*** 임베딩 대신 HuggingFace 임베딩 사용 (권한 문제 해결)
        emb = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True}
        )
        _RAGAS_LLM = llm
        _RAGAS_EMB = emb
        
        # RAGAS Wrapper 설정 (SalesRAGAS 방식)
        global _RAGAS_LLM_WRAPPER, _RAGAS_EMB_WRAPPER
        _RAGAS_LLM_WRAPPER = LangchainLLMWrapper(_RAGAS_LLM)
        _RAGAS_EMB_WRAPPER = LangchainEmbeddingsWrapper(_RAGAS_EMB)
        
        print(f"   - 🔑 RAGAS 백엔드=OpenAI LLM + HF Embeddings · LLM={RAGAS_OPENAI_LLM}, EMB={EMBED_MODEL_NAME}")
    except Exception as e:
        print(f"   - ⚠️ RAGAS 백엔드 초기화 실패: {e}")

_init_ragas_backend()

# =========[ 유틸 ]=========
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

def minmax_norm(scores: List[float]) -> List[float]:
    if not scores: return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-8: return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))

def build_temporal_meta() -> Dict[str, Any]:
    today = now_kst().date()
    y = today.year
    return {"today": today.isoformat(), "this_year": y, "last_year": y - 1, "two_years_ago": y - 2}

_RELATIVE_PATTERNS = [
    (r"\b올해\b",      lambda t: f"{t['this_year']}년"),
    (r"\b작년\b",      lambda t: f"{t['last_year']}년"),
    (r"\b재작년\b",    lambda t: f"{t['two_years_ago']}년"),
]

_KOREAN_REGIONS = [
    "강원", "경기", "경남", "경북", "광주", "대구", "대전", "부산", "서울", "세종",
    "울산", "인천", "전남", "전북", "제주", "충남", "충북"
]

def resolve_relative_years_kst(question: str, temporal: Dict[str, Any]) -> str:
    resolved = question
    for pat, repl in _RELATIVE_PATTERNS:
        resolved = re.sub(pat, repl(temporal), resolved)
    return resolved

def extract_region_from_question(question: str) -> Optional[str]:
    for region in _KOREAN_REGIONS:
        if region in question:
            if region in ["강원", "경기", "경남", "경북", "전남", "전북", "충남", "충북", "제주"] and not question.endswith("도"):
                 return region + "도"
            return region
    return None

# =========[ 상태/LLM ]=========
class GraphState(TypedDict):
    question: Optional[str]
    question_resolved: Optional[str]
    db_context: Optional[str]
    web_context: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    answer_draft: Optional[str]
    store_obj: Optional[Any]
    retrieved_docs: Optional[List[Document]]
    retry_count: int
    is_retrieval_sufficient: bool
    is_answer_sufficient: bool
    temporal: Optional[Dict[str, Any]]

def make_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY=REDACTED ValueError("OPENAI_API_KEY=REDACTED에 없습니다.")
    return ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY=REDACTED = ChatPromptTemplate.from_template(
    """너는 농작물 재해 정보 전문가야.
아래 문맥을 참고하여 질문에 대한 초안 답변을 작성해줘.
답변은 딱딱한 보고서 형식이 아닌, 대화하듯이 친절하게 설명하는 스타일로 작성해야 해.

**답변 형식 규칙:**
1.  **개요**: 어떤 재해로 어떤 피해가 있었는지 2~3문장으로 간결하게 설명해.
2.  **상세 내용**: 아래 문맥에서 찾은 구체적인 피해 내용을 자연스러운 문장으로 서술해줘.
    - 특히 표(pdf_table) 내용이 있으면 이를 중심으로 **시·군별 피해 내역**을 최대한 상세히 설명해.
    - 수치와 단위(ha, 억원, 마리, 개소)는 정확하게 유지해. **단, 문맥에 명시된 단위가 없으면 어떤 단위도 임의로 붙이지 마.**
    - 해당 정보가 없는 경우, "구체적인 내용은 확인되지 않았어요."처럼 솔직하게 답변해줘.
3.  **출처**: 마지막에 "이 정보는 [파일명]의 [페이지]에서 가져온 내용이에요."와 같이 출처를 명확하게 밝혀줘. 여러 개면 `,`로 구분해.
4.  **마무리**: "혹시 다른 궁금한 점이 있으시면 언제든지 물어봐 주세요!"와 같은 친절한 인사로 마무리해줘.
5.  **줄바꿈**: 문단이 바뀔 때마다 줄바꿈을 넣어 답변을 읽기 쉽게 만들어.

**재해 대응 기술 질문 시:**
'사전', '즉시', '사후'로 나누어 어떻게 대응해야 하는지 대화체로 설명해줘.

[문맥]
{context}

[질문 원문] {question_raw}
[질문(상대시점 해석)] {question_resolved}

초안:"""
)

WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template(
    """너는 농업재해 및 농업 정보 전문가야.
아래에는 로컬 인덱스 문맥과 웹 검색 결과가 함께 있어.
이 모든 정보를 활용하여 초안 답변을 작성해줘. 답변은 대화체로, 친절하게 설명하는 스타일로 작성해야 해.

**답변 형식 규칙:**
- 시작, 개요, 상세 내용, 출처, 마무리 규칙은 로컬 인덱스 초안과 동일하게 적용해.
- 웹 검색 결과가 있다면 최신성/구체성을 고려해 반드시 반영해줘.
- 로컬과 웹 검색 내용이 충돌하면 최신 정보인 **웹 결과를 우선**으로 다뤄줘.
- 세부 내용은 **시·군별로 구분하여** 자연스러운 문장으로 설명해.
- 단위(ha, 억원, 마리, 개소)를 정확하게 사용해.
- 줄바꿈을 적절히 사용해 답변을 읽기 쉽게 만들어.

[문맥+웹 검색]
{context}

[질문 원문] {question_raw}
[질문(상대시점 해석)] {question_resolved}

초안:"""
)

REFINE_PROMPT = ChatPromptTemplate.from_template(
    """질문, 초안 답변, 그리고 문맥이 주어졌어.
초안 답변을 검토하여 **최종 답변**을 완전하고 간결하게 한국어로 작성해줘.
답변은 딱딱한 보고서 형식이 아닌, 대화하듯이 친절하게 설명하는 스타일을 유지해야 해.

**규칙:**
- 초안 답변의 내용과 대화체 스타일을 그대로 유지하되, **문맥의 기간/지역/재해유형과 일치하는지 반드시 확인**하고, 더 정확하고 자연스럽게 다듬어.
- 표(pdf_table) 기반 수치는 정확하게 사용하고 단위(ha, 억원, 마리, 개소)를 보존해. **단, 문맥에 명시된 단위가 없으면 어떤 단위도 임의로 붙이지 마.**
- 대상 외 지역/기간의 수치는 제외해. 날짜 정보가 없으면 "날짜 불명"임을 명시해.
- 답변이 끝나면 줄바꿈을 한 번 추가해줘.

[문맥]
{context}

[질문 원문] {question_raw}
[질문(상대시점 해석)] {question_resolved}

초안: {answer_draft}

최종 답변:"""
)

WEB_SEARCH_REFINE_PROMPT = ChatPromptTemplate.from_template(
    """다음은 문맥(로컬+웹 검색), 질문, 그리고 초안 답변이야.
초안을 검토해서 최종 답변을 완전하고 간결하게 한국어로 작성해줘. 답변은 대화체 스타일을 유지해야 해.

**규칙:**
- 웹 검색 결과가 있으면 최신성/구체성 기준으로 우선 반영해.
- 로컬과 웹 내용이 충돌하면 차이를 자연스럽게 언급할 수도 있어.
- 초안 답변의 대화체 스타일과 내용 구조를 최대한 유지하면서 다듬어.
- **최신 정보인 웹 결과의 기간/지역/재해유형을 우선**으로 반영하여 답변을 수정해.
- 날짜는 문맥에만 근거, 없으면 "날짜 불명"임을 명시해.
- 답변이 끝나면 줄바꿈을 한 번 추가해줘.

[문맥+웹 검색]
{context}

[질문 원문] {question_raw}
[질문(상대시점 해석)] {question_resolved}

초안: {answer_draft}

최종 답변:"""
)

_VALIDATE_RETRIEVAL_PROMPT = ChatPromptTemplate.from_template(
    """당신은 주어진 질문에 대해 검색된 문서가 충분한 정보를 담고 있는지 평가하는 AI 평가자입니다.
오직 'YES' 또는 'NO'로만 대답하세요. 다른 설명은 절대 추가하지 마십시오.

[질문]
{question}

[검색된 문서]
{context}

[평가]
위 '검색된 문서'가 '질문'에 답변하기에 충분하고 관련성이 높습니까?
"""
)

def _has_web_results(context_text: str) -> bool:
    c = (context_text or "")
    return "[웹 검색 결과]" in c

# =========[ RAGAS 결과 파싱 헬퍼 ]=========
def _ragas_overall(result_obj: Any, metric_name: str) -> Optional[float]:
    try:
        val = None
        
        # 1. RAGAS 0.3.x _scores_dict 속성에서 직접 접근 (가장 일반적)
        if hasattr(result_obj, "_scores_dict") and isinstance(result_obj._scores_dict, dict):
            val = result_obj._scores_dict.get(metric_name)
            if val is not None:
                # 리스트인 경우 첫 번째 값 사용
                if isinstance(val, list) and len(val) > 0:
                    val = val[0]
                # JSON 문자열인 경우 파싱 시도
                elif isinstance(val, str) and val.startswith('{'):
                    try:
                        import json
                        json_data = json.loads(val)
                        if "statements" in json_data and isinstance(json_data["statements"], list):
                            # verdict 값들의 평균 계산
                            verdicts = []
                            for stmt in json_data["statements"]:
                                if isinstance(stmt, dict) and "verdict" in stmt:
                                    verdicts.append(float(stmt["verdict"]))
                            if verdicts:
                                val = sum(verdicts) / len(verdicts)
                                print(f"   - ✅ {metric_name}: {val:.4f} (JSON verdict 평균)")
                                return val
                    except:
                        pass
                val = float(val)
                if val == val:  # NaN 체크
                    print(f"   - ✅ {metric_name}: {val:.4f} (_scores_dict)")
                    return val
        
        # 2. RAGAS 0.3.x scores 속성에서 직접 접근
        if hasattr(result_obj, "scores") and hasattr(result_obj.scores, metric_name):
            val = getattr(result_obj.scores, metric_name)
            if val is not None:
                # JSON 문자열인 경우 파싱 시도
                if isinstance(val, str) and val.startswith('{'):
                    try:
                        import json
                        json_data = json.loads(val)
                        if "statements" in json_data and isinstance(json_data["statements"], list):
                            # verdict 값들의 평균 계산
                            verdicts = []
                            for stmt in json_data["statements"]:
                                if isinstance(stmt, dict) and "verdict" in stmt:
                                    verdicts.append(float(stmt["verdict"]))
                            if verdicts:
                                val = sum(verdicts) / len(verdicts)
                                print(f"   - ✅ {metric_name}: {val:.4f} (scores JSON verdict 평균)")
                                return val
                    except:
                        pass
                val = float(val)
                if val == val:  # NaN 체크
                    print(f"   - ✅ {metric_name}: {val:.4f} (scores 속성)")
                    return val
        
        # 3. to_dict() 시도
        if hasattr(result_obj, "to_dict"):
            d = result_obj.to_dict()
            if isinstance(d, dict):
                # scores 딕셔너리 내부 확인 (RAGAS 0.3.x)
                if "scores" in d and isinstance(d["scores"], dict):
                    val = d["scores"].get(metric_name)
                    if val is not None:
                        val = float(val)
                        if val == val:  # NaN 체크
                            print(f"   - ✅ {metric_name}: {val:.4f} (to_dict scores)")
                            return val
                
                # overall 딕셔너리 내부 확인 (구버전)
                if "overall" in d and isinstance(d["overall"], dict):
                    val = d["overall"].get(metric_name)
                    if val is not None:
                        val = float(val)
                        if val == val:  # NaN 체크
                            print(f"   - ✅ {metric_name}: {val:.4f} (to_dict overall)")
                            return val
                
                # 직접 키 접근
                if metric_name in d:
                    val = d[metric_name]
                    if val is not None:
                        val = float(val)
                        if val == val:  # NaN 체크
                            print(f"   - ✅ {metric_name}: {val:.4f} (to_dict 직접)")
                            return val
        
        # 4. __dict__ 시도
        if hasattr(result_obj, "__dict__"):
            d = result_obj.__dict__
            # _scores_dict 딕셔너리 내부 확인 (RAGAS 0.3.x)
            if "_scores_dict" in d and isinstance(d["_scores_dict"], dict):
                val = d["_scores_dict"].get(metric_name)
                if val is not None:
                    # 리스트인 경우 첫 번째 값 사용
                    if isinstance(val, list) and len(val) > 0:
                        val = val[0]
                    # JSON 문자열인 경우 파싱 시도
                    elif isinstance(val, str) and val.startswith('{'):
                        try:
                            import json
                            json_data = json.loads(val)
                            if "statements" in json_data and isinstance(json_data["statements"], list):
                                # verdict 값들의 평균 계산
                                verdicts = []
                                for stmt in json_data["statements"]:
                                    if isinstance(stmt, dict) and "verdict" in stmt:
                                        verdicts.append(float(stmt["verdict"]))
                                if verdicts:
                                    val = sum(verdicts) / len(verdicts)
                                    print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ JSON verdict 평균)")
                                    return val
                        except:
                            pass
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ _scores_dict)")
                        return val
            
            # scores 딕셔너리 내부 확인 (RAGAS 0.3.x)
            if "scores" in d and isinstance(d["scores"], dict):
                val = d["scores"].get(metric_name)
                if val is not None:
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ scores)")
                        return val
            
            if "overall" in d and isinstance(d["overall"], dict):
                val = d["overall"].get(metric_name)
                if val is not None:
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ overall)")
                        return val
            
            if metric_name in d:
                val = d[metric_name]
                if val is not None:
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ 직접)")
                        return val
        
        # 5. 직접 속성 접근
        if hasattr(result_obj, metric_name):
            val = getattr(result_obj, metric_name)
            if val is not None:
                val = float(val)
                if val == val:  # NaN 체크
                    print(f"   - ✅ {metric_name}: {val:.4f} (직접 속성)")
                    return val
        
        print(f"   - ❌ {metric_name} 값을 찾을 수 없음")
        return None
        
    except Exception as e:
        print(f"   - ⚠️ RAGAS 결과 파싱 실패 ({metric_name}): {e}")
        return None

# =========[ 전역 변수 ]=========
_vectorstore = None

# =========[ LangGraph 노드 ]=========
def load_store_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 벡터스토어 연결 (LangChain-Milvus)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )
    # Milvus 객체를 상태에 저장하지 않고 전역 변수로 관리
    global _vectorstore
    _vectorstore = vectorstore
    return {**state, "retry_count": 0}

def temporal_enrich_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 시간/상대시점 해석(KST 기준)")
    temporal = build_temporal_meta()
    q_raw = state.get("question", "")
    q_resolved = resolve_relative_years_kst(q_raw, temporal)
    if q_resolved != q_raw:
        print(f"   - 질문 치환: '{q_raw}'   ->   '{q_resolved}'")
    return {**state, "question": q_raw, "temporal": temporal, "question_resolved": q_resolved}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 검색 (메타데이터 필터링 활용)")
    q = state.get("question_resolved", state.get("question", ""))
    vectorstore = _vectorstore

    region = extract_region_from_question(q)
    results_with_score = []

    if region:
        print(f"   - 지역명 '{region}'을(를) 사용하여 필터링된 검색 수행")
        try:
            expr = f"json_contains(regions, '\"{region}\"')"
            filtered_results_with_score = vectorstore.similarity_search_with_score(q, k=5, expr=expr)
            print(f"   - 필터링된 검색 결과: {len(filtered_results_with_score)}개")
            results_with_score.extend(filtered_results_with_score)
            if len(results_with_score) < 3:
                print("   - 결과가 부족하여 일반 검색을 추가로 수행합니다.")
                unfiltered_results_with_score = vectorstore.similarity_search_with_score(q, k=5)
                existing_content = {doc.page_content for doc, _ in results_with_score}
                for doc, score in unfiltered_results_with_score:
                    if doc.page_content not in existing_content:
                        results_with_score.append((doc, score))
                        existing_content.add(doc.page_content)
        except Exception as e:
            print(f"   - ⚠️ 메타데이터 필터링 검색 실패, 일반 검색으로 대체: {e}")
            results_with_score = vectorstore.similarity_search_with_score(q, k=8)
    else:
        print("   - 지역명이 없어 일반 유사도 검색 수행")
        results_with_score = vectorstore.similarity_search_with_score(q, k=8)

    results_with_score.sort(key=lambda x: x[1])
    docs = [doc for doc, score in results_with_score]

    ctx_parts = []
    for d, score in results_with_score[:8]:
        meta = getattr(d, "metadata", {})
        fname = meta.get("file_name") or meta.get("source") or "unknown"
        page = meta.get("page")
        tag = meta.get("type") or "text"
        header = f"[유사도:{score:.4f}][{tag}][{fname}{f' p.{page}' if page else ''}]"
        ctx_parts.append(f"{header}\n{d.page_content}")

    context = "\n\n".join(ctx_parts) or "관련 문서를 찾을 수 없습니다."
    return {**state, "db_context": context, "retrieved_docs": docs}

def combine_context_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 컨텍스트 결합")
    db_context = state.get("db_context", "")
    web_context = state.get("web_context", "")
    final_context = db_context
    if web_context:
        print("   - DB와 웹 컨텍스트를 결합합니다.")
        final_context = f"[DB 검색 결과]\n{db_context}\n\n[웹 검색 결과]\n{web_context}"
    else:
        print("   - DB 컨텍스트만 사용합니다.")
    return {**state, "context": final_context}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 웹 검색")
    question = state.get("question_resolved", state.get("question", ""))
    try:
        print(f"   - '{question}'에 대한 웹 검색을 시작합니다...")
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        results = tavily.search(query=question, max_results=TAVILY_MAX_RESULTS)
        if not results or not results.get("results"):
            print("   - ⚠️ 웹 검색 결과가 없습니다.")
            return {**state, "web_context": "[웹 검색 결과 없음]"}
        web_context = "\n\n".join([f"- 출처: {res['url']}\n 내용: {res['content']}" for res in results['results']]) or "검색 결과를 찾지 못했습니다."
        print("   - ✅ 웹 검색 완료.")
        return {**state, "web_context": web_context}
    except Exception as e:
        print(f"   - ❌ 웹 검색 중 오류 발생: {e}")
        return {**state, "web_context": f"[웹 검색 실패: {e}]"}

def generate_draft_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 초안 생성")
    if not state.get("context"):
        raise ValueError("context 누락")
    t = state.get("temporal") or {}
    use_web_prompt = _has_web_results(state.get("context", ""))
    prompt = WEB_SEARCH_PROMPT if use_web_prompt else DRAFT_PROMPT
    chain = (
        {
            "context": itemgetter("context"),
            "question_raw": lambda s: s.get("question", ""),
            "question_resolved": lambda s: s.get("question_resolved", s.get("question", "")),
        }
        | prompt.partial(today=t.get("today", ""))
        | make_llm()
        | StrOutputParser()
    )
    ans = chain.invoke({
        "context": state["context"],
        "question_raw": state.get("question", ""),
        "question_resolved": state.get("question_resolved", state.get("question", "")),
    })
    return {**state, "answer_draft": ans.strip()}

def refine_answer_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 답변 개선 및 최종 생성")
    if not state.get("answer_draft"):
        raise ValueError("answer_draft 누락")
    t = state.get("temporal") or {}
    use_web_prompt = _has_web_results(state.get("context", ""))
    prompt = WEB_SEARCH_REFINE_PROMPT if use_web_prompt else REFINE_PROMPT
    chain = (
        {
            "context": itemgetter("context"),
            "question_raw": lambda s: s.get("question", ""),
            "question_resolved": lambda s: s.get("question_resolved", s.get("question", "")),
            "answer_draft": itemgetter("answer_draft"),
        }
        | prompt.partial(today=t.get("today", ""))
        | make_llm()
        | StrOutputParser()
    )
    ans = chain.invoke({
        "context": state.get("context", ""),
        "question_raw": state.get("question", ""),
        "question_resolved": state.get("question_resolved", state.get("question", "")),
        "answer_draft": state["answer_draft"],
    })
    return {**state, "answer": ans.strip()}

# ===== 검증 노드들 =====
# 1차 검증 (검색 품질) 임계값
CONTEXT_PRECISION_THRESHOLD = 0.7

# 2차 검증 (답변 품질) 임계값
FAITHFULNESS_THRESHOLD = 0.5  # RAGAS faithfulness는 일반적으로 낮게 나옴
ANSWER_RELEVANCY_THRESHOLD = 0.7

MAX_RETRIES = 3

def retrieval_validation_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 1차 검증 (검색 품질)")
    question = state.get("question_resolved", state.get("question", ""))
    db_context = state.get("db_context", "")

    if not db_context or "관련 문서를 찾을 수 없습니다." in db_context:
        print("   - ❌ 검색된 문서가 없어 불충분으로 판단합니다.")
        return {**state, "is_retrieval_sufficient": False}

    # RAGAS 평가
    ragas_scores = {"context_precision": 0.0}
    if _HAS_RAGAS and _RAGAS_LLM_WRAPPER:
        try:
            print("   - 📊 RAGAS 검색 품질 평가 중...")

            # 컨텍스트 최적화
            max_context_length = 2500
            optimized_context = db_context[:max_context_length] if len(db_context) > max_context_length else db_context

            # 임시 답변 생성 (LLMContextPrecisionWithoutReference용)
            temp_answer = optimized_context[:1200] if len(optimized_context) > 0 else "정보 부족"

            print(f"   - 📝 SingleTurnSample 준비: 질문={len(question)}자, 컨텍스트={len(optimized_context)}자")

            # SalesRAGAS 방식: SingleTurnSample 사용
            context_precision_scorer = LLMContextPrecisionWithoutReference(llm=_RAGAS_LLM_WRAPPER)
            
            # SingleTurnSample 생성
            context_sample = SingleTurnSample(
                user_input=question,
                response=temp_answer,
                retrieved_contexts=[optimized_context] if optimized_context else [""]
            )

            print("   - 🔄 RAGAS 평가 실행 중...")
            
            # SingleTurnSample 방식으로 평가 (동기 방식으로 변경)
            import asyncio
            context_precision_score = asyncio.run(context_precision_scorer.single_turn_ascore(context_sample))
            ragas_scores["context_precision"] = float(context_precision_score)
            
            print(f"   - 📈 검색 품질 지표:")
            print(f"     • Context Precision (LLM-based): {ragas_scores['context_precision']:.3f}")

        except Exception as e:
            print(f"   - ⚠️ RAGAS 검색 평가 실패: {e}")
    else:
        print("   - ⚠️ RAGAS 백엔드가 준비되지 않아 평가를 건너뜁니다.")

    # 개별 임계값 평가
    precision_sufficient = ragas_scores["context_precision"] >= CONTEXT_PRECISION_THRESHOLD
    is_sufficient = precision_sufficient
    
    print(f"   - 🎯 개별 평가 결과:")
    print(f"     • Context Precision: {ragas_scores['context_precision']:.3f} (임계값: {CONTEXT_PRECISION_THRESHOLD}) {'✅' if precision_sufficient else '❌'}")
    print(f"     • 최종 결과: {'✅ 충분' if is_sufficient else '⚠️ 불충분'}")
    
    return {**state, "is_retrieval_sufficient": is_sufficient}

def answer_validation_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 2차 검증 (답변 품질)")
    retry_count = state.get("retry_count", 0) + 1

    if not _HAS_RAGAS or not (_RAGAS_LLM_WRAPPER and _RAGAS_EMB_WRAPPER):
        print("   - ⚠️ RAGAS 백엔드가 준비되지 않아 검증을 건너뜁니다.")
        return {**state, "is_answer_sufficient": True, "retry_count": retry_count}

    question = state.get("question_resolved", state.get("question", ""))
    context = state.get("context", "")
    answer = state.get("answer", "")

    if not all([question, context, answer]):
        print("   - ❌ 평가 정보가 부족하여 검증을 건너뜁니다.")
        return {**state, "is_answer_sufficient": True, "retry_count": retry_count}

    # 컨텍스트 및 답변 최적화
    max_context_length = 3000
    optimized_context = context[:max_context_length] if len(context) > max_context_length else context
    max_answer_length = 1200
    optimized_answer = answer[:max_answer_length] if len(answer) > max_answer_length else answer

    if len(optimized_context.strip()) < 50 or len(optimized_answer.strip()) < 20:
        print("   - ⚠️ 컨텍스트/답변이 너무 짧아 RAGAS 평가 생략")
        return {**state, "is_answer_sufficient": True, "retry_count": retry_count}

    print(f"   - 📝 답변 품질 평가 준비: 질문={len(question)}자, 컨텍스트={len(optimized_context)}자, 답변={len(optimized_answer)}자")

    try:
        print("   - 📊 RAGAS 답변 품질 평가 중...")
        
        scores = {}
        
        try:
            # Faithfulness (SalesRAGAS 방식)
            faithfulness_scorer = Faithfulness(llm=_RAGAS_LLM_WRAPPER)
            
            # SingleTurnSample 생성 (Faithfulness용)
            faithfulness_sample = SingleTurnSample(
                user_input=question,
                response=optimized_answer,
                retrieved_contexts=[optimized_context] if optimized_context else [""]
            )
            
            # SingleTurnSample 방식으로 평가 (동기 방식으로 변경)
            import asyncio
            faithfulness_score = asyncio.run(faithfulness_scorer.single_turn_ascore(faithfulness_sample))
            scores['faithfulness'] = float(faithfulness_score)
            
        except Exception as e:
            scores['faithfulness'] = 0.0
        
        try:
            # Answer Relevancy (SalesRAGAS 방식)
            answer_relevancy_scorer = ResponseRelevancy(
                llm=_RAGAS_LLM_WRAPPER, 
                embeddings=_RAGAS_EMB_WRAPPER
            )
            
            # SingleTurnSample 생성 (Answer Relevancy용)
            relevancy_sample = SingleTurnSample(
                user_input=question,
                response=optimized_answer,
                retrieved_contexts=[optimized_context] if optimized_context else [""]
            )
            
            # SingleTurnSample 방식으로 평가 (동기 방식으로 변경)
            import asyncio
            answer_relevancy_score = asyncio.run(answer_relevancy_scorer.single_turn_ascore(relevancy_sample))
            scores['answer_relevancy'] = float(answer_relevancy_score)
            
        except Exception as e:
            scores['answer_relevancy'] = 0.0

        f_val = scores.get('faithfulness', 0.0)
        r_val = scores.get('answer_relevancy', 0.0)

        if f_val is None or r_val is None:
            print("   - ⚠️ RAGAS 점수 NaN/None → 이번 라운드 통과로 처리")
            return {**state, "is_answer_sufficient": True, "retry_count": retry_count}

        # 개별 임계값 평가
        faithfulness_sufficient = f_val >= FAITHFULNESS_THRESHOLD
        relevancy_sufficient = r_val >= ANSWER_RELEVANCY_THRESHOLD
        is_sufficient = faithfulness_sufficient and relevancy_sufficient

        print(f"   - 📈 답변 품질 지표:")
        print(f"     • Faithfulness: {f_val:.3f} (임계값: {FAITHFULNESS_THRESHOLD}) {'✅' if faithfulness_sufficient else '❌'}")
        print(f"     • Answer Relevancy: {r_val:.3f} (임계값: {ANSWER_RELEVANCY_THRESHOLD}) {'✅' if relevancy_sufficient else '❌'}")
        print(f"     • 최종 결과: {'✅ 충분' if is_sufficient else '⚠️ 불충분'}")

        return {**state, "is_answer_sufficient": is_sufficient, "retry_count": retry_count}

    except Exception as e:
        print(f"   - ❌ 2차 검증 중 오류 발생: {e}")
        return {**state, "is_answer_sufficient": True, "retry_count": retry_count}

# ===== 대체 답변 노드 =====
def fallback_answer_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 대체 답변 생성")
    fallback_message = "죄송합니다. 해당 질문에 대한 충분한 정보를 찾을 수 없습니다. 다른 질문을 해주세요."
    return {**state, "answer": fallback_message}

# ===== 그래프 빌드 =====
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("load_store", load_store_node)
    g.add_node("temporal_enrich", temporal_enrich_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("retrieval_validation", retrieval_validation_node)
    g.add_node("web_search", web_search_node)
    g.add_node("combine_context", combine_context_node)
    g.add_node("generate_draft", generate_draft_node)
    g.add_node("refine_answer", refine_answer_node)
    g.add_node("answer_validation", answer_validation_node)
    g.add_node("fallback_answer", fallback_answer_node)

    g.set_entry_point("load_store")
    g.add_edge("load_store", "temporal_enrich")
    g.add_edge("temporal_enrich", "retrieve")
    g.add_edge("retrieve", "retrieval_validation")

    # 1차 검증 결과에 따라 웹 검색 여부 결정
    g.add_conditional_edges(
        "retrieval_validation",
        lambda state: "sufficient" if state["is_retrieval_sufficient"] else "insufficient",
        {"sufficient": "combine_context", "insufficient": "web_search"}
    )
    g.add_edge("web_search", "combine_context")
    g.add_edge("combine_context", "generate_draft")
    g.add_edge("generate_draft", "refine_answer")
    g.add_edge("refine_answer", "answer_validation")

    # 2차 검증 결과에 따라 종료/재시도/대체 답변 결정
    def decide_after_answer_validation(state: GraphState) -> str:
        if state["is_answer_sufficient"]:
            return "end"
        elif state["retry_count"] >= MAX_RETRIES:
            return "fallback"
        else:
            return "retry"

    g.add_conditional_edges(
        "answer_validation",
        decide_after_answer_validation,
        {"end": END, "fallback": "fallback_answer", "retry": "web_search"}
    )
    g.add_edge("fallback_answer", END)

    app = g.compile()
    # try:
    #     graph_image_path = "agent_workflow_openai.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(app.get_graph().draw_mermaid_png())
    #     print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류: {e}")
    return app

# =========[ OchestratorTest.py 호환 함수 ]=========
def run(state: dict) -> dict:
    """
    OchestratorTest.py에서 호출되는 재해대응 에이전트 실행 함수
    
    Args:
        state: OchestratorTest.py에서 전달받은 상태 딕셔너리
               - query: 사용자 질문 (필수)
    
    Returns:
        dict: 실행 결과
            - pred_answer: 최종 답변
            - source: "disaster_agent"
    """
    try:
        # 질문 추출
        query = state.get("query", "")
        if not query:
            return {"agent_answer": "질문이 제공되지 않았습니다. 재해 관련 질문을 해주세요."}
        
        print(f"[재해_agent] 질문 처리 시작: {query}")
        
        # 그래프 빌드 및 실행
        app = build_graph()
        
        # 그래프 실행
        result = app.invoke({"question": query})
        
        # 답변 추출
        answer = result.get("answer", "답변을 생성할 수 없습니다.")
        
        print(f"[재해_agent] 답변 생성 완료: {len(answer)}자")
        
        return {"agent_answer": answer}
        
    except Exception as e:
        error_msg = f"재해대응 에이전트 실행 중 오류가 발생했습니다: {e}"
        print(f"[재해_agent] 오류: {e}")
        return {"agent_answer": error_msg}

# =========[ 실행부 ]=========
if __name__ == "__main__":
    parser = ArgumentParser(description="OpenAI 기반 RAG 테스트")
    parser.add_argument("-q", "--question", default=None, help="한 번만 질문하고 종료")
    parser.add_argument("--show-context", action="store_true", help="검색 컨텍스트 및 이유를 출력")

    args = parser.parse_args()

    print("💬 OpenAI 기반 LangGraph RAG 테스트")
    app = build_graph()

    def print_context_and_reason(out: dict):
        ctx = out.get("context", "")
        print("\n=== 컨텍스트(근거) ===")
        print(ctx)

        reasons = []
        if "[웹 검색 결과]" in ctx:
            reasons.append("웹 검색 결과로 부족한 컨텍스트를 보강했습니다.")
        if "[persist]" in ctx or "[pdf_table]" in ctx or "[text]" in ctx:
            reasons.append("로컬 벡터스토어(인덱스) 문서를 근거로 사용했습니다.")
        if "LIVE_STATUS" in ctx or "[live_" in ctx:
            reasons.append("실시간(LIVE) 데이터가 반영되었습니다.")
        if not reasons:
            reasons.append("검색 컨텍스트를 기반으로 답변을 생성했습니다.")

        print("\n--- 왜 이런 답변이 나왔나요? ---")
        print(" · " + "\n · ".join(reasons))

    if args.question:
        q = args.question.strip()
        if not q:
            raise ValueError("질문이 비어 있습니다.")
        try:
            out = app.invoke({"question": q})
            if args.show_context:
                print_context_and_reason(out)
            print("\n=== 답변 ===")
            print(out.get("answer", ""))
            print()
        except Exception as e:
            print(f"❌ 오류: {e}\n")
    else:
        print("질문을 입력하세요. (종료: exit/quit)")
        while True:
            q = input("질문> ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue
            try:
                out = app.invoke({"question": q})
                if args.show_context:
                    print_context_and_reason(out)
                print("\n=== 답변 ===")
                print(out.get("answer", ""))
                print()
            except Exception as e:
                print(f"❌ 오류: {e}\n")
