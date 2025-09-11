import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========[ 벡터스토어 / 임베딩 관련 Import (맨 위) ]=========
from common.milvus_helpers import search_milvus_documents, search_milvus_documents_by_subject, create_context_from_documents

# =========[ 표준/외부 라이브러리 ]=========
import os
import re
import json
import time
import asyncio
import threading
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
from langchain.schema import Document

# =========[ LangChain / LangGraph / LLM ]=========
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# =========[ 환경설정 ]=========
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agri_disaster_docs")

# OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

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
    is_retrieval_sufficient: bool
    temporal: Optional[Dict[str, Any]]
    milvus_data: Optional[Dict[str, Any]]
    milvus_context: Optional[str]

def make_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")
    return ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)

# === 프롬프트들 ===
DRAFT_PROMPT = ChatPromptTemplate.from_template(
    """너는 농작물 재해 정보 전문가야.
아래 문맥을 참고하여 질문에 대한 초안 답변을 작성해줘.
**문맥에 표(pdf_table)가 있으면 이를 중심으로 답변해줘.**
**문맥에 없는 정보는 절대 넣지마.**
**질문에서 요청한 연도와 문맥의 연도가 다르면 반드시 "해당 연도의 정보를 찾을 수 없습니다"라고 답변해줘.**

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
    """너는 농업재해 및 작물 재해대응 정보 전문가야.
아래에는 로컬 인덱스 문맥과 웹 검색 결과가 함께 있어.
이 모든 정보를 활용하여 초안 답변을 작성해줘. 답변은 대화체로, 친절하게 설명하는 스타일로 작성해야 해.

**중요**: 질문에서 요청한 연도와 문맥의 연도가 다르면 반드시 "해당 연도의 정보를 찾을 수 없습니다"라고 답변해줘.

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

# ===== LLM 기반 검증 프롬프트들 =====
LLM_RETRIEVAL_VALIDATION_PROMPT = ChatPromptTemplate.from_template(
    """당신은 농업재해 정보 검색 품질을 평가하는 AI 전문가입니다.
주어진 질문에 대해 검색된 문서가 충분한 정보를 담고 있는지 판단해주세요.

**평가 기준:**
1. **관련성**: 검색된 문서가 질문과 직접적으로 관련이 있는가?
2. **완전성**: 질문에 답하기에 필요한 핵심 정보가 포함되어 있는가?
3. **구체성**: 구체적인 수치, 지역, 기간, 재해 유형 등의 정보가 있는가?
4. **신뢰성**: 출처가 명확하고 신뢰할 수 있는 정보인가?

**답변 형식:**
다음 JSON 형식으로만 답변하세요:
{{
    "judgment": "SUFFICIENT" 또는 "INSUFFICIENT",
    "reason": "판단 이유를 간단히 설명"
}}

[질문]
{question}

[검색된 문서]
{context}

[판단]:"""
)

# (삭제됨) LLM_ANSWER_VALIDATION_PROMPT
# 답변 품질에 대한 2차 검증 프롬프트는 요구사항에 따라 제거되었습니다.

def _has_web_results(context_text: str) -> bool:
    c = (context_text or "")
    return "[웹 검색 결과]" in c

# =========[ LangGraph 노드 ]=========
def load_store_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: MilvusDB 연결 확인")
    milvus_data = state.get("milvus_data", {})
    
    if milvus_data.get("connection_status", False):
        print("   - ✅ MilvusDB 연결됨")
    else:
        print("   - ⚠️ MilvusDB 연결 안됨")
    
    return {**state}

def temporal_enrich_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 시간/상대시점 해석(KST 기준)")
    temporal = build_temporal_meta()
    q_raw = state.get("question", "")
    q_resolved = resolve_relative_years_kst(q_raw, temporal)
    if q_resolved != q_raw:
        print(f"   - 질문 치환: '{q_raw}'   ->   '{q_resolved}'")
    return {**state, "question": q_raw, "temporal": temporal, "question_resolved": q_resolved}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 검색 (MilvusDB 통합 + 메타데이터 필터링)")
    q = state.get("question_resolved", state.get("question", ""))
    milvus_data = state.get("milvus_data", {})
    
    # MilvusDB 연결 상태 확인
    if not milvus_data.get("connection_status", False):
        print("   - ⚠️ MilvusDB 연결 안됨 - 빈 컨텍스트로 진행")
        return {**state, "db_context": "관련 문서를 찾을 수 없습니다.", "retrieved_docs": []}

    region = extract_region_from_question(q)
    # ✅ 질문에서 연도 추출 (기존 기능 유지)
    year_match = re.search(r"(19|20)\d{2}", q)
    year = int(year_match.group()) if year_match else None

    try:
        print("   - 🔍 MilvusDB에서 문서 검색 중...")
        
        # 검색 쿼리 구성 (지역명과 연도 정보 포함)
        search_query = q
        if region:
            search_query = f"{q} {region}"
            print(f"   - 지역명 '{region}'을(를) 검색 쿼리에 포함")
        if year:
            search_query = f"{search_query} {year}년"
            print(f"   - 연도 '{year}'을(를) 검색 쿼리에 포함")
        
        # MilvusDB에서 재해 관련 문서 검색
        documents = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name=COLLECTION_NAME,
            query=search_query,
            k=30
        )
        
        # 필터링된 검색 결과가 부족한 경우 일반 검색 추가
        if (region or year) and documents and len(documents) < 10:
            print("   - 필터링된 검색 결과가 부족하여 일반 검색을 추가로 수행합니다.")
            additional_docs = search_milvus_documents(
                milvus_data=milvus_data,
                collection_name=COLLECTION_NAME,
                query=q,
                k=20
            )
            
            # 중복 제거하면서 추가
            existing_content = {doc.page_content for doc in documents}
            for doc in additional_docs:
                if doc.page_content not in existing_content:
                    documents.append(doc)
                    existing_content.add(doc.page_content)
        
        if documents:
            print(f"   - ✅ MilvusDB 검색 완료: {len(documents)}개 문서")
            
            # 검색 결과를 상세하게 포맷팅 (기존 형식 유지)
            ctx_parts = []
            for i, doc in enumerate(documents[:30], 1):
                meta = getattr(doc, "metadata", {})
                fname = meta.get("file_name") or meta.get("source") or f"문서{i}"
                page = meta.get("page")
                tag = meta.get("type") or "text"
                years = meta.get("years", [])
                
                # 기존 형식과 유사하게 유사도 점수 표시 (MilvusDB에서는 직접 점수를 얻기 어려움)
                header = f"[문서{i}][{tag}][{fname}{f' p.{page}' if page else ''}][years={years}]"
                ctx_parts.append(f"{header}\n{doc.page_content}")
            
            context = "\n\n".join(ctx_parts)
            print(f"   - 📄 상세 컨텍스트 생성: {len(context)}자")
            
            return {**state, "db_context": context, "retrieved_docs": documents}
        else:
            print("   - ⚠️ MilvusDB에서 관련 문서를 찾지 못함")
            return {**state, "db_context": "관련 문서를 찾을 수 없습니다.", "retrieved_docs": []}
            
    except Exception as e:
        print(f"   - ❌ MilvusDB 검색 실패: {e}")
        return {**state, "db_context": f"검색 중 오류가 발생했습니다: {e}", "retrieved_docs": []}


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
    print("🧩 노드: 최종 답변 생성")
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
    return {**state, "answer": ans.strip()}

# refine_answer_node 제거됨 - 초안 답변을 그대로 최종 답변으로 사용

# ===== LLM 기반 검증 노드들 =====
# (삭제됨) MAX_RETRIES
# 2차 검증/재시도 로직 제거로 인해 MAX_RETRIES 상수 또한 제거합니다.

def llm_retrieval_validation_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 1차 검증 (LLM 기반 검색 품질 평가)")
    question = state.get("question_resolved", state.get("question", ""))
    db_context = state.get("db_context", "")

    if not db_context or "관련 문서를 찾을 수 없습니다." in db_context:
        print("   - ❌ 검색된 문서가 없어 불충분으로 판단합니다.")
        return {**state, "is_retrieval_sufficient": False}

    try:
        print("   - 🤖 LLM이 검색 품질을 평가 중...")
        
        # LLM 기반 검증 체인
        validation_chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context")
            }
            | LLM_RETRIEVAL_VALIDATION_PROMPT
            | make_llm()
            | StrOutputParser()
        )
        
        result = validation_chain.invoke({
            "question": question,
            "context": db_context
        })
        
        # JSON 결과 파싱
        try:
            import json
            result_json = json.loads(result.strip())
            judgment = result_json.get("judgment", "INSUFFICIENT")
            reason = result_json.get("reason", "파싱 실패")

            is_sufficient = str(judgment).upper() == "SUFFICIENT"

            print(f"   - 📊 LLM 검증 결과:")
            print(f"     • LLM 판단: {judgment}")
            print(f"     • 최종 판단: {'SUFFICIENT' if is_sufficient else 'INSUFFICIENT'}")
            print(f"     • 이유: {reason}")
            print(f"   - 🎯 최종 결과: {'✅ 충분' if is_sufficient else '⚠️ 불충분'}")

        except json.JSONDecodeError:
            # JSON 파싱 실패 시 기존 방식으로 폴백
            result_clean = result.strip().upper()
            is_sufficient = "SUFFICIENT" in result_clean
            print(f"   - 📊 LLM 검증 결과 (폴백): {result.strip()}")
            print(f"   - 🎯 최종 판단: {'✅ 충분' if is_sufficient else '⚠️ 불충분'}")
        
        return {**state, "is_retrieval_sufficient": is_sufficient}
        
    except Exception as e:
        print(f"   - ❌ LLM 검증 중 오류 발생: {e}")
        # 오류 시 기본적으로 불충분으로 판단
        return {**state, "is_retrieval_sufficient": False}

# (삭제됨) llm_answer_validation_node
# 2차 답변 품질 평가 노드는 제거되었습니다.

# (삭제됨) fallback_answer_node
# 2차 검증 실패 시 사용되던 폴백 노드는 제거되었습니다.

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
    g.add_node("llm_retrieval_validation", llm_retrieval_validation_node)
    g.add_node("web_search", web_search_node)
    g.add_node("combine_context", combine_context_node)
    g.add_node("generate_draft", generate_draft_node)
    # refine_answer 노드 제거됨
    # 2차 검증/폴백 노드 제거

    g.set_entry_point("load_store")
    g.add_edge("load_store", "temporal_enrich")
    g.add_edge("temporal_enrich", "retrieve")
    g.add_edge("retrieve", "llm_retrieval_validation")

    # 1차 검증 결과에 따라 웹 검색 여부 결정
    g.add_conditional_edges(
        "llm_retrieval_validation",
        lambda state: "sufficient" if state["is_retrieval_sufficient"] else "insufficient",
        {"sufficient": "combine_context", "insufficient": "web_search"}
    )
    g.add_edge("web_search", "combine_context")
    g.add_edge("combine_context", "generate_draft")
    # refine_answer 제거: generate_draft에서 바로 최종 답변 생성 후 종료
    g.add_edge("generate_draft", END)

    # 2차 검증 흐름 제거에 따라 관련 분기 제거 (복구 가능)
    # 필요 시 추후 복구 가능

    app = g.compile()
    try:
        graph_image_path = "agent_workflow_openai.png"
        with open(graph_image_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류: {e}")
    return app

# =========[ 지연 로딩을 위한 전역 변수 ]=========
_disaster_app = None

def _get_disaster_app():
    """재해대응 에이전트 애플리케이션을 지연 로딩으로 가져오기"""
    global _disaster_app
    if _disaster_app is None:
        print("⚠️ 재해_agent 모듈 로딩 중...")
        _disaster_app = build_graph()
        print("✅ 재해_agent 모듈 로딩 완료")
    return _disaster_app

# =========[ OchestratorTest.py 호환 함수 ]=========
def run(state: dict) -> dict:
    """
    OchestratorTest.py에서 호출되는 재해대응 에이전트 실행 함수 (비동기)
    
    Args:
        state: OchestratorTest.py에서 전달받은 상태 딕셔너리
               - query: 사용자 질문 (필수)
               - milvus_data: MilvusDB 연결 정보 (선택)
               - milvus_context: 기존 Milvus 컨텍스트 (선택)
    
    Returns:
        dict: 실행 결과
            - agent_answer: 최종 답변
            - source: "disaster_agent"
    """
    try:
        # 질문 추출
        query = state.get("query", "")
        milvus_data = state.get("milvus_data", {})
        milvus_context = state.get("milvus_context", "")
        
        if not query:
            return {"agent_answer": "질문이 제공되지 않았습니다. 재해 관련 질문을 해주세요."}
        
        print(f"[재해_agent_LLM] 질문 처리 시작: {query}")
        print(f"[재해_agent_LLM] MilvusDB 연결: {'연결됨' if milvus_data.get('connection_status') else '연결 안됨'}")
        
        # MilvusDB 연결 상태 확인 및 로깅
        if milvus_data.get("connection_status", False):
            print(f"[재해_agent_LLM] MilvusDB 컬렉션: {COLLECTION_NAME}")
        else:
            print(f"[재해_agent_LLM] ⚠️ MilvusDB 연결 안됨 - 제한된 기능으로 진행")
        
        # 그래프를 지연 로딩으로 가져오기
        app = _get_disaster_app()
        
        # 그래프 실행을 위한 초기 상태 구성
        initial_state = {
            "question": query,
            "milvus_data": milvus_data,
            "milvus_context": milvus_context
        }
        
        # 그래프 실행
        result = app.invoke(initial_state)
        
        # 답변 추출
        answer = result.get("answer", "답변을 생성할 수 없습니다.")
        
        print(f"[재해_agent_LLM] 답변 생성 완료: {len(answer)}자")
        
        return {"agent_answer": answer}
        
    except Exception as e:
        error_msg = f"재해대응 에이전트 실행 중 오류가 발생했습니다: {e}"
        print(f"[재해_agent_LLM] 오류: {e}")
        return {"agent_answer": error_msg}

# =========[ 실행부 ]=========
if __name__ == "__main__":
    parser = ArgumentParser(description="LLM 기반 검증 RAG 테스트")
    parser.add_argument("-q", "--question", default=None, help="한 번만 질문하고 종료")
    parser.add_argument("--show-context", action="store_true", help="검색 컨텍스트 및 이유를 출력")

    args = parser.parse_args()

    print("💬 LLM 기반 검증 LangGraph RAG 테스트")
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
