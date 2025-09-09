# -*- coding: utf-8 -*-
# ============================================================
# [개선 최종版] PDF -> Milvus -> (부족하면 웹 검색) -> 답변 -> RAGAS 평가
# ============================================================

import os  # 운영체제와 상호작용하기 위한 라이브러리 (파일 경로 등)
import re  # 정규 표현식을 사용하기 위한 라이브러리
import json  # JSON 데이터를 다루기 위한 라이브러리
import logging  # 로그를 기록하기 위한 라이브러리
import random  # 무작위 샘플링 등을 위한 라이브러리
from datetime import datetime  # 날짜와 시간을 다루기 위한 라이브러리
from typing import List, Dict, Any, Optional, TypedDict  # 타입 힌팅을 위한 라이브러리

# Third-party
import pandas as pd  # 데이터 분석 및 조작을 위한 라이브러리 (CSV 저장에 사용)
from dotenv import load_dotenv, find_dotenv  # .env 파일에서 환경 변수를 불러오기 위한 라이브러리
from tavily import TavilyClient  # Tavily 웹 검색 서비스를 사용하기 위한 클라이언트

from langchain_core.prompts import ChatPromptTemplate  # LLM에 전달할 프롬프트 템플릿을 만들기 위한 클래스
from langchain_huggingface import HuggingFaceEmbeddings  # 허깅페이스 모델을 임베딩용으로 사용하기 위한 래퍼
from langchain_milvus import Milvus as MilvusVectorStore  # Milvus 벡터 DB를 LangChain과 함께 사용하기 위한 래퍼
from langchain_openai import ChatOpenAI  # OpenAI의 챗봇 모델을 사용하기 위한 래퍼
from langgraph.graph import StateGraph, END  # 작업 흐름(그래프)을 정의하기 위한 LangGraph 클래스
from pymilvus import connections  # Milvus DB에 직접 연결하기 위한 라이브러리

from datasets import Dataset  # RAGAS 평가에 필요한 데이터셋 형식으로 변환하기 위한 클래스
from ragas import evaluate  # RAGAS 평가를 실행하는 함수
from ragas.metrics import (  # RAGAS에서 사용할 평가지표들
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_similarity,
)

# PDF
import fitz  # PyMuPDF 라이브러리로, PDF 파일을 열고 텍스트를 추출하는 데 사용
import argparse  # 커맨드 라인 인자(argument)를 파싱하기 위한 라이브러리

# ==================== 로깅 설정 ====================
logging.basicConfig(
    level=logging.INFO,  # 로그 출력 레벨을 INFO로 설정 (DEBUG, WARNING, ERROR 등)
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',  # 로그 출력 형식 지정
    datefmt='%Y-%m-%d %H:%M:%S'  # 로그 시간 형식 지정
)
logger = logging.getLogger("pdf2golden_ragas_final")  # 이 스크립트 전용 로거 객체 생성

# ==================== 환경 변수 로드 ====================
load_dotenv(find_dotenv())  # 현재 디렉토리나 상위 디렉토리에서 .env 파일을 찾아 환경 변수로 로드

# Milvus 연결 정보
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")  # .env 파일에서 Milvus 주소를 가져오고, 없으면 기본값 사용
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")  # .env 파일에서 Milvus 토큰을 가져오고, 없으면 기본값 사용
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "crop_info")  # .env 파일에서 사용할 컬렉션 이름을 가져옴

# 임베딩 모델 정보
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")  # 사용할 허깅페이스 임베딩 모델 이름

# OpenAI API 정보
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # .env 파일에서 OpenAI API 키를 가져옴
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 사용할 OpenAI 모델 이름

# Tavily 웹 검색 API 정보
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # .env 파일에서 Tavily API 키를 가져옴

# 평가 파라미터 및 검색 설정
MIN_DB_CONTEXT_CHARS = int(os.getenv("MIN_DB_CONTEXT_CHARS", "800"))  # 웹 검색을 트리거할 DB 검색 결과의 최소 글자 수
EVALUATION_THRESHOLD = {  # RAGAS 평가 점수의 합격/불합격(PASS/FAIL) 기준선
    "faithfulness": float(os.getenv("THRESH_FAITHFULNESS", 0.7)),
    "answer_relevancy": float(os.getenv("THRESH_ANSWER_RELEVANCY", 0.7)),
    "context_recall": float(os.getenv("THRESH_CONTEXT_RECALL", 0.7)),
    "answer_similarity": float(os.getenv("THRESH_ANSWER_SIMILARITY", 0.8)),
}
TOPK_RETRIEVE = int(os.getenv("TOPK_RETRIEVE", "10"))  # Milvus에서 처음에 검색할 문서 개수
TOPK_USE = int(os.getenv("TOPK_USE", "5"))  # 실제로 사용할 상위 문서 개수

# 필수 환경 변수 검증
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env에 필요합니다.")  # OpenAI API 키가 없으면 에러 발생
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY가 .env에 없습니다. 웹 검색 기능이 비활성화됩니다.")  # Tavily API 키가 없으면 경고 메시지 출력

# ==================== 전역 객체 생성 ====================
_vectorstore = None  # Milvus 벡터스토어 객체를 저장할 전역 변수 초기화
embedding_model = HuggingFaceEmbeddings(  # 허깅페이스 임베딩 모델 로드
    model_name=EMBED_MODEL_NAME,  # 사용할 모델 이름 지정
    model_kwargs={"device": "cpu"}  # 모델을 CPU에서 실행하도록 설정 (GPU 사용 시 "cuda")
)
# 역할에 따라 LLM 객체를 분리하여 생성 (temperature 등 파라미터 조절)
llm_question = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.7, api_key=OPENAI_API_KEY)  # 창의성이 필요한 질문 생성용
llm_gt = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.5, api_key=OPENAI_API_KEY)  # 사실 기반의 정답(GT) 생성용
llm_answer = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.5, api_key=OPENAI_API_KEY)  # 최종 답변 생성용

# ==================== 프롬프트 정의 ====================
# (1) PDF 텍스트 조각(chunk)으로부터 질문을 생성하기 위한 프롬프트
QUESTION_SYSTEM_PROMPT = """너는 농업 현장의 실제 질문을 만들어주는 도우미다. 아래 '컨텍스트'를 영감으로 삼아, 초보~중급 농업인이 할 법한 **한 줄 질문** 1개를 만들어라. 규칙: 한글, 한 문장, 존댓말. 과도한 전문용어 최소화. 맥락은 일반화하되 현실적. 질문만 출력(따옴표/머릿말/번호 금지)."""
question_prompt = ChatPromptTemplate.from_messages([("system", QUESTION_SYSTEM_PROMPT), ("user", "컨텍스트:\n{chunk_text}\n\n질문:")])

# (2) 검색된 DB 컨텍스트로부터 모범 답안(Ground Truth)을 생성하기 위한 프롬프트
GT_SYSTEM_PROMPT = """너는 '골든셋 정답 작성자'다. 아래 컨텍스트만 사용해서 질문에 대한 간결하고 정확한 **정답**을 한국어로 작성하라. 규칙: 컨텍스트에 없는 내용은 쓰지 말 것(추측 금지). 단계/조건/수치가 있으면 명확히. 5~8문장 이내."""
gt_prompt = ChatPromptTemplate.from_messages([("system", GT_SYSTEM_PROMPT), ("user", "[컨텍스트]\n{contexts}\n\n[질문]\n{question}\n\n[정답]:")])

# (3) 최종 답변을 생성하기 위한 RAG 프롬프트 (DB + 웹 검색 결과 활용)
RAG_PROMPT_TMPL = """당신은 대한민국 농업 작물 재배 전문가입니다.
아래 제공된 [DB 검색 결과]와 [웹 검색 결과]를 종합하여 질문에 답변하세요.

[DB 검색 결과]
{db_context}

[웹 검색 결과]
{web_context}

[질문]
{question}

---
**규칙 (반드시 엄수):**
1.  **근거 기반 답변:** 제공된 [DB 검색 결과]와 [웹 검색 결과] 내용만으로 답변해야 합니다. 당신의 사전 지식은 절대 사용하지 마세요.
2.  **DB 우선 활용:** DB 정보가 충분하다면 DB를 우선적으로 사용하세요. 웹 검색 결과는 DB 정보가 부족하거나 없을 때 보충하는 용도로 사용하세요.
3.  **인용 필수:** 모든 문장 끝에는 근거가 된 정보의 라벨을 반드시 붙여야 합니다.
    - DB 근거: `[C1]`, `[C2]` ...
    - 웹 근거: `[W1]`, `[W2]` ...
    - 예시: `~하는 것이 좋습니다. [C1]`, `전문가들은 ~라고 말합니다. [W2]`
4.  **근거 없으면 답변 불가:** 두 검색 결과 모두에 근거가 없다면, "제공된 정보로는 답변할 수 없습니다."라고만 답변하세요.
5.  **형식:** 답변은 간결하게, 단계별로 줄을 바꿔 정리하세요.
---
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL)  # 문자열 템플릿으로부터 프롬프트 객체 생성

# ==================== LangGraph 상태 정의 ====================
class GraphState(TypedDict, total=False):  # 작업 흐름(그래프)의 각 단계에서 공유될 데이터 구조 정의
    question: Optional[str]  # 현재 처리 중인 질문
    db_context_str: Optional[str]  # DB에서 검색된 문맥 (라벨 포함)
    web_context_str: Optional[str]  # 웹에서 검색된 문맥 (라벨 포함)
    answer: Optional[str]  # LLM이 생성한 최종 답변
    retrieved_docs: Optional[List[str]]  # DB에서 검색된 순수 텍스트 문서 리스트
    web_search_docs: Optional[List[str]]  # 웹에서 검색된 순수 텍스트 문서 리스트
    final_contexts: Optional[List[str]]  # DB와 웹 문서를 합친, RAGAS 평가에 사용될 최종 컨텍스트 리스트
    retrieved_meta: Optional[List[Dict[str, Any]]]  # DB 검색 결과의 메타데이터 (소스, 점수 등)
    web_meta: Optional[List[Dict[str, Any]]]  # 웹 검색 결과의 메타데이터 (URL 등)
    labeled_contexts: Optional[List[Dict[str, Any]]]  # DB 검색 결과에 [C#] 라벨을 붙인 객체 리스트
    labeled_web_contexts: Optional[List[Dict[str, Any]]]  # 웹 검색 결과에 [W#] 라벨을 붙인 객체 리스트
    used_contexts: Optional[List[Dict[str, Any]]]  # 최종 답변 생성에 실제로 인용된 컨텍스트 리스트

# ==================== PDF 처리 및 Milvus 관련 함수 ====================
def extract_chunks_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """단일 PDF 파일에서 텍스트를 추출하고 적절한 크기의 조각(chunk)으로 자르는 함수"""
    chunks: List[Dict[str, str]] = []  # 추출된 텍스트 조각들을 저장할 리스트
    try:
        doc = fitz.open(pdf_path)  # PyMuPDF를 사용해 PDF 파일 열기
        full_text = []  # PDF의 모든 페이지 텍스트를 저장할 리스트
        for page in doc:  # 각 페이지를 순회
            h = page.rect.height  # 페이지 높이 계산
            # 페이지의 상하 10%를 제외하고 텍스트 추출 (머리말/꼬리말 제거 목적)
            text = page.get_text("text", clip=fitz.Rect(0, h * 0.1, page.rect.width, h * 0.9))
            full_text.append(text)  # 추출된 텍스트를 리스트에 추가
        text = re.sub(r'\s+', ' ', "\n".join(full_text)).strip()  # 모든 페이지 텍스트를 합치고 공백 정리
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)  # 문장 단위로 텍스트 분리
        buf = ""  # 텍스트 조각을 임시로 만들 버퍼
        for s in sentences:  # 각 문장을 순회
            if len(buf) + len(s) < 1200:  # 버퍼와 현재 문장의 길이를 합쳐 1200자가 안 되면
                buf += s + " "  # 버퍼에 현재 문장 추가
            else:  # 1200자가 넘으면
                if len(buf) > 800:  # 버퍼의 길이가 800자를 넘는 유효한 조각이면
                    chunks.append({"source": os.path.basename(pdf_path), "text": buf.strip()})  # 청크 리스트에 추가
                buf = s + " "  # 버퍼를 현재 문장으로 초기화
        if buf and len(buf.strip()) > 800:  # 마지막에 남은 버퍼도 유효하면
            chunks.append({"source": os.path.basename(pdf_path), "text": buf.strip()})  # 청크 리스트에 추가
    except Exception as e:
        logger.warning(f"PDF 처리 실패: {pdf_path} - {e}")  # PDF 처리 중 에러 발생 시 경고 로그
    return chunks  # 최종 텍스트 조각 리스트 반환

def collect_pdf_chunks(input_dir: str) -> List[Dict[str, str]]:
    """지정된 디렉토리의 모든 PDF 파일에서 텍스트 조각을 수집하는 함수"""
    pdfs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]  # 디렉토리 내 모든 PDF 파일 경로 수집
    all_chunks: List[Dict[str, str]] = []  # 모든 PDF의 텍스트 조각을 저장할 리스트
    for p in pdfs:  # 각 PDF 파일에 대해
        all_chunks.extend(extract_chunks_from_pdf(p))  # 텍스트 조각을 추출하여 리스트에 추가
    return all_chunks  # 최종 텍스트 조각 리스트 반환

def ensure_milvus():
    """Milvus 벡터스토어 객체가 없으면 생성하고 연결하는 함수"""
    global _vectorstore  # 전역 변수 _vectorstore를 사용
    try:
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)  # Milvus 서버에 연결
    except Exception as e:
        logger.warning(f"Milvus 연결 경고: {e}")
    _vectorstore = MilvusVectorStore(  # LangChain용 Milvus 벡터스토어 객체 생성
        embedding_function=embedding_model,  # 텍스트를 벡터로 변환할 임베딩 모델 지정
        collection_name=MILVUS_COLLECTION,  # 사용할 컬렉션 이름 지정
        connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN},  # 연결 정보 전달
    )

def retrieve_from_milvus(query: str, topk_retrieve: int = TOPK_RETRIEVE, topk_use: int = TOPK_USE):
    """주어진 쿼리로 Milvus에서 유사도 높은 문서를 검색하는 함수"""
    if _vectorstore is None:  # 벡터스토어 객체가 없으면
        ensure_milvus()  # 생성 및 연결
    pairs = _vectorstore.similarity_search_with_score(query, k=topk_retrieve)  # 유사도 검색 실행 (점수 포함)
    pairs = [(doc, score) for doc, score in pairs if len((doc.page_content or "").strip()) > 100]  # 너무 짧은 문서는 필터링
    final = pairs[:topk_use]  # 상위 k개만 선택
    contents = [doc.page_content for doc, _ in final]  # 문서 내용만 추출
    metas = [{"id": doc.metadata.get("id") or doc.metadata.get("pk"), "source": doc.metadata.get("source"), "score": float(score)} for doc, score in final]  # 메타데이터 추출
    ctx_str = "\n\n".join(contents) if contents else "관련 문서를 찾을 수 없습니다."  # 문서 내용을 하나의 문자열로 합침
    return contents, metas, ctx_str  # (내용 리스트, 메타데이터 리스트, 전체 문자열) 반환

# ==================== LangGraph 노드 정의 ====================
def route_after_retrieve(state: "GraphState") -> str:
    """DB 검색 후 다음 단계(웹 검색 또는 컨텍스트 결합)를 결정하는 라우팅 함수"""
    db_context_len = len((state.get("db_context_str") or "").strip())  # DB 검색 결과의 글자 수 계산
    if db_context_len < MIN_DB_CONTEXT_CHARS:  # 글자 수가 기준보다 적으면
        logger.info(f"DB 컨텍스트 길이({db_context_len})가 기준({MIN_DB_CONTEXT_CHARS}) 미만. 웹 검색을 시작합니다.")
        return "web_search"  # 'web_search' 노드로 분기
    logger.info("충분한 DB 컨텍스트를 확보하여 웹 검색을 건너뜁니다.")
    return "combine_context"  # 기준을 넘으면 'combine_context' 노드로 분기

def load_milvus_node(state: GraphState) -> Dict[str, Any]:
    """그래프의 시작점으로, Milvus 연결을 확인하는 노드"""
    ensure_milvus()  # Milvus 연결 확인
    return {**state}  # 상태를 그대로 다음 노드로 전달

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """질문을 받아 Milvus에서 관련 문서를 검색하는 노드"""
    q = state.get("question") or ""  # 현재 상태에서 질문을 가져옴
    enhanced_q = f"{q} 농사, 재배, 병충해"  # 검색 성능을 높이기 위해 질문에 키워드 추가
    logger.info(f"Milvus 검색 시작: '{enhanced_q}'")
    contents, metas, _ = retrieve_from_milvus(enhanced_q)  # Milvus 검색 실행

    labeled = []  # 라벨링된 컨텍스트를 저장할 리스트
    for i, (doc_content, doc_meta) in enumerate(zip(contents, metas), 1):  # 검색 결과와 메타데이터를 순회
        labeled.append({"label": f"C{i}", "text": doc_content, "meta": doc_meta})  # [C1], [C2]... 라벨과 함께 저장

    ctx_str = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled else "내부 DB에서 관련 정보를 찾지 못했습니다."  # 프롬프트에 넣을 문자열 생성
    return {**state, "db_context_str": ctx_str, "retrieved_docs": contents, "retrieved_meta": metas, "labeled_contexts": labeled}  # 상태 업데이트 후 반환

def web_search_node(state: GraphState) -> Dict[str, Any]:
    """DB 정보가 부족할 때 Tavily API로 웹 검색을 수행하는 노드"""
    if not TAVILY_API_KEY:  # Tavily API 키가 없으면
        return {**state, "web_context_str": "웹 검색 비활성화", "web_search_docs": [], "web_meta": [], "labeled_web_contexts": []}  # 웹 검색 비활성화 상태 반환

    q = state.get("question") or ""  # 현재 질문 가져오기
    logger.info(f"Tavily 웹 검색 시작: '{q}'")

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)  # Tavily 클라이언트 생성
        res = client.search(query=q, max_results=5, search_depth="advanced")  # 웹 검색 실행

        docs, meta, labeled = [], [], []  # 결과 저장용 리스트 초기화
        for i, r in enumerate(res.get("results", []), 1):  # 검색 결과를 순회
            content = (r.get("content") or "").strip()  # 내용 추출 및 정리
            if content:  # 내용이 있으면
                url = r.get("url")  # URL 추출
                docs.append(content)  # 문서 리스트에 추가
                meta.append({"url": url, "score": r.get("score")})  # 메타데이터 리스트에 추가
                labeled.append({"label": f"W{i}", "text": content, "meta": {"url": url}})  # [W1], [W2]... 라벨과 함께 저장

        web_ctx = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled else "웹에서 관련 정보를 찾지 못했습니다."  # 프롬프트용 문자열 생성
        logger.info(f"{len(docs)}개의 웹 문서 검색 완료.")
        return {**state, "web_context_str": web_ctx, "web_search_docs": docs, "web_meta": meta, "labeled_web_contexts": labeled}  # 상태 업데이트 후 반환
    except Exception as e:
        logger.error(f"웹 검색 실패: {e}")  # 에러 발생 시 로그 기록
        return {**state, "web_context_str": "웹 검색 중 오류 발생", "web_search_docs": [], "web_meta": [], "labeled_web_contexts": []}  # 에러 상태 반환

def combine_context_node(state: GraphState) -> Dict[str, Any]:
    """DB와 웹 검색 결과를 최종 컨텍스트 리스트로 결합하는 노드"""
    db_docs = state.get("retrieved_docs") or []  # DB 검색 문서 리스트
    web_docs = state.get("web_search_docs") or []  # 웹 검색 문서 리스트
    final_contexts = db_docs + web_docs  # 두 리스트를 합쳐 RAGAS 평가에 사용할 최종 리스트 생성
    logger.info(f"컨텍스트 결합: DB {len(db_docs)}개, 웹 {len(web_docs)}개 -> 총 {len(final_contexts)}개")
    return {**state, "final_contexts": final_contexts}  # 상태 업데이트 후 반환

def generate_draft_node(state: GraphState) -> Dict[str, Any]:
    """최종적으로 결합된 컨텍스트를 바탕으로 LLM이 답변을 생성하는 노드"""
    q = state.get("question") or ""  # 질문 가져오기
    db_ctx = state.get("db_context_str") or "정보 없음"  # DB 컨텍스트 문자열 가져오기
    web_ctx = state.get("web_context_str") or "정보 없음"  # 웹 컨텍스트 문자열 가져오기

    labeled_db = state.get("labeled_contexts", [])  # 라벨링된 DB 컨텍스트 리스트
    labeled_web = state.get("labeled_web_contexts", [])  # 라벨링된 웹 컨텍스트 리스트

    msgs = rag_prompt.format_messages(question=q, db_context=db_ctx, web_context=web_ctx)  # 최종 프롬프트 생성
    resp = llm_answer.invoke(msgs)  # LLM을 호출하여 답변 생성
    draft = (resp.content or "").strip()  # 생성된 답변 텍스트 추출

    # 답변에서 실제로 인용된 컨텍스트만 추출하는 로직
    used_labels = set(re.findall(r'\[(C|W)\d+\]', draft))  # 답변에서 [C#] 또는 [W#] 형식의 라벨 추출
    used_contexts = []  # 사용된 컨텍스트를 저장할 리스트
    all_labeled_contexts = {d["label"]: d for d in labeled_db}  # DB 컨텍스트를 딕셔너리로 변환 (빠른 조회용)
    all_labeled_contexts.update({d["label"]: d for d in labeled_web})  # 웹 컨텍스트도 추가

    for label in sorted(list(used_labels)):  # 추출된 라벨들을 순회
        if label in all_labeled_contexts:  # 해당 라벨의 원본 컨텍스트가 있으면
            used_contexts.append(all_labeled_contexts[label])  # 사용된 컨텍스트 리스트에 추가

    return {**state, "answer": draft, "used_contexts": used_contexts}  # 최종 답변과 사용된 컨텍스트를 상태에 저장

def build_graph():
    """지금까지 정의한 노드들을 연결하여 작업 흐름(그래프)을 구성하는 함수"""
    g = StateGraph(GraphState)  # 상태 그래프 객체 생성
    g.add_node("load_milvus", load_milvus_node)  # 'load_milvus' 노드 추가
    g.add_node("retrieve", retrieve_node)  # 'retrieve' 노드 추가
    g.add_node("web_search", web_search_node)  # 'web_search' 노드 추가
    g.add_node("combine_context", combine_context_node)  # 'combine_context' 노드 추가
    g.add_node("generate_draft", generate_draft_node)  # 'generate_draft' 노드 추가

    g.set_entry_point("load_milvus")  # 시작점을 'load_milvus'로 지정
    g.add_edge("load_milvus", "retrieve")  # 'load_milvus' 다음에 'retrieve' 실행
    g.add_conditional_edges(  # 'retrieve' 다음에는 조건에 따라 분기
        "retrieve",
        route_after_retrieve,  # 'route_after_retrieve' 함수의 결과에 따라
        {
            "need_web": "web_search",  # 결과가 "need_web"이면 'web_search'로
            "have_db": "combine_context"  # 결과가 "have_db"이면 'combine_context'로
        }
    )
    g.add_edge("web_search", "combine_context")  # 'web_search' 다음에는 'combine_context' 실행
    g.add_edge("combine_context", "generate_draft")  # 'combine_context' 다음에는 'generate_draft' 실행
    g.add_edge("generate_draft", END)  # 'generate_draft'가 끝나면 전체 흐름 종료
    return g.compile()  # 완성된 그래프를 실행 가능한 객체로 컴파일하여 반환

# ==================== 질문/GT 생성 함수 ====================
def gen_question_from_chunk(chunk_text: str) -> str:
    """텍스트 조각으로부터 LLM을 이용해 질문을 생성하는 함수"""
    msgs = question_prompt.format_messages(chunk_text=chunk_text)  # 질문 생성 프롬프트 채우기
    resp = llm_question.invoke(msgs)  # LLM 호출
    q = resp.content.strip().replace("\n", " ").replace('"', '').replace("'", "")  # 결과 정리
    return re.sub(r'[「」『』“”`]+', '', q).strip()  # 불필요한 따옴표 제거 후 반환

def gen_ground_truth_from_db(question: str) -> (str, List[str], Dict[str, Any]):
    """질문과 DB 컨텍스트로부터 모범 답안(GT)을 생성하는 함수"""
    enhanced = f"{question} 재배 방법 키우기 팁"  # 검색어 보강
    contents, metas, _ = retrieve_from_milvus(enhanced)  # Milvus에서 문서 검색
    contexts_block = "\n\n".join(contents) if contents else ""  # 검색된 내용을 하나의 문자열로 합침
    if not contexts_block:  # 검색된 내용이 없으면
        return "DB 정보 부족", [], {"retrieved_meta": metas}  # '정보 부족' 상태 반환
    msgs = gt_prompt.format_messages(contexts=contexts_block, question=question)  # GT 생성 프롬프트 채우기
    resp = llm_gt.invoke(msgs)  # LLM 호출
    gt = resp.content.strip()  # 생성된 GT 텍스트
    return gt, contents, {"retrieved_meta": metas}  # (GT, 근거 문서, 메타데이터) 반환

# ==================== CLI 인자 파싱 및 메인 실행 로직 ====================
def parse_args():
    """커맨드 라인에서 입력받은 인자를 파싱하는 함수"""
    ap = argparse.ArgumentParser(description="PDF → 질문 생성 → RAGAS 평가 (웹 검색 추가)")  # 파서 객체 생성
    ap.add_argument("--input-dir", default=r"C:\Rookies_project\pdf", help="PDF 폴더 또는 단일 PDF 파일 경로")  # PDF 경로 인자
    ap.add_argument("--num-chunks", type=int, default=3, help="무작위 선택할 청크 수(=질문 개수)")  # 생성할 질문 개수 인자
    ap.add_argument("--seed", type=int, default=42)  # 재현성을 위한 랜덤 시드 인자
    return ap.parse_args()  # 파싱된 인자 반환

if __name__ == "__main__":  # 이 스크립트가 직접 실행될 때만 아래 코드 블록 실행
    args = parse_args()  # 커맨드 라인 인자 파싱
    random.seed(args.seed)  # 랜덤 시드 설정 (매번 같은 순서로 샘플링하기 위함)

    app = build_graph()  # LangGraph 작업 흐름 컴파일

    # PDF 파일 경로 처리 및 청크 수집
    input_dir_raw = args.input_dir or ""
    input_path = os.path.normpath(os.path.expandvars(os.path.expanduser(input_dir_raw)))  # 경로 정규화
    if not (os.path.isdir(input_path) or (os.path.isfile(input_path) and input_path.lower().endswith(".pdf"))):
        env_fallback = os.getenv("PDF_INPUT_DIR", "").strip()  # .env 파일의 대체 경로 확인
        if env_fallback: input_path = os.path.normpath(os.path.expandvars(os.path.expanduser(env_fallback)))
    if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):  # 단일 PDF 파일인 경우
        logger.info(f"단일 PDF 파일로부터 청크를 추출합니다: {input_path}")
        chunks = extract_chunks_from_pdf(input_path)
    elif os.path.isdir(input_path):  # 디렉토리인 경우
        logger.info(f"PDF 폴더로부터 청크를 수집합니다: {input_path}")
        chunks = collect_pdf_chunks(input_path)
    else:
        logger.error(f"입력 경로를 찾을 수 없습니다: '{input_path}'")
        raise SystemExit(1)  # 프로그램 비정상 종료
    if not chunks:
        logger.error("PDF에서 유효한 청크를 찾지 못했습니다.")
        raise SystemExit(1)

    # 지정된 개수만큼 텍스트 조각을 무작위로 샘플링
    sample = chunks if args.num_chunks >= len(chunks) else random.sample(chunks, args.num_chunks)

    golden_items = []  # 생성된 (질문, GT) 쌍을 저장할 리스트
    # 샘플링된 각 텍스트 조각에 대해 질문과 GT 생성
    for i, ch in enumerate(sample, 1):
        q = gen_question_from_chunk(ch["text"])  # 질문 생성
        gt, ctxs_for_gt, meta = gen_ground_truth_from_db(q)  # GT 생성
        if gt == "DB 정보 부족":  # GT 생성에 실패하면
            logger.warning(f"GT 생성 건너뜀 (DB 정보 부족): {q}")
            continue  # 다음 조각으로 넘어감
        golden_items.append({"question": q, "ground_truth": gt, "gt_contexts": ctxs_for_gt, "meta": meta, "source_pdf": ch["source"]})
        logger.info(f"[{i}/{len(sample)}] 질문/GT 생성: {q[:50]}...")

    if not golden_items:  # 생성된 골든셋이 하나도 없으면
        logger.error("생성된 골든셋이 없습니다. 프로그램을 종료합니다.")
        raise SystemExit(1)

    # RAG 실행 및 평가 데이터 준비
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 파일명에 사용할 타임스탬프 생성
    evaluation_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}  # RAGAS 평가용 데이터 딕셔너리
    final_states_log = []  # 각 질문 처리 후의 최종 상태를 저장할 리스트

    for item in golden_items:  # 생성된 각 골든셋에 대해
        st = app.invoke({"question": item["question"]})  # LangGraph 앱 실행하여 답변 생성

        # 평가용 데이터 딕셔너리에 결과 추가
        evaluation_data["question"].append(item["question"])
        evaluation_data["answer"].append(st.get("answer", "답변 생성 실패"))
        evaluation_data["contexts"].append(st.get("final_contexts", []))
        evaluation_data["ground_truth"].append(item["ground_truth"])
        final_states_log.append(st)  # 최종 상태 저장

    # RAGAS 평가 실행
    logger.info("RAGAS 평가 시작...")
    dataset = Dataset.from_dict(evaluation_data)  # 딕셔너리를 RAGAS용 데이터셋으로 변환
    metrics = [faithfulness, answer_relevancy, context_recall, answer_similarity]  # 사용할 평가지표 리스트
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm_answer, embeddings=embedding_model)  # 평가 실행
    results_df = result.to_pandas()  # 평가 결과를 판다스 데이터프레임으로 변환
    logger.info("RAGAS 평가 완료")

    # 최종 결과를 CSV 파일로 저장
    golden_df = pd.DataFrame.from_dict(evaluation_data)  # 평가 데이터를 데이터프레임으로 변환
    for m in metrics:  # 각 평가지표에 대해
        golden_df[f"ragas_{m.name}"] = results_df[m.name].values  # 점수 컬럼 추가

    out_csv_main = f"ragas_results_{timestamp}.csv"  # 출력 CSV 파일명 생성
    golden_df.to_csv(out_csv_main, index=False, encoding='utf-8-sig')  # 데이터프레임을 CSV로 저장 (엑셀에서 한글 깨짐 방지)

    # 콘솔에 평가 요약 정보 출력
    print("\n" + "=" * 58)
    print(" " * 12 + "RAGAS 평가 요약 (웹 검색 보강)")
    print("=" * 58)
    overall = results_df.mean(numeric_only=True)  # 모든 점수의 평균 계산
    for m in metrics:  # 각 평가지표에 대해
        name = m.name  # 지표 이름
        avg = overall.get(name, 0.0)  # 평균 점수
        thr = EVALUATION_THRESHOLD.get(name, 0.0)  # 합격 기준 점수
        passes = (results_df[name] >= thr).sum()  # 합격 개수
        fails = len(results_df) - passes  # 불합격 개수
        rate = (passes / len(results_df) * 100) if results_df.empty is False else 0  # 합격률
        print(f"- {name}: 평균 {avg:.4f} | 기준 {thr} | 통과율 {rate:.2f}% ({passes}/{len(results_df)})")
    print("=" * 58)
    print(f"CSV 파일 저장 완료: {out_csv_main}")