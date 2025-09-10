# -*- coding: utf-8 -*-
# =================================================================================
# [개선 최종판 v2.2] PDF -> Milvus -> Rerank -> (부족하면 웹 검색) -> 답변 -> RAGAS 평가
# - 기능 확장: LLM 캐싱, 리랭커(Re-ranker) 추가
# - 성능 개선: Asyncio를 이용한 비동기 처리로 실행 시간 단축
# - 수정: 프롬프트 내 마크다운(**) 및 주석 한자 제거
# =================================================================================

import os
import re
import json
import logging
import random
import asyncio  # 비동기 처리를 위한 라이브러리
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict

# Third-party
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from tavily import TavilyClient
import fitz  # PyMuPDF
import argparse
import torch # GPU 사용 가능 여부 확인용

# LangChain & RAG ecosystem
import langchain
from langchain.cache import InMemoryCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus as MilvusVectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pymilvus import connections
from sentence_transformers import CrossEncoder # 리랭커 모델

# Evaluation
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_similarity,
)

# ==================== 로깅 및 캐시 설정 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("pdf2golden_ragas_final_v2_2")

# LLM 호출 결과를 메모리에 캐싱하여 반복 비용 절약
logger.info("LLM 인메모리 캐시를 활성화합니다.")
langchain.llm_cache = InMemoryCache()

# ==================== 환경 변수 로드 ====================
load_dotenv(find_dotenv())

# Milvus
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "crop_info")

# Models
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "bongsoo/korean-cross-encoder-v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Tavily Web Search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Parameters
MIN_DB_CONTEXT_CHARS = int(os.getenv("MIN_DB_CONTEXT_CHARS", "800"))
EVALUATION_THRESHOLD = {
    "faithfulness": float(os.getenv("THRESH_FAITHFULNESS", 0.7)),
    "answer_relevancy": float(os.getenv("THRESH_ANSWER_RELEVANCY", 0.7)),
    "context_recall": float(os.getenv("THRESH_CONTEXT_CALL", 0.7)),
    "answer_similarity": float(os.getenv("THRESH_ANSWER_SIMILARITY", 0.8)),
}
TOPK_RETRIEVE = int(os.getenv("TOPK_RETRIEVE", "10"))
TOPK_USE = int(os.getenv("TOPK_USE", "5"))

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env에 필요합니다.")
if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY가 .env에 없습니다. 웹 검색 기능이 비활성화됩니다.")

# ==================== 전역 객체 생성 ====================
_vectorstore = None

# 자동 장치 감지 (GPU 우선 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"임베딩 및 리랭커 모델을 위한 장치로 '{device}'를 사용합니다.")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": device}
)
reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device) # 리랭커 모델 로드

llm_question = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.7, api_key=OPENAI_API_KEY)
llm_gt = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.5, api_key=OPENAI_API_KEY)
llm_answer = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.5, api_key=OPENAI_API_KEY)

# ==================== 프롬프트 정의 ====================
# (1) PDF 텍스트 조각(chunk)으로부터 질문을 생성하기 위한 프롬프트
QUESTION_SYSTEM_PROMPT = """너는 농업 현장의 실제 질문을 만들어주는 도우미다. 아래 '컨텍스트'를 영감으로 삼아, 초보~중급 농업인이 할 법한 한 줄 질문 1개를 만들어라. 규칙: 한글, 한 문장, 존댓말. 과도한 전문용어 최소화. 맥락은 일반화하되 현실적. 질문만 출력(따옴표/머릿말/번호 금지)."""
question_prompt = ChatPromptTemplate.from_messages([("system", QUESTION_SYSTEM_PROMPT), ("user", "컨텍스트:\n{chunk_text}\n\n질문:")])

# (2) 검색된 DB 컨텍스트로부터 모범 답안(Ground Truth)을 생성하기 위한 프롬프트
GT_SYSTEM_PROMPT = """너는 '골든셋 정답 작성자'다. 아래 컨텍스트만 사용해서 질문에 대한 간결하고 정확한 정답을 한국어로 작성하라. 규칙: 컨텍스트에 없는 내용은 쓰지 말 것(추측 금지). 단계/조건/수치가 있으면 명확히. 5~8문장 이내."""
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
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL)

# ==================== LangGraph 상태 정의 ====================
class GraphState(TypedDict, total=False):
    question: Optional[str]
    db_context_str: Optional[str]
    web_context_str: Optional[str]
    answer: Optional[str]
    retrieved_docs: Optional[List[str]]
    web_search_docs: Optional[List[str]]
    final_contexts: Optional[List[str]]
    retrieved_meta: Optional[List[Dict[str, Any]]]
    web_meta: Optional[List[Dict[str, Any]]]
    labeled_contexts: Optional[List[Dict[str, Any]]]
    labeled_web_contexts: Optional[List[Dict[str, Any]]]
    used_contexts: Optional[List[Dict[str, Any]]]

# ==================== 핵심 함수 (PDF, Milvus, Rerank) ====================
def extract_chunks_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    try:
        doc = fitz.open(pdf_path)
        if doc.is_encrypted:
            logger.warning(f"암호화된 PDF 파일은 건너뜁니다: {pdf_path}")
            return []
        full_text = []
        for page in doc:
            h = page.rect.height
            text = page.get_text("text", clip=fitz.Rect(0, h * 0.1, page.rect.width, h * 0.9))
            full_text.append(text)
        text = re.sub(r'\s+', ' ', "\n".join(full_text)).strip()
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) < 1200:
                buf += s + " "
            else:
                if len(buf) > 800:
                    chunks.append({"source": os.path.basename(pdf_path), "text": buf.strip()})
                buf = s + " "
        if buf and len(buf.strip()) > 800:
            chunks.append({"source": os.path.basename(pdf_path), "text": buf.strip()})
    except Exception as e:
        logger.warning(f"PDF 처리 실패: {pdf_path} - {e}")
    return chunks

def collect_pdf_chunks(input_dir: str) -> List[Dict[str, str]]:
    pdfs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    all_chunks: List[Dict[str, str]] = []
    for p in pdfs:
        all_chunks.extend(extract_chunks_from_pdf(p))
    return all_chunks

def ensure_milvus():
    global _vectorstore
    if _vectorstore: return
    try:
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)
    except Exception as e:
        logger.warning(f"Milvus 연결 경고: {e}")
    _vectorstore = MilvusVectorStore(
        embedding_function=embedding_model,
        collection_name=MILVUS_COLLECTION,
        connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN},
    )

def retrieve_from_milvus(query: str, top_k: int = TOPK_RETRIEVE):
    """Milvus에서 문서를 검색하는 함수 (리랭킹 미적용)"""
    ensure_milvus()
    return _vectorstore.similarity_search_with_score(query, k=top_k)

def rerank_documents(query: str, pairs: List[tuple]) -> List[tuple]:
    """검색된 문서들을 Cross-Encoder로 재정렬하는 함수"""
    if not pairs:
        return []
    logger.info(f"리랭커 모델({RERANKER_MODEL_NAME})로 {len(pairs)}개 문서 재정렬 시작...")
    sentence_pairs = [(query, doc.page_content) for doc, score in pairs]
    scores = reranker.predict(sentence_pairs)
    scored_pairs = list(zip(scores, pairs))
    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    reranked_pairs = [pair for score, pair in scored_pairs]
    logger.info("문서 재정렬 완료.")
    return reranked_pairs

# ==================== LangGraph 노드 정의 ====================
def route_after_retrieve(state: "GraphState") -> str:
    db_context_len = len((state.get("db_context_str") or "").strip())
    if db_context_len < MIN_DB_CONTEXT_CHARS:
        logger.info(f"DB 컨텍스트 길이({db_context_len})가 기준({MIN_DB_CONTEXT_CHARS}) 미만. 웹 검색을 시작합니다.")
        return "web_search"
    logger.info("충분한 DB 컨텍스트를 확보하여 웹 검색을 건너뜁니다.")
    return "combine_context"

def load_milvus_node(state: GraphState) -> Dict[str, Any]:
    ensure_milvus()
    return {**state}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    q = state.get("question") or ""
    enhanced_q = f"{q} 농사, 재배, 병충해"
    logger.info(f"Milvus 검색 시작: '{enhanced_q}'")
    
    pairs = retrieve_from_milvus(enhanced_q, top_k=TOPK_RETRIEVE)
    pairs = [(doc, score) for doc, score in pairs if len((doc.page_content or "").strip()) > 100]
    
    reranked_pairs = rerank_documents(q, pairs)
    
    final = reranked_pairs[:TOPK_USE]
    
    contents = [doc.page_content for doc, _ in final]
    metas = [{"id": doc.metadata.get("id") or doc.metadata.get("pk"), "source": doc.metadata.get("source"), "score": float(score)} for doc, score in final]
    
    labeled = []
    for i, (doc_content, doc_meta) in enumerate(zip(contents, metas), 1):
        labeled.append({"label": f"C{i}", "text": doc_content, "meta": doc_meta})

    ctx_str = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled else "내부 DB에서 관련 정보를 찾지 못했습니다."
    return {**state, "db_context_str": ctx_str, "retrieved_docs": contents, "retrieved_meta": metas, "labeled_contexts": labeled}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        return {**state, "web_context_str": "웹 검색 비활성화", "web_search_docs": [], "web_meta": [], "labeled_web_contexts": []}

    q = state.get("question") or ""
    logger.info(f"Tavily 웹 검색 시작: '{q}'")
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        res = client.search(query=q, max_results=5, search_depth="advanced")
        docs, meta, labeled = [], [], []
        for i, r in enumerate(res.get("results", []), 1):
            content = (r.get("content") or "").strip()
            if content:
                url = r.get("url")
                docs.append(content)
                meta.append({"url": url, "score": r.get("score")})
                labeled.append({"label": f"W{i}", "text": content, "meta": {"url": url}})
        web_ctx = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled else "웹에서 관련 정보를 찾지 못했습니다."
        logger.info(f"{len(docs)}개의 웹 문서 검색 완료.")
        return {**state, "web_context_str": web_ctx, "web_search_docs": docs, "web_meta": meta, "labeled_web_contexts": labeled}
    except Exception as e:
        logger.error(f"웹 검색 실패: {e}")
        return {**state, "web_context_str": "웹 검색 중 오류 발생", "web_search_docs": [], "web_meta": [], "labeled_web_contexts": []}

def combine_context_node(state: GraphState) -> Dict[str, Any]:
    db_docs = state.get("retrieved_docs") or []
    web_docs = state.get("web_search_docs") or []
    final_contexts = db_docs + web_docs
    logger.info(f"컨텍스트 결합: DB {len(db_docs)}개, 웹 {len(web_docs)}개 -> 총 {len(final_contexts)}개")
    return {**state, "final_contexts": final_contexts}

def generate_draft_node(state: GraphState) -> Dict[str, Any]:
    q = state.get("question") or ""
    db_ctx = state.get("db_context_str") or "정보 없음"
    web_ctx = state.get("web_context_str") or "정보 없음"
    labeled_db = state.get("labeled_contexts", [])
    labeled_web = state.get("labeled_web_contexts", [])
    msgs = rag_prompt.format_messages(question=q, db_context=db_ctx, web_context=web_ctx)
    resp = llm_answer.invoke(msgs)
    draft = (resp.content or "").strip()
    used_labels = set(re.findall(r'\[(C|W)\d+\]', draft))
    used_contexts = []
    all_labeled_contexts = {d["label"]: d for d in labeled_db}
    all_labeled_contexts.update({d["label"]: d for d in labeled_web})
    for label in sorted(list(used_labels)):
        if label in all_labeled_contexts:
            used_contexts.append(all_labeled_contexts[label])
    return {**state, "answer": draft, "used_contexts": used_contexts}

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("load_milvus", load_milvus_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("web_search", web_search_node)
    g.add_node("combine_context", combine_context_node)
    g.add_node("generate_draft", generate_draft_node)
    g.set_entry_point("load_milvus")
    g.add_edge("load_milvus", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"web_search": "web_search", "combine_context": "combine_context"})
    g.add_edge("web_search", "combine_context")
    g.add_edge("combine_context", "generate_draft")
    g.add_edge("generate_draft", END)
    return g.compile()

# ==================== 질문/GT 생성 함수 ====================
def gen_question_from_chunk(chunk_text: str) -> str:
    msgs = question_prompt.format_messages(chunk_text=chunk_text)
    resp = llm_question.invoke(msgs)
    q = resp.content.strip().replace("\n", " ").replace('"', '').replace("'", "")
    return re.sub(r'[「」『』“”`]+', '', q).strip()

def gen_ground_truth_from_db(question: str) -> (str, List[str], Dict[str, Any]):
    enhanced = f"{question} 재배 방법 키우기 팁"
    pairs = retrieve_from_milvus(enhanced, top_k=TOPK_USE)
    contents = [doc.page_content for doc, _ in pairs]
    metas = [{"source": doc.metadata.get("source")} for doc, _ in pairs]
    
    contexts_block = "\n\n".join(contents) if contents else ""
    if not contexts_block:
        return "DB 정보 부족", [], {"retrieved_meta": metas}
    
    msgs = gt_prompt.format_messages(contexts=contexts_block, question=question)
    resp = llm_gt.invoke(msgs)
    gt = resp.content.strip()
    return gt, contents, {"retrieved_meta": metas}

# ==================== 메인 실행 로직 (비동기) ====================
def parse_args():
    ap = argparse.ArgumentParser(description="PDF → 질문 생성 → RAGAS 평가 (리랭커, 웹 검색, 비동기 처리)")
    ap.add_argument("--input-dir", default=r"C:\Rookies_project\pdf", help="PDF 폴더 또는 단일 PDF 파일 경로")
    ap.add_argument("--num-chunks", type=int, default=3, help="무작위 선택할 청크 수(=질문 개수)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

async def main():
    args = parse_args()
    random.seed(args.seed)

    app = build_graph()

    # --- 1. PDF에서 청크 수집 ---
    input_path = os.path.normpath(os.path.expandvars(os.path.expanduser(args.input_dir or "")))
    if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        logger.info(f"단일 PDF 파일로부터 청크를 추출합니다: {input_path}")
        chunks = extract_chunks_from_pdf(input_path)
    elif os.path.isdir(input_path):
        logger.info(f"PDF 폴더로부터 청크를 수집합니다: {input_path}")
        chunks = collect_pdf_chunks(input_path)
    else:
        logger.error(f"입력 경로를 찾을 수 없습니다: '{input_path}'")
        raise SystemExit(1)
    if not chunks:
        logger.error("PDF에서 유효한 청크를 찾지 못했습니다.")
        raise SystemExit(1)

    # --- 2. Golden Set 생성 ---
    sample = chunks if args.num_chunks >= len(chunks) else random.sample(chunks, args.num_chunks)
    golden_items = []
    for i, ch in enumerate(sample, 1):
        q = gen_question_from_chunk(ch["text"])
        gt, ctxs_for_gt, meta = gen_ground_truth_from_db(q)
        if gt == "DB 정보 부족":
            logger.warning(f"GT 생성 건너뜀 (DB 정보 부족): {q}")
            continue
        golden_items.append({"question": q, "ground_truth": gt, "gt_contexts": ctxs_for_gt, "meta": meta, "source_pdf": ch["source"]})
        logger.info(f"[{i}/{len(sample)}] 질문/GT 생성: {q[:50]}...")
    
    if not golden_items:
        logger.error("생성된 골든셋이 없습니다. 프로그램을 종료합니다.")
        raise SystemExit(1)

    # --- 3. RAG 파이프라인 비동기 실행 ---
    logger.info(f"{len(golden_items)}개 질문에 대해 RAG 파이프라인을 비동기로 실행합니다...")
    tasks = [app.ainvoke({"question": item["question"]}) for item in golden_items]
    final_states_log = await asyncio.gather(*tasks)
    logger.info("모든 질문에 대한 답변 생성이 완료되었습니다.")

    # --- 4. RAGAS 평가 데이터 준비 ---
    evaluation_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item, st in zip(golden_items, final_states_log):
        evaluation_data["question"].append(item["question"])
        evaluation_data["answer"].append(st.get("answer", "답변 생성 실패"))
        evaluation_data["contexts"].append(st.get("final_contexts", []))
        evaluation_data["ground_truth"].append(item["ground_truth"])

    # --- 5. RAGAS 평가 실행 ---
    logger.info("RAGAS 평가 시작...")
    dataset = Dataset.from_dict(evaluation_data)
    metrics = [faithfulness, answer_relevancy, context_recall, answer_similarity]
    try:
        result = evaluate(dataset=dataset, metrics=metrics, llm=llm_answer, embeddings=embedding_model)
        results_df = result.to_pandas()
        logger.info("RAGAS 평가 완료")
    except Exception as e:
        logger.error(f"RAGAS 평가 중 오류 발생: {e}")
        results_df = pd.DataFrame() # 평가 실패 시 빈 데이터프레임 생성

    # --- 6. 최종 결과 저장 및 출력 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    golden_df = pd.DataFrame.from_dict(evaluation_data)
    
    if not results_df.empty:
        for m in metrics:
            if m.name in results_df.columns:
                golden_df[f"ragas_{m.name}"] = results_df[m.name].values

    out_csv_main = f"ragas_results_{timestamp}.csv"
    golden_df.to_csv(out_csv_main, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 58)
    print(" " * 12 + "RAGAS 평가 요약 (리랭커 + 웹 보강)")
    print("=" * 58)
    if not results_df.empty:
        overall = results_df.mean(numeric_only=True)
        for m in metrics:
            name = m.name
            avg = overall.get(name, 0.0)
            thr = EVALUATION_THRESHOLD.get(name, 0.0)
            passes = (results_df[name] >= thr).sum()
            fails = len(results_df) - passes
            rate = (passes / len(results_df) * 100) if not results_df.empty else 0
            print(f"- {name}: 평균 {avg:.4f} | 기준 {thr} | 통과율 {rate:.2f}% ({passes}/{len(results_df)})")
    else:
        print("RAGAS 평가가 실패하여 요약 정보를 출력할 수 없습니다.")
    print("=" * 58)
    print(f"CSV 파일 저장 완료: {out_csv_main}")

if __name__ == "__main__":
    # 비동기 main 함수 실행
    asyncio.run(main())