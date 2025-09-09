# -*- coding: utf-8 -*-
# ============================================================
# PDF -> 질문 생성 -> Milvus 컨텍스트 -> Ground Truth -> 답변 -> RAGAS 평가
# (CSV 없이 메모리로 처리, 최종 Excel만 저장)
# ============================================================

import os
import re
import json
import logging
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict

# Third-party
import pandas as pd
from dotenv import load_dotenv, find_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus as MilvusVectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pymilvus import connections

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_similarity,
)

# PDF
import fitz  # PyMuPDF
import argparse

# ==================== 로깅 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("pdf2golden_ragas")

# ==================== 환경 변수 ====================
load_dotenv(find_dotenv())

# Milvus
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "crop_info")

# Embedding
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))

# Tavily (옵션, 답변 보강/컨텍스트 결합 시 사용)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# 평가 파라미터
MIN_DB_CONTEXT_CHARS = int(os.getenv("MIN_DB_CONTEXT_CHARS", "800"))
EVALUATION_THRESHOLD = {
    "faithfulness": float(os.getenv("THRESH_FAITHFULNESS", 0.7)),
    "answer_relevancy": float(os.getenv("THRESH_ANSWER_RELEVANCY", 0.7)),
    "context_recall": float(os.getenv("THRESH_CONTEXT_RECALL", 0.7)),
    "answer_similarity": float(os.getenv("THRESH_ANSWER_SIMILARITY", 0.8)),
}
USE_WEB_IN_CONTEXTS_FOR_EVAL = os.getenv("USE_WEB_IN_CONTEXTS_FOR_EVAL", "0").strip() == "1"

# Milvus 검색 파라미터
TOPK_RETRIEVE = int(os.getenv("TOPK_RETRIEVE", "10"))
TOPK_USE = int(os.getenv("TOPK_USE", "5"))

# 검증
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env에 필요합니다.")

# ==================== 전역 ====================
_vectorstore = None
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"}  # 필요시 cuda
)
llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)

# ==================== 프롬프트 ====================
# (1) PDF → 질문 생성
QUESTION_SYSTEM_PROMPT = """
너는 농업 현장의 실제 질문을 만들어주는 도우미다.
아래 '컨텍스트'를 영감으로 삼아, 초보~중급 농업인이 할 법한 **한 줄 질문** 1개를 만들어라.

규칙:
- 한글, 한 문장, 존댓말.
- 과도한 전문용어 최소화.
- 맥락은 일반화하되 현실적(재배법/병해충/토양/기상/시설/수확/친환경/작물추천 등).
- 질문만 출력(따옴표/머릿말/번호 금지).
"""

question_prompt = ChatPromptTemplate.from_messages([
    ("system", QUESTION_SYSTEM_PROMPT),
    ("user", "컨텍스트:\n{chunk_text}\n\n질문:")
])

# (2) Ground Truth 생성 (Milvus DB 컨텍스트만 기반)
GT_SYSTEM_PROMPT = """
너는 '골든셋 정답 작성자'다. 아래 컨텍스트만 사용해서 질문에 대한 간결하고 정확한 **정답**을 한국어로 작성하라.
규칙:
- 컨텍스트에 없는 내용은 쓰지 말 것(추측 금지).
- 단계/조건/수치가 있으면 명확히.
- 5~8문장 이내.
"""

gt_prompt = ChatPromptTemplate.from_messages([
    ("system", GT_SYSTEM_PROMPT),
    ("user", "[컨텍스트]\n{contexts}\n\n[질문]\n{question}\n\n[정답]:")
])

# (3) 최종 답변용 RAG 프롬프트
RAG_PROMPT_TMPL = """
당신은 대한민국 농업 작물 추천·재배 전문가입니다.
아래 '문맥'만 사용해 질문에 답하세요. 문맥에 근거가 부족하면 '근거가 부족합니다'라고 명시하세요.

[문맥]
{context}

질문: {question}

규칙:
- 문맥에 있는 정보만 사용.
- 추천/이유/재배 조건·시기·관리 방법을 구체적으로.
- 한국어, 단계는 줄바꿈으로 구분.
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL)

# ==================== LangGraph State ====================
class GraphState(TypedDict, total=False):
    question: Optional[str]
    db_context_str: Optional[str]
    web_context_str: Optional[str]
    final_context_str: Optional[str]
    answer_draft: Optional[str]
    answer: Optional[str]
    retrieved_docs: Optional[List[str]]
    web_search_docs: Optional[List[str]]
    final_contexts: Optional[List[str]]
    retrieved_meta: Optional[List[Dict[str, Any]]]
    web_meta: Optional[List[Dict[str, Any]]]

# ==================== 유틸: PDF → 청크 ====================
def extract_chunks_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """PDF에서 텍스트 추출 후 800~1500자 내외 청크로 분할"""
    chunks: List[Dict[str, str]] = []
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            # 상하단 노이즈 제거(페이지번호/머리말 등)
            h = page.rect.height
            text = page.get_text("text", clip=fitz.Rect(0, h*0.1, page.rect.width, h*0.9))
            full_text.append(text)
        text = re.sub(r'\s+', ' ', "\n".join(full_text)).strip()
        # 문장 단위로 조립
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

# ==================== Milvus / 검색 ====================
def ensure_milvus():
    global _vectorstore
    try:
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)
    except Exception as e:
        logger.warning(f"Milvus 연결 경고: {e}")
    _vectorstore = MilvusVectorStore(
        embedding_function=embedding_model,
        collection_name=MILVUS_COLLECTION,
        connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN},
    )

def retrieve_from_milvus(query: str, topk_retrieve: int = TOPK_RETRIEVE, topk_use: int = TOPK_USE):
    if _vectorstore is None:
        ensure_milvus()
    pairs = _vectorstore.similarity_search_with_score(query, k=topk_retrieve)
    pairs = [(doc, score) for doc, score in pairs if len((doc.page_content or "").strip()) > 100]
    final = pairs[:topk_use]
    contents = [doc.page_content for doc, _ in final]
    metas = [{"id": doc.metadata.get("id") or doc.metadata.get("pk"),
              "source": doc.metadata.get("source"),
              "score": float(score)} for doc, score in final]
    ctx_str = "\n\n".join(contents) if contents else "관련 문서를 찾을 수 없습니다."
    return contents, metas, ctx_str

# ==================== LangGraph 노드 ====================
def route_after_retrieve(state: "GraphState") -> str:
    db_context = (state.get("db_context_str") or "").strip()
    if (not db_context) or ("관련 문서를 찾을 수 없습니다." in db_context) or (len(db_context) < MIN_DB_CONTEXT_CHARS):
        return "need_web"
    return "have_db"

def load_milvus_node(state: GraphState) -> Dict[str, Any]:
    ensure_milvus()
    return {**state}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    q = state.get("question") or ""
    enhanced = f"{q} 재배 방법 키우기 팁"
    contents, metas, ctx_str = retrieve_from_milvus(enhanced)
    return {**state,
            "db_context_str": ctx_str,
            "retrieved_docs": contents,
            "retrieved_meta": metas}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    # 옵션: 웹 보강 (키 없으면 비활성)
    if not TAVILY_API_KEY:
        return {**state, "web_context_str": "웹 검색 비활성화", "web_search_docs": [], "web_meta": []}
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        res = client.search(query=state.get("question") or "", max_results=5, include_raw_content=True)
        docs, meta, parts = [], [], []
        for r in res.get("results", []):
            url = (r.get("url") or "").strip()
            content = (r.get("content") or "").strip()
            if content:
                docs.append(content)
                meta.append({"url": url})
                parts.append(f"- 출처: {url}\n- 내용: {content}")
        web_ctx = "\n\n".join(parts) if parts else "검색 결과 없음"
        return {**state, "web_context_str": web_ctx, "web_search_docs": docs, "web_meta": meta}
    except Exception as e:
        logger.warning(f"웹 검색 실패: {e}")
        return {**state, "web_context_str": "웹 검색 실패", "web_search_docs": [], "web_meta": []}

def combine_context_node(state: GraphState) -> Dict[str, Any]:
    db_ctx = state.get("db_context_str") or ""
    web_ctx = state.get("web_context_str") or ""
    db_docs = state.get("retrieved_docs") or []
    web_docs = state.get("web_search_docs") or []

    if web_ctx and web_ctx not in ["웹 검색 비활성화", "웹 검색 실패", "검색 결과 없음"]:
        final_ctx_str = f"[DB 검색 결과]\n{db_ctx}\n\n[웹 검색 결과]\n{web_ctx}"
    else:
        final_ctx_str = db_ctx

    final_contexts = db_docs + web_docs
    return {**state, "final_context_str": final_ctx_str, "final_contexts": final_contexts}

def generate_draft_node(state: GraphState) -> Dict[str, Any]:
    ctx = state.get("final_context_str") or ""
    q = state.get("question") or ""
    if not ctx or (("관련 문서를 찾을 수 없습니다." in ctx) and ("웹 검색 실패" in ctx)):
        return {**state, "answer_draft": "주어진 정보로는 답변할 수 없습니다."}
    msgs = rag_prompt.format_messages(context=ctx, question=q)
    resp = llm.invoke(msgs)
    return {**state, "answer_draft": resp.content}

def refine_answer_node(state: GraphState) -> Dict[str, Any]:
    return {**state, "answer": state.get("answer_draft", "")}

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("load_milvus", load_milvus_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("web_search", web_search_node)
    g.add_node("combine_context", combine_context_node)
    g.add_node("generate_draft", generate_draft_node)
    g.add_node("refine_answer", refine_answer_node)

    g.set_entry_point("load_milvus")
    g.add_edge("load_milvus", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"need_web": "web_search", "have_db": "combine_context"})
    g.add_edge("web_search", "combine_context")
    g.add_edge("combine_context", "generate_draft")
    g.add_edge("generate_draft", "refine_answer")
    g.add_edge("refine_answer", END)
    return g.compile()

# ==================== 질문 생성 / GT 생성 ====================
def gen_question_from_chunk(chunk_text: str) -> str:
    msgs = question_prompt.format_messages(chunk_text=chunk_text)
    resp = llm.invoke(msgs)
    # 한 줄만 기대
    q = resp.content.strip().replace("\n", " ")
    # 과도한 특수문자 제거
    q = re.sub(r'[「」『』“”"\'`]+', '', q).strip()
    return q

def gen_ground_truth_from_db(question: str) -> (str, List[str], Dict[str, Any]):
    # DB 컨텍스트만으로 GT 생성
    enhanced = f"{question} 재배 방법 키우기 팁"
    contents, metas, _ctx_str = retrieve_from_milvus(enhanced)
    contexts_block = "\n\n".join(contents) if contents else ""
    msgs = gt_prompt.format_messages(contexts=contexts_block, question=question)
    resp = llm.invoke(msgs)
    gt = resp.content.strip()
    return gt, contents, {"retrieved_meta": metas}

# ==================== CLI ====================
def parse_args():
    ap = argparse.ArgumentParser(description="PDF → 질문 생성 → RAGAS 평가 (CSV 없음)")
    ap.add_argument("--input-dir", default="C:\Rookies_project\pdf", help="PDF 폴더 또는 단일 PDF 파일 경로")
    ap.add_argument("--num-chunks", type=int, default=10, help="무작위 선택할 청크 수(=질문 개수)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ==================== 메인 ====================
if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    # 0) 그래프 준비
    app = build_graph()
    try:
        png = "agent_workflow_goldenset.png"
        with open(png, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        logger.info(f"워크플로 PNG 저장: {png}")
    except Exception as e:
        logger.warning(f"Graphviz 미설치 또는 렌더 실패: {e}")

    # 1) PDF → 청크 수집 (경로 보정/검증 + 단일 PDF 지원)  <<< 수정된 구간
    input_dir_raw = args.input_dir or ""
    # 경로 보정
    input_path = os.path.normpath(os.path.expandvars(os.path.expanduser(input_dir_raw)))

    # 폴더/파일 존재 검증, 폴백: 환경변수 PDF_INPUT_DIR
    if not (os.path.isdir(input_path) or (os.path.isfile(input_path) and input_path.lower().endswith(".pdf"))):
        env_fallback = os.getenv("PDF_INPUT_DIR", "").strip()
        if env_fallback:
            input_path = os.path.normpath(os.path.expandvars(os.path.expanduser(env_fallback)))

    if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        logger.info(f"단일 PDF 파일로부터 청크를 추출합니다: {input_path}")
        chunks = extract_chunks_from_pdf(input_path)
    elif os.path.isdir(input_path):
        logger.info(f"PDF 폴더로부터 청크를 수집합니다: {input_path}")
        chunks = collect_pdf_chunks(input_path)
    else:
        logger.error(f"입력 경로를 찾을 수 없습니다. 전달값='{input_dir_raw}', 보정값='{input_path}'. "
                     f"옵션 --input-dir 또는 .env의 PDF_INPUT_DIR를 정확히 설정하세요.")
        raise SystemExit(1)

    if not chunks:
        logger.error("PDF에서 유효한 청크를 찾지 못했습니다.")
        raise SystemExit(1)

    total_chunks = len(chunks)
    if args.num_chunks > total_chunks:
        logger.warning(f"--num-chunks({args.num_chunks}) > 전체 청크({total_chunks}) → 전체 사용")
        sample = chunks
    else:
        sample = random.sample(chunks, args.num_chunks)

    # 2) 질문 & Ground Truth 생성 (DB 컨텍스트만 기반)
    golden_items = []
    for i, ch in enumerate(sample, 1):
        q = gen_question_from_chunk(ch["text"])
        gt, ctxs_for_gt, meta = gen_ground_truth_from_db(q)
        golden_items.append({
            "question": q,
            "ground_truth": gt,
            "gt_contexts": ctxs_for_gt,
            "meta": meta,
            "source_pdf": ch["source"],
        })
        logger.info(f"[{i}/{len(sample)}] 질문 생성: {q[:60]}... | GT 길이: {len(gt)}")

    # 3) RAG(답변 생성) + RAGAS 평가 준비
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    evaluation_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    final_states_log: List[Dict[str, Any]] = []
    retrieval_meta_log: List[List[Dict[str, Any]]] = []
    web_meta_log: List[List[Dict[str, Any]]] = []

    for idx, item in enumerate(golden_items, 1):
        q = item["question"]
        gt = item["ground_truth"]
        # LangGraph 실행
        st = app.invoke({"question": q})
        ans = st.get("answer", "답변 생성 실패")

        # 평가 contexts: DB만 또는 DB+웹
        if USE_WEB_IN_CONTEXTS_FOR_EVAL:
            ctxs = st.get("final_contexts", []) or []
        else:
            # GT 생성 때의 DB 컨텍스트를 그대로 사용하면 재현성↑
            ctxs = item["gt_contexts"] or st.get("retrieved_docs", []) or []

        evaluation_data["question"].append(q)
        evaluation_data["answer"].append(ans)
        evaluation_data["contexts"].append(ctxs)
        evaluation_data["ground_truth"].append(gt)

        final_states_log.append(st)
        retrieval_meta_log.append(st.get("retrieved_meta", []) or item["meta"].get("retrieved_meta", []))
        web_meta_log.append(st.get("web_meta", []))

    # 4) RAGAS 평가
    logger.info("RAGAS 평가 시작...")
    dataset = Dataset.from_dict(evaluation_data)
    metrics = [faithfulness, answer_relevancy, context_recall, answer_similarity]
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embedding_model)
    results_df = result.to_pandas()
    logger.info("RAGAS 평가 완료")

    # 5) 엑셀 병합 출력
    golden_df = pd.DataFrame({
        "question": evaluation_data["question"],
        "ground_truth": evaluation_data["ground_truth"],
        "contexts": [json.dumps(c, ensure_ascii=False) for c in evaluation_data["contexts"]],
        "answer": evaluation_data["answer"],
        "evolution_type": "",  # 필요시 수동 태깅
        "metadata": "",
        "episode_done": True,
        "source_pdf": [it["source_pdf"] for it in golden_items],
    })

    # 메타 주입
    run_meta = {
        "collection": MILVUS_COLLECTION,
        "embedding_model": EMBED_MODEL_NAME,
        "openai_model": OPENAI_MODEL,
        "topk_retrieve": TOPK_RETRIEVE,
        "topk_use": TOPK_USE,
        "use_web_in_eval": USE_WEB_IN_CONTEXTS_FOR_EVAL,
        "timestamp": timestamp,
    }
    merged_meta = []
    for i in range(len(golden_df)):
        st = final_states_log[i]
        row_meta = {
            **run_meta,
            "retrieved_meta": retrieval_meta_log[i],
            "web_meta": web_meta_log[i],
            "db_context_chars": len((st.get("db_context_str") or "")),
            "used_web": st.get("web_context_str", "") not in ["", "웹 검색 비활성화", "웹 검색 실패", "검색 결과 없음"],
        }
        merged_meta.append(json.dumps(row_meta, ensure_ascii=False))
    golden_df["metadata"] = merged_meta

    # 점수 컬럼
    for m in metrics:
        golden_df[f"ragas_{m.name}"] = results_df[m.name].values

    # PASS/FAIL
    def _passfail(i: int) -> str:
        ok = True
        for m in metrics:
            thr = EVALUATION_THRESHOLD.get(m.name, 0.0)
            val = results_df.loc[i, m.name]
            try:
                if float(val) < float(thr):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        return "PASS" if ok else "FAIL"

    golden_df["evaluation_status"] = [_passfail(i) for i in range(len(golden_df))]

    out_xlsx = f"goldenset_from_pdf_with_ragas_{timestamp}.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        golden_df.to_excel(w, index=False, sheet_name="golden_with_scores")
        results_df.to_excel(w, index=False, sheet_name="ragas_raw")

    # 콘솔 요약
    print("\n" + "="*58)
    print(" " * 12 + "RAGAS 평가 요약 (PDF→질문 자동 생성)")
    print("="*58)
    overall = results_df.mean(numeric_only=True)
    for m in metrics:
        name = m.name
        avg = overall.get(name, 0.0)
        thr = EVALUATION_THRESHOLD.get(name, 0.0)
        passes = (results_df[name] >= thr).sum()
        fails = (results_df[name] < thr).sum()
        total = int(passes + fails)
        rate = (passes / total * 100) if total else 0.0
        print(f"\n- {name}: 평균 {avg:.4f} | 기준 {thr} | 통과 {passes} / 실패 {fails} | 통과율 {rate:.2f}%")
    print("="*58)
    print(f"엑셀 파일: {out_xlsx}")
