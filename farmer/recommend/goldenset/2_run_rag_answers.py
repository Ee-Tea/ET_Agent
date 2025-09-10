# 2_run_rag_answers.py (v1.6: 문제 해결 및 프롬프트 재구성)

import os
import re
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import torch
from tavily import TavilyClient
from langchain_community.cache import InMemoryCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus as MilvusVectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pymilvus import connections
from sentence_transformers import CrossEncoder
import langchain

# ==================== 설정: 입력 파일 이름 ====================
INPUT_CSV_FILENAME = "1_golden_set_20250910_175515.csv"
# ==========================================================

# ==================== 설정 (공통) ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("run_rag_answers")
langchain.llm_cache = InMemoryCache()
load_dotenv(find_dotenv())

# 환경 변수
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "crop_info")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MIN_DB_CONTEXT_CHARS = int(os.getenv("MIN_DB_CONTEXT_CHARS", "200"))
TOPK_RETRIEVE = int(os.getenv("TOPK_RETRIEVE", "10"))
TOPK_USE = int(os.getenv("TOPK_USE", "5"))
if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY가 .env에 필요합니다.")

# 전역 객체
_vectorstore = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"모델을 위한 장치로 '{device}'를 사용합니다.")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={"device": device})
reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)
llm_answer = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.7, api_key=OPENAI_API_KEY)

# 프롬프트 (재구성)
RAG_PROMPT_TMPL = """당신은 대한민국 농업 작물 재배 전문가입니다. 아래 제공된 [DB 검색 결과]와 [웹 검색 결과]를 종합하여 질문에 답변하세요.
[DB 검색 결과]\n{db_context}\n\n[웹 검색 결과]\n{web_context}\n\n[질문]\n{question}\n\n---
규칙 (반드시 엄수):
1. **근거 기반 답변**: 답변의 모든 내용은 **제공된 [DB]와 [웹 검색 결과]의 사실만을 기반**으로 구성해야 합니다. 추측은 절대 사용하지 마세요. (할루시네이션 방지)
2. **작물 추천 특화**: 질문에 관련된 작물 추천 정보만으로 답변을 구성하세요. 주제에서 벗어난 내용은 제외해야 합니다.
3. **요약 및 정리**: 답변은 5~8 문장으로 간결하게 요약하고 정리하세요. (컨텍스트-답변 일치도 100% 목표)
4. **인용 필수**: 답변에 사용된 모든 문장에는 근거가 된 정보의 라벨(`[C1]`, `[W1]` 등)을 반드시 표시하세요.
5. **형식 준수**: 마크다운, 불릿포인트, 번호매기기 등 **특수 문자를 사용하지 않은**, 간결하고 자연스러운 존댓말 문장으로만 작성하세요.
6. **중복 및 서론 제거**: 동일한 내용이 반복되지 않도록 주의하고, "제공된 정보에 따르면"과 같은 불필요한 서론은 제거하세요.
7. **맞춤법 검사**: 모든 문장은 맞춤법에 맞게 작성되어야 합니다."""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL)

# ==================== RAG 파이프라인 (LangGraph) ====================
class GraphState(TypedDict, total=False):
    question: Optional[str]; db_context_str: Optional[str]; web_context_str: Optional[str]; answer: Optional[str]
    retrieved_docs: Optional[List[str]]; web_search_docs: Optional[List[str]]; final_contexts: Optional[List[str]]

def ensure_milvus():
    global _vectorstore
    if _vectorstore: return
    try:
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)
        _vectorstore = MilvusVectorStore(embedding_function=embedding_model, collection_name=MILVUS_COLLECTION, connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN})
    except Exception as e:
        logger.error(f"Milvus 연결 실패: {e}")
        raise

def retrieve_from_milvus(query: str, top_k: int):
    if not _vectorstore: ensure_milvus()
    return _vectorstore.similarity_search_with_score(query, k=top_k)

def rerank_documents(query: str, pairs: List[tuple]) -> List[tuple]:
    if not pairs: return []
    sentence_pairs = [(query, doc.page_content) for doc, score in pairs]
    scores = reranker.predict(sentence_pairs)
    scored_pairs = sorted(list(zip(scores, pairs)), key=lambda x: x[0], reverse=True)
    return [pair for score, pair in scored_pairs]

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    q = state.get("question", "")
    pairs = retrieve_from_milvus(q, top_k=TOPK_RETRIEVE)
    pairs = [(doc, score) for doc, score in pairs if len(doc.page_content.strip()) > 100]
    reranked_pairs = rerank_documents(q, pairs)
    final = reranked_pairs[:TOPK_USE]
    contents = [doc.page_content for doc, _ in final]
    labeled = [{"label": f"C{i+1}", "text": doc_content} for i, doc_content in enumerate(contents)]
    ctx_str = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled else "내부 DB에서 관련 정보를 찾지 못했습니다."
    return {"db_context_str": ctx_str, "retrieved_docs": contents}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        return {"web_context_str": "웹 검색 비활성화", "web_search_docs": []}
    q = state.get("question", "")
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        res = client.search(query=q, max_results=10, search_depth="advanced")
        docs = [r["content"].strip() for r in res.get("results", []) if r.get("content")]
        labeled = [{"label": f"W{i+1}", "text": content} for i, content in enumerate(docs)]
        web_ctx = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled else "웹에서 관련 정보를 찾지 못했습니다."
        return {"web_context_str": web_ctx, "web_search_docs": docs}
    except Exception as e:
        logger.error(f"웹 검색 실패: {e}")
        return {"web_context_str": "웹 검색 중 오류 발생", "web_search_docs": []}

def combine_context_node(state: GraphState) -> Dict[str, Any]:
    db_docs = state.get("retrieved_docs", [])
    web_docs = state.get("web_search_docs", [])
    return {"final_contexts": db_docs + web_docs}

def generate_draft_node(state: GraphState) -> Dict[str, Any]:
    msgs = rag_prompt.format_messages(question=state.get("question", ""), db_context=state.get("db_context_str", "정보 없음"), web_context=state.get("web_context_str", "정보 없음"))
    resp = llm_answer.invoke(msgs)
    return {"answer": resp.content.strip()}

def route_after_retrieve(state: "GraphState") -> str:
    return "web_search" if len(state.get("db_context_str", "").strip()) < MIN_DB_CONTEXT_CHARS else "combine_context"

def build_graph():
    g = StateGraph(GraphState)
    nodes = [("retrieve", retrieve_node), ("web_search", web_search_node), ("combine_context", combine_context_node), ("generate_draft", generate_draft_node)]
    for name, node in nodes: g.add_node(name, node)
    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"web_search": "web_search", "combine_context": "combine_context"})
    g.add_edge("web_search", "combine_context")
    g.add_edge("combine_context", "generate_draft")
    g.add_edge("generate_draft", END)
    return g.compile()

def remove_markdown_and_special_chars(text: str) -> str:
    text = re.sub(r'#{1,6}\s', '', text)
    text = re.sub(r'[\*\-]', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

async def main():
    logger.info(f"'{INPUT_CSV_FILENAME}' 파일에서 골든셋을 불러옵니다.")
    try:
        df = pd.read_csv(INPUT_CSV_FILENAME)
    except FileNotFoundError:
        logger.error(f"'{INPUT_CSV_FILENAME}' 파일을 찾을 수 없습니다.")
        logger.error("스크립트 상단의 파일 이름이 정확한지, 파일이 스크립트와 같은 폴더에 있는지 확인하세요.")
        return
        
    golden_items = df.to_dict('records')

    app = build_graph()
    
    logger.info("RAG 파이프라인의 워크플로우를 .png 파일로 저장합니다...")
    try:
        graph_image_path = "2_run_rag_answers.png"
        with open(graph_image_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        logger.info(f"LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        logger.error(f"그래프 시각화 중 오류 발생: {e}")
        logger.warning("시각화를 위해서는 playwright가 필요합니다. (pip install playwright && playwright install)")

    logger.info(f"{len(golden_items)}개 질문에 대해 RAG 파이프라인을 비동기로 실행합니다...")
    tasks = [app.ainvoke({"question": item["question"]}) for item in golden_items]
    results = await asyncio.gather(*tasks)
    
    df['answer'] = [res.get('answer', '답변 생성 실패') for res in results]
    df['contexts'] = [str(res.get('final_contexts', [])) for res in results]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"2_rag_answers_{timestamp}.csv"
    
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    logger.info(f"RAG 답변 생성을 완료하여 '{output_filename}' 파일로 저장했습니다.")

if __name__ == "__main__":
    asyncio.run(main())