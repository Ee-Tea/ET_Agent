# 2_run_rag_answers.py (v1.10: 들여쓰기 오류 수정 및 상세 출처 반영)

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
INPUT_CSV_FILENAME = "1_golden_set_20250911_101128.csv"  #<-- 사용할 골든셋 CSV 파일 이름을 여기서 수정하세요.
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
# reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)
llm_answer = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.7, api_key=OPENAI_API_KEY)

# 프롬프트 (재구성)
RAG_PROMPT_TMPL = """
당신은 대한민국 농업 작물 재배에 대해 친절하게 상담해 드리는 전문가입니다. 
아래 제공된 [DB 검색 결과]와 [웹 검색 결과]의 사실만을 근거로 [질문]에 맞는 작물을 정성껏 추천해 주세요.

[DB 검색 결과]
{db_context}

[웹 검색 결과]
{web_context}

[질문]
{question}

---
규칙 (반드시 엄수):

1) **출처 제한**
- 답변은 반드시 [DB 검색 결과]와 [웹 검색 결과]에 포함된 사실만 사용하세요.
- 컨텍스트에 없는 정보, 일반 지식, 추측은 절대 포함하지 마세요.

2) **질문 집중**
- 답변은 [질문]의 의도를 정확하게 충족해야 합니다.
- 불필요한 서론, 결론, 잡설은 절대 포함하지 마세요.

3) **작물 추천 형식**
- 질문이 추천을 요구하면, 각 문장은 반드시 “<작물명>을/를 추천드립니다.”로 시작하고,
  이어서 [컨텍스트]에서 확인된 이유를 한 문장으로 설명하세요.
- 예: 포도를 추천드립니다. 포도는 다양한 품종과 용도로 재배할 수 있어 농가 소득 증대에 기여할 수 있습니다.
- 예: 사과를 추천드립니다. 사과는 국내 소비가 꾸준하고, 가공품 수요도 높아 안정적인 판매가 가능합니다.
- 예: 배추를 추천드립니다. 배추는 고랭지 지역에서 재배가 가능하고, 김장철 수요로 인해 가격이 상승하는 경향이 있습니다.

4) **작물명 정규화**
- 품종명, 숫자코드, 외래어,영어,일본어 표기는 모두 제거하고, 일반적인 작물명만 사용하세요.
- 예: 캠벨얼리 → 포도, 홍로 → 사과, 101-14 → 포도

5) **중복 제거**
- 같은 작물이 여러 번 언급되면 한 번만 답변에 포함하세요.

6) **형식 및 톤**
- 전체 답변은 5~8 문장으로 작성하세요.
- 불릿, 번호, 마크다운, 일본어, 영어, 기호는 절대 사용하지 마세요.
- 모든 문장은 한국어 맞춤법에 맞는 자연스러운 존댓말로 작성하세요.

"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL)

# ==================== RAG 파이프라인 (LangGraph) ====================
class GraphState(TypedDict, total=False):
    question: Optional[str]
    db_context_str: Optional[str]
    web_context_str: Optional[str]
    answer: Optional[str]
    retrieved_docs: Optional[List[str]]
    web_search_docs: Optional[List[str]]
    final_contexts: Optional[List[str]]
    final_sources: Optional[List[str]]

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

# def rerank_documents(query: str, pairs: List[tuple]) -> List[tuple]:
#     if not pairs: return []
#     sentence_pairs = [(query, doc.page_content) for doc, score in pairs]
#     scores = reranker.predict(sentence_pairs)
#     scored_pairs = sorted(list(zip(scores, pairs)), key=lambda x: x[0], reverse=True)
#     return [pair for score, pair in scored_pairs]

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    q = state.get("question", "")
    pairs = retrieve_from_milvus(q, top_k=TOPK_RETRIEVE)
    pairs = [(doc, score) for doc, score in pairs if len(doc.page_content.strip()) > 100]
    # reranked_pairs = rerank_documents(q, pairs)
    # final = reranked_pairs[:TOPK_USE]
    final = pairs[:TOPK_USE]  # 재순위 없이 바로 상위 K개 사용
    contents = [doc.page_content for doc, _ in final]
    # 문서 메타데이터에서 원본 파일명 추출
    sources = [doc.metadata.get('source', 'unknown_pdf') for doc, _ in final]
    
    labeled = [{"label": f"C{i+1}", "text": doc_content} for i, doc_content in enumerate(contents)]
    ctx_str = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled else "내부 DB에서 관련 정보를 찾지 못했습니다."
    return {"db_context_str": ctx_str, "retrieved_docs": contents, "final_sources": sources}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        return {"web_context_str": "웹 검색 비활성화", "web_search_docs": [], "web_sources": []}
    q = state.get("question", "")
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        res = client.search(query=q, max_results=10, search_depth="advanced")
        docs = [r["content"].strip() for r in res.get("results", []) if r.get("content")]
        labeled = [{"label": f"W{i+1}", "text": content} for i, content in enumerate(docs)]
        web_ctx = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled else "웹에서 관련 정보를 찾지 못했습니다."
        # 웹 검색은 'web_search'로 출처 통일
        sources = ['web_search'] * len(docs)
        return {"web_context_str": web_ctx, "web_search_docs": docs, "web_sources": sources}
    except Exception as e:
        logger.error(f"웹 검색 실패: {e}")
        return {"web_context_str": "웹 검색 중 오류 발생", "web_search_docs": []}

def combine_context_node(state: GraphState) -> Dict[str, Any]:
    db_docs = state.get("retrieved_docs", [])
    web_docs = state.get("web_search_docs", [])
    
    db_sources = state.get("final_sources", [])
    web_sources = state.get("web_sources", [])
    
    # 두 소스를 합치고, 웹 검색 노드에서 온 소스를 추가합니다.
    final_sources = db_sources + web_sources
    final_contexts = db_docs + web_docs
    
    return {"final_contexts": final_contexts, "final_sources": final_sources}

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