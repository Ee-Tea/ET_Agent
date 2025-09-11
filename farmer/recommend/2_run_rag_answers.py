# 2_run_rag_answers.py (v1.11: 이벤트 루프/밀부스 연결 안정화, 재순위 버그 수정, 상세 출처 컬럼 추가)

import os
import re
import sys
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

# --- [윈도우 이벤트 루프 정책: 중요] ---
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ==================== 설정: 입력 파일 이름 ====================
INPUT_CSV_FILENAME = "1_golden_set_20250911_180809.csv"
# ==========================================================

# ==================== 설정 (공통) ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("run_rag_answers")
langchain.llm_cache = InMemoryCache()
load_dotenv(find_dotenv())

# 환경 변수
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "crop_info")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")
# RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L6-v2") # 영어 모델
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "upskyy/ko-reranker") #한국 데이터 재학습 모델
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MIN_DB_CONTEXT_CHARS = int(os.getenv("MIN_DB_CONTEXT_CHARS", "250"))
TOPK_RETRIEVE = int(os.getenv("TOPK_RETRIEVE", "15"))
TOPK_USE = int(os.getenv("TOPK_USE", "5"))

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env에 필요합니다.")

# 전역 객체
_vectorstore = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"모델을 위한 장치로 '{device}'를 사용합니다.")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={"device": device})
reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)
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

2) **답변 형식**
- 질문이 작물 추천을 요구하면, 답변은 **"~을/를 추천드립니다."** 또는 **"~는 ~에 적합합니다."**와 같은 존댓말 문장으로 자연스럽게 연결하며, 추천 이유를 간결하게 설명하세요.
- 여러 작물을 추천할 경우, 한 문장에 콤마(,)나 '및', '그리고' 등을 사용해 자연스럽게 나열할 수 있습니다.
- 답변은 서론과 결론을 포함하지 않고, 추천 내용을 바로 제시하세요.

3) **예시 (이런 톤과 형식으로 답변하세요)**
- 여름철 텃밭 재배에는 오이, 토마토, 고추, 상추 등을 추천드립니다. 오이는 더위에 강하고 수확량이 풍부하며, 토마토는 햇빛을 많이 받아야 잘 자라 다양한 요리에 활용될 수 있습니다. 고추는 여름철 고온다습한 환경에 잘 적응하고, 상추는 생육이 빨라 텃밭에서 쉽게 기를 수 있습니다. 이 작물들은 여름철 기후에 잘 적응하며 재배가 용이합니다.

4) **작물명 정규화**
- 품종명, 숫자코드, 외래어,영어,일본어 표기는 모두 제거하고, 일반적인 작물명만 사용하세요.

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
    db_sources: Optional[List[str]]
    web_sources: Optional[List[str]]

def ensure_milvus():
    """
    메인 스레드에서 1회 연결 후 alias="default" 재사용.
    """
    global _vectorstore
    if _vectorstore:
        return
    try:
        # 기존 연결 재사용 시도
        try:
            connections.get_connection(alias="default")
        except Exception:
            connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)

        _vectorstore = MilvusVectorStore(
            embedding_function=embedding_model,
            collection_name=MILVUS_COLLECTION,
            connection_args={
                "alias": "default",      # 이미 연결된 alias 사용
                "uri": MILVUS_URI,       # (명시 유지 - 안전)
                "token": MILVUS_TOKEN,
            },
        )
        logger.info("Milvus 벡터스토어 초기화 완료.")
    except Exception as e:
        logger.error(f"Milvus 연결 실패: {e}")
        raise

def retrieve_from_milvus(query: str, top_k: int):
    ensure_milvus()
    return _vectorstore.similarity_search_with_score(query, k=top_k)

def rerank_documents(query: str, pairs: List[tuple]) -> List[tuple]:
    if not pairs:
        return []
    # pairs: List[(Document, score)]
    sentence_pairs = [(query, doc.page_content) for doc, _ in pairs]
    scores = reranker.predict(sentence_pairs)
    # zip(scores, pairs) -> (score, (doc, score_from_vectorstore))
    return [p for _, p in sorted(zip(scores, pairs), key=lambda x: x[0], reverse=True)]

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    q = state.get("question", "") or ""
    # 1) 검색
    pairs = retrieve_from_milvus(q, top_k=TOPK_RETRIEVE)
    # 2) 너무 짧은 청크 제외
    pairs = [(doc, score) for doc, score in pairs if len(doc.page_content.strip()) > 100]
    # 3) 재순위
    reranked_pairs = rerank_documents(q, pairs)
    final = reranked_pairs[:TOPK_USE]

    contents = [doc.page_content for doc, _ in final]
    # 파일/페이지 등 원본 출처 (가능하면 'source' 또는 'file_path' 메타 키 사용)
    db_sources = [
        (doc.metadata.get("source")
         or doc.metadata.get("file_path")
         or "unknown_pdf")
        for doc, _ in final
    ]

    labeled = [{"label": f"C{i+1}", "text": c} for i, c in enumerate(contents)]
    ctx_str = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled \
              else "내부 DB에서 관련 정보를 찾지 못했습니다."
    return {
        "db_context_str": ctx_str,
        "retrieved_docs": contents,
        "db_sources": db_sources,
        "final_sources": db_sources[:]  # 초기에는 DB 출처만
    }

def web_search_node(state: GraphState) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        return {"web_context_str": "웹 검색 비활성화", "web_search_docs": [], "web_sources": []}
    q = state.get("question", "") or ""
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        res = client.search(query=q, max_results=10, search_depth="advanced")
        results = res.get("results", []) or []

        docs = []
        web_sources = []
        for i, r in enumerate(results):
            content = (r.get("content") or "").strip()
            if not content:
                continue
            docs.append(content)
            # URL을 상세 출처로 저장
            web_sources.append(r.get("url", "web_search"))

        labeled = [{"label": f"W{i+1}", "text": c} for i, c in enumerate(docs)]
        web_ctx = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled \
                  else "웹에서 관련 정보를 찾지 못했습니다."

        return {
            "web_context_str": web_ctx,
            "web_search_docs": docs,
            "web_sources": web_sources
        }
    except Exception as e:
        logger.error(f"웹 검색 실패: {e}")
        return {"web_context_str": "웹 검색 중 오류 발생", "web_search_docs": [], "web_sources": []}

def combine_context_node(state: GraphState) -> Dict[str, Any]:
    db_docs = state.get("retrieved_docs", []) or []
    web_docs = state.get("web_search_docs", []) or []

    db_sources = state.get("db_sources", []) or []
    web_sources = state.get("web_sources", []) or []

    final_contexts = db_docs + web_docs
    final_sources = db_sources + web_sources

    return {
        "final_contexts": final_contexts,
        "final_sources": final_sources,
        "db_sources": db_sources,
        "web_sources": web_sources
    }

def generate_draft_node(state: GraphState) -> Dict[str, Any]:
    msgs = rag_prompt.format_messages(
        question=state.get("question", "") or "",
        db_context=state.get("db_context_str", "정보 없음"),
        web_context=state.get("web_context_str", "정보 없음")
    )
    resp = llm_answer.invoke(msgs)
    return {"answer": (resp.content or "").strip()}

def route_after_retrieve(state: "GraphState") -> str:
    return "web_search" if len(state.get("db_context_str", "").strip()) < MIN_DB_CONTEXT_CHARS else "combine_context"

def build_graph():
    g = StateGraph(GraphState)
    nodes = [
        ("retrieve", retrieve_node),
        ("web_search", web_search_node),
        ("combine_context", combine_context_node),
        ("generate_draft", generate_draft_node),
    ]
    for name, node in nodes:
        g.add_node(name, node)
    g.set_entry_point("retrieve")
    g.add_conditional_edges(
        "retrieve", route_after_retrieve,
        {"web_search": "web_search", "combine_context": "combine_context"}
    )
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

    # 상세 출처 컬럼 추가
    # df['db_sources'] = [str(res.get('db_sources', [])) for res in results]
    # df['web_sources'] = [str(res.get('web_sources', [])) for res in results]
    # df['final_sources'] = [str(res.get('final_sources', [])) for res in results]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"2_rag_answers_{timestamp}.csv"

    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    logger.info(f"RAG 답변 생성을 완료하여 '{output_filename}' 파일로 저장했습니다.")

if __name__ == "__main__":
    asyncio.run(main())
