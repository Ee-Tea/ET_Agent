# 2_run_rag_answers.py (v1.0: 초안 작성)
# 2_run_rag_answers.py (v1.1: 프롬프트 수정)
# 2_run_rag_answers.py (v1.2: Milvus 연결 오류 처리 추가)
# 2_run_rag_answers.py (v1.3: 웹 검색 실패 처리 추가)
# 2_run_rag_answers.py (v1.4: JSON 파일에서 추가 컨텍스트 로드 기능 추가)
# 2_run_rag_answers.py (v1.5: JSON 컨텍스트 통합 및 프롬프트 수정)
# 2_run_rag_answers.py (v1.6: JSON 파일 경로 설정 환경 변수화)
# 2_run_rag_answers.py (v1.7: JSON 컨텍스트가 없을 때 처리 추가)
# 2_run_rag_answers.py (v1.8: JSON 파일 추가, 프롬프트 수정)
# 2_run_rag_answers.py (v1.9: JSON 컨텍스트가 없을 때 처리 수정)
# 2_run_rag_answers.py (v2.0: 완료) 


import os  # 운영 체제와 상호 작용하기 위한 모듈 임포트
import re  # 정규 표현식 작업을 위한 모듈 임포트
import sys  # 파이썬 인터프리터와 상호 작용하기 위한 모듈 임포트
import logging  # 로깅 기능을 위한 모듈 임포트
import asyncio  # 비동기 I/O 작업을 위한 모듈 임포트
import json  # 추가: JSON 파일 처리를 위한 라이브러리 임포트
from datetime import datetime  # 날짜 및 시간 처리를 위한 클래스 임포트
from typing import List, Dict, Any, Optional, TypedDict  # 타입 힌팅을 위한 클래스 임포트
import pandas as pd  # 데이터 조작 및 분석을 위한 pandas 라이브러리 임포트
from dotenv import load_dotenv, find_dotenv  # .env 파일에서 환경 변수를 로드하기 위한 함수 임포트
import torch  # PyTorch 라이브러리 임포트 (GPU 사용 확인용)
from tavily import TavilyClient  # Tavily 웹 검색 API 클라이언트 임포트
from langchain_community.cache import InMemoryCache  # LangChain의 인메모리 캐시 임포트
from langchain_core.prompts import ChatPromptTemplate  # LangChain의 채팅 프롬프트 템플릿 임포트
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace 임베딩 모델을 위한 LangChain 래퍼 임포트
from langchain_milvus import Milvus as MilvusVectorStore  # Milvus 벡터 저장소를 위한 LangChain 래퍼 임포트
from langchain_openai import ChatOpenAI  # OpenAI 채팅 모델을 위한 LangChain 래퍼 임포트
from langgraph.graph import StateGraph, END  # LangGraph에서 상태 그래프와 종료 지점을 임포트
from pymilvus import connections  # Milvus 데이터베이스 연결을 위한 모듈 임포트
from sentence_transformers import CrossEncoder  # 재순위화를 위한 CrossEncoder 모델 임포트
import langchain  # LangChain 라이브러리 메인 모듈 임포트

# --- [윈도우 이벤트 루프 정책: 중요] ---
if sys.platform.startswith("win"):  # 현재 운영 체제가 윈도우인지 확인
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # 윈도우용 비동기 이벤트 루프 정책 설정

# ==================== 설정: 입력 파일 이름 ====================
INPUT_CSV_FILENAME = "1_golden_set_20250912_151447.csv"  # 질문이 포함된 입력 CSV 파일의 이름
# 추가: JSON 데이터 파일 경로
# JSON 파일의 실제 경로를 여기에 입력하세요.
JSON_DATA_PATH = r"C:\Users\user\Documents\GitHub\ET_Agent\farmer\recommend\data\1_golden_set_20250912_155358.json"  # 추가 컨텍스트로 사용할 JSON 파일의 경로

# ==========================================================

# ==================== 설정 (공통) ====================
logging.basicConfig(  # 로깅 기본 설정 구성
    level=logging.INFO,  # 로그 레벨을 INFO로 설정
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',  # 로그 메시지 형식 지정
    datefmt='%Y-%m-%d %H:%M:%S'  # 날짜 및 시간 형식 지정
)
logger = logging.getLogger("run_rag_answers")  # "run_rag_answers"라는 이름의 로거 객체 생성
langchain.llm_cache = InMemoryCache()  # LangChain의 LLM 응답을 메모리에 캐시하도록 설정
load_dotenv(find_dotenv())  # .env 파일을 찾아 환경 변수를 로드

# 환경 변수
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")  # Milvus 서버 주소 환경 변수 로드 (기본값 설정)
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")  # Milvus 인증 토큰 환경 변수 로드 (기본값 설정)
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "crop_info")  # Milvus 컬렉션 이름 환경 변수 로드 (기본값 설정)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")  # 임베딩 모델 이름 환경 변수 로드 (기본값 설정)
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "upskyy/ko-reranker")  # 재순위화 모델 이름 환경 변수 로드 (기본값 설정)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API 키 환경 변수 로드

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # OpenAI 모델 이름 환경 변수 로드 (기본값 설정)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # Tavily API 키 환경 변수 로드 (기본값 설정)
MIN_DB_CONTEXT_CHARS = int(os.getenv("MIN_DB_CONTEXT_CHARS", "350"))  # 웹 검색을 트리거할 최소 DB 컨텍스트 문자 수
TOPK_RETRIEVE = int(os.getenv("TOPK_RETRIEVE", "30"))  # Milvus에서 검색할 초기 문서 수
TOPK_USE = int(os.getenv("TOPK_USE", "5"))  # 재순위화 후 최종적으로 사용할 문서 수

if not OPENAI_API_KEY:  # OpenAI API 키가 없는 경우
    raise ValueError("OPENAI_API_KEY가 .env에 필요합니다.")  # 에러 발생 및 스크립트 중지

# 전역 객체
_vectorstore = None  # 전역 Milvus 벡터 저장소 객체를 None으로 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"  # CUDA 사용 가능 여부에 따라 장치(device) 설정
logger.info(f"모델을 위한 장치로 '{device}'를 사용합니다.")  # 설정된 장치 정보 로그 기록
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={"device": device})  # 지정된 모델과 장치로 임베딩 모델 초기화
reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)  # 지정된 모델과 장치로 재순위화 모델(CrossEncoder) 초기화
llm_answer = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.7, api_key=OPENAI_API_KEY)  # 지정된 모델과 온도로 OpenAI LLM 초기화

# 프롬프트 (재구성)
# 프롬프트에 [추가 참고 문서] 섹션 추가
RAG_PROMPT_TMPL = """
당신은 대한민국 농업 작물 재배에 대해 친절하게 상담해 드리는 전문가입니다.
아래 제공된 [DB 검색 결과], [웹 검색 결과], 그리고 [추가 참고 문서]의 사실만을 근거로 [질문]에 맞는 작물을 정성껏 추천해 주세요.

[DB 검색 결과]
{db_context}

[웹 검색 결과]
{web_context}

[추가 참고 문서]
{json_context}

[질문]
{question}

---
규칙 (반드시 엄수):

1) **출처 제한**
- 답변은 반드시 [DB 검색 결과], [웹 검색 결과], [추가 참고 문서]에 포함된 사실 전부 사용하세요.
- 컨텍스트에 없는 정보, 일반 지식, 추측은 절대 포함하지 마세요.

2) **답변 형식**
- 질문이 작물 추천을 요구하면, 답변은 **"~을/를 추천드립니다."** 또는 **"~는 ~에 적합합니다."**와 같은 존댓말 문장으로 자연스럽게 연결하며, 추천 이유를 간결하게 설명하세요.
- 여러 작물을 추천할 경우, 한 문장에 콤마(,)나 '및', '그리고' 등을 사용해 자연스럽게 나열할 수 있습니다.
- 답변은 서론과 결론을 포함하지 않고, 추천 내용을 바로 제시하세요.
- 정보가 여러 문서에 나뉘어 있더라도, 하나의 일관된 답변으로 통합하여 제시하세요."
- 작물추천은 1~5개 이내로 제한하세요.

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

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL)  # 문자열 템플릿으로부터 채팅 프롬프트 객체 생성

# ==================== RAG 파이프라인 (LangGraph) ====================
# GraphState에 json_context_str 필드 추가
class GraphState(TypedDict, total=False):  # LangGraph의 상태를 관리하는 TypedDict 클래스 정의
    question: Optional[str]  # 사용자의 질문
    db_context_str: Optional[str]  # DB에서 검색된 컨텍스트 문자열
    web_context_str: Optional[str]  # 웹에서 검색된 컨텍스트 문자열
    json_context_str: Optional[str]  # JSON 파일에서 로드된 컨텍스트 문자열
    answer: Optional[str]  # 생성된 최종 답변
    retrieved_docs: Optional[List[str]]  # DB에서 검색된 문서 목록
    web_search_docs: Optional[List[str]]  # 웹에서 검색된 문서 목록
    final_contexts: Optional[List[str]]  # 최종적으로 사용될 컨텍스트 목록
    final_sources: Optional[List[str]]  # 최종 컨텍스트의 출처 목록
    db_sources: Optional[List[str]]  # DB 컨텍스트의 출처 목록
    web_sources: Optional[List[str]]  # 웹 컨텍스트의 출처 목록

def ensure_milvus():  # Milvus 연결을 확인하고 설정하는 함수
    """
    메인 스레드에서 1회 연결 후 alias="default" 재사용.
    """
    global _vectorstore  # 전역 변수 _vectorstore를 사용하도록 선언
    if _vectorstore:  # 벡터 저장소 객체가 이미 존재하면
        return  # 함수를 종료
    try:  # 예외 처리를 위한 try 블록 시작
        try:  # 중첩된 try 블록 시작
            connections.get_connection(alias="default")  # 'default' 별칭의 연결이 있는지 확인
        except Exception:  # 연결이 없을 경우 예외 발생
            connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)  # 'default' 별칭으로 Milvus에 연결

        _vectorstore = MilvusVectorStore(  # MilvusVectorStore 객체 생성
            embedding_function=embedding_model,  # 임베딩 함수로 지정된 모델 사용
            collection_name=MILVUS_COLLECTION,  # 지정된 컬렉션 이름 사용
            connection_args={  # 연결 인자 설정
                "alias": "default",  # 연결 별칭
                "uri": MILVUS_URI,  # Milvus 서버 주소
                "token": MILVUS_TOKEN,  # Milvus 인증 토큰
            },
        )
        logger.info("Milvus 벡터스토어 초기화 완료.")  # 초기화 완료 로그 기록
    except Exception as e:  # Milvus 연결 중 예외 발생 시
        logger.error(f"Milvus 연결 실패: {e}")  # 에러 로그 기록
        raise  # 예외를 다시 발생시켜 프로그램 중지

def retrieve_from_milvus(query: str, top_k: int):  # Milvus에서 문서를 검색하는 함수
    ensure_milvus()  # Milvus 연결 보장
    return _vectorstore.similarity_search_with_score(query, k=top_k)  # 유사도 검색을 수행하고 결과 반환

def rerank_documents(query: str, pairs: List[tuple]) -> List[tuple]:  # 검색된 문서를 재순위화하는 함수
    if not pairs:  # 문서 쌍(pairs)이 비어있으면
        return []  # 빈 리스트 반환
    sentence_pairs = [(query, doc.page_content) for doc, _ in pairs]  # (질문, 문서 내용) 형태의 쌍 생성
    scores = reranker.predict(sentence_pairs)  # CrossEncoder 모델로 재순위화 점수 예측
    return [p for _, p in sorted(zip(scores, pairs), key=lambda x: x[0], reverse=True)]  # 점수가 높은 순으로 정렬하여 반환

def retrieve_node(state: GraphState) -> Dict[str, Any]:  # DB 검색 및 재순위화를 수행하는 그래프 노드
    q = state.get("question", "") or ""  # 상태에서 질문을 가져옴
    # 1) 검색
    pairs = retrieve_from_milvus(q, top_k=TOPK_RETRIEVE)  # Milvus에서 TOPK_RETRIEVE 개수만큼 문서 검색
    # 2) 너무 짧은 청크 제외
    pairs = [(doc, score) for doc, score in pairs if len(doc.page_content.strip()) > 100]  # 내용이 100자 이상인 문서만 필터링
    # 3) 재순위
    reranked_pairs = rerank_documents(q, pairs)  # 검색된 문서를 재순위화
    final = reranked_pairs[:TOPK_USE]  # 상위 TOPK_USE개의 문서를 최종 선택

    contents = [doc.page_content for doc, _ in final]  # 최종 선택된 문서의 내용을 리스트로 저장
    db_sources = [  # 최종 선택된 문서의 출처를 리스트로 저장
        (doc.metadata.get("source")  # 메타데이터에서 'source' 키 값 가져오기
         or doc.metadata.get("file_path")  # 없으면 'file_path' 키 값 가져오기
         or "unknown_pdf")  # 둘 다 없으면 'unknown_pdf'로 설정
        for doc, _ in final
    ]

    labeled = [{"label": f"C{i+1}", "text": c} for i, c in enumerate(contents)]  # 문서에 C1, C2... 레이블 붙이기
    ctx_str = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled \
        else "내부 DB에서 관련 정보를 찾지 못했습니다."  # 레이블링된 텍스트를 하나의 문자열로 결합
    return {  # 노드의 결과로 상태 업데이트 값을 반환
        "db_context_str": ctx_str,  # DB 컨텍스트 문자열
        "retrieved_docs": contents,  # 검색된 문서 내용
        "db_sources": db_sources,  # DB 출처
        "final_sources": db_sources[:]  # 최종 출처 (초기값으로 DB 출처 복사)
    }

def web_search_node(state: GraphState) -> Dict[str, Any]:  # 웹 검색을 수행하는 그래프 노드
    if not TAVILY_API_KEY:  # Tavily API 키가 없으면
        return {"web_context_str": "웹 검색 비활성화", "web_search_docs": [], "web_sources": []}  # 웹 검색 비활성화 상태 반환
    q = state.get("question", "") or ""  # 상태에서 질문을 가져옴
    try:  # 예외 처리 블록 시작
        client = TavilyClient(api_key=TAVILY_API_KEY)  # Tavily 클라이언트 초기화
        res = client.search(query=q, max_results=10, search_depth="advanced")  # 고급 웹 검색 수행
        results = res.get("results", []) or []  # 검색 결과를 가져옴

        docs = []  # 웹 문서 내용을 저장할 리스트
        web_sources = []  # 웹 문서 출처(URL)를 저장할 리스트
        for i, r in enumerate(results):  # 각 검색 결과에 대해 반복
            content = (r.get("content") or "").strip()  # 결과에서 내용(content)을 추출하고 공백 제거
            if not content:  # 내용이 없으면
                continue  # 다음 결과로 넘어감
            docs.append(content)  # 문서 내용 리스트에 추가
            web_sources.append(r.get("url", "web_search"))  # 문서 출처 리스트에 추가

        labeled = [{"label": f"W{i+1}", "text": c} for i, c in enumerate(docs)]  # 웹 문서에 W1, W2... 레이블 붙이기
        web_ctx = "\n\n".join([f"[{d['label']}] {d['text']}" for d in labeled]) if labeled \
            else "웹에서 관련 정보를 찾지 못했습니다."  # 레이블링된 텍스트를 하나의 문자열로 결합

        return {  # 노드의 결과로 상태 업데이트 값을 반환
            "web_context_str": web_ctx,  # 웹 컨텍스트 문자열
            "web_search_docs": docs,  # 웹 검색 문서 내용
            "web_sources": web_sources  # 웹 출처
        }
    except Exception as e:  # 웹 검색 중 예외 발생 시
        logger.error(f"웹 검색 실패: {e}")  # 에러 로그 기록
        return {"web_context_str": "웹 검색 중 오류 발생", "web_search_docs": [], "web_sources": []}  # 오류 상태 반환

def combine_context_node(state: GraphState) -> Dict[str, Any]:  # DB와 웹 컨텍스트를 결합하는 노드
    db_docs = state.get("retrieved_docs", []) or []  # 상태에서 DB 문서 목록을 가져옴
    web_docs = state.get("web_search_docs", []) or []  # 상태에서 웹 문서 목록을 가져옴

    db_sources = state.get("db_sources", []) or []  # 상태에서 DB 출처 목록을 가져옴
    web_sources = state.get("web_sources", []) or []  # 상태에서 웹 출처 목록을 가져옴

    final_contexts = db_docs + web_docs  # DB 문서와 웹 문서를 결합하여 최종 컨텍스트 생성
    final_sources = db_sources + web_sources  # DB 출처와 웹 출처를 결합하여 최종 출처 생성

    return {  # 노드의 결과로 상태 업데이트 값을 반환
        "final_contexts": final_contexts,  # 최종 컨텍스트
        "final_sources": final_sources,  # 최종 출처
        "db_sources": db_sources,  # DB 출처 (그대로 유지)
        "web_sources": web_sources  # 웹 출처 (그대로 유지)
    }

# generate_draft_node를 수정하여 JSON 컨텍스트를 프롬프트에 전달
def generate_draft_node(state: GraphState) -> Dict[str, Any]:  # 최종 답변 초안을 생성하는 노드
    msgs = rag_prompt.format_messages(  # 프롬프트 템플릿에 동적으로 값을 채워 메시지 생성
        question=state.get("question", "") or "",  # 질문
        db_context=state.get("db_context_str", "정보 없음"),  # DB 컨텍스트
        web_context=state.get("web_context_str", "정보 없음"),  # 웹 컨텍스트
        json_context=state.get("json_context_str", "정보 없음")  # 추가: JSON 컨텍스트
    )
    resp = llm_answer.invoke(msgs)  # 생성된 메시지를 LLM에 전달하여 답변 생성
    return {"answer": (resp.content or "").strip()}  # 생성된 답변의 내용만 추출하여 반환

def route_after_retrieve(state: "GraphState") -> str:  # 검색 후 다음 단계를 결정하는 라우팅 함수
    return "web_search" if len(state.get("db_context_str", "").strip()) < MIN_DB_CONTEXT_CHARS else "combine_context"  # DB 컨텍스트가 충분하지 않으면 'web_search', 충분하면 'combine_context'로 분기

def build_graph():  # LangGraph 워크플로우를 구성하는 함수
    g = StateGraph(GraphState)  # GraphState를 사용하는 상태 그래프 객체 생성
    nodes = [  # 그래프에 추가할 노드 목록 정의
        ("retrieve", retrieve_node),  # 검색 노드
        ("web_search", web_search_node),  # 웹 검색 노드
        ("combine_context", combine_context_node),  # 컨텍스트 결합 노드
        ("generate_draft", generate_draft_node),  # 답변 생성 노드
    ]
    for name, node in nodes:  # 각 노드에 대해 반복
        g.add_node(name, node)  # 그래프에 노드 추가
    g.set_entry_point("retrieve")  # 그래프의 시작점을 'retrieve' 노드로 설정
    g.add_conditional_edges(  # 조건부 엣지(분기) 추가
        "retrieve", route_after_retrieve,  # 'retrieve' 노드 이후 'route_after_retrieve' 함수의 결과에 따라 분기
        {"web_search": "web_search", "combine_context": "combine_context"}  # 결과값에 따른 목적지 노드 매핑
    )
    g.add_edge("web_search", "combine_context")  # 'web_search' 노드에서 'combine_context' 노드로 엣지 추가
    g.add_edge("combine_context", "generate_draft")  # 'combine_context' 노드에서 'generate_draft' 노드로 엣지 추가
    g.add_edge("generate_draft", END)  # 'generate_draft' 노드에서 그래프의 종료 지점(END)으로 엣지 추가
    return g.compile()  # 구성된 그래프를 컴파일하여 실행 가능한 객체로 반환

def remove_markdown_and_special_chars(text: str) -> str:  # 마크다운 및 특수 문자를 제거하는 함수
    text = re.sub(r'#{1,6}\s', '', text)  # 마크다운 제목(#) 제거
    text = re.sub(r'[\*\-]', '', text)  # 마크다운 강조(*, -) 제거
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # 마크다운 링크 제거
    text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나의 공백으로 축소
    return text.strip()  # 앞뒤 공백 제거 후 반환

# 추가: JSON 데이터 로드 함수
def load_json_data(file_path: str) -> Optional[str]:  # 지정된 경로의 JSON 파일을 읽어 문자열로 반환하는 함수
    """
    지정된 JSON 파일에서 데이터를 읽어 문자열로 변환합니다.
    파일을 찾을 수 없거나 읽기 오류가 발생하면 None을 반환합니다.
    """
    if not os.path.exists(file_path):  # 파일 경로가 존재하지 않으면
        logger.warning(f"JSON 파일 '{file_path}'을 찾을 수 없습니다. 이 소스는 답변에 포함되지 않습니다.")  # 경고 로그 기록
        return None  # None 반환
    
    try:  # 파일 읽기 중 발생할 수 있는 예외 처리
        with open(file_path, 'r', encoding='utf-8') as f:  # 파일을 UTF-8 인코딩으로 열기
            data = json.load(f)  # JSON 파일의 내용을 파싱하여 파이썬 객체로 변환
            # JSON 데이터 구조에 따라 이 부분은 수정이 필요할 수 있습니다.
            # 예시: 딕셔너리 리스트에서 'text' 키의 값들을 합치는 경우
            if isinstance(data, list):  # 데이터가 리스트 형태인 경우
                # 텍스트가 여러 객체에 나뉘어 있을 경우 하나로 합칩니다.
                return "\n\n".join(item.get("text", "") for item in data if isinstance(item, dict))  # 각 딕셔셔너리의 'text' 값을 합쳐 반환
            else:  # 데이터가 리스트가 아닌 경우
                # 단일 객체일 경우 JSON 문자열로 변환합니다.
                return json.dumps(data, ensure_ascii=False, indent=2)  # JSON 객체를 보기 좋게 포맷팅된 문자열로 변환하여 반환
    except Exception as e:  # 파일 읽기 또는 파싱 중 예외 발생 시
        logger.error(f"JSON 파일 '{file_path}' 읽기 실패: {e}")  # 에러 로그 기록
        return None  # None 반환


async def main():  # 메인 비동기 실행 함수
    logger.info(f"'{INPUT_CSV_FILENAME}' 파일에서 골든셋을 불러옵니다.")  # CSV 파일 로딩 시작 로그
    try:  # 파일 로딩 중 예외 처리
        df = pd.read_csv(INPUT_CSV_FILENAME)  # 입력 CSV 파일을 pandas DataFrame으로 로드
    except FileNotFoundError:  # 파일을 찾을 수 없을 때
        logger.error(f"'{INPUT_CSV_FILENAME}' 파일을 찾을 수 없습니다.")  # 에러 로그 기록
        logger.error("스크립트 상단의 파일 이름이 정확한지, 파일이 스크립트와 같은 폴더에 있는지 확인하세요.")  # 사용자 안내 메시지
        return  # 함수 실행 종료

    # 추가: JSON 데이터 미리 로드
    json_data_content = load_json_data(JSON_DATA_PATH)  # 지정된 경로의 JSON 데이터 로드
    if json_data_content is None:  # JSON 데이터 로드에 실패하면
        json_data_content = "추가 참고 문서 없음."  # 기본값 설정
    
    golden_items = df.to_dict('records')  # DataFrame을 딕셔너리 리스트로 변환
    app = build_graph()  # LangGraph 애플리케이션(워크플로우) 생성

    logger.info("RAG 파이프라인의 워크플로우를 .png 파일로 저장합니다...")  # 그래프 시각화 시작 로그
    try:  # 그래프 이미지 생성 중 예외 처리
        graph_image_path = "2_run_rag_answers.png"  # 저장할 이미지 파일 이름
        with open(graph_image_path, "wb") as f:  # 파일을 바이너리 쓰기 모드로 열기
            f.write(app.get_graph().draw_mermaid_png())  # 그래프를 Mermaid 다이어그램 PNG로 그려서 파일에 저장
        logger.info(f"LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")  # 저장 완료 로그
    except Exception as e:  # 이미지 생성 중 예외 발생 시
        logger.error(f"그래프 시각화 중 오류 발생: {e}")  # 에러 로그 기록
        logger.warning("시각화를 위해서는 playwright가 필요합니다. (pip install playwright && playwright install)")  # 해결 방법 안내

    logger.info(f"{len(golden_items)}개 질문에 대해 RAG 파이프라인을 비동기로 실행합니다...")  # RAG 파이프라인 실행 시작 로그
    # 각 질문에 JSON 컨텍스트를 포함하여 Graph에 전달
    tasks = [app.ainvoke({  # 비동기 실행을 위한 작업 목록 생성
        "question": item["question"],  # 각 항목의 질문
        "json_context_str": json_data_content  # 미리 로드한 JSON 컨텍스트
    }) for item in golden_items]  # 모든 질문 항목에 대해 반복
    results = await asyncio.gather(*tasks)  # 모든 작업을 병렬로 실행하고 결과를 기다림

    df['answer'] = [res.get('answer', '답변 생성 실패') for res in results]  # 결과에서 'answer'를 추출하여 DataFrame에 새 컬럼으로 추가
    df['contexts'] = [str(res.get('final_contexts', [])) for res in results]  # 결과에서 'final_contexts'를 추출하여 DataFrame에 새 컬럼으로 추가
    
    # 상세 출처 컬럼 추가
    # df['db_sources'] = [str(res.get('db_sources', [])) for res in results]
    # df['web_sources'] = [str(res.get('web_sources', [])) for res in results]
    # df['final_sources'] = [str(res.get('final_sources', [])) for res in results]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 시간을 이용해 타임스탬프 문자열 생성
    output_filename = f"2_rag_answers_{timestamp}.csv"  # 출력 파일 이름에 타임스탬프 포함

    df.to_csv(output_filename, index=False, encoding='utf-8-sig')  # DataFrame을 CSV 파일로 저장 (UTF-8-SIG 인코딩 사용)
    logger.info(f"RAG 답변 생성을 완료하여 '{output_filename}' 파일로 저장했습니다.")  # 최종 저장 완료 로그

if __name__ == "__main__":  # 이 스크립트가 직접 실행될 때만 아래 코드 블록 실행
    asyncio.run(main())  # 메인 비동기 함수 실행