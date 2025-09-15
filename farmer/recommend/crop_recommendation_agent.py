
# =========[ 표준/외부 라이브러리 ]=========
import os
import re
import argparse
from typing import List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv

# =========[ LangChain / LangGraph / LLM ]=========
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# =========[ 외부 서비스 ]=========
from tavily import TavilyClient

# =========[ 공통 모듈 ]=========
from common.milvus_helpers import search_milvus_documents, create_context_from_documents

# =========[ 지연 로딩을 위한 전역 변수 ]=========
_crop_recommend_app = None
_llm_instance = None
_tavily_client = None
_embedding_model = None
_vectorstore = None

# ==================== 환경 변수 로드 ====================
load_dotenv() # .env 파일을 찾아 환경 변수들을 로드합니다.

# ==================== 환경 설정 ====================
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530") # 환경 변수에서 Milvus DB URI를 가져옵니다. 기본값 설정.
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus") # 환경 변수에서 Milvus 접속 토큰을 가져옵니다. 기본값 설정.
MILVUS_COLLECTION = "crop_info" # 사용할 Milvus 컬렉션의 이름입니다.
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask") # 환경 변수에서 임베딩 모델 이름을 가져옵니다.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # OpenAI API 키를 환경 변수에서 가져옵니다.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini") # 사용할 OpenAI 모델 이름입니다.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6")) # LLM의 창의성을 제어하는 온도를 설정합니다.

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # Tavily API 키를 환경 변수에서 가져옵니다.

if not OPENAI_API_KEY: # OpenAI API 키가 설정되지 않았다면
    raise ValueError("OPENAI_API_KEY가 .env에 설정되어야 합니다.") # 오류를 발생시킵니다.


# =========[ 지연 로딩 함수들 ]=========
def _get_llm():
    """LLM 인스턴스를 지연 로딩으로 가져오기"""
    global _llm_instance
    if _llm_instance is None:
        print("🤖 LLM 모듈 로딩 중...")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 .env에 설정되어야 합니다.")
        _llm_instance = ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)
        print("✅ LLM 모듈 로딩 완료")
    return _llm_instance

def _get_tavily_client():
    """Tavily 클라이언트를 지연 로딩으로 가져오기"""
    global _tavily_client
    if _tavily_client is None:
        print("🔍 Tavily 클라이언트 로딩 중...")
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        print("✅ Tavily 클라이언트 로딩 완료")
    return _tavily_client

def _get_embedding_model():
    """임베딩 모델을 지연 로딩으로 가져오기"""
    global _embedding_model
    if _embedding_model is None:
        print("🧠 임베딩 모델 로딩 중...")
        from langchain_huggingface import HuggingFaceEmbeddings
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"}
        )
        print("✅ 임베딩 모델 로딩 완료")
    return _embedding_model

def _get_vectorstore():
    """벡터스토어를 지연 로딩으로 가져오기"""
    global _vectorstore
    if _vectorstore is None:
        print("🗄️ 벡터스토어 로딩 중...")
        from langchain_milvus import Milvus as MilvusVectorStore
        from pymilvus import connections
        
        # Milvus 연결이 없으면 새로 연결을 시도합니다.
        if "default" not in connections.list_connections() or not connections.has_connection("default"):
            connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)
        
        embedding_model = _get_embedding_model()
        _vectorstore = MilvusVectorStore(
            embedding_function=embedding_model,
            collection_name=MILVUS_COLLECTION,
            connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN},
        )
        print(f"✅ 벡터스토어 로딩 완료: {MILVUS_COLLECTION}")
    return _vectorstore


# 통합 프롬프트
UNIFIED_PROMPT_TEMPLATE = """
당신은 농업 작물 추천 전문가입니다.
다음은 검색된 정보입니다. 이 정보를 활용하여 사용자의 질문에 답변해 주세요.

# 검색 결과:
{db_context}

답변을 생성하는데 참고해야할 규칙은 다음과 같습니다.
1. 전문성 및 친절함: 농작물 전문가처럼 친절하고 명확한 문체로 작성해 주세요.
2. 추천 작물 제한: 추천 작물은 적합한 수준(1~5개 이내)으로 제한하세요.
3. 작물 설명 체계화: 각 작물의 재배 특성·환경 조건·장점을 학문적으로 설명하세요.
4. 선택 이유 명시: 추천 작물의 선택 이유와 장점을 명확하게 설명하세요.
5. 재배 정보 체계화: 재배 조건, 시기, 관리 방법은 단계별로 정리하되, 논리적이고 객관적인 표현을 사용하세요.
6. 재배 방법 포맷팅: 재배 방법은 각 단계마다 한 줄씩 띄워서 작성하세요.
7. 언어 사용: 반드시 한국어 존댓말을 사용하며, 설명은 명료하고 체계적으로 하세요.
8. 용어 설명: 필요할 경우 농학적 용어를 포함하되, 일반인도 이해할 수 있도록 쉽게 풀어서 설명하세요.
9. 답변 길이: 답변은 6~10문장 이내로 작성하세요.
10. 정보 활용: 제공된 검색 결과를 종합하여 답변하세요.
11. 핵심 요약 및 정리: 여러 검색 결과에서 중복되는 핵심 내용들을 종합하여 간결하게 요약해 주세요.
12. 작물 추천 시 번호 매기기: 추천하는 작물은 1., 2., 3.... 순서로 번호를 매겨주세요.
13. **절대 마크다운 형식을 사용하지 마세요. #, ##, ###, **, *, -, ` 등의 기호를 사용하지 마세요.**
14. **평문으로만 작성하세요. 제목이나 헤더는 사용하지 마세요.**

질문: {question}
답변:
"""

# 검증 프롬프트
VALIDATION_PROMPT = """
주어진 맥락을 바탕으로 다음 질문에 대한 유용한 답변을 제공할 수 있는지 여부를 '네' 또는 '아니오'로만 답변하세요.

기준:
- 질문과 관련된 정보가 맥락에 포함되어 있으면 '네'
- 완전하지 않더라도 부분적인 답변을 제공할 수 있으면 '네'
- 전혀 관련이 없거나 아무 정보가 없을 때만 '아니오'

질문: {question}
맥락: {db_context}
답변:
"""
def _get_unified_prompt():
    """통합 프롬프트를 지연 로딩으로 가져오기"""
    return ChatPromptTemplate.from_template(UNIFIED_PROMPT_TEMPLATE)

def _get_validation_prompt():
    """검증 프롬프트를 지연 로딩으로 가져오기"""
    return ChatPromptTemplate.from_template(VALIDATION_PROMPT)


class GraphState(TypedDict, total=False):  # LangGraph의 상태를 정의하는 딕셔너리 타입입니다.
    question: Optional[str]  # 사용자의 질문을 저장하는 필드입니다.
    answer: Optional[str]  # 최종 답변을 저장하는 필드입니다.
    db_context: Optional[str]  # DB 검색 결과를 저장하는 필드입니다.
    web_sources: Optional[List[Dict[str, Any]]]  # 웹 검색 소스 정보를 저장하는 필드입니다.
    db_sources: Optional[List[Dict[str, Any]]]  # DB 검색 소스 정보를 저장하는 필드입니다.
    context: Optional[str]  # 결합된 최종 컨텍스트를 저장하는 필드입니다.
    is_sufficient: Optional[str]  # DB 검색 결과 충분성 여부를 저장하는 필드입니다.
    milvus_data: Optional[Dict[str, Any]]  # MilvusDB 연결 정보를 저장하는 필드입니다.
    milvus_context: Optional[str]  # 기존 Milvus 컨텍스트를 저장하는 필드입니다.

# ==================== 라우팅 유틸 ====================




# ==================== 노드들 ====================
def process_topics_and_retrieve_content_node(state: GraphState) -> Dict[str, Any]:
    """DB 검색 및 충분성 검증 노드"""
    print("\n---노드: DB 검색 실행---")
    question = state["question"]
    milvus_data = state.get("milvus_data", {})

    # MilvusDB에서 검색
    if not milvus_data.get("connection_status", False):
        print("⚠️ MilvusDB 연결 안됨 - 빈 컨텍스트 반환")
        return {**state, "db_context": "", "is_sufficient": "no"}

    try:
        # MilvusDB에서 작물 정보 검색
        enhanced_query = f"{question} 재배 방법 키우기 팁"
        documents = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name=MILVUS_COLLECTION,
            query=enhanced_query,
            k=5
        )
        
        if not documents:
            print("⚠️ MilvusDB에서 관련 문서를 찾지 못함")
            return {**state, "db_context": "", "is_sufficient": "no"}
        
        # 기존 Milvus 컨텍스트가 있으면 추가
        existing_milvus_context = state.get("milvus_context", "")
        if existing_milvus_context:
            db_context = f"{existing_milvus_context}\n\n[MilvusDB 검색 결과]\n{create_context_from_documents(documents, max_length=2000)}"
            print("✅ 기존 Milvus 컨텍스트와 결합")
        else:
            db_context = create_context_from_documents(documents, max_length=2000)
        
        print(f"✅ MilvusDB 검색 완료: {len(documents)}개 문서")
        
        # DB 내용 충분성 검증
        is_sufficient = "no"
        if db_context.strip():
            print("🔍 DB 내용 충분성 검증 중...")
            llm = _get_llm()
            validation_prompt = _get_validation_prompt()
            response = llm.invoke(validation_prompt.format(question=question, db_context=db_context))
            validation_result = response.content.strip()
            is_sufficient = "yes" if "네" in validation_result else "no"
            print(f"✅ DB 충분성 검증 결과: {is_sufficient}")
        
        # 소스 정보 생성
        db_sources = []
        for doc in documents:
            db_sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "content": doc.page_content
            })
        
        return {**state, "db_context": db_context, "is_sufficient": is_sufficient, "db_sources": db_sources}
        
    except Exception as e:
        print(f"❌ MilvusDB 검색 실패: {e}")
        return {**state, "db_context": "", "is_sufficient": "no"}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    """웹 검색 노드"""
    print("--- 노드: 웹 검색 ---")
    question = state.get("question")
    if not question:
        raise ValueError("질문이 누락되었습니다.")
    
    if not TAVILY_API_KEY:
        print("⚠️ TAVILY API 키가 없어 웹 검색을 건너뜁니다.")
        return {**state, "web_sources": []}
    
    try:
        search_tool = _get_tavily_client()
        response = search_tool.search(query=question, max_results=5)
        
        web_sources = []
        if isinstance(response, dict) and "results" in response:
            results = response["results"]
            for r in results:
                if isinstance(r, dict):
                    web_sources.append({
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "title": r.get("title", "")
                    })
        
        print(f"✅ 웹 검색 완료: {len(web_sources)}개 결과")
        return {**state, "web_sources": web_sources}
        
    except Exception as e:
        print(f"❌ 웹 검색 실패: {e}")
        return {**state, "web_sources": []}

def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """DB 전용 답변 생성 노드"""
    print("--- 노드: DB 전용 답변 생성 ---")
    question = state.get("question", "")
    db_context = state.get("db_context", "")
    
    if not db_context:
        print("❌ DB 컨텍스트가 없어 답변을 생성할 수 없습니다.")
        return {**state, "answer": "주어진 정보로는 답변할 수 없습니다."}

    llm = _get_llm()
    db_prompt = _get_db_only_prompt()
    response = llm.invoke(db_prompt.format(db_context=db_context, question=question))
    answer = response.content
    
    print("✅ DB 전용 답변 생성 완료.")
    return {**state, "answer": answer}

def combine_context_node(state: GraphState) -> Dict[str, Any]:
    """컨텍스트를 결합하는 노드 함수"""
    print("--- 노드: 컨텍스트 결합 ---")
    db_context = state.get("db_context", "")
    web_sources = state.get("web_sources", [])
    
    final_context = db_context
    
    if web_sources:
        print("✅ DB와 웹 컨텍스트를 결합합니다.")
        # 웹 검색 결과를 텍스트로 변환
        web_search_results = []
        for source in web_sources:
            web_search_results.append(f"제목: {source.get('title', 'N/A')}\nURL: {source.get('url', 'N/A')}\n내용: {source.get('content', 'N/A')}")
        
        web_context = "\n\n".join(web_search_results)
        final_context = f"[DB 검색 결과]\n{db_context}\n\n[웹 검색 결과]\n{web_context}"
    else:
        print("ℹ️ DB 컨텍스트만 사용합니다.")
    
    return {**state, "context": final_context}

def generate_final_answer_node(state: GraphState) -> Dict[str, Any]:
    """최종 답변 생성 노드"""
    print("--- 노드: 최종 답변 생성 ---")
    question = state.get("question", "")
    context = state.get("context", "")
    
    if not context:
        print("❌ 컨텍스트가 없어 답변을 생성할 수 없습니다.")
        return {**state, "answer": "주어진 정보로는 답변할 수 없습니다."}

    llm = _get_llm()
    unified_prompt = _get_unified_prompt()
    response = llm.invoke(unified_prompt.format(db_context=context, question=question))
    answer = response.content
    
    print("✅ 최종 답변 생성 완료.")
    return {**state, "answer": answer}



# 중복된 함수 정의 제거됨 - 위의 함수들이 사용됨

# ==================== 그래프 빌드 ====================
def build_graph(): # LangGraph 워크플로우를 구축하는 함수입니다.
    g = StateGraph(GraphState)
    
    # 노드 추가
    g.add_node("process_topics_and_retrieve_content", process_topics_and_retrieve_content_node)
    g.add_node("web_search", web_search_node)
    g.add_node("combine_context", combine_context_node)
    g.add_node("generate_final_answer", generate_final_answer_node)

    # 시작점 설정
    g.set_entry_point("process_topics_and_retrieve_content")

    # 조건부 분기
    g.add_conditional_edges(
        "process_topics_and_retrieve_content",
        lambda state: state.get("is_sufficient"),
        {
            "yes": "combine_context",
            "no": "web_search"
        }
    )
    
    # 웹 검색 후 컨텍스트 결합
    g.add_edge("web_search", "combine_context")
    
    # 컨텍스트 결합 후 최종 답변 생성
    g.add_edge("combine_context", "generate_final_answer")
    
    # 최종 답변 생성 후 종료
    g.add_edge("generate_final_answer", END)

    return g.compile()

def _get_crop_recommend_app():
    """작물추천 에이전트 애플리케이션을 지연 로딩으로 가져오기"""
    global _crop_recommend_app
    if _crop_recommend_app is None:
        print("🌾 작물추천_agent 모듈 로딩 중...")
        _crop_recommend_app = build_graph()
        print("✅ 작물추천_agent 모듈 로딩 완료")
    return _crop_recommend_app

def run(state: dict) -> dict:  # 에이전트를 실행하는 함수입니다.
    try:
        query = state.get("query", "")  # 상태에서 쿼리를 가져옵니다.
        milvus_data = state.get("milvus_data", {})  # MilvusDB 연결 정보를 가져옵니다.
        milvus_context = state.get("milvus_context", "")  # 기존 Milvus 컨텍스트를 가져옵니다.
        
        if not query:  # 쿼리가 없으면
            return {"agent_answer": "질문이 제공되지 않았습니다. 작물추천 관련 질문을 해주세요."}  # 오류 메시지를 반환합니다.
        print(f"[작물추천_agent] 질문 처리 시작: {query}")  # 질문 처리 시작을 알립니다.
        print(f"[작물추천_agent] MilvusDB 연결: {'연결됨' if milvus_data.get('connection_status') else '연결 안됨'}")

        # 그래프를 지연 로딩으로 가져오기
        app = _get_crop_recommend_app()
        final_state = app.invoke({
            "question": query,
            "milvus_data": milvus_data,
            "milvus_context": milvus_context
        })  # LangGraph 애플리케이션을 호출하여 그래프를 실행합니다.

        if isinstance(final_state, dict):
            answer = final_state.get("answer", "답변 생성에 실패했습니다.")
        elif isinstance(final_state, str):
            answer = final_state
        else:
            answer = "답변 형식이 올바르지 않습니다."

        print(f"[작물추천_agent] 답변 생성 완료: {len(answer)}자")
        return {"agent_answer": answer}
    except Exception as e:
        error_msg = f"작물추천 에이전트 실행 중 오류가 발생했습니다: {e}"
        print(f"[작물추천_agent] 오류: {e}")
        return {"agent_answer": error_msg}

# ==================== 출력 유틸 ====================
def remove_markdown_and_special_chars(text: str) -> str: # 마크다운 및 특수문자를 제거하는 함수입니다.
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE) # 헤더(#) 제거 - 공백 없이도 제거
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE) # 앞에 공백이 있는 헤더도 제거
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE) # 불릿 포인트 제거 (- *)
    text = re.sub(r'[\*\-]', '', text) # 불릿(*, -) 제거
    text = re.sub(r'\[.*?\]\(.*?\)', '', text) # 링크 제거
    text = re.sub(r'\s+', ' ', text) # 여러 공백을 하나로 축소
    return text.strip() # 양 끝 공백 제거


# ==================== main ====================
if __name__ == "__main__":  # 스크립트가 직접 실행될 때의 진입점입니다.
    parser = argparse.ArgumentParser(description="RAG 파이프라인 - 농업 작물 추천 시스템 (DB 우선 + 조건부 웹검색)")
    parser.add_argument("-q", "--query", type=str, help="한 번만 실행할 질문 (예: -q '주말농장에 키울 작물 추천해줘')")
    args = parser.parse_args()
    
    agent_app = _get_crop_recommend_app()  # 그래프를 지연 로딩으로 빌드합니다.

    # 그래프 시각화 (선택 사항)
    # try:
    #     graph_image_path = "agent_workflow.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(agent_app.get_graph().draw_mermaid_png())
    #     print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류 발생: {e}")
    
    # ==================== 터미널 전용 ====================
    if args.query:  # 명령줄 인자로 질문이 주어진 경우
        print(f"\n🧐 질문: '{args.query}'")
        print("-" * 20)
        final_state = agent_app.invoke({"question": args.query})
        answer = final_state.get("answer", "답변 생성에 실패했습니다.")
        cleaned_answer = remove_markdown_and_special_chars(answer)
        print("\n🌾 최종 답변 🌾")
        print("=" * 20)
        print(cleaned_answer)
        print("=" * 20)
    else:  # 대화형 모드
        print("(종료: exit/quit)")
        while True:
            q = input("\n질문: ").strip() # 기본 version
            # q = input("\n👨‍🌾이봐, 젊은이! 무엇을 도와줄까?🌱 :  ").strip() #이장님 version
            if not q or q.lower() in ("exit", "quit"):
                break
            final_state = agent_app.invoke({"question": q})
            answer = final_state.get("answer", "답변 생성에 실패했습니다.")
            cleaned_answer = remove_markdown_and_special_chars(answer)
            print("\n🍀 최종 답변 🍀")
            print("=" * 20)
            print(cleaned_answer)
            print("=" * 20)
        print("👋 프로그램을 종료합니다.")
