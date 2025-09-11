import os
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict

# Langchain 및 LangGraph 관련 라이브러리
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus as LangChainMilvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from common.milvus_helpers import search_milvus_documents, create_context_from_documents

# --- 1. 환경 설정 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    exit()
if not TAVILY_API_KEY:
    print("오류: TAVILY_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    exit()

# Milvus 연결 정보 및 컬렉션 이름 정의
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME_INFO = "crop_info"
COLLECTION_NAME_GROW = "crop_grow"

# --- 2. LLM 및 프롬프트 설정 ---
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7
)

# ❗ DB 정보만 사용할 때의 답변 프롬프트
DB_ONLY_PROMPT_TEMPLATE = """
당신은 농작물 재배 및 관리에 대한 전문 지식을 갖춘 친절하고 정확한 정보를 제공하는 챗봇입니다.
다음은 내부 전문가 자료(DB)에서 검색된 내용입니다. 이 정보를 활용하여 사용자의 질문에 답변해 주세요.

# DB 검색 결과 (내부 전문가 자료):
{db_context}

답변을 생성하는데 참고해야할 규칙은 다음과 같습니다.
1. 전문성 및 친절함: 농작물 전문가처럼 친절하고 명확한 문체로 작성해 주세요. 전문 용어는 이해하기 쉽게 풀어 설명하고, 사용자가 직접 실행할 수 있는 실질적인 조언을 포함해 주세요.
2. 정보의 출처 명시: 답변은 반드시 제공된 DB에 제시된 정보만을 사용해야 합니다. 제공된 정보에 없는 정보, 저의 상식, 추측, 거짓 정보 등은 절대 답변에 넣지 마세요. 만약 질문에 대한 답변이 검색 결과에 없다면, '검색 결과에 해당 정보가 없습니다.'라고 명확하게 말해야 합니다.
3. 핵심 요약 및 정리: 여러 검색 결과에서 중복되는 핵심 내용들을 종합하여 간결하게 요약해 주세요.
4. 구체적이고 상세하게: 답변은 가능한 한 구체적인 정보(예: 날짜, 숫자, 기관명, 재배 단계 등)를 포함하여 작성해 주세요.
5. 한글로만 답변: 모든 답변은 한글로만 제공해야 합니다.

질문: {question}
답변:
"""
db_only_prompt = ChatPromptTemplate.from_template(DB_ONLY_PROMPT_TEMPLATE)

# ❗ 웹 검색 정보만 사용할 때의 답변 프롬프트 
WEB_ONLY_PROMPT_TEMPLATE = """
당신은 농작물 재배 및 관리에 대한 전문 지식을 갖춘 친절하고 정확한 정보를 제공하는 챗봇입니다.
다음은 웹 검색을 통해 얻은 정보입니다. 이 정보를 활용하여 사용자의 질문에 답변해 주세요.

# 웹 검색 결과:
{web_search_results}

답변을 생성하는데 참고해야할 규칙은 다음과 같습니다.
1. 전문성 및 친절함: 농작물 전문가처럼 친절하고 명확한 문체로 작성해 주세요.
2. 정보의 출처 명시: 답변은 반드시 제공된 웹 검색 결과에 제시된 정보만을 사용해야 합니다.
3. 핵심 요약 및 정리: 여러 검색 결과에서 중복되는 핵심 내용들을 종합하여 간결하게 요약해 주세요.
4. 구체적이고 상세하게: 답변은 가능한 한 구체적인 정보(예: 날짜, 숫자, 기관명, 재배 단계 등)를 포함하여 작성해 주세요.
5. 한글로만 답변: 모든 답변은 한글로만 제공해야 합니다.

질문: {question}
답변:
"""
web_only_prompt = ChatPromptTemplate.from_template(WEB_ONLY_PROMPT_TEMPLATE)

VALIDATION_PROMPT = """
주어진 맥락만 사용하여 다음 질문에 대한 완전하고 상세한 답변을 생성할 수 있는지 여부를 '네' 또는 '아니오'로만 답변하세요.
질문: {question}
맥락: {db_context}
답변:
"""
validation_prompt = ChatPromptTemplate.from_template(VALIDATION_PROMPT)

tavily_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)

# --- 3. LangGraph 상태 정의 ---
class GraphState(TypedDict):
    question: Optional[str]
    answer: Optional[str]
    topics: Optional[List[str]]
    db_context: Optional[str]
    web_sources: Optional[List[Dict[str, Any]]]
    db_sources: Optional[List[Dict[str, Any]]]
    is_sufficient: Optional[str]
    milvus_data: Optional[Dict[str, Any]]
    milvus_context: Optional[str]

# --- 4. MilvusDB 검색 함수 (common.milvus_helpers 사용) ---
def retrieve_top_k_from_collections(question: str, milvus_data: Dict[str, Any], k: int = 3) -> Dict[str, Any]:
    """MilvusDB에서 작물 정보와 재배 정보 검색"""
    
    if not milvus_data.get("connection_status", False):
        print("⚠️ MilvusDB 연결 안됨 - 빈 컨텍스트 반환")
        return {"context": "", "db_sources": []}
    
    try:
        # 작물 정보 컬렉션에서 검색
        crop_info_docs = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name=COLLECTION_NAME_INFO,
            query=question,
            k=k
        )
        
        # 작물 재배 정보 컬렉션에서 검색
        crop_grow_docs = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name=COLLECTION_NAME_GROW,
            query=question,
            k=k
        )
        
        # 두 컬렉션 결과 합치기
        all_docs = crop_info_docs + crop_grow_docs
        
        # 컨텍스트 생성
        context = create_context_from_documents(all_docs, max_length=2000)
        
        # 소스 정보 생성
        db_sources = []
        for doc in all_docs:
            db_sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "content": doc.page_content
            })
        
        print(f"✅ MilvusDB 검색 완료: {len(all_docs)}개 문서")
        return {"context": context, "db_sources": db_sources}
        
    except Exception as e:
        print(f"❌ MilvusDB 검색 실패: {e}")
        return {"context": "", "db_sources": []}

# --- 5. LangGraph 노드 함수 정의 ---
def process_topics_and_retrieve_content_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: DB 검색 실행---")
    question = state["question"]
    milvus_data = state.get("milvus_data", {})

    # MilvusDB에서 검색
    retrieval_result = retrieve_top_k_from_collections(question, milvus_data, k=3)
    db_context = retrieval_result["context"]
    db_sources = retrieval_result["db_sources"]
    print("✅ DB 검색 완료.")

    # 기존 Milvus 컨텍스트가 있으면 추가
    existing_milvus_context = state.get("milvus_context", "")
    if existing_milvus_context and db_context:
        db_context = f"{existing_milvus_context}\n\n{db_context}"
        print("✅ 기존 Milvus 컨텍스트와 결합")

    is_sufficient = "no"
    if db_context.strip():
        print("🔍 DB 내용 충분성 검증 중...")
        validation_chain = validation_prompt | llm | StrOutputParser()
        validation_result = validation_chain.invoke({"question": question, "db_context": db_context})
        is_sufficient = "yes" if "네" in validation_result.strip() else "no"
        print(f"❗ DB 내용만으로 답변 가능 여부: {is_sufficient}")
        
    return {**state, "db_context": db_context, "db_sources": db_sources, "is_sufficient": is_sufficient}

def retrieve_from_web_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 웹 검색 실행---")
    question = state["question"]

    print("🌐 웹 검색으로 답변 생성 중...")
    search_results = tavily_tool.invoke({"query": question})
    web_sources = [{"url": res["url"], "content": res["content"]} for res in search_results]
    print("✅ 웹 검색 완료.")
    
    return {**state, "web_sources": web_sources}

def generate_final_answer_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 최종 답변 생성 실행---")
    question = state["question"]
    is_sufficient = state.get("is_sufficient")

    web_sources = state.get("web_sources", [])
    web_search_results = "웹 검색 결과가 없습니다." if not web_sources else "\n".join([str(res) for res in web_sources])

    if is_sufficient == "yes":
        db_context = state.get("db_context", "내부 DB에서 검색된 정보가 없습니다.")
        final_chain = db_only_prompt | llm | StrOutputParser()
        inputs = {"question": question, "db_context": db_context}
    else:
        final_chain = web_only_prompt | llm | StrOutputParser()
        inputs = {"question": question, "web_search_results": web_search_results}

    answer = final_chain.invoke(inputs)
    return {**state, "answer": answer}

def remove_markdown_and_special_chars(text: str) -> str:
    text = re.sub(r'#{1,6}\s', '', text)
    text = re.sub(r'[\*\-]', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    return text.strip()

# --- 6. LangGraph 워크플로우 빌드 ---
def build_query_graph():
    query_builder = StateGraph(GraphState)

    query_builder.add_node("db_retrieval", process_topics_and_retrieve_content_node)
    query_builder.add_node("web_search", retrieve_from_web_node)
    query_builder.add_node("generate_answer", generate_final_answer_node)

    query_builder.set_entry_point("db_retrieval")

    query_builder.add_conditional_edges(
        "db_retrieval",
        lambda state: state.get("is_sufficient"),
        {
            "yes": "generate_answer",
            "no": "web_search"
        }
    )

    query_builder.add_edge("web_search", "generate_answer")
    query_builder.add_edge("generate_answer", END)

    return query_builder.compile()

# --- 7. OchestratorTest.py와 호환되는 run 함수 ---
def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    OchestratorTest.py에서 호출되는 메인 실행 함수 (비동기)
    
    Args:
        state: OchestratorTest.py에서 전달받은 상태 딕셔너리
               - query: 사용자 질문 (필수)
               - milvus_data: MilvusDB 연결 정보 (선택)
               - milvus_context: 기존 Milvus 컨텍스트 (선택)
               - 기타 필요한 상태 정보들
    
    Returns:
        dict: 실행 결과
            - agent_answer: 최종 응답
            - status: 실행 상태
            - error: 오류 정보 (있는 경우)
    """
    try:
        # 입력 검증
        if not state or not state.get("query"):
            return {
                "agent_answer": "질문이 제공되지 않았습니다.",
                "status": "error",
                "error": "query 필드가 없습니다."
            }
        
        query = state["query"]
        milvus_data = state.get("milvus_data", {})
        milvus_context = state.get("milvus_context", "")
        
        print(f"\n=== 🌱 작물재배_agent 실행 시작 ===")
        print(f"질문: {query}")
        print(f"MilvusDB 연결: {'연결됨' if milvus_data.get('connection_status') else '연결 안됨'}")
        
        # RAG 애플리케이션 빌드
        rag_app = build_query_graph()
        
        # LangGraph가 자동으로 재시도 및 fallback 처리
        print("답변 생성 중...")
        try:
            current_state = {
                "question": query, 
                "milvus_data": milvus_data,
                "milvus_context": milvus_context,
                "web_search_count": 0
            }
            final_state = rag_app.invoke(current_state)
            
            # 최종 답변 추출
            final_response = final_state.get('answer', '답변을 생성할 수 없습니다.')
            print("✅ 답변 생성 완료")
            
            return {
                "agent_answer": final_response,
                "status": "success",
                "error": None
            }
                        
        except Exception as e:
            print(f"❌ 답변 생성 중 오류: {e}")
            return {
                "agent_answer": f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {e}",
                "status": "error",
                "error": str(e)
            }
            
    except Exception as e:
        print(f"❌ 작물재배_agent 실행 중 치명적 오류: {e}")
        return {
            "agent_answer": f"작물재배_agent 실행 중 오류가 발생했습니다: {e}",
            "status": "error",
            "error": str(e)
        }

# --- 8. 메인 실행 로직 ---
if __name__ == "__main__":
    print("🌱 농작물 챗봇 에이전트 시작...")
    print("--------------------------------------------------")

    rag_app = build_query_graph()

    print("이제 질문을 입력하세요. (종료하려면 'exit' 또는 'quit' 입력)")
    print("--------------------------------------------------")

    while True:
        prompt = input("질문을 입력하세요: ")
        if prompt.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break

        print("답변을 생성하는 중...")
        try:
            final_state = rag_app.invoke({"question": prompt})
            response = final_state.get('answer', "죄송합니다. 답변을 생성하지 못했습니다.")
            
            db_sources = final_state.get('db_sources', [])
            web_sources = final_state.get('web_sources', [])
            
            cleaned_response = remove_markdown_and_special_chars(response)

            print("\n------------------- 답변 -------------------")
            print(cleaned_response)
            print("-------------------------------------------\n")

            if db_sources:
                print("--- 참고한 DB 내용 ---")
                for i, source in enumerate(db_sources, 1):
                    file_name = os.path.basename(source.get('source', '')).rsplit('.', 1)[0]
                    page_num = source.get('page')
                    print(f"**[{i}]** 출처: {file_name}", end="")
                    if page_num is not None:
                        print(f", 페이지: {page_num + 1}", end="")
                    print(f"\n내용: {source.get('content', '내용 없음')[:100]}...\n")
            
            if web_sources:
                print("--- 참고한 웹 검색 결과 ---")
                for i, source in enumerate(web_sources, 1):
                    print(f"**[{i}]** URL: {source.get('url', 'URL 없음')}")
                    print(f"내용: {source.get('content', '내용 없음')[:100]}...\n")

            print("-------------------------------------------\n")

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")