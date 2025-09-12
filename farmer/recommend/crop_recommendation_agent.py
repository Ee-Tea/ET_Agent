# crop _recommendation_agent_optimized.py (v1.1: generate_draft=최종답변, 조건부 웹검색 고정)
# crop_recommendation_agent_optimized.py (v1.2: query 값 추가)
# crop_recommendation_agent_optimized.py (v1.3: route_after_retrieve 개선)
# crop_recommendation_agent_optimized.py (v1.4: generate_draft_node 개선, graph 시각화 주석처리)
# crop_recommendation_agent_optimized.py (v1.5: prompt 개선)


# 이 스크립트는 RAG 파이프라인을 구축하여 사용자 질문에 대한 최적의 답변을 생성합니다.

# ==================== 라이브러리 불러오기 ====================
import os # 파일 경로, 환경 변수 등 운영체제 상호작용 모듈
import re # 정규표현식 모듈
import argparse # 명령줄 인자를 파싱하는 모듈
from typing import List, Dict, Any, Optional, TypedDict # 타입 힌트 모듈

from dotenv import load_dotenv, find_dotenv # .env 파일에서 환경 변수를 로드
from langchain_core.prompts import ChatPromptTemplate # LLM 프롬프트 템플릿 생성 모듈
from langchain_huggingface import HuggingFaceEmbeddings # Hugging Face 임베딩 모델 사용 모듈
from langchain_milvus import Milvus as MilvusVectorStore # Milvus 벡터 데이터베이스와 연동
from tavily import TavilyClient # Tavily 웹 검색 API 사용 모듈
from langchain_openai import ChatOpenAI # OpenAI LLM 모델 사용
from langgraph.graph import StateGraph, END # LangGraph 워크플로우를 정의하는 핵심 모듈
from pymilvus import connections # Milvus DB 연결 관리 모듈

# ==================== 환경 변수 로드 ====================
load_dotenv(find_dotenv()) # .env 파일을 찾아 환경 변수들을 로드합니다.

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

MIN_DB_CONTEXT_CHARS = int(os.getenv("MIN_DB_CONTEXT_CHARS", "1000")) # DB 컨텍스트의 최소 글자 수를 설정합니다.

# ==================== 전역 변수 및 초기화 ====================
_vectorstore = None # Milvus 벡터스토어 객체를 저장할 전역 변수
embedding_model = HuggingFaceEmbeddings( # Hugging Face 임베딩 모델을 초기화합니다.
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"} # 모델을 CPU에서 실행하도록 설정합니다.
)
llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY) # OpenAI LLM을 초기화합니다.
agent_app = None # LangGraph 앱 객체를 저장할 전역 변수

# ==================== 프롬프트 (구수 version) ====================
# RAG_PROMPT_TMPL = """ 
# 그려, 나는 평생을 흙이랑 같이 살아온 농사꾼이네. 👨‍🌾  
# 내가 살아온 경험하고, 아래 '문맥'을 참고해서 자네가 물어본 데 맞는 작물을 골라주고 재배법도 알려주겠네.  

# [문맥]  
# {context}  

# 규칙:  
# - 추천 작물은 여러 개 나오면 좋고, 많으면 5~7가지까지 알려주게. 🌾🍠🍅🥬  
# - 작물 하나하나에 대해서, 왜 좋은지 이유를 덧붙여 주게.  
# - 재배 시기, 흙 관리, 물 주는 법 같은 건 내가 자식 가르치듯 단계별로 알려주면 되네.  
# - 따뜻하고 정겨운 어르신 말투로 해주게.  
# - 단계별 안내가 필요하면 줄바꿈을 하지말고 일관된 답변으로 이야기 하게.
# - 중간중간에 🌱😊 같은 이모티콘도 섞어서 정겹게 해주면 더 좋네.  
# - '문맥'에 없으면 웹 검색해서 보태서 알려주게.  
# - 답변은 6~10문장 정도로 풀어주되, 너무 딱딱하지 않게 풀어주게.
# - 반복되는 말은 빼주게. (예: 겨울철에 키우기 좋은 작물로는 겨울철에도 잘 자라는)

# 질문: {question}  
# 답변:  
# """
#---------------------------------------- RAG 프롬프트 ----------------------------------------

RAG_PROMPT_TMPL = """ 
당신은 농업 분야의 박사로서, 깊은 전문 지식을 바탕으로 질문에 답변하는 전문가입니다. 🎓  
아래 '문맥'을 참고하여 사용자의 질문에 과학적 근거와 연구 기반 지식을 종합해 설명해주세요.  

[문맥]  
{context}  

규칙:  
- 추천 작물은 최대 7개까지 제시하되, 각 작물의 재배 특성·환경 조건·장점을 학문적으로 설명하세요.  
- 재배 조건, 시기, 관리 방법은 단계별로 정리하되, 논리적이고 객관적인 표현을 사용하세요.  
- 반드시 한국어 존댓말을 사용하며, 설명은 명료하고 체계적으로 하세요.  
- 필요할 경우 농학적 용어를 포함하되, 일반인도 이해할 수 있도록 쉽게 풀어서 설명하세요.  
- 답변은 6~10문장 이내로 작성하세요.  
- '문맥'에 없는 내용은 웹 검색 결과로 보완하여 활용하세요.  

🟢 질문: {question}  
✨ 답변:  
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL) # RAG 프롬프트 템플릿 객체를 생성합니다.


#---------------------------------------- 프롬프트 (구수 version) ----------------------------------------
# WEB_PROMPT_TMPL = """ 
# 나는 오랫동안 농사 지어온 어르신이네 👨‍🌾  
# 아래 '웹 검색 결과'를 살펴보고, 자네 질문에 맞는 답을 정성껏 풀어주겠네.  

# [웹 검색 결과]  
# {web_context_parts}  

# 규칙:  
# - 반드시 검색 결과를 토대로 답변해야 하네.  
# - 검색된 사실들을 잘 요약하고, 자네 질문에 맞게 구체적으로 풀어주게.  
# - 따뜻하고 정겨운 말투를 쓰게 😊  
# - 핵심 위주로 6~10문장 정도로 이야기하세.  
# - 질문이 농업과 관련 있다면 작물 추천과 재배 방법까지 꼭 포함하게. 🌱  
# - 추천 작물이 많으면 최대 7개까지 알려주되 🍅🥒🌽, 각각 이유와 재배 팁을 곁들이게.  
# - 단계별 안내가 필요하면 줄바꿈을 하지말고 일관된 답변으로 이야기 하게.
# - 이모티콘을 적절히 써서 정겹고 친근하게 표현하세.  

# 🟢 질문: {question}  
# ✨ 답변:  
# """
#---------------------------------------- 웹 검색 프롬프트 ----------------------------------------
WEB_PROMPT_TMPL = """
당신은 대한민국 농업 작물 추천 전문가입니다.
아래 '웹 검색 결과'를 바탕으로 사용자의 질문에 맞는 작물을 추천하고 재배 정보를 종합하여 안내해주세요.

[웹 검색 결과]
{search_results}

규칙:
- 검색 결과를 바탕으로 작물 추천과 재배 방법을 종합하여 답변하세요.
- 추천 작물의 선택 이유와 장점을 명확하게 설명하세요.
- 재배 조건, 시기, 관리 방법 등을 단계별로 정리하세요.
- 재배 방법은 각 단계마다 한 줄씩 띄워서 작성하세요.
- 모든 답변은 반드시 한국어로 작성하세요.
- 추천 작물의 개수는 적합한 수준(1~5개 이내)으로 제한하세요.

🟢 질문: {question}
✨ 답변:
"""
web_prompt = ChatPromptTemplate.from_template(WEB_PROMPT_TMPL) # 웹 전용 프롬프트 템플릿 객체를 생성합니다.

# ==================== 상태 타입 ====================
class GraphState(TypedDict, total=False): # LangGraph 워크플로우의 상태를 정의합니다.
    question: Optional[str] # 사용자 질문
    db_context: Optional[str] # DB에서 검색된 컨텍스트
    web_context: Optional[str] # 웹에서 검색된 컨텍스트
    context: Optional[str] # 최종 결합된 컨텍스트
    answer: Optional[str] # ✅ 최종 답변만 유지
    answer_source: Optional[str] # 답변의 출처를 나타냅니다.
    


# ==================== 라우팅 유틸 ====================
def format_answer_for_terminal(answer: str) -> str:
    """
    LLM 답변을 터미널에서 볼 때 줄바꿈 대신 <br>로 치환
    """
    return answer.replace("\n", "<br>")

def _normalize_ko(s: str) -> str:
    return (s or "").lower()

def _extract_tokens_ko(s: str) -> List[str]:
    # 한글/영문/숫자 토큰만, 길이 2자 이상(조사/불용어 최소화)
    return [t for t in re.findall(r"[가-힣A-Za-z0-9]+", s or "") if len(t) >= 2]

def has_keyword_overlap(question: str, context: str, min_hits: int = 1) -> bool:
    q_tokens = _extract_tokens_ko(question)
    ctx = _normalize_ko(context)
    # 질문 토큰이 컨텍스트에 포함된 횟수
    hits = sum(1 for t in q_tokens if t.lower() in ctx)
    return hits >= min_hits

def route_after_retrieve(state: "GraphState") -> str:
    db = (state.get("db_context") or "").strip()
    docs_with_scores = state.get("docs_with_scores", []) # ✅ 추가: state에서 docs_with_scores 가져오기
    
    # 1.만약에 DB가 비었거나 실패 문구가 포함된 경우
    if (not db) or ("관련 문서를 찾을 수 없습니다." in db):
        return "need_web"

    # 2. DB 컨텍스트 길이가 너무 짧은 경우
    if len(db) < MIN_DB_CONTEXT_CHARS:
        return "need_web"

    # 3. 질문과 컨텍스트의 내용적 관련성을 추가로 판단 (강화된 로직)
    #    a) 키워드 중복이 1개도 없으면
    if not has_keyword_overlap(state.get("question"), db, min_hits=1):
        return "need_web"
        
    #    b) 가장 높은 유사도 점수가 낮으면
    if docs_with_scores:
        max_score = max(score for doc, score in docs_with_scores)
        if max_score < 0.5: # 0.5는 예시, 필요에 따라 조정 가능
            return "need_web"

    # 4. 모든 조건 통과 시 DB만 사용
    return "have_db"


# ==================== 노드들 ====================
def load_milvus_node(state: GraphState) -> Dict[str, Any]: # Milvus를 로드하는 노드 함수입니다.
    print("\n--- 노드: Milvus 로드 ---")
    global _vectorstore
    # Milvus 연결이 없으면 새로 연결을 시도합니다.
    if "default" not in connections.list_connections() or not connections.has_connection("default"):
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)
    try:
        _vectorstore = MilvusVectorStore( # MilvusVectorStore 객체를 생성합니다.
            embedding_function=embedding_model,
            collection_name=MILVUS_COLLECTION,
            connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN},
        )
        print(f"✅ Milvus 로드 완료: {MILVUS_COLLECTION}")
        return {**state}
    except Exception as e:
        print(f"❌ Milvus 로드 실패: {e}")
        raise ConnectionError("Milvus 벡터스토어 로드 실패")

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("--- 노드: 문서 검색 ---")
    question = state.get("question")
    global _vectorstore
    vectorstore = _vectorstore
    if not question or not vectorstore:
        raise ValueError("질문 또는 벡터스토어가 누락되었습니다.")
    
    # enhanced_query = f"{question} 재배 방법 키우기 팁"
    enhanced_query = question  # NEW: 사용자의 질문 그대로 검색
    docs_with_scores = vectorstore.similarity_search_with_score(enhanced_query, k=10)
    
    filtered_docs = []
    # docs_with_scores 리스트를 순회하며 필터링
    for doc, score in docs_with_scores:
        content = doc.page_content or ""
        if score > 0.5 and len(content.strip()) > 100:
            filtered_docs.append(doc)
    
    final_docs = filtered_docs[:5]

    context = ""
    if final_docs:
        print(f"✅ {len(final_docs)}개 문서 검색 완료.")
        for i, doc in enumerate(final_docs):
            preview = (doc.page_content or "")[:100].replace("\n", " ")
            print(f"   - 문서 {i+1}: '{preview}...'")
            context += f"\n\n{doc.page_content}"
    else:
        print("⚠️ 검색된 문서가 없습니다.")
        context = "관련 문서를 찾을 수 없습니다."
        
    print(f"   - 컨텍스트 길이: {len(context)}자")
    # ✅ 수정된 부분: 'docs_with_scores'와 'final_docs'를 state에 저장
    return {**state, "db_context": context, "docs_with_scores": docs_with_scores, "final_docs": final_docs}

def web_search_node(state: GraphState) -> Dict[str, Any]: # 웹 검색 노드입니다.
    print("--- 노드: 웹 검색 ---")
    question = state.get("question")
    if not question:
        raise ValueError("질문이 누락되었습니다.")
    if not TAVILY_API_KEY:
        print("⚠️ TAVILY API 키가 없어 웹 검색을 건너뜁니다.")
        return {**state, "web_context": "웹 검색 비활성화"} # API 키가 없으면 검색을 건너뜁니다.
    
    search_tool = TavilyClient(api_key=TAVILY_API_KEY)
    web_context_parts = []
    
    try:
        response = search_tool.search(query=question, max_results=5) # Tavily API로 웹 검색을 수행합니다.
        if isinstance(response, dict) and "results" in response:
            results = response["results"]
            for r in results:
                if isinstance(r, dict):
                    content = (r.get("content") or r.get("snippet") or "").strip() # 웹 문서의 내용을 추출합니다.
                    url = (r.get("url") or "").strip() # 출처 URL을 추출합니다.
                    web_context_parts.append(f"- 출처: {url or 'N/A'}\n 내용: {content}")
        else:
            print(f"⚠️ 예상치 못한 Tavily 응답 형식: {type(response)}")
            web_context_parts.append(f"- 출처: N/A\n 내용: 웹 검색 응답 형식 오류")
    except Exception as e:
        print(f"❌ Tavily 검색 오류: {e}")
        web_context_parts.append(f"- 출처: N/A\n 내용: 웹 검색 실패 - {str(e)}")

    web_context = "\n\n".join(web_context_parts)
    print("✅ 웹 검색 완료.")
    return {**state, "web_context": web_context}

def combine_context_node(state: GraphState) -> Dict[str, Any]: # 컨텍스트를 결합하는 노드입니다.
    print("--- 노드: 컨텍스트 결합 ---")
    db_context = state.get("db_context", "")
    web_context = state.get("web_context", "")
    final_context = db_context # 최종 컨텍스트를 DB 내용으로 초기화합니다.
    if web_context and web_context != "웹 검색 비활성화": # 웹 검색 결과가 있을 경우
        print("✅ DB와 웹 컨텍스트를 결합합니다.")
        # DB와 웹 컨텍스트를 결합하여 최종 컨텍스트를 만듭니다.
        final_context = f"[DB 검색 결과]\n{db_context}\n\n[웹 검색 결과]\n{web_context}"
    else:
        print("ℹ️ DB 컨텍스트만 사용합니다.")
    return {**state, "context": final_context}

def generate_draft_node(state: GraphState) -> Dict[str, Any]: # 최종 답변을 생성하는 노드입니다.
    print("--- 노드: 답변 생성 ---")
    context = state.get("context", "")
    question = state.get("question", "")

    if not context:
        print("❌ 컨텍스트가 없어 답변을 생성할 수 없습니다.")
        return {**state, "answer": "제공된 자료만으로는 답변이 어렵습니다."}

    # 웹 컨텍스트만 있을 경우와 DB 컨텍스트가 있을 경우 다른 프롬프트를 사용합니다.
    if context.startswith("[웹 검색 결과]"):
        messages = web_prompt.format_messages(web_context_parts=context, question=question)
    else:
        messages = rag_prompt.format_messages(context=context, question=question)

    response = llm.invoke(messages) # LLM을 호출하여 답변을 생성합니다.
    ans = response.content
    print("✅ 최종 답변 생성 완료.")
    return {**state, "answer": ans, "answer_source": "DB/웹"}

# ==================== 그래프 빌드 ====================
def build_graph(): # LangGraph 워크플로우를 구축하는 함수입니다.
    global agent_app
    if agent_app is not None:
        return agent_app
    
    g = StateGraph(GraphState)
    g.add_node("load_milvus", load_milvus_node) # 노드: Milvus 연결
    g.add_node("retrieve", retrieve_node) # 노드: DB 검색
    g.add_node("web_search", web_search_node) # 노드: 웹 검색
    g.add_node("combine_context", combine_context_node) # 노드: 컨텍스트 결합
    g.add_node("generate_draft", generate_draft_node) # ✅ 노드: 최종 답변 생성

    g.set_entry_point("load_milvus") # 워크플로우의 시작점
    g.add_edge("load_milvus", "retrieve") # Milvus 로드 후 검색으로 이동

    g.add_conditional_edges( # 검색 결과에 따라 조건부 이동을 정의합니다.
        "retrieve",
        route_after_retrieve, # 라우팅 함수를 사용합니다.
        {
            "need_web": "web_search", # 웹 검색이 필요한 경우 웹 검색 노드로 이동합니다.
            "have_db": "combine_context", # DB 컨텍스트만으로 충분할 경우 컨텍스트 결합 노드로 이동합니다.
        },
    )
    g.add_edge("web_search", "combine_context") # 웹 검색 후 컨텍스트 결합으로 이동합니다.
    g.add_edge("combine_context", "generate_draft") # 컨텍스트 결합 후 답변 생성으로 이동합니다.
    g.add_edge("generate_draft", END) # ✅ 답변 생성 후 워크플로우를 종료합니다.

    agent_app = g.compile() # 정의된 그래프를 컴파일하여 실행 가능한 앱으로 만듭니다.
    return agent_app

# ==================== 실행 래퍼 ====================
async def run(state: dict) -> dict: # 에이전트를 실행하고 결과를 반환하는 래퍼 함수입니다.
    try:
        query = state.get("query", "")
        if not query:
            return {"agent_answer": "질문이 제공되지 않았습니다. 어떤 주제든 질문해 주세요."}
        print(f"[작물추천_agent] 질문 처리 시작: {query}")

        final_state = agent_app.invoke({"question": query}) # LangGraph 앱을 실행합니다.

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
    text = re.sub(r'#{1,6}\s', '', text) # 헤더(#) 제거
    text = re.sub(r'[\*\-]', '', text) # 불릿(*, -) 제거
    text = re.sub(r'\[.*?\]\(.*?\)', '', text) # 링크 제거
    text = re.sub(r'\s+', ' ', text) # 여러 공백을 하나로 축소
    return text.strip() # 양 끝 공백 제거


# ==================== main ====================
if __name__ == "__main__":  # 스크립트가 직접 실행될 때의 진입점입니다.
    parser = argparse.ArgumentParser(description="RAG 파이프라인 - 농업 작물 추천 시스템 (DB 우선 + 조건부 웹검색)")
    parser.add_argument("-q", "--query", type=str, help="한 번만 실행할 질문 (예: -q '주말농장에 키울 작물 추천해줘')")
    args = parser.parse_args()
    
    agent_app = build_graph()  # 그래프를 빌드합니다.

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
        # 터미널에서 줄바꿈 보기 좋게 <br> → \n 치환
        print(cleaned_answer.replace("<br>", "\n"))
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
            print(cleaned_answer.replace("<br>", "\n"))
            print("=" * 20)
        print("👋 프로그램을 종료합니다.")

    # ==================== 웹 UI용 (FastAPI 예시, 주석 처리) ====================
    # """
    # from fastapi import FastAPI, Request
    # app = FastAPI()

    # @app.post("/ask")
    # async def ask(request: Request):
    #     body = await request.json()
    #     query = body.get("query", "")
    #     final_state = agent_app.invoke({"question": query})
    #     answer = final_state.get("answer", "답변 생성에 실패했습니다.")
    #     # 웹에서는 <br> 태그를 유지해 프론트에서 줄바꿈 처리 가능
    #     return {"answer": answer}
    # """
