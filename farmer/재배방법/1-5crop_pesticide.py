import os
import re
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, Optional, TypedDict

# --- 1. 환경 설정 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PESTICIDE_API_KEY = os.getenv("PESTICIDE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
if not PESTICIDE_API_KEY:
    raise ValueError("PESTICIDE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

print(f"OPENAI_API_KEY 로드 완료: {OPENAI_API_KEY[:5]}...")
print(f"PESTICIDE_API_KEY 로드 완료: {PESTICIDE_API_KEY[:5]}...")

# --- 2. LLM 및 프롬프트 설정 ---
# 답변 생성을 위한 LLM
llm_answer = ChatGroq(model_name="llama3-70b-8192", 
                      temperature=0.7, 
                      api_key=OPENAI_API_KEY)

# 키워드 추출을 위한 LLM
llm_keyword = ChatGroq(model_name="llama3-8b-8192",
                       temperature=0.0,
                       api_key=OPENAI_API_KEY)

# 농약 전문가 프롬프트 템플릿 (API 정보만 사용)
RAG_PROMPT_TEMPLATE = """
당신은 대한민국 농약 전문가입니다.

다음은 외부 API에서 얻은 농약 관련 정보입니다. 이 정보를 활용하여 사용자의 질문에 답변해 주세요.

# 외부 API에서 얻은 정보:
{api_result}

당신이 지킬 규칙들은 다음과 같습니다:
- 제공된 정보에 없는 정보, 저의 상식, 추측, 거짓 정보 등은 절대 답변에 넣지 마세요.
- 모든 답변은 한글로만 답해야 합니다. 한글이 아니면 절대 출력하지 말고, 한글로만 답변을 완성해야 합니다.
- 각 설명은 반드시 "한 문장씩 줄바꿈"해서 써주세요.
- 답변이 불가능하거나 제공된 정보에 관련 정보가 없으면 "주어진 정보로는 답변할 수 없습니다."라고만 답해야 합니다.
- 친근하고 전문적인 태도로, 자연스럽게 설명해주세요.

질문: {question}
답변:
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# 키워드 추출을 위한 프롬프트
KEYWORD_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    "사용자의 질문에서 작물명, 병해충명을 쉼표로 구분하여 추출해. 다른 말은 하지마.\n질문: {question}\n키워드:"
)

# --- 3. LangGraph 상태 정의 ---
class GraphState(TypedDict):
    """
    LangGraph 상태를 나타내는 TypedDict.
    API 호출에 필요한 정보만 포함합니다.
    """
    question: Optional[str]
    api_result: Optional[str]
    keywords: Optional[str]
    answer: Optional[str]

# --- 4. 핵심 기능 함수 정의 (사용자 제공 API 호출 코드 통합) ---
def call_pesticide_api(crop_name: str = "", disease_name: str = ""):
    """농약 API를 호출하여 농약 정보를 DataFrame으로 반환합니다."""
    BASE_URL = "https://psis.rda.go.kr/openApi/service.do"
    
    if not PESTICIDE_API_KEY:
        return pd.DataFrame()
        
    params = {
        "apiKey": PESTICIDE_API_KEY,
        "serviceCode": "SVC01",
        "serviceType": "AA001",
        "displayCount": 10,
        "startPoint": 1,
        "cropName": crop_name,
        "diseaseWeedName": disease_name,
        "similarFlag" : "Y",
    }
    
    try:
        res = requests.get(BASE_URL, params=params)
        res.raise_for_status()
        root = ET.fromstring(res.text)
        
        rows = []
        for item in root.findall(".//item"):
            rows.append({
                "작물명": item.findtext("cropName"),
                "병해충": item.findtext("diseaseWeedName"),
                "용도": item.findtext("useName"),
                "상표명": item.findtext("pestiBrandName"),
                "사용방법": item.findtext("pestiUse"),
                "희석배수": item.findtext("dilutUnit"),
                "안전사용기준(수확 일 전)": item.findtext("useSuittime"),
                "안전사용기준(회 이내)": item.findtext("useNum"),
            })
        
        df = pd.DataFrame(rows)
        return df

    except requests.exceptions.RequestException as e:
        print(f"API 호출 오류: {e}")
        return pd.DataFrame()
    except ET.ParseError as e:
        print(f"XML 파싱 오류: {e}")
        return pd.DataFrame()

# --- 5. LangGraph 노드 함수 정의 ---
def extract_keywords_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 질문에서 키워드 추출"""
    print("\n---노드: 키워드 추출 (Extract Keywords) 실행---")
    question = state["question"]
    keyword_chain = KEYWORD_EXTRACT_PROMPT | llm_keyword | StrOutputParser()
    keywords = keyword_chain.invoke({"question": question})
    print(f"추출된 키워드: {keywords}")
    return {**state, "keywords": keywords}

def call_api_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 키워드를 이용해 API 호출"""
    print("\n---노드: API 호출 (Call API) 실행---")
    keywords = state.get("keywords")
    api_result = "외부 API에서 얻은 추가 정보가 없습니다."

    if keywords:
        parsed_keywords = [k.strip() for k in keywords.split(',') if k.strip()]
        crop_name = parsed_keywords[0] if len(parsed_keywords) > 0 else ""
        disease_name = parsed_keywords[1] if len(parsed_keywords) > 1 else ""

        print(f"API 호출에 사용될 작물명: {crop_name}, 병해충명: {disease_name}")
        df = call_pesticide_api(crop_name=crop_name, disease_name=disease_name)
        
        if not df.empty:
            api_result = "외부 API 결과:\n" + df.to_string(index=False)
        else:
            api_result = "외부 API 결과: 관련 정보를 찾을 수 없습니다."
    
    print(f"API 호출 결과: \n{api_result}")
    return {**state, "api_result": api_result}

def generate_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 답변 생성"""
    print("\n---노드: 생성 (Generate) 실행---")
    if ("question" not in state or not state["question"] or
        "api_result" not in state or not state["api_result"]):
        raise ValueError("답변 생성을 위해 질문과 API 결과가 모두 필요합니다.")
    
    question = state["question"]
    api_result = state["api_result"]
    
    rag_chain = (
        {"question": RunnablePassthrough(), "api_result": RunnablePassthrough()}
        | rag_prompt
        | llm_answer
        | StrOutputParser()
    )
    inputs = {"question": question, "api_result": api_result}
    answer = rag_chain.invoke(inputs)
    return {**state, "answer": answer}

# --- 6. LangGraph 워크플로우 빌드 및 컴파일 ---
def build_rag_workflow():
    """API 전용 RAG 워크플로우를 빌드하고 컴파일합니다."""
    workflow = StateGraph(GraphState) 

    workflow.add_node("extract_keywords", extract_keywords_node)
    workflow.add_node("call_api", call_api_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge("extract_keywords", "call_api")
    workflow.add_edge("call_api", "generate")
    workflow.add_edge("generate", END)

    workflow.set_entry_point("extract_keywords")

    return workflow.compile()

# --- 7. 메인 실행 부분 ---
if __name__ == "__main__":
    print("\n---농약 전문가 챗봇 에이전트 시작 (종료하려면 'exit' 또는 'quit' 입력)---")

    app = build_rag_workflow()

    while True:
        user_question = input("\n질문을 입력하세요: ")
        if user_question.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break

        if not user_question.strip():
            print("질문을 입력해주세요.")
            continue

        try:
            final_state = app.invoke({"question": user_question})

            print("\n---챗봇 답변---")
            print(final_state['answer'])

        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")