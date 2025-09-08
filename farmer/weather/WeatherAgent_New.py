# -*- coding: utf-8 -*-
"""
모듈화된 날씨 에이전트
기상특보, 단기예보, 중기예보 노드를 모듈화하여 사용하는 새로운 WeatherAgent
"""

import os
import re
import time
from typing import TypedDict, Optional, Any, Dict
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# LangChain / LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# 모듈화된 노드들 import
from .advisory_node import AdvisoryNode
from .short_forecast_node import ShortForecastNode
from .mid_forecast_node import MidForecastNode
from .utils import combine_weather_data, search_similar_documents

load_dotenv()

# 환경 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
KST = ZoneInfo("Asia/Seoul")

# 그래프 상태 정의
class GraphState(TypedDict):
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    answer_draft: Optional[str]
    advisory_data: Optional[list]
    short_forecast_data: Optional[list]
    mid_forecast_data: Optional[list]

def make_llm() -> ChatOpenAI:
    """LLM 인스턴스 생성"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")
    return ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY
    )

# 프롬프트 템플릿
DRAFT_PROMPT = ChatPromptTemplate.from_template(
    """너는 대한민국 기상청 기준 용어를 사용하는 '현직 예보관'이야.
문맥에는 KMA 특보/단기/중기 예보 정보가 섞여 있다.
다음 '출력 규격'을 반드시 만족하는 한국어 자연문으로 초안 답변을 작성해.

[출력 규격 - 조건부 구조]

**질문이 오늘인 경우:**
1) 개요: 한 문장 핵심 요약
2) 기상특보 현황: 
   - 문맥의 [live_advisory] 데이터로 현재 특보 상태 표시
   - 특보 데이터가 없으면 "특보 없음"
3) 단기예보 상세: 
   - 문맥의 [live_forecast] 데이터만 사용
   - 하늘상태, 기온, 강수확률, 바람, 발표시각 포함해서 한줄로 작성해
4) 기상 정보 요약: 제공된 기상 데이터를 종합한 요약

**질문이 내일~3일 이내인 경우:**
1) 개요: 한 문장 핵심 요약
2) 단기예보 상세: 
   - 문맥의 [live_forecast] 데이터만 사용
   - 하늘상태, 기온, 강수확률, 바람, 발표시각 포함해서 한줄로 작성해
3) 기상 정보 요약: 제공된 기상 데이터를 종합한 요약

**질문이 오늘+4일 이후인 경우:**
1) 개요: 한 문장 핵심 요약
2) 중기예보 전망:
   - 문맥의 [region_forecast] 정보만 사용
   - 향후 기간의 날씨 전망과 강수 가능성
   - 기온 변화 경향
3) 기상 정보 요약: 제공된 기상 데이터를 종합한 요약

금지: 표/코드/JSON/불릿(번호는 허용), 과도한 추측

[문맥]
{context}

질문: {question}
초안 답변(위 구조 포함):"""
)

REFINE_PROMPT = ChatPromptTemplate.from_template(
    """다음은 초안과 근거 문맥이야.
예보관 관점에서 사실성(문맥 일치), 시간/지역 명시, 위험도 판단의 타당성, 조치의 구체성을 강화해 최종 답변을 작성해.
문맥 밖 정보 추가 금지. 표/코드/JSON 금지. 번호는 허용.

특히 다음 조건부 구조를 반드시 따르고 시간 기반 정보를 정확히 반영해:

**질문이 오늘인 경우:**
1) 개요
2) 기상특보 현황 ([live_advisory] 데이터로 현재 특보 상태)
3) 단기예보 상세 ([live_forecast]만)
4) 기상 정보 요약

**질문이 내일~3일 이내인 경우:**
1) 개요
2) 단기예보 상세 ([live_forecast]만)
3) 기상 정보 요약

**질문이 오늘+4일 이후인 경우:**
1) 개요
2) 중기예보 전망 ([region_forecast]만)
3) 기상 정보 요약

※ 질문 날짜에 따라 섹션 개수와 번호가 달라짐
 
[문맥]
{context}

질문: {question}
초안: {answer_draft}

최종 답변(위 구조 유지):"""
)

# 노드 구현
def load_store_node(state: GraphState) -> Dict[str, Any]:
    """초기화 노드"""
    print("🧩 노드: 초기화")
    return {
        **state,
        "advisory_data": [],
        "short_forecast_data": [],
        "mid_forecast_data": []
    }

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """데이터 수집 노드"""
    print("🧩 노드: 날씨 데이터 수집")
    
    # 모듈화된 노드들 실행
    advisory_node = AdvisoryNode()
    short_forecast_node = ShortForecastNode()
    mid_forecast_node = MidForecastNode()
    
    # 각 노드 실행
    state = advisory_node.run(state)
    state = short_forecast_node.run(state)
    state = mid_forecast_node.run(state)
    
    # 데이터 통합
    context = combine_weather_data(state)
    
    return {**state, "context": context}

def generate_draft_node(state: GraphState) -> Dict[str, Any]:
    """초안 생성 노드"""
    print("🧩 노드: 초안 생성")
    
    if not state.get("question"):
        raise ValueError("question 누락")
    if not state.get("context"):
        raise ValueError("context 누락")
    
    chain = (
        {"context": lambda x: x["context"], "question": lambda x: x["question"]}
        | DRAFT_PROMPT
        | make_llm()
        | StrOutputParser()
    )
    
    answer = chain.invoke({"context": state["context"], "question": state["question"]})
    time.sleep(1)  # API 요청 간격
    
    txt = re.sub(r'\n{3,}', '\n\n', answer or "").strip()
    return {**state, "answer_draft": txt}

def refine_answer_node(state: GraphState) -> Dict[str, Any]:
    """답변 개선 노드"""
    print("🧩 노드: 답변 개선/최종")
    
    if not state.get("question"):
        raise ValueError("question 누락")
    if not state.get("context"):
        raise ValueError("context 누락")
    if not state.get("answer_draft"):
        raise ValueError("answer_draft 누락")
    
    chain = (
        {
            "context": lambda x: x["context"],
            "question": lambda x: x["question"],
            "answer_draft": lambda x: x["answer_draft"]
        }
        | REFINE_PROMPT
        | make_llm()
        | StrOutputParser()
    )
    
    answer = chain.invoke({
        "context": state["context"],
        "question": state["question"],
        "answer_draft": state["answer_draft"]
    })
    time.sleep(1)  # API 요청 간격
    
    txt = re.sub(r'\n{3,}', '\n\n', answer or "").strip()
    return {**state, "answer": txt}

# 그래프 빌드
def build_graph():
    """그래프 빌드"""
    g = StateGraph(GraphState)
    
    # 노드 추가
    g.add_node("load_store", load_store_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate_draft", generate_draft_node)
    g.add_node("refine_answer", refine_answer_node)
    
    # 엣지 설정
    g.set_entry_point("load_store")
    g.add_edge("load_store", "retrieve")
    g.add_edge("retrieve", "generate_draft")
    g.add_edge("generate_draft", "refine_answer")
    g.add_edge("refine_answer", END)
    
    app = g.compile()
    return app

# OchestratorTest.py 호환 함수
def run(state: dict) -> dict:
    """
    OchestratorTest.py에서 호출되는 날씨 에이전트 실행 함수
    
    Args:
        state: OchestratorTest.py에서 전달받은 상태 딕셔너리
               - query: 사용자 질문 (필수)
    
    Returns:
        dict: 실행 결과
            - agent_answer: 최종 답변
    """
    try:
        # 질문 추출
        query = state.get("query", "")
        if not query:
            return {"agent_answer": "질문이 제공되지 않았습니다. 날씨 관련 질문을 해주세요."}
        
        print(f"[날씨_agent] 질문 처리 시작: {query}")
        
        # 그래프 빌드 및 실행
        app = build_graph()
        
        # 그래프 실행
        result = app.invoke({"question": query})
        
        # 답변 추출
        answer = result.get("answer", "답변을 생성할 수 없습니다.")
        
        print(f"[날씨_agent] 답변 생성 완료: {len(answer)}자")
        
        return {"agent_answer": answer}
        
    except Exception as e:
        error_msg = f"날씨 에이전트 실행 중 오류가 발생했습니다: {e}"
        print(f"[날씨_agent] 오류: {e}")
        return {"agent_answer": error_msg}

# 메인 실행부
def main():
    """메인 함수"""
    import sys
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="모듈화된 기상 전문가 그래프")
    parser.add_argument("-q", "--question", default=None, help="질문 1회 실행 후 종료")
    parser.add_argument("--show-context", action="store_true", help="컨텍스트(근거) 출력")
    args = parser.parse_args()

    print("💬 모듈화된 기상 전문가 그래프")
    app = build_graph()

    if args.question:
        q = args.question.strip()
        if not q:
            raise ValueError("질문이 비어 있습니다.")
        try:
            out = app.invoke({"question": q})
            if args.show_context:
                print("\n=== 컨텍스트 ===")
                print(out.get("context", ""))
            print("\n=== 답변 ===")
            print(out.get("answer", ""))
            print()
        except Exception as e:
            print(f"❌ 오류: {e}\n")
    else:
        print("질문을 입력하세요. (종료: exit/quit)")
        while True:
            q = input("질문> ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue
            try:
                out = app.invoke({"question": q})
                if args.show_context:
                    print("\n=== 컨텍스트 ===")
                    print(out.get("context", ""))
                print("\n=== 답변 ===")
                print(out.get("answer", ""))
                print()
            except Exception as e:
                print(f"❌ 오류: {e}\n")

if __name__ == "__main__":
    main()
