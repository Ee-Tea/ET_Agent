# -*- coding: utf-8 -*-
"""
모듈화된 날씨 에이전트 실행 파일 (절대 import 버전)
기상특보, 단기예보, 중기예보 노드를 모듈화하여 사용하는 새로운 WeatherAgent
"""

import os
import sys
import re
import time
from typing import TypedDict, Optional, Any, Dict, Annotated
from operator import add
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# 프로젝트 루트를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

# LangChain / LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# 모듈화된 노드들 import (절대 import)
from farmer.weather.advisory_node import AdvisoryNode
from farmer.weather.short_forecast_node import ShortForecastNode
from farmer.weather.mid_forecast_node import MidForecastNode
from farmer.weather.utils import combine_weather_data, search_similar_documents

load_dotenv()

# 환경 설정
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
KST = ZoneInfo("Asia/Seoul")

# 그래프 상태 정의
class GraphState(TypedDict):
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    answer_draft: Optional[str]
    advisory_data: Annotated[Optional[list], add]
    short_forecast_data: Annotated[Optional[list], add]
    mid_forecast_data: Annotated[Optional[list], add]
    # 새로운 필드들
    need_advisory: Optional[bool]
    need_short_forecast: Optional[bool]
    need_mid_forecast: Optional[bool]
    analysis_result: Optional[str]
    # 지역 정보 관련 필드들
    has_region_info: Optional[bool]
    has_date_info: Optional[bool]
    default_region: Optional[str]
    processing_mode: Optional[str]
    waiting_for_region: Optional[bool]
    waiting_for_date: Optional[bool]
    original_question: Optional[str]
    detected_region: Optional[str]
    detected_date: Optional[str]
    retry_count: Optional[int]
    # 추가 질문 관련 필드들
    is_follow_up_question: Optional[bool]
    should_continue_flow: Optional[bool]
    # 자동 서울 처리 플래그
    auto_seoul_processed: Optional[bool]
    # 추가질문 관련 필드들
    should_ask_follow_up: Optional[bool]
    waiting_for_follow_up: Optional[bool]
    is_weather_related: Optional[bool]
    follow_up_analysis: Optional[str]

def make_llm() -> ChatOpenAI:
    """LLM 인스턴스 생성"""
    if not OPENAI_API_KEY=REDACTED ValueError("OPENAI_API_KEY=REDACTED에 없습니다.")
    return ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY=REDACTED = ChatPromptTemplate.from_template(
    """다음 질문을 분석해서 어떤 기상 데이터가 필요한지 판단하세요.

질문: {question}

반드시 다음 JSON 형식으로만 답변하세요 (다른 텍스트 없이):
{{
    "need_advisory": true,
    "need_short_forecast": true,
    "need_mid_forecast": false,
    "has_region_info": true,
    "has_date_info": true,
    "default_region": "서울",
    "processing_mode": "auto_seoul",
    "reason": "판단 이유"
}}

주의: true/false는 반드시 boolean 값이어야 합니다 (문자열 "true"/"false"가 아님)

판단 기준:
- need_advisory: 특보, 경보, 주의보 관련 질문이거나 "오늘" 날씨 질문
- need_short_forecast: "오늘", "내일", "모레", "3일" 이내 날씨 질문
- need_mid_forecast: "4일", "5일", "6일", "7일", "일주일", "주간" 날씨 질문
- has_region_info: 질문에 지역명이 포함되어 있는지 (서울, 부산, 대구, 인천, 광주, 대전, 울산, 세종, 경기, 강원, 충북, 충남, 전북, 전남, 경북, 경남, 제주)
- has_date_info: 질문에 구체적인 날짜/시간 정보가 있는지 (오늘, 내일, 모레, 3일, 4일, 5일, 6일, 7일, 일주일, 주간 등)
- default_region: 지역 정보가 없을 때 기본 지역 (서울, 부산, 대구 중 하나)
- processing_mode: "auto_seoul" (기본 서울 처리) 또는 "ask_user" (사용자에게 질문)

JSON만 답변하세요:"""
)

# 답변 생성 프롬프트
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """기상청 예보관으로서 다음 데이터를 바탕으로 종합적인 답변을 작성하세요.

질문: {question}

[기상 데이터]
{context}

답변 구조:
1) 개요: 핵심 요약
2) 기상특보 현황: (특보 데이터가 있을 때만)
3) 단기예보: (단기 데이터가 있을 때만) 
4) 중기예보: (중기 데이터가 있을 때만)
5) 종합 요약

주의사항:
- 지역 정보가 명확하지 않은 경우, 서울/수도권 기준으로 답변하세요
- 답변 시작 부분에 "서울/수도권 기준"이라고 명시하세요
- 답변 끝에 다른 지역 정보도 제공 가능하다고 안내하세요
- 예시: "다른 지역(부산, 대구, 인천 등)의 날씨 정보도 필요하시면 말씀해주세요!"

답변:"""
)

# 노드 구현
def analyze_question_node(state: GraphState) -> Dict[str, Any]:
    """질문 분석 노드"""
    print("🧩 노드: 질문 분석")
    
    question = state.get("question", "")
    if not question:
        raise ValueError("question 누락")
    
    # LLM으로 질문 분석
    chain = (
        {"question": lambda x: x["question"]}
        | ANALYSIS_PROMPT
        | make_llm()
        | StrOutputParser()
    )
    
    analysis = chain.invoke({"question": question})
    
    # JSON 파싱 (개선된 버전)
    import json
    import re
    
    # JSON 부분만 추출
    json_match = re.search(r'\{.*\}', analysis, re.DOTALL)
    if json_match:
        json_str = json_match.group()
    else:
        json_str = analysis
    
    try:
        analysis_data = json.loads(json_str)
        
        # 문자열 boolean을 실제 boolean으로 변환
        def str_to_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'on']
            return False
        
        need_advisory = str_to_bool(analysis_data.get("need_advisory", False))
        need_short_forecast = str_to_bool(analysis_data.get("need_short_forecast", False))
        need_mid_forecast = str_to_bool(analysis_data.get("need_mid_forecast", False))
        has_region_info = str_to_bool(analysis_data.get("has_region_info", False))
        has_date_info = str_to_bool(analysis_data.get("has_date_info", False))
        default_region = analysis_data.get("default_region", "서울")
        processing_mode = analysis_data.get("processing_mode", "auto_seoul")
        reason = analysis_data.get("reason", "")
        
        print(f"   - LLM 분석 성공: {reason}")
    except Exception as e:
        # 파싱 실패 시 기본값 (더 안전한 기본값)
        print(f"   - JSON 파싱 실패: {e}")
        print(f"   - 원본 응답: {analysis}")
        
        # 질문 내용으로 간단한 판단
        question_lower = question.lower()
        if "오늘" in question:
            need_advisory = True
            need_short_forecast = True
            need_mid_forecast = False
            has_date_info = True
        elif "내일" in question or "모레" in question:
            need_advisory = False
            need_short_forecast = True
            need_mid_forecast = False
            has_date_info = True
        elif "일주일" in question or "주간" in question:
            need_advisory = False
            need_short_forecast = False
            need_mid_forecast = True
            has_date_info = True
        else:
            need_advisory = True
            need_short_forecast = True
            need_mid_forecast = False
            has_date_info = False
        
        # 지역 정보 판단
        region_keywords = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
        has_region_info = any(region in question for region in region_keywords)
        
        default_region = "서울"
        processing_mode = "auto_seoul"
        reason = f"파싱 실패로 기본값 사용 (지역: {has_region_info}, 날짜: {has_date_info})"
    
    print(f"   - 분석 결과: 특보={need_advisory}, 단기={need_short_forecast}, 중기={need_mid_forecast}")
    print(f"   - 지역 정보: {has_region_info}, 날짜 정보: {has_date_info}")
    print(f"   - 처리 모드: {processing_mode}, 기본 지역: {default_region}")
    print(f"   - 이유: {reason}")
    
    # 자동 서울 처리 방식 (중복 처리 방지)
    if not has_region_info and default_region not in question:
        print(f"   - 자동 서울 처리: 기본 지역({default_region})으로 처리")
        enhanced_question = f"{default_region} {question}"
        print(f"   - 질문 강화: {enhanced_question}")
        processing_mode = "auto_seoul_with_options"
        auto_seoul_processed = True
        has_region_info = True  # 자동 서울 처리 후 지역 정보 있음으로 설정
    else:
        enhanced_question = question
        auto_seoul_processed = False
        print(f"   - 지역 정보 이미 포함됨: {question}")
    
    return {
        **state,
        "question": enhanced_question,  # 지역 정보가 추가된 질문으로 업데이트
        "original_question": question,  # 원본 질문 저장
        "need_advisory": need_advisory,
        "need_short_forecast": need_short_forecast,
        "need_mid_forecast": need_mid_forecast,
        "has_region_info": has_region_info,  # 자동 서울 처리 후 True로 업데이트
        "has_date_info": has_date_info,
        "default_region": default_region,
        "processing_mode": processing_mode,
        "analysis_result": analysis,
        "retry_count": 0,  # 재시도 카운터 초기화
        "auto_seoul_processed": auto_seoul_processed,
        "advisory_data": [],
        "short_forecast_data": [],
        "mid_forecast_data": []
    }

def advisory_node_wrapper(state: GraphState) -> Dict[str, Any]:
    """기상특보 노드 래퍼"""
    if not state.get("need_advisory", False):
        print("🧩 노드: 기상특보 스킵")
        return {"advisory_data": []}
    
    print("🧩 노드: 기상특보 데이터 수집")
    advisory_node = AdvisoryNode()
    result = advisory_node.run(state)
    # Annotated 필드만 반환
    return {"advisory_data": result.get("advisory_data", [])}

def short_forecast_node_wrapper(state: GraphState) -> Dict[str, Any]:
    """단기예보 노드 래퍼"""
    if not state.get("need_short_forecast", False):
        print("🧩 노드: 단기예보 스킵")
        return {"short_forecast_data": []}
    
    print("🧩 노드: 단기예보 데이터 수집")
    short_forecast_node = ShortForecastNode()
    result = short_forecast_node.run(state)
    # Annotated 필드만 반환
    return {"short_forecast_data": result.get("short_forecast_data", [])}

def mid_forecast_node_wrapper(state: GraphState) -> Dict[str, Any]:
    """중기예보 노드 래퍼"""
    if not state.get("need_mid_forecast", False):
        print("🧩 노드: 중기예보 스킵")
        return {"mid_forecast_data": []}
    
    print("🧩 노드: 중기예보 데이터 수집")
    mid_forecast_node = MidForecastNode()
    result = mid_forecast_node.run(state)
    # Annotated 필드만 반환
    return {"mid_forecast_data": result.get("mid_forecast_data", [])}

def parallel_execution_node(state: GraphState) -> Dict[str, Any]:
    """병렬 실행 노드 - LangGraph가 자동으로 병렬 처리"""
    print("🧩 노드: 병렬 실행 시작")
    
    need_advisory = state.get("need_advisory", False)
    need_short_forecast = state.get("need_short_forecast", False)
    need_mid_forecast = state.get("need_mid_forecast", False)
    
    print(f"   - 병렬 실행 대상: 특보={need_advisory}, 단기={need_short_forecast}, 중기={need_mid_forecast}")
    print("   - LangGraph가 자동으로 병렬 처리합니다")
    
    # 상태만 반환 (실제 병렬 처리는 LangGraph가 엣지를 통해 처리)
    return state

def combine_data_node(state: GraphState) -> Dict[str, Any]:
    """데이터 통합 노드"""
    print("🧩 노드: 데이터 통합")
    
    context = combine_weather_data(state)
    return {**state, "context": context}

def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """답변 생성 노드"""
    print("🧩 노드: 답변 생성")
    
    # 이미 답변이 있는 경우 (특별한 경우)
    if state.get("answer"):
        print("   - 기존 답변 반환")
        return state
    
    if not state.get("question"):
        raise ValueError("question 누락")
    if not state.get("context"):
        raise ValueError("context 누락")
    
    chain = (
        {"context": lambda x: x["context"], "question": lambda x: x["question"]}
        | ANSWER_PROMPT
        | make_llm()
        | StrOutputParser()
    )
    
    answer = chain.invoke({"context": state["context"], "question": state["question"]})
    time.sleep(1)  # API 요청 간격
    
    txt = re.sub(r'\n{3,}', '\n\n', answer or "").strip()
    
    # 자동 서울 처리 모드인 경우 추가 옵션 제공
    processing_mode = state.get("processing_mode", "")
    original_question = state.get("original_question", "")
    
    if processing_mode == "auto_seoul_with_options" and original_question:
        additional_options = f"""

💡 다른 지역의 날씨 정보도 필요하시다면 말씀해주세요!
예시: "부산 오늘 날씨 어때?", "대구 내일 날씨는?", "인천 이번주 날씨는?"

🏙️ 제공 가능한 지역:
- 서울/수도권, 부산, 대구, 인천, 광주, 대전, 울산, 세종
- 경기도, 강원도, 충청북도, 충청남도, 전라북도, 전라남도, 경상북도, 경상남도, 제주도

📅 날짜 옵션:
- 오늘, 내일, 모레, 3일 후, 4일 후, 5일 후, 6일 후, 7일 후/일주일 후"""
        
        txt += additional_options
        print("   - 추가 지역 옵션 제공")
    
    return {**state, "answer": txt}

def ask_missing_info_node(state: GraphState) -> Dict[str, Any]:
    """부족한 정보 요청 노드 - 통합 처리 (추가질문 처리 포함)"""
    print("🧩 노드: 부족한 정보 요청 및 추가질문 처리")
    
    original_question = state.get("original_question", state.get("question", ""))
    has_region_info = state.get("has_region_info", False)
    has_date_info = state.get("has_date_info", False)
    retry_count = state.get("retry_count", 0)
    
    # 재시도 3번 초과 시 fallback
    if retry_count >= 3:
        return {
            **state,
            "answer": "fallback_answer"  # fallback_answer 노드로 라우팅
        }
    
    # 추가질문 처리: 사용자가 새로운 질문을 했는지 확인
    current_question = state.get("question", "")
    if current_question != original_question and retry_count > 0:
        # 새로운 질문이 들어온 경우 - 지역/날짜 정보 추출
        print("   - 추가질문 감지, 정보 추출 중...")
        
        # 지역 정보 추출
        region_keywords = {
            "서울": "서울", "수도권": "서울", "경기": "경기",
            "부산": "부산", "대구": "대구", "인천": "인천",
            "광주": "광주", "대전": "대전", "울산": "울산",
            "세종": "세종", "강원": "강원", "충북": "충북",
            "충남": "충남", "전북": "전북", "전남": "전남",
            "경북": "경북", "경남": "경남", "제주": "제주"
        }
        
        # 날짜 정보 추출
        date_keywords = {
            "오늘": "오늘", "내일": "내일", "모레": "모레",
            "3일": "3일 후", "4일": "4일 후", "5일": "5일 후",
            "6일": "6일 후", "7일": "7일 후", "일주일": "7일 후", "주간": "7일 후"
        }
        
        detected_region = None
        detected_date = None
        
        # 지역 정보 추출
        for keyword, region in region_keywords.items():
            if keyword in current_question:
                detected_region = region
                break
        
        # 날짜 정보 추출
        for keyword, date in date_keywords.items():
            if keyword in current_question:
                detected_date = date
                break
        
        # 새로운 질문 구성
        enhanced_question = original_question
        
        if detected_region:
            enhanced_question = f"{detected_region} {enhanced_question}"
            print(f"   - 지역 감지: {detected_region}")
        
        if detected_date:
            enhanced_question = f"{enhanced_question.replace('날씨', detected_date + ' 날씨')}"
            print(f"   - 날짜 감지: {detected_date}")
        
        # 정보가 충분한지 확인
        has_region = detected_region or has_region_info
        has_date = detected_date or has_date_info
        
        if has_region and has_date:
            print(f"   - 새로운 질문으로 재처리: {enhanced_question}")
            return {
                **state,
                "question": enhanced_question,
                "has_region_info": True,
                "has_date_info": True,
                "retry_count": 0,  # 성공 시 재시도 카운터 리셋
                "answer": None  # 기존 답변 초기화
            }
    
    # 부족한 정보에 따른 질문 생성
    missing_parts = []
    if not has_region_info:
        missing_parts.append("지역")
    if not has_date_info:
        missing_parts.append("날짜")
    
    follow_up_question = f"""더 정확한 날씨 정보를 위해 {'/'.join(missing_parts)} 정보가 필요합니다.

현재 질문: "{original_question}"

"""
    
    if not has_region_info and not has_date_info:
        follow_up_question += """📍 지역과 날짜를 모두 선택해주세요:

🏙️ 지역 선택:
- 서울/수도권, 부산, 대구, 인천, 광주, 대전, 울산, 세종
- 경기도, 강원도, 충청북도, 충청남도, 전라북도, 전라남도, 경상북도, 경상남도, 제주도

📅 날짜 선택:
- 오늘, 내일, 모레, 3일 후, 4일 후, 5일 후, 6일 후, 7일 후/일주일 후

예시: "서울 오늘 날씨 어때?" 또는 "부산 내일 날씨는?" """
    
    elif not has_region_info:
        follow_up_question += """📍 지역을 선택해주세요:

🏙️ 지역 선택:
- 서울/수도권, 부산, 대구, 인천, 광주, 대전, 울산, 세종
- 경기도, 강원도, 충청북도, 충청남도, 전라북도, 전라남도, 경상북도, 경상남도, 제주도

예시: "서울" 또는 "부산" """
    
    elif not has_date_info:
        follow_up_question += """📅 날짜를 선택해주세요:

📅 날짜 선택:
- 오늘, 내일, 모레, 3일 후, 4일 후, 5일 후, 6일 후, 7일 후/일주일 후

예시: "오늘" 또는 "내일" """
    
    follow_up_question += f"\n\n선택해주시면 정확한 날씨 정보를 제공해드리겠습니다! 🎯 (시도 {retry_count + 1}/3)"
    
    return {
        **state,
        "answer": follow_up_question,
        "waiting_for_region": not has_region_info,
        "waiting_for_date": not has_date_info,
        "original_question": original_question,
        "retry_count": retry_count + 1
    }



def condition_check_node(state: GraphState) -> Dict[str, Any]:
    """조건분기 노드 - 자동 서울 처리 여부 확인"""
    print("🧩 노드: 조건분기 확인")
    
    auto_seoul_processed = state.get("auto_seoul_processed", False)
    
    if auto_seoul_processed:
        print("   - 자동 서울 처리됨: 추가질문 옵션 제공")
        return {
            **state,
            "should_ask_follow_up": True
        }
    else:
        print("   - 일반 처리: 바로 종료")
        return {
            **state,
            "should_ask_follow_up": False
        }

def follow_up_question_node(state: GraphState) -> Dict[str, Any]:
    """추가질문 노드 - Human-in-the-Loop로 사용자 입력 대기"""
    print("🧩 노드: 추가질문 처리 - 사용자 입력 대기")
    
    # 추가질문 옵션 제공
    follow_up_message = """

💡 다른 지역의 날씨 정보도 필요하시다면 말씀해주세요!
예시: "부산 오늘 날씨 어때?", "대구 내일 날씨는?", "인천 이번주 날씨는?"

🏙️ 제공 가능한 지역:
- 서울/수도권, 부산, 대구, 인천, 광주, 대전, 울산, 세종
- 경기도, 강원도, 충청북도, 충청남도, 전라북도, 전라남도, 경상북도, 경상남도, 제주도

📅 날짜 옵션:
- 오늘, 내일, 모레, 3일 후, 4일 후, 5일 후, 6일 후, 7일 후/일주일 후"""
    
    return {
        **state,
        "answer": state.get("answer", "") + follow_up_message,
        "waiting_for_follow_up": True,
        "interrupt_before": ["analyze_follow_up"]  # Human-in-the-Loop: 사용자 입력 대기
    }

def analyze_follow_up_node(state: GraphState) -> Dict[str, Any]:
    """추가질문 분석 노드 - LLM으로 질문 판별"""
    print("🧩 노드: 추가질문 분석")
    
    # Human-in-the-Loop에서 받은 사용자 입력
    user_question = state.get("question", "")
    
    if not user_question:
        print("   - 사용자 입력이 없습니다.")
        return {
            **state,
            "is_weather_related": False,
            "follow_up_analysis": "사용자 입력 없음"
        }
    
    print(f"   - 사용자 추가질문: {user_question}")
    
    # LLM으로 추가질문 분석
    follow_up_analysis_prompt = ChatPromptTemplate.from_template(
        """다음은 사용자가 날씨 에이전트에게 한 추가 질문입니다.

사용자 추가 질문: {user_question}

이 추가 질문이 날씨 관련 질문인지 판단하세요.

다음 JSON 형식으로 답변하세요:
{{
    "is_weather_related": true/false,
    "reason": "판단 이유"
}}

판단 기준:
- is_weather_related: 날씨, 기상, 온도, 강수, 바람 등 날씨 관련 키워드가 포함된 질문
- reason: 판단 이유

답변:"""
    )
    
    chain = (
        {"user_question": lambda x: x["user_question"]}
        | follow_up_analysis_prompt
        | make_llm()
        | StrOutputParser()
    )
    
    analysis = chain.invoke({"user_question": user_question})
    
    # JSON 파싱
    import json
    try:
        analysis_data = json.loads(analysis)
        is_weather_related = analysis_data.get("is_weather_related", False)
        reason = analysis_data.get("reason", "")
    except:
        is_weather_related = False
        reason = "파싱 실패"
    
    print(f"   - 날씨 관련: {is_weather_related}")
    print(f"   - 이유: {reason}")
    
    return {
        **state,
        "is_weather_related": is_weather_related,
        "follow_up_analysis": analysis
    }

def fallback_answer_node(state: GraphState) -> Dict[str, Any]:
    """Fallback 답변 노드"""
    print("🧩 노드: Fallback 답변")
    
    fallback_message = """정확한 날짜와 지역이 없어서 답변을 내지 못합니다.

다시 시도해주시거나 구체적인 지역명과 날짜를 포함해서 질문해주세요.
예시: "서울 오늘 날씨 어때?", "부산 내일 날씨는?" 등"""
    
    return {
        **state,
        "answer": fallback_message
    }


# 그래프 빌드
def build_graph():
    """그래프 빌드 (병렬 처리 포함)"""
    g = StateGraph(GraphState)
    
    # 노드 추가
    g.add_node("analyze_question", analyze_question_node)
    g.add_node("ask_missing_info", ask_missing_info_node)
    g.add_node("advisory", advisory_node_wrapper)
    g.add_node("short_forecast", short_forecast_node_wrapper)
    g.add_node("mid_forecast", mid_forecast_node_wrapper)
    g.add_node("parallel_execution", parallel_execution_node)
    g.add_node("combine_data", combine_data_node)
    g.add_node("generate_answer", generate_answer_node)
    g.add_node("condition_check", condition_check_node)
    g.add_node("follow_up_question", follow_up_question_node)
    g.add_node("analyze_follow_up", analyze_follow_up_node)
    g.add_node("fallback_answer", fallback_answer_node)
    
    # 엣지 설정
    g.set_entry_point("analyze_question")
    
    # 단순하고 명확한 워크플로우
    def route_after_analysis(state: GraphState) -> str:
        """분석 후 라우팅 함수"""
        need_advisory = state.get("need_advisory", False)
        need_short_forecast = state.get("need_short_forecast", False)
        need_mid_forecast = state.get("need_mid_forecast", False)
        has_region_info = state.get("has_region_info", False)
        has_date_info = state.get("has_date_info", False)
        retry_count = state.get("retry_count", 0)
        
        # 정보 부족 시 추가 질문
        if (not has_region_info or not has_date_info) and retry_count < 3:
            return "ask_missing_info"
        
        # 데이터 수집 결정
        needed_nodes = sum([need_advisory, need_short_forecast, need_mid_forecast])
        
        if needed_nodes == 0:
            return "combine_data"
        elif needed_nodes == 1:
            if need_advisory:
                return "advisory"
            elif need_short_forecast:
                return "short_forecast"
            elif need_mid_forecast:
                return "mid_forecast"
        else:
            return "parallel_execution"
    
    def route_after_ask_missing_info(state: GraphState) -> str:
        """부족한 정보 요청 후 라우팅 함수"""
        answer = state.get("answer", "")
        retry_count = state.get("retry_count", 0)
        
        # fallback_answer인 경우 fallback_answer 노드로
        if answer == "fallback_answer":
            return "fallback_answer"
        # 새로운 질문으로 재처리하는 경우 (재시도 횟수 제한)
        elif answer is None and retry_count < 3:  # 새로운 질문으로 재처리
            return "analyze_question"
        else:
            # 재시도 횟수 초과 또는 일반적인 경우 fallback
            return "fallback_answer"
    
    def route_after_condition_check(state: GraphState) -> str:
        """조건분기 후 라우팅 함수"""
        should_ask_follow_up = state.get("should_ask_follow_up", False)
        
        if should_ask_follow_up:
            return "follow_up_question"
        else:
            return "END"
    
    def route_after_analyze_follow_up(state: GraphState) -> str:
        """추가질문 분석 후 라우팅 함수"""
        is_weather_related = state.get("is_weather_related", False)
        
        if is_weather_related:
            return "analyze_question"  # 날씨 관련 질문이면 재처리
        else:
            return "END"  # 다른 질문이면 supervisor로 전달
    
    # 조건부 엣지 설정
    g.add_conditional_edges(
        "analyze_question",
        route_after_analysis,
        {
            "advisory": "advisory",
            "short_forecast": "short_forecast", 
            "mid_forecast": "mid_forecast",
            "parallel_execution": "parallel_execution",
            "combine_data": "combine_data",
            "ask_missing_info": "ask_missing_info"
        }
    )
    
    # 단일 노드 실행 후 데이터 통합
    g.add_edge("advisory", "combine_data")
    g.add_edge("short_forecast", "combine_data")
    g.add_edge("mid_forecast", "combine_data")
    
    # 병렬 실행에서 각 노드로 분기
    g.add_edge("parallel_execution", "advisory")
    g.add_edge("parallel_execution", "short_forecast")
    g.add_edge("parallel_execution", "mid_forecast")
    
    # 답변 생성 후 조건분기
    g.add_edge("combine_data", "generate_answer")
    g.add_edge("generate_answer", "condition_check")
    
    # 조건분기 후 라우팅
    g.add_conditional_edges(
        "condition_check",
        route_after_condition_check,
        {
            "follow_up_question": "follow_up_question",
            "END": END
        }
    )
    
    # 추가질문 처리 후 분석
    g.add_edge("follow_up_question", "analyze_follow_up")
    
    # 추가질문 분석 후 라우팅
    g.add_conditional_edges(
        "analyze_follow_up",
        route_after_analyze_follow_up,
        {
            "analyze_question": "analyze_question",
            "END": END
        }
    )
    
    # 부족한 정보 요청 후 조건부 라우팅
    g.add_conditional_edges(
        "ask_missing_info",
        route_after_ask_missing_info,
        {
            "fallback_answer": "fallback_answer",
            "analyze_question": "analyze_question"
        }
    )
    
    # Fallback 답변 후 종료
    g.add_edge("fallback_answer", END)
    
    app = g.compile()
    try:
        graph_image_path = "agent_workflow_v3.png"
        with open(graph_image_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류: {e}")
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
        
        # 그래프 실행 (Human-in-the-Loop 지원)
        config = {"configurable": {"thread_id": "weather_agent_session"}}
        
        # 첫 번째 실행
        result = app.invoke({"question": query}, config=config)
        
        # Human-in-the-Loop 처리
        while result.get("waiting_for_follow_up", False):
            print(f"[날씨_agent] 사용자 추가질문 대기 중...")
            print(f"[날씨_agent] 현재 답변: {result.get('answer', '')}")
            
            # 사용자 입력 받기 (실제 구현에서는 외부에서 받아야 함)
            follow_up_question = input("\n추가 질문을 입력하세요 (엔터만 누르면 종료): ").strip()
            
            if not follow_up_question:
                print("[날씨_agent] 추가질문 없이 종료")
                break
            
            # 추가질문으로 그래프 재실행
            result = app.invoke({"question": follow_up_question}, config=config)
        
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
