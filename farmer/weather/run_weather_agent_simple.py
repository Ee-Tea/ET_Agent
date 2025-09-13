# -*- coding: utf-8 -*-
"""
단순화된 날씨 에이전트 실행 파일
- 시/군/구만 지원 (도 단위 안내/차단 로직 제거)
- 지역 미지정 시 서울 + 오늘 기본값
"""

# =========[ 표준/외부 라이브러리 ]=========
import os
import sys
import re
import time
import json
from typing import TypedDict, Optional, Any, Dict, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# 프로젝트 루트를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

# =========[ LangChain / LangGraph / LLM ]=========
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# =========[ 외부 서비스 ]=========
from tavily import TavilyClient

# =========[ 지연 로딩을 위한 전역 변수 ]=========
_weather_app = None
_llm_instance = None
_tavily_client = None

# 모듈화된 노드들 import (절대 import)
from farmer.weather.advisory_node import AdvisoryNode
from farmer.weather.short_forecast_node import ShortForecastNode
from farmer.weather.mid_forecast_node import MidForecastNode
from farmer.weather.utils import combine_weather_data, REGION_extract_datetime_from_question, REGION_extract_date_range_from_question, match_region_with_default

load_dotenv()

# 환경 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Tavily 웹검색 설정
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
KST = ZoneInfo("Asia/Seoul")

# 시/군/구 키워드 (필요하면 계속 추가)
CITY_KEYWORDS = [
    "서울","부산","대구","인천","광주","대전","울산","세종",
    "수원","성남","용인","고양","부천","안산","남양주","의정부","화성","평택","파주","김포","광명","시흥",
    "군포","하남","오산","이천","안성","의왕","구리","과천","여주","양주",
    "춘천","강릉","청주","천안","전주","광양","여수","순천","목포","익산","군산",
    "창원","진주","통영","거제","사천","밀양","양산","포항","구미","안동",
    "제주","서귀포"
]

def _normalize_question(q: str) -> str:
    return re.sub(r'[\/\s]+$', '', (q or '').strip())

def _extract_city(text: str) -> Optional[str]:
    if not text:
        return None
    for c in CITY_KEYWORDS:
        if c in text:
            return c
    return None

def _parse_kst_maybe(s: str) -> Optional[datetime]:
    """여러 포맷 시도 → KST datetime"""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    fmts = [
        "%Y-%m-%d %H:%M KST",
        "%Y-%m-%d %H:%M",
        "%Y%m%d%H%M",
        "%Y/%m/%d %H:%M",
        "%Y년 %m월 %d일 %H:%M"
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            return dt.replace(tzinfo=KST)
        except Exception:
            pass
    return None

# 그래프 상태
class GraphState(TypedDict):
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    advisory_data: Optional[list]
    short_forecast_data: Optional[list]
    mid_forecast_data: Optional[list]
    need_advisory: Optional[bool]
    need_short_forecast: Optional[bool]
    need_mid_forecast: Optional[bool]
    analysis_result: Optional[str]
    has_region_info: Optional[bool]
    has_date_info: Optional[bool]
    default_region: Optional[str]
    target_region: Optional[str]
    processing_mode: Optional[str]
    context_ok: Optional[bool]
    web_context: Optional[str]
    question_date: Optional[datetime]
    question_date_range: Optional[Tuple[datetime, datetime, bool]]
    is_weekly_request: Optional[bool]
    region_from_default: Optional[bool]

def _get_llm() -> ChatOpenAI:
    """LLM 인스턴스를 지연 로딩으로 가져오기"""
    global _llm_instance
    if _llm_instance is None:
        print("🤖 LLM 모듈 로딩 중...")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")
        _llm_instance = ChatOpenAI(
            model_name=OPENAI_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
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

def _get_simple_analysis_prompt():
    """단순 분석 프롬프트를 지연 로딩으로 가져오기"""
    return ChatPromptTemplate.from_template(
    """다음 질문을 분석해서 어떤 기상 데이터가 필요한지 판단하세요.

질문: {question}

[날짜 정보]
- "오늘", "내일", "모레", "현재", "지금" → has_date_info: true
- 구체적인 날짜(예: "9월 11일", "내일 오후") → has_date_info: true
- 날짜 관련 단어가 전혀 없으면 → has_date_info: false
- 날짜 계산 시 현재 날짜를 기준으로 정확히 계산
- 이번 주는 일요일~토요일 단위로 계산
  (예: 이번 주가 9월 7일~9월 13일이면, "이번 주" → 9월 7일~9월 13일 전체 포함)
- "이번 주"라고 하면 오늘이 며칠이든 이번 주 토요일까지의 모든 데이터 포함
- "다음 주"라고 하면 다음 주 일요일~토요일 전체 포함

[예보 종류]
- 1~3일 이내 → 단기예보
- 4일 이후 → 중기예보
- "이번 주 주간 예보", "다음 주" 같은 포괄 질문 → 단기 + 중기 + 기상특보 전부 포함

[기상특보 판단 기준]
- "특보", "경보", "주의보", "기상특보" 등의 단어가 있으면 need_advisory: true
- "날씨", "기상", "예보" 등의 일반적인 질문이면 need_advisory: true (현재 발효 중인 특보가 있을 수 있음)
- "기온", "온도"만 물어보면 need_advisory: false

지역 정보 판단 기준:
- 구체적인 도시명(서울, 부산, 대구 등)이 있으면 has_region_info: true
- "수도권", "경기" 등 지역명이 있으면 has_region_info: true
- 지역 관련 단어가 전혀 없으면 has_region_info: false

반드시 다음 JSON 형식으로만 답변하세요 (다른 텍스트 없이):
{{
    "need_advisory": true,
    "need_short_forecast": true,
    "need_mid_forecast": false,
    "has_region_info": false,
    "has_date_info": false,
    "default_region": "서울",
    "reason": "판단 이유"
}}
"""
    )

def _get_answer_prompt():
    """답변 생성 프롬프트를 지연 로딩으로 가져오기"""
    return ChatPromptTemplate.from_template(

    """기상청 예보관으로서 다음 데이터를 바탕으로 종합적인 답변을 작성하세요.

현재 날짜: {current_date}

질문: {question}

[기상 데이터]
{context}

답변 구조:
순번) 개요: 핵심 요약
순번) 기상특보 현황: (컨텍스트에 [live_advisory] 데이터가 있으면 반드시 포함)
순번) 단기예보: (컨텍스트에 [live_forecast] 데이터가 있으면 반드시 포함)
순번) 중기예보: (컨텍스트에 [mid_forecast] 데이터가 있으면 반드시 포함)
순번) 종합 요약

주의사항:
- 지역 정보가 명확하지 않은 경우, 서울/수도권 기준으로 답변하세요
- 데이터가 없는 섹션은 아예 생략하세요. "없습니다"라고 표시하지 마세요
- **중요**: 컨텍스트에 [live_advisory] 데이터가 있으면 기상특보 현황 섹션을 반드시 포함하세요
- **중요**: 컨텍스트에 [live_forecast] 데이터가 있으면 단기예보 섹션을 반드시 포함하세요
- 단기예보는 나열식 불릿이 아니라, 날짜와 시간대를 연결하여 자연스러운 문장 형태로 서술하세요. 대신 날짜별로 줄바꿈도 해야함
  (예: "12일 오전에는 구름이 많고 강수 확률은 20%입니다. 오후에는 흐리고 기온은 28도로 올라가며 비가 올 확률은 60%입니다.")
- 날짜 계산은 현재 날짜를 기준으로 정확하게 계산하세요
  (예: 이번 주가 일요일~토요일이면 "이번 주" → 해당 주간 전체 포함,
       다음 주는 다음 주의 일요일~토요일 전체 포함)
- "이번 주" 또는 "다음 주"와 같은 주간 예보 요청은 포괄적으로 판단하세요
  → 단기예보와 중기예보, 기상특보를 함께 사용해야 합니다
- 주간 예보 요청 시 해당 기간의 모든 데이터를 빠짐없이 나열하듯이 답변하세요
- 발표 예정인 특보는 반드시 "발표 예정" 또는 "예정"이라고 명시하세요
- **절대 금지**: 데이터가 없을 경우 "없음", "데이터가 없습니다" 등의 문구를 출력하지 말고, 그 섹션 자체를 출력하지 마세요.
- **중요**: 데이터가 없는 섹션은 아예 제외하세요
- **중요**: 불릿 구조는 반드시 유지하세요 (번호 대신 - 로 구분)
- **중요**: 기상특보 데이터가 없으면 "기상특보 현황" 섹션을 아예 제외하세요
- **중요**: 중기예보 데이터가 없으면 "중기예보" 섹션을 아예 제외하세요
- **중요**: 단기예보 데이터가 없으면 "단기예보" 섹션을 아예 제외하세요
- 웹검색 결과가 있으면: 기상청 공식 데이터를 우선하되, 웹검색 결과의 최신 정보도 참고하세요
- 마크다운 문법을 사용하지 말고 일반 텍스트로 작성하세요
- 답변 끝에 안내 문구는 추가하지 마세요

답변:"""
)

# 1) 질문 분석
def simple_analyze_question_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 질문 분석 (단순화)")
    question = state.get("question", "")
    if not question:
        raise ValueError("question 누락")

    question = _normalize_question(question)

    chain = (
        {"question": lambda x: x["question"]}
        | _get_simple_analysis_prompt()
        | _get_llm()
        | StrOutputParser()
    )
    analysis = chain.invoke({"question": question})

    # JSON 파싱
    m = re.search(r'\{.*\}', analysis, re.DOTALL)
    js = m.group() if m else analysis
    try:
        data = json.loads(js)
        def b(x, dv=False): return bool(x) if isinstance(x, bool) else dv
        need_advisory = b(data.get("need_advisory"), True)
        need_short_forecast = b(data.get("need_short_forecast"), True)
        need_mid_forecast = b(data.get("need_mid_forecast"), False)
        has_date_info = b(data.get("has_date_info"), False)
        default_region = data.get("default_region", "서울")
        print("   - LLM 분석 성공")
    except Exception as e:
        print(f"   - JSON 파싱 실패: {e} → 기본값 사용")
        need_advisory = True
        need_short_forecast = True
        need_mid_forecast = False
        has_date_info = False
        default_region = "서울"

    # 시/군/구 추출
    city = _extract_city(question)
    if city:
        target_region = city
        has_region_info = True
        region_from_default = False   # ✅ 직접 입력된 지역
    else:
        print("   - 지역 정보 없음: 기본값 '서울' 적용")
        target_region = "서울"
        has_region_info = False
        region_from_default = True    # ✅ 자동 기본값

    # 날짜 기본값/범위 처리
    enhanced_question = question
    date_range = REGION_extract_date_range_from_question(enhanced_question)
    if date_range:
        sdt, edt, is_week = date_range
        has_date_info = True
        is_weekly_request = bool(is_week)
        print(f"   - 추출된 기간: {sdt.strftime('%Y-%m-%d')} ~ {edt.strftime('%Y-%m-%d')} (주간요청={is_weekly_request})")
    else:
        # 없으면 '오늘' 주입
        enhanced_question = f"{enhanced_question} 오늘".strip()
        has_date_info = True
        is_weekly_request = False
        # 단일 날짜도 state에 보관
        sdt = REGION_extract_datetime_from_question(enhanced_question)
        edt = sdt
        date_range = (sdt, edt, False) if sdt else None
        if sdt:
            print(f"   - 단일 날짜 추출: {sdt.strftime('%Y-%m-%d')}")

    # 주간 요청이면 중기예보도 필요 플래그 보정
    if date_range:
        start_days_from_now = (sdt.date() - datetime.now(tz=KST).date()).days
        end_days_from_now = (edt.date() - datetime.now(tz=KST).date()).days
        if end_days_from_now >= 4:
            need_mid_forecast = True

    # 현재 시간 출력
    now = datetime.now(tz=KST)
    print(f"   - 현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S KST')}")

    print(f"   - 최종 질문: {enhanced_question}")
    print(f"   - 타깃 지역(시): {target_region}")
    print(f"   - 분석 결과: 특보={need_advisory}, 단기={need_short_forecast}, 중기={need_mid_forecast}")

    return {
        **state,
        "question": enhanced_question,
        "question_date": sdt,
        "question_date_range": date_range,
        "is_weekly_request": is_weekly_request,
        "need_advisory": need_advisory,
        "need_short_forecast": need_short_forecast,
        "need_mid_forecast": need_mid_forecast,
        "has_region_info": has_region_info,
        "has_date_info": has_date_info,
        "default_region": target_region,
        "target_region": target_region,
        "region_from_default": region_from_default,  # ✅ 추가
        "processing_mode": "auto_defaults",
        "analysis_result": analysis,
        "advisory_data": [],
        "short_forecast_data": [],
        "mid_forecast_data": []
    }

# 2) 특보 래퍼
def advisory_node_wrapper(state: GraphState) -> Dict[str, Any]:
    if not state.get("need_advisory", False):
        print("🧩 노드: 기상특보 스킵")
        return {"advisory_data": []}

    print("🧩 노드: 기상특보 데이터 수집")
    advisory_node = AdvisoryNode()
    result = advisory_node.run(state)

    target_region = state.get("target_region", "서울")
    now = datetime.now(tz=KST)
    date_range = state.get("question_date_range")

    filtered = []
    for it in result.get("advisory_data", []) or []:
        if not isinstance(it, dict):
            continue
        try:
            j = json.loads(it.get("json", "{}"))
            name = j.get("region_name", "")
            end_s = j.get("window_end_kst") or ""
            start_s = j.get("window_start_kst") or ""
            cmd_name = j.get("command_name", "")

            # 시간/해제 방어
            end_dt = _parse_kst_maybe(end_s)
            start_dt = _parse_kst_maybe(start_s)
            if end_dt and end_dt < now:
                continue
            if "해제" in cmd_name:
                continue

            # ✅ 지역 매칭 (공통 유틸 함수 사용)
            if match_region_with_default(target_region, name, state.get("region_from_default", False)):
                filtered.append(it)

            # ✅ 기간 매칭
            if date_range:
                sdt, edt, _ = date_range
                ok = True
                if start_dt and end_dt:
                    ok = not (end_dt < sdt or start_dt > edt)
                elif start_dt and not end_dt:
                    ok = (start_dt <= edt)
                elif end_dt and not start_dt:
                    ok = (end_dt >= sdt)
                if not ok:
                    continue

            filtered.append(it)
        except Exception:
            pass

    # ✅ 서울일 경우 우선순위 정렬
    if target_region == "서울" and filtered:
        def _rank(it):
            n = json.loads(it["json"]).get("region_name", "")
            if "서울" in n: return (0, n)
            if "경기" in n: return (1, n)
            if "인천" in n and not any(x in n for x in ["서해5도", "백령"]): return (2, n)
            if any(x in n for x in ["서해5도", "백령"]): return (3, n)
            return (4, n)
        filtered.sort(key=_rank)

    print(f"   - {target_region} 지역 필터링: {len(filtered)}개")
    return {"advisory_data": filtered}

# 3) 단기예보 래퍼
def short_forecast_node_wrapper(state: GraphState) -> Dict[str, Any]:
    if not state.get("need_short_forecast", False):
        print("🧩 노드: 단기예보 스킵")
        return {"short_forecast_data": []}

    print("🧩 노드: 단기예보 데이터 수집")
    short_forecast_node = ShortForecastNode()
    result = short_forecast_node.run(state)

    target_region = state.get("target_region", "서울")
    raw = result.get("short_forecast_data", []) or []
    filtered = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        try:
            j = json.loads(it.get("json", "{}"))
            name = j.get("region_name", "")

            # ✅ 지역 매칭
            if match_region_with_default(target_region, name, state.get("region_from_default", False)):
                filtered.append(it)
        except Exception:
            pass

    # ✅ 서울일 경우 우선순위 + 날짜 필터링
    if target_region == "서울" and filtered:
        def _rank(it):
            n = json.loads(it["json"]).get("region_name", "")
            if "서울" in n: return (0, n)
            if "경기" in n: return (1, n)
            if "인천" in n and not any(x in n for x in ["서해5도", "백령"]): return (2, n)
            if any(x in n for x in ["서해5도", "백령"]): return (3, n)
            return (4, n)
        filtered.sort(key=_rank)

        # ✅ 날짜 범위 필터링
        date_range = state.get("question_date_range")
        if date_range:
            sdt, edt, _ = date_range
            start_str, end_str = sdt.strftime("%Y-%m-%d"), edt.strftime("%Y-%m-%d")
            date_filtered = []
            for it in filtered:
                try:
                    j = json.loads(it["json"])
                    eff = j.get("effective_time", "")
                    d = eff.split(" ")[0] if " " in eff else eff
                    if start_str <= d <= end_str:
                        date_filtered.append(it)
                except:
                    date_filtered.append(it)
            if date_filtered:
                filtered = date_filtered
                print(f"   - 날짜 범위 필터링: {len(filtered)}개")

    print(f"   - 최종 데이터: {len(filtered)}개")
    return {"short_forecast_data": filtered}

# 4) 중기예보 래퍼
def mid_forecast_node_wrapper(state: GraphState) -> Dict[str, Any]:
    if not state.get("need_mid_forecast", False):
        print("🧩 노드: 중기예보 스킵")
        return {"mid_forecast_data": []}

    print("🧩 노드: 중기예보 데이터 수집")
    mid_forecast_node = MidForecastNode()
    result = mid_forecast_node.run(state)

    target_region = state.get("target_region", "서울")
    raw = result.get("mid_forecast_data", []) or []
    filtered = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        try:
            j = json.loads(it.get("json", "{}"))
            name = j.get("region_name", "")

            # ✅ 지역 매칭
            if match_region_with_default(target_region, name, state.get("region_from_default", False)):
                filtered.append(it)
        except Exception:
            pass

    print(f"   - {target_region} 지역 필터링: {len(filtered)}개 (전체: {len(raw)}개)")
    return {"mid_forecast_data": filtered}

# 5) 병렬 실행 노드
def parallel_execution_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 병렬 실행 시작")
    na = state.get("need_advisory", False)
    ns = state.get("need_short_forecast", False)
    nm = state.get("need_mid_forecast", False)
    
    print(f"   - 기상특보: {'실행' if na else '스킵'}")
    print(f"   - 단기예보: {'실행' if ns else '스킵'}")
    print(f"   - 중기예보: {'실행' if nm else '스킵'}")
    
    return {**state}

# 6) 데이터 통합
def combine_data_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 데이터 통합")
    
    # 병렬 처리된 데이터들이 모두 준비되었는지 확인
    advisory_data = state.get("advisory_data", [])
    short_forecast_data = state.get("short_forecast_data", [])
    mid_forecast_data = state.get("mid_forecast_data", [])
    
    print(f"   - 기상특보: {len(advisory_data)}개")
    print(f"   - 단기예보: {len(short_forecast_data)}개") 
    print(f"   - 중기예보: {len(mid_forecast_data)}개")
    
    context = combine_weather_data(state)
    
    
    return {**state, "context": context}

# 7) 컨텍스트 결합 (웹검색 후)
def combine_context_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 컨텍스트 결합")
    context = state.get("context", "")
    web_context = state.get("web_context", "")

    # ✅ 웹검색 결과가 있으면 결합, 없으면 기상청 데이터만
    if web_context and web_context not in ["[웹검색 결과 없음]", ""]:
        print("   - 기상 데이터와 웹검색 결과를 결합합니다.")
        final_context = f"[기상청 공식 데이터]\n{context}\n\n[웹검색 결과]\n{web_context}"
    else:
        print("   - 기상청 공식 데이터만 사용합니다.")
        final_context = context

    return {**state, "context": final_context}


# 8) 컨텍스트 충분성 판단 및 라우팅
def assess_and_route_context_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 컨텍스트 충분성 판단 및 라우팅")
    question = state.get("question", "")
    context = state.get("context", "")

    if not context or context == "NO_DATA_AVAILABLE":
        print("   - ❌ 컨텍스트 없음")
        return {**state, "context_ok": False, "context": "NO_DATA_AVAILABLE"}

    prompt = ChatPromptTemplate.from_template(
        """질문과 컨텍스트만으로 충분히 구체적인 답변을 작성할 수 있는지 판단하세요.

질문: {question}
컨텍스트: {context}

판단 기준:
- 날씨 데이터가 있으면 context_ok: true
- 지역과 날짜 정보가 명확하면 context_ok: true
- 기온, 하늘상태, 강수확률 등이 있으면 context_ok: true

JSON만:
{{"context_ok": true/false, "reason": "한 줄 이유"}}
"""
)

    chain = (
        {"question": lambda x: x["question"], "context": lambda x: x["context"]}
        | prompt
        | _get_llm()
        | StrOutputParser()
    )

    raw = chain.invoke({"question": question, "context": context})
    
    try:
        data = json.loads(raw)
        ok = bool(data.get("context_ok", False))
        reason = data.get("reason", "이유 없음")
        print(f"   - 컨텍스트 충분성: {ok} ({reason})")
    except Exception as e:
        print(f"   - JSON 파싱 실패: {e}")
        ok = False

    if ok:
        print("   - ✅ 충분함 → 답변 생성으로 이동")
    else:
        print("   - ⚠️ 불충분함 → 웹검색으로 이동")

    return {**state, "context_ok": ok}

def route_context_sufficiency(state: GraphState) -> str:
    context_ok = state.get("context_ok", False)
    return "combine_context" if context_ok else "web_search"

# 8) 웹검색 노드
def web_search_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 웹검색")
    question = state.get("question", "")
    target_region = state.get("target_region", "서울")
    date_range = state.get("question_date_range")
    date_str = ""
    if date_range:
        sdt, edt, _ = date_range
        date_str = f"{sdt.strftime('%Y-%m-%d')}~{edt.strftime('%Y-%m-%d')}"
    try:
        search_query = f"{target_region} {date_str} 날씨 기상예보"
        print(f"   - 웹검색 쿼리: {search_query}")
        tavily = _get_tavily_client()
        results = tavily.search(query=search_query, max_results=TAVILY_MAX_RESULTS)
        if not results or not results.get("results"):
            print("   - ⚠️ 웹검색 결과가 없습니다.")
            return {**state, "web_context": "[웹검색 결과 없음]"}
        web_context = "\n\n".join([
            f"- 출처: {res['url']}\n 내용: {res['content']}" 
            for res in results['results']
        ]) or "검색 결과를 찾지 못했습니다."
        print(f"   - ✅ 웹검색 완료: {len(results['results'])}개 결과")
        return {**state, "web_context": web_context}
    except Exception as e:
        print(f"   - ❌ 웹검색 중 오류 발생: {e}")
        return {**state, "web_context": f"[웹검색 실패: {e}]"}

# 9) 컨텍스트 결합 노드
def combine_context_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 컨텍스트 결합")
    context = state.get("context", "")
    web_context = state.get("web_context", "")
    
    if web_context and web_context not in ["[웹검색 결과 없음]", ""]:
        print("   - 기상 데이터와 웹검색 결과를 결합합니다.")
        final_context = f"[기상청 공식 데이터]\n{context}\n\n[웹검색 결과]\n{web_context}"
    else:
        print("   - 기상청 공식 데이터만 사용합니다.")
        final_context = context
    
    return {**state, "context": final_context}

# 10) 답변 생성
def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 답변 생성")

    if not state.get("question"):
        raise ValueError("question 누락")
    if not state.get("context"):
        raise ValueError("context 누락")

    if state["context"] == "NO_DATA_AVAILABLE":
        print("   - 데이터 없음: 적절한 메시지 반환")
        return {
            **state,
            "answer": "죄송합니다. 요청하신 날짜와 지역에 대한 기상 데이터를 찾을 수 없습니다. 다른 날짜나 지역으로 다시 문의해주세요."
        }

    current_date = datetime.now(tz=KST).strftime("%Y년 %m월 %d일")

    chain = (
        {"context": lambda x: x["context"], "question": lambda x: x["question"], "current_date": lambda x: current_date}
        | _get_answer_prompt()
        | _get_llm()
        | StrOutputParser()
    )

    answer = chain.invoke({"context": state["context"], "question": state["question"], "current_date": current_date})
    time.sleep(1)
    txt = re.sub(r'\n{3,}', '\n\n', answer or "").strip()
    print("   - 답변 생성 완료")
    return {**state, "answer": txt}

# 그래프 빌드
def build_simple_graph():
    g = StateGraph(GraphState)

    g.add_node("analyze_question", simple_analyze_question_node)
    g.add_node("parallel_execution", parallel_execution_node)
    g.add_node("advisory", advisory_node_wrapper)
    g.add_node("short_forecast", short_forecast_node_wrapper)
    g.add_node("mid_forecast", mid_forecast_node_wrapper)
    g.add_node("combine_data", combine_data_node)
    g.add_node("assess_and_route_context", assess_and_route_context_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate_answer", generate_answer_node)

    g.set_entry_point("analyze_question")

    def route_after_analysis(state: GraphState) -> str:
        na = state.get("need_advisory", False)
        ns = state.get("need_short_forecast", False)
        nm = state.get("need_mid_forecast", False)
        needed = sum([1 if na else 0, 1 if ns else 0, 1 if nm else 0])
        
        if needed == 0:
            return "combine_data"
        elif needed == 1:
            if na: return "advisory"
            if ns: return "short_forecast"
            if nm: return "mid_forecast"
        else:
            return "parallel_execution"

    g.add_conditional_edges(
        "analyze_question",
        route_after_analysis,
        {
            "advisory": "advisory",
            "short_forecast": "short_forecast", 
            "mid_forecast": "mid_forecast",
            "parallel_execution": "parallel_execution",
            "combine_data": "combine_data",
        }
    )

    g.add_edge("parallel_execution", "advisory")
    g.add_edge("parallel_execution", "short_forecast")
    g.add_edge("parallel_execution", "mid_forecast")

    g.add_edge("advisory", "combine_data")
    g.add_edge("short_forecast", "combine_data")
    g.add_edge("mid_forecast", "combine_data")

    g.add_edge("combine_data", "assess_and_route_context")
    
    # 웹검색 후 컨텍스트 결합 노드 추가
    g.add_node("combine_context", combine_context_node)
    
    g.add_conditional_edges(
        "assess_and_route_context",
        route_context_sufficiency,
        {"combine_context": "combine_context", "web_search": "web_search"}
    )
    
    g.add_edge("web_search", "combine_context")
    g.add_edge("combine_context", "generate_answer")
    g.add_edge("generate_answer", END)

    app = g.compile()
    # try:
    #     graph_image_path = "agent_workflow_simple.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(app.get_graph().draw_mermaid_png())
    #     print(f"\n단순화된 LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류: {e}")
    return app

def _get_weather_app():
    """날씨 에이전트 애플리케이션을 지연 로딩으로 가져오기"""
    global _weather_app
    if _weather_app is None:
        print("🌤️ 날씨_agent 모듈 로딩 중...")
        _weather_app = build_simple_graph()
        print("✅ 날씨_agent 모듈 로딩 완료")
    return _weather_app

# OchestratorTest.py 호환 함수
def run(state: dict) -> dict:
    try:
        query = state.get("query", "")
        if not query:
            return {"agent_answer": "질문이 제공되지 않았습니다. 날씨 관련 질문을 해주세요."}

        print(f"[날씨_agent_simple] 질문 처리 시작: {query}")
        app = _get_weather_app()
        result = app.invoke({"question": query})
        answer = result.get("answer", "답변을 생성할 수 없습니다.")
        print(f"[날씨_agent_simple] 답변 생성 완료: {len(answer)}자")
        return {"agent_answer": answer}

    except Exception as e:
        error_msg = f"날씨 에이전트 실행 중 오류가 발생했습니다: {e}"
        print(f"[날씨_agent_simple] 오류: {e}")
        return {"agent_answer": error_msg}

# 메인 실행부
def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="단순화된 기상 전문가 그래프")
    parser.add_argument("-q", "--question", default=None, help="질문 1회 실행 후 종료")
    parser.add_argument("--show-context", action="store_true", help="컨텍스트(근거) 출력")
    args = parser.parse_args()

    print("💬 단순화된 기상 전문가 그래프")
    app = _get_weather_app()

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
    print("=== 단순화된 날씨 에이전트 시작 ===")
    main()
