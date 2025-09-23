# -*- coding: utf-8 -*-
"""
단순화된 날씨 에이전트 실행 파일
- 시/군/구만 지원 (도 단위 안내/차단 로직 제거)
- 지역 미지정 시 서울 + 오늘 기본값
"""

import os
import sys
import re
import time
import json
from typing import TypedDict, Optional, Any, Dict
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
from farmer.weather.utils import combine_weather_data

load_dotenv()

# 환경 설정
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
KST = ZoneInfo("Asia/Seoul")

# 시/군/구 키워드 (필요하면 계속 추가)
CITY_KEYWORDS = [
    # 특별/광역시
    "서울","부산","대구","인천","광주","대전","울산","세종",
    # 경기
    "수원","성남","용인","고양","부천","안산","남양주","의정부","화성","평택","파주","김포","광명","시흥",
    "군포","하남","오산","이천","안성","의왕","구리","과천","여주","양주",
    # 강원/충청/전라/경상/제주 주요 시
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

def make_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY=REDACTED ValueError("OPENAI_API_KEY=REDACTED에 없습니다.")
    return ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY=REDACTED: 어떤 데이터가 필요한지(특보/단기/중기)만 판별
SIMPLE_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """다음 질문을 분석해서 어떤 기상 데이터가 필요한지 판단하세요.

질문: {question}

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

# 답변 생성 프롬프트
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """기상청 예보관으로서 다음 데이터를 바탕으로 종합적인 답변을 작성하세요.

현재 날짜: {current_date}

질문: {question}

[기상 데이터]
{context}

답변 구조:
순번) 개요: 핵심 요약
순번) 기상특보 현황: (해당 지역에 현재 발효 중이거나 발표 예정인 특보가 있을 때만)
순번) 단기예보: (단기 데이터가 있을 때만)
순번) 중기예보: (중기 데이터가 있을 때만)
순번) 종합 요약

주의사항:
- 지역 정보가 명확하지 않은 경우, 서울/수도권 기준으로 답변하세요
- 데이터가 없는 섹션은 아예 생략하세요. "없습니다"라고 표시하지 마세요.
- 섹션을 생략할 때는 번호를 연속으로 맞추세요
- 발표 예정인 특보는 "발표 예정" 또는 "예정"이라고 명시하세요
- **중요**: 데이터가 없는 섹션은 아예 생략하세요. "없습니다"라고 표시하지 마세요.
- **중요**: 섹션을 생략할 때는 번호를 연속으로 맞추세요 (예: 중기예보가 없으면 1,2,3,5가 아닌 1,2,3,4로)
- **중요**: 기상특보 데이터가 없으면 "기상특보 현황" 섹션을 아예 제외하세요
- **중요**: 중기예보 데이터가 없으면 "중기예보" 섹션을 아예 제외하세요
- **중요**: 단기예보 데이터가 없으면 "단기예보" 섹션을 아예 제외하세요
- 마크다운 문법을 사용하지 말고 일반 텍스트로 작성하세요
- 답변 끝에 안내 문구는 추가하지 마세요
- 날짜 계산 시 현재 날짜를 기준으로 정확히 계산하세요

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
        | SIMPLE_ANALYSIS_PROMPT
        | make_llm()
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

    # 시/군/구 추출(없으면 서울)
    city = _extract_city(question)
    target_region = city if city else default_region
    has_region_info = True
    if not city:
        print("   - 지역 정보 없음: 기본값 '서울' 적용")
        target_region = "서울"

    # 날짜 기본값
    enhanced_question = question
    if not has_date_info:
        enhanced_question = f"{enhanced_question} 오늘".strip()
        has_date_info = True
        print("   - 날짜 정보 없음: '오늘' 주입")

    print(f"   - 최종 질문: {enhanced_question}")
    print(f"   - 타깃 지역(시): {target_region}")
    print(f"   - 분석 결과: 특보={need_advisory}, 단기={need_short_forecast}, 중기={need_mid_forecast}")

    return {
        **state,
        "question": enhanced_question,
        "need_advisory": need_advisory,
        "need_short_forecast": need_short_forecast,
        "need_mid_forecast": need_mid_forecast,
        "has_region_info": has_region_info,
        "has_date_info": has_date_info,
        "default_region": target_region,
        "target_region": target_region,
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

    filtered = []
    for it in result.get("advisory_data", []) or []:
        if not isinstance(it, dict):
            continue
        try:
            j = json.loads(it.get("json", "{}"))
            name = j.get("region_name", "")
            end_s = j.get("window_end_kst") or ""
            cmd_name = j.get("command_name", "")  # 발표/변경/해제 등

            # 시간/해제 방어
            end_dt = _parse_kst_maybe(end_s)
            if end_dt and end_dt < now:
                continue
            if "해제" in cmd_name:
                continue

            # 지역 매칭
            if target_region == "서울":
                hit = any(k in name for k in ["서울","경기","인천","수도권"])
            else:
                hit = (target_region in name)

            if hit:
                filtered.append(it)
        except Exception:
            pass

    # 서울 우선순위: 서울 > 경기 > 인천(본섬) > 서해5도/백령
    if target_region == "서울" and filtered:
        def _rank(it):
            n = json.loads(it["json"]).get("region_name", "")
            if "서울" in n: return (0, n)
            if "경기" in n: return (1, n)
            if "인천" in n and not any(x in n for x in ["서해5도","백령"]): return (2, n)
            if any(x in n for x in ["서해5도","백령"]): return (3, n)
            return (4, n)
        filtered.sort(key=_rank)

    print(f"   - {target_region} 지역 필터링: {len(filtered)}개 (전체: {len(result.get('advisory_data', []) or [])}개)")
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
    print(f"   - 대상 지역(시): {target_region}")

    raw = result.get("short_forecast_data", []) or []
    filtered = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        try:
            j = json.loads(it.get("json", "{}"))
            name = j.get("region_name", "")
            if target_region == "서울":
                if any(k in name for k in ["서울","경기","인천","수도권"]):
                    filtered.append(it)
            else:
                if target_region in name:
                    filtered.append(it)
        except Exception:
            pass

    if target_region == "서울" and filtered:
        def _rank(it):
            n = json.loads(it["json"]).get("region_name", "")
            if "서울" in n: return (0, n)
            if "경기" in n: return (1, n)
            if "인천" in n and not any(x in n for x in ["서해5도","백령"]): return (2, n)
            if any(x in n for x in ["서해5도","백령"]): return (3, n)
            return (4, n)
        filtered.sort(key=_rank)

    print(f"   - {target_region} 지역 필터링: {len(filtered)}개 (전체: {len(raw)}개)")
    if len(filtered) == 0:
        print("   - ⚠️ 필터링된 데이터가 없습니다. 원본 상위 3개 출력:")
        for i, it in enumerate(raw[:3]):
            print(f"     {i+1}: {it}")

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
    print(f"   - 대상 지역(시): {target_region}")

    raw = result.get("mid_forecast_data", []) or []
    filtered = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        try:
            j = json.loads(it.get("json", "{}"))
            name = j.get("region_name", "")
            if target_region == "서울":
                if any(k in name for k in ["서울","경기","인천","수도권"]):
                    filtered.append(it)
            else:
                if target_region in name:
                    filtered.append(it)
        except Exception:
            pass

    print(f"   - {target_region} 지역 필터링: {len(filtered)}개 (전체: {len(raw)}개)")
    return {"mid_forecast_data": filtered}

# 5) 데이터 통합
def combine_data_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 데이터 통합")
    context = combine_weather_data(state)
    return {**state, "context": context}

# 6) 컨텍스트 충분성 판단 (LLM 간단 체크)
def assess_context_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 컨텍스트 충분성 판단")
    question = state.get("question", "")
    context = state.get("context", "")

    if not context or context == "NO_DATA_AVAILABLE":
        return {**state, "context_ok": False, "context": "NO_DATA_AVAILABLE"}

    prompt = ChatPromptTemplate.from_template(
        """질문과 컨텍스트만으로 충분히 구체적인 답변을 작성할 수 있는지 판단하세요.

질문: {question}
컨텍스트: {context}

JSON만:
{{"context_ok": true/false, "reason": "한 줄 이유"}}
"""
    )

    chain = (
        {"question": lambda x: x["question"], "context": lambda x: x["context"]}
        | prompt
        | make_llm()
        | StrOutputParser()
    )

    raw = chain.invoke({"question": question, "context": context})
    try:
        data = json.loads(raw)
        ok = bool(data.get("context_ok", False))
    except Exception:
        ok = False

    if not ok:
        return {**state, "context_ok": False, "context": "NO_DATA_AVAILABLE"}
    return {**state, "context_ok": True}

# 7) 답변 생성
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
        | ANSWER_PROMPT
        | make_llm()
        | StrOutputParser()
    )

    answer = chain.invoke({"context": state["context"], "question": state["question"], "current_date": current_date})
    time.sleep(1)  # API 요청 간격

    txt = re.sub(r'\n{3,}', '\n\n', answer or "").strip()
    print("   - 답변 생성 완료")
    return {**state, "answer": txt}

# 그래프 빌드
def build_simple_graph():
    g = StateGraph(GraphState)

    g.add_node("analyze_question", simple_analyze_question_node)
    g.add_node("advisory", advisory_node_wrapper)
    g.add_node("short_forecast", short_forecast_node_wrapper)
    g.add_node("mid_forecast", mid_forecast_node_wrapper)
    g.add_node("combine_data", combine_data_node)
    g.add_node("assess_context", assess_context_node)
    g.add_node("generate_answer", generate_answer_node)

    g.set_entry_point("analyze_question")

    def route_after_analysis(state: GraphState) -> str:
        na = state.get("need_advisory", False)
        ns = state.get("need_short_forecast", False)
        nm = state.get("need_mid_forecast", False)
        needed = sum([1 if na else 0, 1 if ns else 0, 1 if nm else 0])
        if needed == 0:
            return "combine_data"
        if needed == 1:
            if na: return "advisory"
            if ns: return "short_forecast"
            if nm: return "mid_forecast"
        # 2개 이상 필요한 경우: 순차 연결
        return "advisory"

    g.add_conditional_edges(
        "analyze_question",
        route_after_analysis,
        {
            "advisory": "advisory",
            "short_forecast": "short_forecast",
            "mid_forecast": "mid_forecast",
            "combine_data": "combine_data",
        }
    )

    # 순차 연결 (간단화)
    g.add_edge("advisory", "short_forecast")
    g.add_edge("short_forecast", "mid_forecast")
    g.add_edge("mid_forecast", "combine_data")

    g.add_edge("combine_data", "assess_context")
    g.add_edge("assess_context", "generate_answer")
    g.add_edge("generate_answer", END)

    app = g.compile()

    try:
        graph_image_path = "agent_workflow_simple.png"
        with open(graph_image_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print(f"\n단순화된 LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류: {e}")

    return app

# OchestratorTest.py 호환 함수
def run(state: dict) -> dict:
    try:
        query = state.get("query", "")
        if not query:
            return {"agent_answer": "질문이 제공되지 않았습니다. 날씨 관련 질문을 해주세요."}

        print(f"[날씨_agent_simple] 질문 처리 시작: {query}")
        app = build_simple_graph()
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
    app = build_simple_graph()

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
