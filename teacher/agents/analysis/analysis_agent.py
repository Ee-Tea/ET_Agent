import json
import os
from typing import Dict, List, TypedDict, Annotated, Any, Literal, Union, TypeGuard
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from groq import Groq

from ..base_agent import BaseAgent

class AnalysisState(TypedDict):
    """LangGraph 노드 간에 주고받는 분석 상태 컨테이너
    - grade_result: 0/1 정오 배열(ScoreEngine 결과)
    - detailed_analysis/overall_assessment: LLM이 생성한 분석 결과(분리 저장)
    """
    messages: Annotated[List[BaseMessage], "그래프 실행 중 생성되는 대화 메시지 로그"]
    problem: List[str]  # 원문 문항 텍스트
    problem_type: List[Dict[str, Any]]  # 계층형 개념 태그(예: 과목명/주요항목/세부항목/세세항목)
    user_answer: List[int]  # 사용자 답
    solution_answer: List[int]  # 정답
    solution: List[str]  # 해설(선택)
    grade_result: List[int]  # 각 문항 정오(1/0)
    detailed_analysis: List[Dict[str, Any]]  # LLM 생성: 문항 단위 오답 분석 리스트
    overall_assessment: Dict[str, Any]  # LLM 생성: 종합 평가/권장 학습 계획

# 결과 페이로드 타입(최소 스키마)
class AnalysisSuccessResult(TypedDict):
    """성공 시: 분석 결과만 반환"""
    status: Literal["success"]
    analysis: Dict[str, Any]  # {"detailed_analysis": [...], "overall_assessment": {...}}

class AnalysisErrorResult(TypedDict):
    """오류 시: 메시지 최소 반환"""
    status: Literal["error"]
    error_message: str

AnalysisResult = Union[AnalysisSuccessResult, AnalysisErrorResult]

# 결과 생성 헬퍼
def _success(*, analysis: Dict[str, Any]) -> AnalysisSuccessResult:
    return {
        "status": "success",
        "analysis": analysis,
    }

def _error(error_message: str) -> AnalysisErrorResult:
    return {
        "status": "error",
        "error_message": error_message,
    }

# 호출 측에서 타입 내로잉에 사용하는 가드
def is_success(result: AnalysisResult) -> TypeGuard[AnalysisSuccessResult]:
    return result.get("status") == "success"

class AnalysisAgent(BaseAgent):
    """분석 에이전트
    - 입력: 문제/개념태그/사용자답/정답/해설 + grade_result(ScoreEngine의 [0,1])
    - 처리: 문항 단위(items)로 재구성 → LLM 분석 요청 → 분석 결과를 상태에 저장
    - 출력: analysis만 반환(detailed_analysis, overall_assessment)
    """
    
    @property
    def name(self) -> str:
        """에이전트의 고유한 이름을 반환합니다."""
        return "analysis"
    
    @property
    def description(self) -> str:
        """에이전트의 역할에 대한 설명을 반환합니다."""
        return "학습자 답안을 분석하고 개인화된 피드백을 생성합니다"
    
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY=REDACTED = "meta-llama/llama-4-scout-17b-16e-instruct"  # 또는 "meta-llama/llama-3.1-8b-instant"
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """분석 그래프 구성
        - 단일 노드(generate_feedback)로 구성
        - entry → generate_feedback → END
        """
        # 상태 정의에 기반한 그래프 생성
        workflow = StateGraph(AnalysisState)
        
        # 노드 추가 - analyze_mistakes 제거하고 직접 generate_feedback으로 연결
        workflow.add_node("grade_answers", self._grade_answers)
        workflow.add_node("generate_feedback", self._generate_feedback)
        
        # 엣지 수정 - grade_answers에서 바로 generate_feedback으로 연결
        workflow.set_entry_point("grade_answers")
        workflow.add_edge("grade_answers", "generate_feedback")
        workflow.add_edge("generate_feedback", END)
        
        return workflow.compile()
    
    def _grade_answers(self, state: AnalysisState) -> AnalysisState:
        """사용자 답안과 정답을 비교하여 채점"""
        user_answers = state["user_answer"]
        solution_answers = state["solution_answer"]
        
        # 정답과 사용자 답안을 비교하여 채점 (정답: 1, 오답: 0)
        grade_result = [1 if ua == sa else 0 for ua, sa in zip(user_answers, solution_answers)]
        state["grade_result"] = grade_result
        
        # 메시지 기록 추가
        state["messages"].append(
            AIMessage(content="채점이 완료되었습니다.")
        )
        return state
    
    
    def _generate_feedback(self, state: AnalysisState) -> AnalysisState:
        """LLM 피드백 생성
        - 입력: problem/problem_type/user_answer/solution_answer/solution/grade_result
        - 준비: problem_type 평탄화 → items(문항 단위 구조) 생성
        - 출력: detailed_analysis / overall_assessment 만 상태에 저장
        """
        problems = state["problem"]
        problem_types = state["problem_type"]
        user_answers = state["user_answer"]
        solution_answers = state["solution_answer"]
        solutions = state["solution"]
        grade_result = state["grade_result"]

        # problem_type 평탄화(계층형 → '과목명 > 주요항목 > 세부항목 > 세세항목' 경로 문자열)
        def flatten_problem_type(pt: Any) -> str:
            if isinstance(pt, dict):
                keys = ["과목명", "주요항목", "세부항목", "세세항목"]
                parts = [str(pt.get(k)) for k in keys if pt.get(k)]
                return " > ".join(parts) if parts else json.dumps(pt, ensure_ascii=False)
            return str(pt)

        flattened_types: List[str] = [flatten_problem_type(pt) for pt in problem_types]

        # 문항 단위(items) 데이터 구성(LLM 입력 최적화)
        items = [
            {
                "number": i + 1,
                "problem": problem,
                "problem_type": p_type,
                "problem_type_path": p_type_flat,
                "user_answer": user_ans,
                "solution_answer": correct_ans,
                "is_correct": bool(is_correct),
                "solution": solution,
            }
            for i, (problem, p_type, p_type_flat, user_ans, correct_ans, solution, is_correct) in enumerate(
                zip(problems, problem_types, flattened_types, user_answers, solution_answers, solutions, grade_result)
            )
        ]
        mistakes = [it for it in items if not it["is_correct"]]

        # LLM 입력 페이로드(간결/일관)
        analysis_data = {
            "items": items,
            "summary": {
                "correct_count": sum(grade_result),
                "total_count": len(grade_result),
                "incorrect_numbers": [it["number"] for it in mistakes],
            },
        }

        # 오답 존재 시: 오답 분석 프롬프트
        if len(mistakes) > 0:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 학생의 학습 데이터를 분석하는 전문 학습 코치입니다.
각 문항 데이터는 'items' 배열에 문항 단위 객체로 제공됩니다.
problem_type 은 각 문항의 개념적 계층 정보를 담는 객체입니다.
예시: {"과목명":"소프트웨어 설계","주요항목":"요구사항 확인","세부항목":"요구사항 확인","세세항목":"요구분석기법"}
필요 시 'problem_type_path'를 사용해 '과목명 > 주요항목 > 세부항목 > 세세항목' 경로로 개념을 활용하십시오.
응답은 지정된 JSON 스키마만 출력하고, 불필요한 자연어 설명은 포함하지 마십시오."""
                    },
                    {
                        "role": "user",
                        "content": f"""다음 학생의 풀이 결과를 문항 단위로 제공합니다. 오답 패턴을 분석하고 맞춤 피드백을 생성하세요.

                {json.dumps(analysis_data, ensure_ascii=False, indent=2)}

분석 지침:
- items[*].problem_type_path를 활용해 개념 경로 기반 패턴을 도출
- 동일/유사 경로 반복 오답은 묶어서 패턴 설명
- 실수 유형을 구체화하고 교정 전략을 제시

아래 JSON 형식을 그대로 따르세요.
```json
{{
  "detailed_analysis": [
    {{
      "problem_number": "틀린 문제 번호",
      "concept_path": "문제의 개념 경로 (problem_type_path 활용)",
      "mistake_type": "실수 유형 (예: 개념 이해 부족, 계산 실수, 조건 누락)",
      "analysis": "왜 틀렸는지에 대한 구체적 원인 분석 (학생의 사고 과정 추정)"
    }}
  ],
  "overall_assessment": {{
    "strengths": "학생이 잘한 점",
    "weaknesses": "취약점과 반복 패턴",
    "action_plan": {{
      "title": "맞춤 학습 계획",
      "short_term_goal": "1~2주 내 실행 목표",
      "long_term_goal": "장기적 성장 목표",
      "recommended_strategies": ["구체적 전략 1", "구체적 전략 2"],
      "recommended_resources": ["자료/강의 (선택)"]
    }},
    "final_message": "격려 메시지"
  }}
}}
```
모든 내용은 한국어로 작성."""
                    }
                ],
                temperature=0,
                max_completion_tokens=4096,
                top_p=1,
                stream=False,
                response_format={"type": "json_object"},
                stop=None
            )
            
            # JSON 응답 파싱 및 저장
            feedback_content = completion.choices[0].message.content
            try:
                parsed_feedback = json.loads(feedback_content)
            except json.JSONDecodeError:
                parsed_feedback = {"detailed_analysis": [], "overall_assessment": {}}
            # LLM JSON 응답 파싱 실패 시 안전 기본값 사용
            # 중복 저장 없음: 두 키만 보관
            state["detailed_analysis"] = parsed_feedback.get("detailed_analysis", [])
            state["overall_assessment"] = parsed_feedback.get("overall_assessment", {})
        else:
            # 전부 정답: 종합 평가만 요청
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 학생의 잠재력을 파악하고 더 높은 단계로 이끌어주는 전문 학습 코치입니다. 학생이 모든 문제를 맞혔을 때, 칭찬과 함께 심화 학습 방향을 구체적으로 제시해주세요."
                    },
                    {
                        "role": "user",
                        "content": f"""학생은 모든 문제({len(grade_result)}문제)를 정답 처리했습니다.
items 배열의 문항 단위 데이터를 활용하여 개념적 강점을 구조적으로 설명하고 다음 학습 단계를 제안하세요.
개념 경로는 각 item의 problem_type_path를 사용하세요.

{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

                피드백은 아래 JSON 형식에 맞춰, 학생의 자신감을 높이고 도전 의식을 자극하는 내용으로 작성해주세요.
                ```json
                {{
                  "overall_assessment": {{
                    "title": "완벽한 결과! 다음 도전을 위한 제안",
                    "strengths_analysis": "문제 유형별 정답률 100%를 달성한 것을 바탕으로, 학생이 어떤 개념과 문제 해결 능력이 뛰어난지 구체적으로 분석하고 칭찬해주세요.",
                    "deepen_learning_plan": {{
                      "title": "실력 유지를 위한 심화 학습 계획",
                      "recommendations": [
                        "현재 지식을 더 깊게 만들기 위한 구체적인 학습 활동 제안 (예: '관련 심화 문제집 풀이', '유사한 개념을 다른 과목과 연결해보기')",
                        "새로운 도전 과제 제안 (예: '경시대회 문제 맛보기', '관련 주제에 대한 프로젝트 학습')"
                      ],
                      "recommended_resources": ["심화 학습에 도움이 될 만한 자료나 책, 강의 링크 (있을 경우)"]
                    }},
                    "final_message": "학생의 성취를 축하하고, 앞으로의 성장을 응원하는 격려의 메시지."
                  }}
                }}
                ```
                모든 내용은 한국어로 작성해주세요.
                """
                    }
                ],
                temperature=0,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                response_format={"type": "json_object"},
                stop=None
            )
            
            feedback_content = completion.choices[0].message.content
            try:
                parsed_feedback = json.loads(feedback_content)
            except json.JSONDecodeError:
                parsed_feedback = {"overall_assessment": {}}
            # 상세 분석은 빈 배열로 유지
            state["detailed_analysis"] = []
            state["overall_assessment"] = parsed_feedback.get("overall_assessment", {})

        # 실행 로그 메시지(간단 표식)
        state["messages"].append(AIMessage(content="분석 및 피드백 생성 완료"))
        return state

    def execute(self, input_data: Dict) -> AnalysisResult:
        """메인 실행
        1) 입력 검증: 필수 필드 유무/길이 일치 확인
        2) 상태 구성: grade_result는 ScoreEngine의 results([0,1]) 사용
        3) 그래프 실행: generate_feedback
        4) 반환: analysis만 포함한 최소 스키마
        성공:
          {"status":"success","analysis":{"detailed_analysis":[...],"overall_assessment":{...}}}
        오류:
          {"status":"error","error_message":"..."}
        """
        try:
            # 입력 데이터 검증
            required_fields = ["problem", "problem_type", "user_answer", "solution_answer", "solution", "results"]
            missing_fields = [field for field in required_fields if field not in input_data]
            if missing_fields:
                return _error(f"필수 필드가 누락되었습니다: {missing_fields}")

            # 데이터 길이 일치 확인
            lengths = [len(input_data[field]) for field in required_fields]
            if len(set(lengths)) > 1:
                return _error(f"모든 필드의 데이터 길이가 일치하지 않습니다: {dict(zip(required_fields, lengths))}")

            # 초기 상태 설정(LLM 호출 전 준비 데이터)
            initial_state = AnalysisState(
                messages=[HumanMessage(content="분석을 시작합니다.")],
                problem=input_data.get("problem", []),
                problem_type=input_data.get("problem_type", []),
                user_answer=input_data.get("user_answer", []),
                solution_answer=input_data.get("solution_answer", []),
                solution=input_data.get("solution", []),
                grade_result=input_data.get("results", []),
                detailed_analysis=[],            # LLM 상세 분석(초기값)
                overall_assessment={},           # LLM 종합 평가(초기값)
            )

            # 그래프 실행
            result = self.graph.invoke(initial_state)

            # 최소 결과 반환: analysis만
            return _success(
                analysis={
                    "detailed_analysis": result.get("detailed_analysis", []),
                    "overall_assessment": result.get("overall_assessment", {}),
                }
            )

        except Exception as e:
            # 기타 예외 처리(간단 메시지)
            return _error(f"분석 실행 중 오류 발생: {str(e)}")

# 사용 예제(콘솔 출력용 유틸리티)
def print_analysis_result(result):
    """분석 결과 간단 출력(현재 스키마: {"status","analysis"} 만 사용)
    - 오류: 메시지만 출력
    - 성공: overall_assessment 요약 + detailed_analysis 요약
    """
    print("\n" + "="*20 + " 분석 결과 " + "="*20)

    # 오류 처리
    if result.get("status") == "error":
        print(f"❌ 오류: {result.get('error_message') or result.get('message') or '알 수 없는 오류'}")
        return

    analysis = result.get("analysis", {}) or {}
    oa = analysis.get("overall_assessment", {}) or {}
    da = analysis.get("detailed_analysis", []) or []

    # 종합 평가(전부 정답/오답 혼재 모두 대응)
    title = oa.get("title") or "분석 요약"
    print(f"\n[ 📋 {title} ]")

    # 전부 정답 케이스(심화 계획 키 사용)
    if "strengths_analysis" in oa:
        print("\n[ 💪 강점 분석 ]")
        print(f"  {oa.get('strengths_analysis', '')}".strip() or "  -")

        deepen = oa.get("deepen_learning_plan", {})
        if deepen:
            print(f"\n[ 📚 {deepen.get('title', '심화 학습 계획')} ]")
            for rec in deepen.get("recommendations", []):
                print(f"  • {rec}")
            if deepen.get("recommended_resources"):
                print("  - 참고 자료:")
                for res in deepen["recommended_resources"]:
                    print(f"    • {res}")

        if oa.get("final_message"):
            print("\n[ 💌 최종 메시지 ]")
            print(f"  {oa['final_message']}")
    else:
        # 오답 분석 케이스(강점/약점/학습 계획 키 사용)
        if oa.get("strengths"):
            print("\n[ 💪 강점 ]")
            print(f"  {oa['strengths']}")
        if oa.get("weaknesses"):
            print("\n[ 🔧 보완점 ]")
            print(f"  {oa['weaknesses']}")
        action = oa.get("action_plan", {})
        if action:
            print(f"\n[ 📈 {action.get('title','학습 계획')} ]")
            if action.get("short_term_goal"):
                print(f"  - 단기 목표: {action['short_term_goal']}")
            if action.get("long_term_goal"):
                print(f"  - 장기 목표: {action['long_term_goal']}")
            for strat in action.get("recommended_strategies", []):
                print(f"  • {strat}")
            if action.get("recommended_resources"):
                print("  - 참고 자료:")
                for res in action["recommended_resources"]:
                    print(f"    • {res}")
        if oa.get("final_message"):
            print("\n[ 💌 최종 메시지 ]")
            print(f"  {oa['final_message']}")

    # 오답 상세 요약
    if da:
        print("\n[ ❗ 오답 상세 ]")
        for item in da:
            num = item.get("problem_number", "-")
            mtype = item.get("mistake_type", "-")
            rec = item.get("recommendation", "-")
            print(f"  · 문제 {num}: {mtype} / {rec}")

    print("\n" + "="*50)