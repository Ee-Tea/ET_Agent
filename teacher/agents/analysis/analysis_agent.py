import json
import os
from typing import Dict, List, TypedDict, Annotated, Any, Literal, Union, TypeGuard, Optional
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from ..base_agent import BaseAgent

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

class AnalysisState(TypedDict):
    """LangGraph 노드 간에 주고받는 분석 상태 컨테이너
    - grade_result: 0/1 정오 배열(ScoreEngine 결과)
    - detailed_analysis/overall_assessment: LLM이 생성한 분석 결과(분리 저장)
    """
    messages: Annotated[List[BaseMessage], "그래프 실행 중 생성되는 대화 메시지 로그"]
    problem: List[str]  # 원문 문항 텍스트
    problem_types: List[str]  # 과목명 리스트(예: ["소프트웨어설계", "소프트웨어개발", ...])
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
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        )
        self.model = OPENAI_LLM_MODEL
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
        """LLM을 사용하여 분석 및 피드백 생성"""
        print(f"\n🔍 [AnalysisAgent] _generate_feedback 시작")
        print(f"  - state 키: {list(state.keys())}")
        
        problems = state.get("problem", [])
        subjects = state.get("problem_types", [])
        user_answers = state.get("user_answer", [])
        solution_answers = state.get("solution_answer", [])
        solutions = state.get("solution", [])
        grade_result = state.get("grade_result", [])
        
        print(f"  - problems: {len(problems)}개")
        print(f"  - subjects: {len(subjects)}개")
        print(f"  - user_answers: {len(user_answers)}개")
        print(f"  - solution_answers: {len(solution_answers)}개")
        print(f"  - solutions: {len(solutions)}개")
        print(f"  - grade_result: {len(grade_result)}개")
        
        # 데이터 길이 검증
        lengths = [len(problems), len(subjects), len(user_answers), len(solution_answers), len(solutions), len(grade_result)]
        print(f"  - 각 필드 길이: {lengths}")
        
        if len(set(lengths)) > 1:
            print(f"❌ [AnalysisAgent] 데이터 길이 불일치: {lengths}")
            # 최소 길이로 맞춤
            min_length = min(lengths)
            problems = problems[:min_length]
            subjects = subjects[:min_length]
            user_answers = user_answers[:min_length]
            solution_answers = solution_answers[:min_length]
            solutions = solutions[:min_length]
            grade_result = grade_result[:min_length]
            print(f"  - 최소 길이({min_length})로 맞춤")
        
        # 데이터 타입 검증 및 변환
        try:
            # user_answers를 정수 리스트로 변환
            if user_answers and isinstance(user_answers[0], str):
                user_answers = [int(ans) if ans.isdigit() else 0 for ans in user_answers]
                print(f"  - user_answers를 정수로 변환: {user_answers}")
            
            # solution_answers를 정수 리스트로 변환
            if solution_answers and isinstance(solution_answers[0], str):
                solution_answers = [int(ans) if ans.isdigit() else 0 for ans in solution_answers]
                print(f"  - solution_answers를 정수로 변환: {solution_answers}")
                
        except Exception as e:
            print(f"⚠️ [AnalysisAgent] 데이터 변환 중 오류: {e}")
            # 기본값 설정
            user_answers = [0] * len(problems)
            solution_answers = [0] * len(problems)
        
        print(f"✅ [AnalysisAgent] 데이터 전처리 완료")
        
        # items 구성
        items = [
            {
                "number": i + 1,
                "problem": problem,
                "subject": subject,
                "user_answer": user_ans,
                "correct_answer": correct_ans,
                "solution": solution,
                "is_correct": bool(is_correct),
            }
            for i, (problem, subject, user_ans, correct_ans, solution, is_correct) in enumerate(
                zip(problems, subjects, user_answers, solution_answers, solutions, grade_result)
            )
        ]
        
        print(f"  - 생성된 items: {len(items)}개")
        mistakes = [it for it in items if not it["is_correct"]]
        print(f"  - 오답 개수: {len(mistakes)}개")

        analysis_data = {
            "items": items,
            "summary": {
                "correct_count": sum(grade_result),
                "total_count": len(grade_result),
                "incorrect_numbers": [it["number"] for it in mistakes],
            },
        }
        
        print(f"  - analysis_data 구성 완료")
        print(f"    - correct_count: {analysis_data['summary']['correct_count']}")
        print(f"    - total_count: {analysis_data['summary']['total_count']}")
        print(f"    - incorrect_numbers: {analysis_data['summary']['incorrect_numbers']}")

        if len(mistakes) > 0:
            print(f"🚀 [AnalysisAgent] 오답 분석 LLM 호출 시작")
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": """당신은 학생의 학습 데이터를 분석하는 전문 학습 코치입니다.
각 문항 데이터는 'items' 배열에 문항 단위 객체로 제공됩니다.
subject 는 각 문항의 과목명(문자열)입니다.
응답은 지정된 JSON 스키마만 출력하고, 불필요한 자연어 설명은 포함하지 마십시오."""
                        },
                        {
                            "role": "user",
                            "content": f"""다음 학생의 풀이 결과를 문항 단위로 제공합니다. 오답 패턴을 분석하고 맞춤 피드백을 생성하세요.

{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

분석 지침:
- items[*].subject를 활용해 과목 기반 오답 패턴을 도출
- 동일 과목에서 반복되는 오답은 묶어서 패턴 설명
- 실수 유형을 구체화하고 교정 전략을 제시

아래 JSON 형식을 그대로 따르세요.
```json
{{
  "detailed_analysis": [
    {{
      "problem_number": "틀린 문제 번호",
      "subject": "과목명",
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
모든 내용은 한국어로 작성해주세요."""
                        }
                    ],
                    temperature=LLM_TEMPERATURE,
                    max_completion_tokens=LLM_MAX_TOKENS,
                    top_p=1,
                    stream=False,
                    response_format={"type": "json_object"},
                    stop=None
                )

                feedback_content = completion.choices[0].message.content
                print(f"✅ [AnalysisAgent] LLM 응답 완료: {len(feedback_content)}자")
                
                try:
                    parsed_feedback = json.loads(feedback_content)
                    print(f"✅ [AnalysisAgent] JSON 파싱 성공")
                except json.JSONDecodeError as e:
                    print(f"⚠️ [AnalysisAgent] JSON 파싱 실패: {e}")
                    parsed_feedback = {"detailed_analysis": [], "overall_assessment": {}}
                    
                state["detailed_analysis"] = parsed_feedback.get("detailed_analysis", [])
                state["overall_assessment"] = parsed_feedback.get("overall_assessment", {})
                
            except Exception as e:
                print(f"❌ [AnalysisAgent] LLM 호출 실패: {e}")
                state["detailed_analysis"] = []
                state["overall_assessment"] = {"error": f"LLM 분석 실패: {str(e)}"}
        else:
            print(f"🚀 [AnalysisAgent] 전부 정답 - 심화 학습 LLM 호출 시작")
            try:
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
items 배열의 문항 단위 데이터를 활용하여 과목 기반 강점을 구조적으로 설명하고 다음 학습 단계를 제안하세요.
과목명은 각 item의 subject를 사용하세요.

{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

피드백은 아래 JSON 형식에 맞춰, 학생의 자신감을 높이고 도전 의식을 자극하는 내용으로 작성해주세요.
```json
{{
  "overall_assessment": {{
    "title": "완벽한 성취!",
    "strengths_analysis": "과목별 강점 분석 (구체적이고 구체적으로)",
    "deepen_learning_plan": {{
      "title": "심화 학습 계획",
      "recommendations": ["구체적 권장사항 1", "구체적 권장사항 2", "구체적 권장사항 3"],
      "recommended_resources": ["심화 자료/강의 (선택)"]
    }},
    "final_message": "격려와 도전 의식을 자극하는 메시지"
  }}
}}
```
모든 내용은 한국어로 작성해주세요."""
                        }
                    ],
                    temperature=LLM_TEMPERATURE,
                    max_completion_tokens=LLM_MAX_TOKENS,
                    top_p=1,
                    stream=False,
                    response_format={"type": "json_object"},
                    stop=None
                )

                feedback_content = completion.choices[0].message.content
                print(f"✅ [AnalysisAgent] 심화 학습 LLM 응답 완료: {len(feedback_content)}자")
                
                try:
                    parsed_feedback = json.loads(feedback_content)
                    print(f"✅ [AnalysisAgent] 심화 학습 JSON 파싱 성공")
                except json.JSONDecodeError as e:
                    print(f"⚠️ [AnalysisAgent] 심화 학습 JSON 파싱 실패: {e}")
                    parsed_feedback = {"overall_assessment": {}}
                    
                state["detailed_analysis"] = []
                state["overall_assessment"] = parsed_feedback.get("overall_assessment", {})
            except Exception as e:
                print(f"❌ [AnalysisAgent] 심화 학습 LLM 호출 실패: {e}")
                state["detailed_analysis"] = []
                state["overall_assessment"] = {"error": f"심화 학습 LLM 분석 실패: {str(e)}"}

        state["messages"].append(AIMessage(content="분석 및 피드백 생성 완료"))
        print(f"✅ [AnalysisAgent] _generate_feedback 완료")
        return state

    def invoke(self, input_data: Dict) -> AnalysisResult:
        """메인 실행
        1) 입력 검증: 필수 필드 유무/길이 일치 확인
        2) 상태 구성: grade_result는 ScoreEngine의 results([0,1]) 사용
        3) 그래프 실행: generate_feedback
        4) 반환: analysis만 포함한 최소 스키마
        """
        try:
            print(f"\n🔍 [AnalysisAgent] invoke 시작")
            print(f"  - 입력 데이터 키: {list(input_data.keys())}")
            print(f"  - 입력 데이터 타입: {type(input_data)}")
            
            # 각 필드별 상세 로깅
            for field in ["problem", "problem_types", "user_answer", "solution_answer", "solution", "results"]:
                value = input_data.get(field)
                if value is not None:
                    print(f"  - {field}: {type(value)} = {len(value) if isinstance(value, (list, dict)) else value}")
                else:
                    print(f"  - {field}: None (누락)")
            
            # 입력 데이터 검증
            required_fields = ["problem", "problem_types", "user_answer", "solution_answer", "results"]
            missing_fields = [field for field in required_fields if field not in input_data]
            if missing_fields:
                print(f"❌ [AnalysisAgent] 필수 필드 누락: {missing_fields}")
                return _error(f"필수 필드가 누락되었습니다: {missing_fields}")

            # 데이터 길이 일치 확인
            lengths = [len(input_data[field]) for field in required_fields]
            print(f"  - 각 필드 길이: {dict(zip(required_fields, lengths))}")
            if len(set(lengths)) > 1:
                print(f"❌ [AnalysisAgent] 데이터 길이 불일치: {dict(zip(required_fields, lengths))}")
                return _error(f"모든 필드의 데이터 길이가 일치하지 않습니다: {dict(zip(required_fields, lengths))}")

            print(f"✅ [AnalysisAgent] 입력 데이터 검증 통과")
            
            # 초기 상태 설정
            initial_state = AnalysisState(
                messages=[HumanMessage(content="분석을 시작합니다.")],
                problem=input_data.get("problem", []),
                problem_types=input_data.get("problem_types", []),
                user_answer=input_data.get("user_answer", []),
                solution_answer=input_data.get("solution_answer", []),
                solution=input_data.get("solution", []),  # 선택적 필드
                grade_result=input_data.get("results", []),
                detailed_analysis=[],
                overall_assessment={},
            )
            
            print(f"✅ [AnalysisAgent] 초기 상태 설정 완료")
            print(f"  - problem: {len(initial_state['problem'])}개")
            print(f"  - problem_types: {len(initial_state['problem_types'])}개")
            print(f"  - user_answer: {len(initial_state['user_answer'])}개")
            print(f"  - solution_answer: {len(initial_state['solution_answer'])}개")
            print(f"  - solution: {len(initial_state['solution'])}개")
            print(f"  - grade_result: {len(initial_state['grade_result'])}개")

            # 그래프 실행
            print(f"🚀 [AnalysisAgent] 그래프 실행 시작")
            result = self.graph.invoke(initial_state)
            print(f"✅ [AnalysisAgent] 그래프 실행 완료")
            print(f"  - 결과 타입: {type(result)}")
            print(f"  - 결과 키: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")

            return _success(
                analysis={
                    "detailed_analysis": result.get("detailed_analysis", []),
                    "overall_assessment": result.get("overall_assessment", {}),
                }
            )

        except Exception as e:
            print(f"❌ [AnalysisAgent] 예외 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return _error(f"분석 실행 중 오류 발생: {str(e)}")

# 사용 예제(콘솔 출력용 유틸리티)
def print_analysis_result(result):
    """분석 결과 간단 출력(현재 스키마: {"status","analysis"} 만 사용)
    - 오류: 메시지만 출력
    - 성공: overall_assessment 요약 + detailed_analysis 요약(subject/analysis 출력)
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
            subject = item.get("subject")
            mtype = item.get("mistake_type", "-")
            detail = (item.get("analysis") or item.get("recommendation") or "").strip()
            header = f"  · 문제 {num}"
            if subject:
                header += f" [과목: {subject}]"
            header += f" - 실수 유형: {mtype}"
            print(header)
            if detail:
                print(f"    원인 분석: {detail}")

    print("\n" + "="*50)
