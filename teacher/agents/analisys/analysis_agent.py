import json
import os
from typing import Dict, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from groq import Groq
from langchain_teddynote import logging
logging.langsmith("analysis-agent")
# 환경 변수 로드
load_dotenv()

from ..base_agent import BaseAgent

class AnalysisState(TypedDict):
    """분석 상태를 정의하는 클래스"""
    messages: Annotated[List[BaseMessage], "메시지 목록"]
    problem: List[str]
    problem_type: List[str]
    user_answer: List[int]
    solution_answer: List[int]
    solution: List[str]
    grade_result: List[int]
    mistake_summary: str
    final_feedback: str

class AnalysisAgent(BaseAgent):
    """LangGraph 기반 분석 에이전트"""
    
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"  # 또는 "meta-llama/llama-3.1-8b-instant"
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """분석 워크플로우 그래프 생성"""
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
        """오답 분석과 개인화된 피드백을 함께 생성"""
        problems = state["problem"]
        problem_types = state["problem_type"]
        user_answers = state["user_answer"]
        solution_answers = state["solution_answer"]
        solutions = state["solution"]
        grade_result = state["grade_result"]
        
        # 오답 문제들 추출 (_analyze_mistakes 기능 통합)
        mistakes = []
        for i, (is_correct, problem, p_type, user_ans, correct_ans, solution) in enumerate(
            zip(grade_result, problems, problem_types, user_answers, solution_answers, solutions)
        ):
            if not is_correct:
                mistakes.append({
                    "problem_number": i + 1,
                    "problem": problem,
                    "problem_type": p_type,
                    "user_answer": user_ans,
                    "correct_answer": correct_ans,
                    "solution": solution
                })
        
        # 분석용 데이터 구조화 (전체 문제와 오답 정보 모두 포함)
        analysis_data = {
            "all_problems": {
                "problem": problems,
                "problem_type": problem_types,
                "user_answer": user_answers,
                "solution_answer": solution_answers,
                "solution": solutions,
                "result": grade_result
            },
            "mistakes": mistakes,
            "correct_count": sum(grade_result),
            "total_count": len(grade_result)
        }
        
        # 통합된 분석 요청: 오답 분석과 종합 피드백을 한 번에 요청
        if len(mistakes) > 0:  # 오답이 있는 경우
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 학생의 학습 데이터를 분석하여 개인화된 피드백을 제공하는 최고의 학습 코치입니다. 학생의 성장을 돕기 위해, 긍정적이고 격려하는 어조를 사용하되, 분석은 매우 구체적이고 전문적이어야 합니다. 제공된 JSON 형식에 맞춰 답변해주세요."""
                    },
                    {
                        "role": "user",
                        "content": f"""다음 학생의 문제 풀이 결과를 심층적으로 분석하고, 전문가 수준의 맞춤형 피드백을 생성해주세요.

                {json.dumps(analysis_data, ensure_ascii=False, indent=2)}

                분석 결과는 반드시 아래 JSON 형식에 맞춰, 각 항목을 구체적이고 실행 가능한 내용으로 채워주세요.
                ```json
                {{
                  "performance_summary": {{
                    "total_problems": "전체 문항 수",
                    "correct_count": "정답 개수",
                    "score": "점수 (100점 만점)",
                    "correctness_by_type": {{
                      "유형A": "정답률 (예: 50%)"
                    }}
                  }},
                  "detailed_analysis": [
                    {{
                      "problem_number": "틀린 문제 번호",
                      "mistake_type": "실수 유형 (예: 개념 이해 부족, 계산 실수, 조건 누락)",
                      "analysis": "왜 틀렸는지에 대한 구체적인 원인 분석. 학생의 풀이 과정을 추측하며 설명해주세요.",
                      "recommendation": "해당 실수를 바로잡기 위한 명확하고 실천적인 조언. (예: 'X 개념을 다시 학습하고, 관련 예제 3개를 풀어보세요.')"
                    }}
                  ],
                  "overall_assessment": {{
                    "strengths": "학생이 보여준 강점과 잘한 점에 대한 구체적인 칭찬.",
                    "weaknesses": "데이터를 기반으로 파악된 전반적인 취약점과 반복되는 실수 패턴.",
                    "action_plan": {{
                      "title": "성장을 위한 맞춤 학습 계획",
                      "short_term_goal": "1-2주 안에 달성할 수 있는 구체적인 단기 목표.",
                      "long_term_goal": "궁극적으로 도달해야 할 장기적인 학습 목표.",
                      "recommended_strategies": ["오답 노트 작성법, 개념 정리법 등 구체적인 학습 전략 제안"],
                      "recommended_resources": ["도움이 될 만한 자료나 강의 링크 (있을 경우)"]
                    }},
                    "final_message": "학생에게 용기를 주는 따뜻한 격려와 응원의 메시지."
                  }}
                }}
                ```
                모든 내용은 한국어로 작성해주세요.
                """
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
            parsed_feedback = json.loads(feedback_content)
            
            # 오답 분석 및 피드백 저장
            state["mistake_summary"] = json.dumps(parsed_feedback.get("detailed_analysis", {}), ensure_ascii=False, indent=2)
            state["final_feedback"] = json.dumps(parsed_feedback, ensure_ascii=False, indent=2)
        else:  # 모든 문제를 맞춘 경우
            # 모든 문제를 맞춘 경우의 분석 생성
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 학생의 잠재력을 파악하고 더 높은 단계로 이끌어주는 전문 학습 코치입니다. 학생이 모든 문제를 맞혔을 때, 칭찬과 함께 심화 학습 방향을 구체적으로 제시해주세요."
                    },
                    {
                        "role": "user",
                        "content": f"""이 학생은 모든 문제({len(grade_result)}문제)를 완벽하게 풀었습니다. 데이터를 기반으로 학생의 강점을 분석하고, 다음 단계로 나아갈 수 있는 심화 학습 계획을 제안해주세요.

                {json.dumps(analysis_data["all_problems"], ensure_ascii=False, indent=2)}

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
            state["mistake_summary"] = "모든 문제를 정답으로 해결했습니다."
            state["final_feedback"] = feedback_content
        
        # 메시지 기록 추가
        state["messages"].append(
            AIMessage(content="오답 분석 및 개인화된 피드백 생성 완료")
        )
        
        return state
    
    def execute(self, input_data: Dict) -> Dict:
        """메인 분석 메서드"""
        try:
            # 입력 데이터 검증
            required_fields = ["problem", "problem_type", "user_answer", "solution_answer", "solution"]
            missing_fields = [field for field in required_fields if field not in input_data]
            
            if missing_fields:
                return {
                    "status": "error",
                    "error_message": f"필수 필드가 누락되었습니다: {missing_fields}",
                    "metadata": {"total_problems": 0, "correct_count": 0, "score": 0, "has_mistakes": False},
                    "grading": {"results": [], "details": []},
                    "analysis": {},
                    "raw_data": {"missing_fields": missing_fields}
                }
            
            # 데이터 길이 일치 확인
            lengths = [len(input_data[field]) for field in required_fields]
            if len(set(lengths)) > 1:
                return {
                    "status": "error", 
                    "error_message": f"모든 필드의 데이터 길이가 일치하지 않습니다: {dict(zip(required_fields, lengths))}",
                    "metadata": {"total_problems": 0, "correct_count": 0, "score": 0, "has_mistakes": False},
                    "grading": {"results": [], "details": []},
                    "analysis": {},
                    "raw_data": {"field_lengths": dict(zip(required_fields, lengths))}
                }
                
            # 초기 상태 설정
            initial_state = AnalysisState(
                messages=[HumanMessage(content="분석을 시작합니다.")],
                problem=input_data.get("problem", []),
                problem_type=input_data.get("problem_type", []),
                user_answer=input_data.get("user_answer", []),
                solution_answer=input_data.get("solution_answer", []),
                solution=input_data.get("solution", []),
                grade_result=[],
                mistake_summary="",
                final_feedback=""
            )
            
            # 그래프 실행
            result = self.graph.invoke(initial_state)
            
            # 결과 정리 - 구조화된 형태로 반환
            try:
                # final_feedback JSON 파싱
                feedback_data = json.loads(result["final_feedback"]) if result["final_feedback"] else {}
                
                # 표준화된 응답 구조 생성
                analysis_result = {
                    "status": "success",
                    "metadata": {
                        "total_problems": len(result["grade_result"]),
                        "correct_count": sum(result["grade_result"]),
                        "incorrect_count": len(result["grade_result"]) - sum(result["grade_result"]),
                        "score": round((sum(result["grade_result"]) / len(result["grade_result"])) * 100, 1) if result["grade_result"] else 0,
                        "analysis_timestamp": "generated",
                        "has_mistakes": sum(result["grade_result"]) < len(result["grade_result"])
                    },
                    "grading": {
                        "results": result["grade_result"],
                        "details": [
                            {
                                "problem_number": i + 1,
                                "is_correct": bool(grade),
                                "user_answer": input_data.get("user_answer", [])[i] if i < len(input_data.get("user_answer", [])) else None,
                                "correct_answer": input_data.get("solution_answer", [])[i] if i < len(input_data.get("solution_answer", [])) else None
                            }
                            for i, grade in enumerate(result["grade_result"])
                        ]
                    },
                    "analysis": feedback_data,
                    "raw_data": {
                        "mistake_summary": result["mistake_summary"],
                        "messages": [msg.content for msg in result["messages"]]
                    }
                }
                
                return analysis_result
                
            except (json.JSONDecodeError, KeyError) as e:
                # JSON 파싱 오류 처리
                return {
                    "status": "error",
                    "error_message": f"결과 파싱 중 오류 발생: {str(e)}",
                    "metadata": {
                        "total_problems": len(result.get("grade_result", [])),
                        "correct_count": sum(result.get("grade_result", [])),
                        "score": round((sum(result.get("grade_result", [])) / len(result.get("grade_result", []))) * 100, 1) if result.get("grade_result") else 0,
                        "has_mistakes": True if result.get("grade_result") else False
                    },
                    "grading": {
                        "results": result.get("grade_result", []),
                        "details": []
                    },
                    "analysis": {},
                    "raw_data": {
                        "mistake_summary": result.get("mistake_summary", ""),
                        "final_feedback": result.get("final_feedback", ""),
                        "messages": [msg.content for msg in result.get("messages", [])],
                        "parsing_error": str(e)
                    }
                }
                
        except Exception as e:
            # 기타 예외 처리
            return {
                "status": "error",
                "error_message": f"분석 실행 중 오류 발생: {str(e)}",
                "metadata": {
                    "total_problems": 0,
                    "correct_count": 0,
                    "score": 0,
                    "has_mistakes": False
                },
                "grading": {
                    "results": [],
                    "details": []
                },
                "analysis": {},
                "raw_data": {
                    "error_details": str(e),
                    "input_data_keys": list(input_data.keys()) if isinstance(input_data, dict) else "invalid_input",
                    "error_type": type(e).__name__
                }
            }

# 사용 예제
def print_analysis_result(result):
    """개선된 분석 결과 출력 함수"""
    print("\n" + "="*20 + " 분석 결과 " + "="*20)
    
    # 상태 확인
    if result.get("status") == "error":
        print(f"❌ 오류 발생: {result.get('error_message', '알 수 없는 오류')}")
        return
    
    # 메타데이터 출력
    metadata = result.get("metadata", {})
    print(f"\n[ 📊 종합 성취도 ]")
    print(f"  - 점수: {metadata.get('score', 0)}점 / 100점")
    print(f"  - 정답률: {metadata.get('correct_count', 0)} / {metadata.get('total_problems', 0)}")
    print(f"  - 오답 개수: {metadata.get('incorrect_count', 0)}개")
    
    # 분석 데이터 출력
    analysis_data = result.get("analysis", {})
    
    if not metadata.get("has_mistakes", False):
        # 모든 문제를 맞춘 경우
        if "overall_assessment" in analysis_data:
            assessment = analysis_data.get("overall_assessment", {})
            print(f"\n🎉 {assessment.get('title', '완벽한 결과!')}")
            print(f"\n[ 💪 강점 분석 ]")
            print(f"  {assessment.get('strengths_analysis', 'N/A')}")

            deepen_plan = assessment.get("deepen_learning_plan", {})
            if deepen_plan:
                print(f"\n[ 📚 {deepen_plan.get('title', '심화 학습 계획')} ]")
                if deepen_plan.get("recommendations"):
                    print("  - 추천 활동:")
                    for rec in deepen_plan["recommendations"]:
                        print(f"    • {rec}")
            
            print(f"\n[ 💌 최종 메시지 ]")
            print(f"  {assessment.get('final_message', 'N/A')}")
    else:
        # 오답이 있는 경우
        if "performance_summary" in analysis_data:
            summary = analysis_data.get("performance_summary", {})
            if summary.get("correctness_by_type"):
                print("  - 유형별 정답률:")
                for p_type, rate in summary["correctness_by_type"].items():
                    print(f"    - {p_type}: {rate}")

            print("\n" + "-"*15 + " 🔍 오답 상세 분석 " + "-"*15)
            detailed_analysis = analysis_data.get("detailed_analysis", [])
            if not detailed_analysis:
                print("  분석할 오답이 없습니다.")
            else:
                for analysis in detailed_analysis:
                    print(f"\n▶ 문제 번호: {analysis.get('problem_number', 'N/A')}")
                    print(f"  - 실수 유형: {analysis.get('mistake_type', 'N/A')}")
                    print(f"  - 원인 분석: {analysis.get('analysis', 'N/A')}")
                    print(f"  - 개선 제안: {analysis.get('recommendation', 'N/A')}")

            assessment = analysis_data.get("overall_assessment", {})
            print("\n" + "-"*15 + " 📋 종합 평가 및 학습 계획 " + "-"*15)
            print(f"\n[ 💪 강점 ]")
            print(f"  {assessment.get('strengths', 'N/A')}")
            print(f"\n[ 🔧 보완점 ]")
            print(f"  {assessment.get('weaknesses', 'N/A')}")

            action_plan = assessment.get("action_plan", {})
            if action_plan:
                print(f"\n[ 📈 {action_plan.get('title', '학습 계획')} ]")
                print(f"  - 단기 목표: {action_plan.get('short_term_goal', 'N/A')}")
                print(f"  - 장기 목표: {action_plan.get('long_term_goal', 'N/A')}")
                if action_plan.get("recommended_strategies"):
                    print("  - 추천 전략:")
                    for strategy in action_plan["recommended_strategies"]:
                        print(f"    • {strategy}")
            
            print(f"\n[ 💌 최종 메시지 ]")
            print(f"  {assessment.get('final_message', 'N/A')}")

    print("\n" + "="*50)


# if __name__ == "__main__":
#     # .env 파일에 OPENAI_API_KEY 설정 필요
#     agent = AnalysisAgent()
    
#     # 테스트 데이터 로드
#     import json
#     import sys
#     from pathlib import Path
    
#     # 기본 input.json 경로 (현재 디렉토리)
#     input_file = Path("input.json")
    
#     # 명령줄 인자로 파일 경로가 제공된 경우 사용
#     if len(sys.argv) > 1:
#         input_file = Path(sys.argv[1])
    
#     # 파일 존재 여부 확인
#     if not input_file.exists():
#         print(f"오류: 입력 파일을 찾을 수 없습니다. ({input_file})")
#         sys.exit(1)
    
#     # 데이터 로드
#     try:
#         with open(input_file, 'r', encoding='utf-8') as f:
#             input_data = json.load(f)
        
#         print(f"파일 '{input_file}' 로드 성공")
        
#         # 필수 필드 확인
#         required_fields = ["problem", "problem_type", "user_answer", "solution_answer", "solution"]
#         for field in required_fields:
#             if field not in input_data:
#                 print(f"오류: 필수 필드 '{field}'가 입력 데이터에 없습니다.")
#                 sys.exit(1)
    
#         # 분석 실행
#         print("분석 시작...")
#         result = agent.execute(input_data)
        
#         # 결과 출력
#         print_analysis_result(result)
        
#         # 결과 저장
#         output_file = input_file.with_name(f"{input_file.stem}_result.json")
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(result, f, ensure_ascii=False, indent=2)
        
#         print(f"\n결과가 '{output_file}'에 저장되었습니다.")
        
#     except json.JSONDecodeError:
#         print(f"오류: '{input_file}'이 올바른 JSON 형식이 아닙니다.")
#         sys.exit(1)
#     except Exception as e:
#         print(f"오류 발생: {str(e)}")
#         sys.exit(1)
