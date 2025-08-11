import json
import os
from typing import Dict, List, TypedDict, Annotated, Any  # 수정: Any 추가
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from groq import Groq
from langchain_teddynote import logging
from ..base_agent import BaseAgent

class AnalysisState(TypedDict):
    """분석 상태를 정의하는 클래스"""
    messages: Annotated[List[BaseMessage], "메시지 목록"]
    problem: List[str]
    # 변경: problem_type 이 구조화된 객체 리스트 형식 지원
    problem_type: List[Dict[str, Any]]  # 예: {"과목명": "...", "주요항목": "...", "세부항목": "...", "세세항목": "..."
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
    
    @property
    def name(self) -> str:
        """에이전트 고유 이름"""
        return "analysis"

    @property
    def description(self) -> str:
        """에이전트 설명"""
        return "사용자 답안을 분석하고 맞춤형 학습 피드백을 생성하는 에이전트입니다."

    def _create_graph(self) -> StateGraph:
        """분석 워크플로우 그래프 생성 (채점 단계 제거)"""
        workflow = StateGraph(AnalysisState)
        
        # 노드 추가 - analyze_mistakes 제거하고 직접 generate_feedback으로 연결
        workflow.add_node("generate_feedback", self._generate_feedback)
        
        # 엣지 수정 - grade_answers에서 바로 generate_feedback으로 연결
        workflow.set_entry_point("generate_feedback")
        workflow.add_edge("generate_feedback", END)
        
        return workflow.compile()
    
    def _generate_feedback(self, state: AnalysisState) -> AnalysisState:
        """사용자 답안 분석 및 피드백 생성 (새 problem_type 구조 반영)"""
        problems = state["problem"]
        problem_types = state["problem_type"]  # 이제 dict 리스트
        user_answers = state["user_answer"]
        solution_answers = state["solution_answer"]
        solutions = state["solution"]

        # 새 helper: problem_type 평탄화
        def flatten_problem_type(pt: Any) -> str:
            if isinstance(pt, dict):
                keys = ["과목명", "주요항목", "세부항목", "세세항목"]
                parts = [str(pt.get(k)) for k in keys if pt.get(k)]
                return " > ".join(parts) if parts else json.dumps(pt, ensure_ascii=False)
            return str(pt)

        flattened_types: List[str] = [flatten_problem_type(pt) for pt in problem_types]

        grade_result = [1 if ua == sa else 0 for ua, sa in zip(user_answers, solution_answers)]
        state["grade_result"] = grade_result

        mistakes = []
        for i, (is_correct, problem, p_type, p_type_flat, user_ans, correct_ans, solution) in enumerate(
            zip(grade_result, problems, problem_types, flattened_types, user_answers, solution_answers, solutions)
        ):
            if not is_correct:
                mistakes.append({
                    "problem_number": i + 1,
                    "problem": problem,
                    "problem_type": p_type,          # 원본 구조
                    "problem_type_path": p_type_flat, # 평탄 경로
                    "user_answer": user_ans,
                    "correct_answer": correct_ans,
                    "solution": solution
                })

        analysis_data = {
            "all_problems": {
                "problem": problems,
                "problem_type": problem_types,          # 원본
                "problem_type_flat": flattened_types,   # 평탄화
                "user_answer": user_answers,
                "solution_answer": solution_answers,
                "solution": solutions,
                "result": grade_result
            },
            "mistakes": mistakes,
            "correct_count": sum(grade_result),
            "total_count": len(grade_result)
        }

        # 프롬프트 수정: problem_type 구조 설명 및 활용 지시
        if len(mistakes) > 0:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 학생의 학습 데이터를 분석하는 전문 학습 코치입니다.
problem_type 은 각 문항의 개념적 계층 정보를 담는 객체입니다.
예시: {"과목명":"소프트웨어 설계","주요항목":"요구사항 확인","세부항목":"요구사항 확인","세세항목":"요구분석기법"}
필요 시 '과목명 > 주요항목 > 세부항목 > 세세항목' 형태로 개념 경로를 구성하여 활용하십시오.
응답은 지정된 JSON 스키마만을 출력하고, 불필요한 자연어 설명은 포함하지 마십시오."""
                    },
                    {
                        "role": "user",
                        "content": f"""다음 학생의 문제 풀이 결과를 심층적으로 분석하고, 전문가 수준의 맞춤형 피드백을 생성해주세요.

{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

분석 시:
- mistakes.problem_type_path 를 사용해 개념 경로 기반으로 패턴을 도출
- 동일/유사 경로 반복 오답은 묶어서 패턴 설명
- 실수 유형은 가능한 한 구체화

아래 JSON 형식을 그대로 따르세요.
```json
{{
  "detailed_analysis": [
    {{
      "problem_number": "틀린 문제 번호",
      "concept_path": "문제의 개념 경로 (problem_type_path 활용)",
      "mistake_type": "실수 유형 (예: 개념 이해 부족, 계산 실수, 조건 누락)",
      "analysis": "왜 틀렸는지에 대한 구체적 원인 분석 (학생의 사고 과정 추정)",
      "recommendation": "실수를 교정하기 위한 구체적 학습/연습 제안"
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
            feedback_content = completion.choices[0].message.content
            parsed_feedback = json.loads(feedback_content)
            state["mistake_summary"] = json.dumps(parsed_feedback.get("detailed_analysis", {}), ensure_ascii=False, indent=2)
            state["final_feedback"] = json.dumps(parsed_feedback, ensure_ascii=False, indent=2)
        else:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 학생의 잠재력을 파악하고 더 높은 단계로 이끌어주는 전문 학습 코치입니다."
                    },
                    {
                        "role": "user",
                        "content": f"""학생은 모든 문제({len(grade_result)}문제)를 정답 처리했습니다.
problem_type 은 계층형 객체이며 flat 경로는 all_problems.problem_type_flat 에 있습니다.
이를 활용하여 개념적 강점을 구조적으로 설명하고 다음 학습 단계를 제안하세요.

{json.dumps(analysis_data["all_problems"], ensure_ascii=False, indent=2)}

JSON 형식:
{{
  "overall_assessment": {{
    "title": "완벽한 결과! 다음 도전을 위한 제안",
    "strengths_analysis": "개념 계층 기반 강점 분석",
    "deepen_learning_plan": {{
      "title": "심화 학습 계획",
      "recommendations": ["추천 1", "추천 2"],
      "recommended_resources": ["자료 1", "자료 2"]
    }},
    "final_message": "격려 메시지"
  }}
}}
한국어로 작성."""
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

        state["messages"].append(AIMessage(content="분석 및 피드백 생성 완료"))
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
                feedback_data = json.loads(result["final_feedback"]) if result["final_feedback"] else {}
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
                                "user_answer": input_data.get("user_answer", [])[i],
                                "correct_answer": input_data.get("solution_answer", [])[i],
                                # 추가: problem_type 경로 포함 (새 구조 추적)
                                "problem_type": input_data.get("problem_type", [])[i],
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
    """채점/분석 결과 출력 (score 엔진 단순 결과 + analysis 에이전트 결과 모두 지원)
    지원 형식 1 (단순 채점):
      {
        "problem": [...],
        "problem_type": [ { "과목명": "...", ... }, ... ],
        "user_answer": [...],
        "solution_answer": [...],
        "solution": [...],
        "status": "success",
        "total": 4,
        "correct": 2,
        "incorrect": 2,
        "score": 50.0,
        "answer_results": [
          {"index":0,"user":3,"solution":3,"correct":true}, ...
        ]
      }
    지원 형식 2 (기존 분석):
      {
        "status":"success",
        "metadata": {...},
        "grading": {"results":[...],"details":[...]},
        "analysis": {...}
      }
    """
    def flatten_problem_type(pt):
        if isinstance(pt, dict):
            keys = ["과목명", "주요항목", "세부항목", "세세항목"]
            parts = [str(pt.get(k)) for k in keys if pt.get(k)]
            return " > ".join(parts) if parts else json.dumps(pt, ensure_ascii=False)
        return str(pt)

    print("\n" + "="*20 + " 결과 출력 " + "="*20)

    # 공통 에러 처리
    if result.get("status") == "error":
        print(f"❌ 오류: {result.get('error_message') or result.get('message') or '알 수 없는 오류'}")
        return

    # 1) 새 단순 채점 구조
    if "answer_results" in result and "total" in result:
        total = result.get("total", 0)
        correct = result.get("correct", 0)
        score = result.get("score", 0)
        incorrect = result.get("incorrect", total - correct)
        print(f"\n[ 📊 채점 요약 ]")
        print(f"  - 총 문항: {total}")
        print(f"  - 정답: {correct}")
        print(f"  - 오답: {incorrect}")
        print(f"  - 점수: {score}점")

        problems = result.get("problem", [])
        problem_types = result.get("problem_type", [])
        solutions = result.get("solution", [])
        answer_results = result.get("answer_results", [])

        # 오답 수집
        wrong = [ar for ar in answer_results if not ar.get("correct")]
        if not wrong:
            print("\n🎉 모든 문항을 맞췄습니다! 훌륭합니다.")
        else:
            print("\n[ ❗ 오답 상세 ]")
            for ar in wrong:
                idx = ar["index"]
                prob_text = problems[idx] if idx < len(problems) else "(문항 없음)"
                user = ar.get("user")
                sol = ar.get("solution")
                concept_path = flatten_problem_type(problem_types[idx]) if idx < len(problem_types) else "-"
                explanation = solutions[idx] if idx < len(solutions) else ""
                # --- 수정: f-string 내부에서 replace('\n    ') 사용 대신 사전 계산 ---
                formatted_problem = prob_text.replace("\n", "\n    ")
                formatted_explanation = explanation.replace("\n", "\n    ") if explanation else ""
                print(f"\n● 문항 #{idx+1}")
                print(f"  - 개념 경로: {concept_path}")
                print(f"  - 사용자 답: {user}")
                print(f"  - 정답: {sol}")
                print("  - 문제:\n    " + formatted_problem)
                if explanation:
                    print("  - 해설:\n    " + formatted_explanation)

        # 정답도 간단 표
        print("\n[ ✅ 전체 문항 결과 ]")
        for ar in answer_results:
            idx = ar["index"]
            mark = "O" if ar["correct"] else "X"
            concept_path = flatten_problem_type(problem_types[idx]) if idx < len(problem_types) else "-"
            print(f"  #{idx+1:02d} {mark}  ({ar['user']} / {ar['solution']})  {concept_path}")

        print("\n" + "="*50)
        return

    # 2) 기존 분석 + grading 구조 (호환)
    metadata = result.get("metadata", {})
    grading = result.get("grading", {})
    analysis = result.get("analysis", {})

    if metadata:
        print(f"\n[ 📊 종합 성취도 ]")
        print(f"  - 점수: {metadata.get('score', 0)}점")
        print(f"  - 정답수: {metadata.get('correct_count', 0)} / {metadata.get('total_problems', 0)}")
        if "incorrect_count" in metadata:
            print(f"  - 오답수: {metadata.get('incorrect_count')}")

    details = grading.get("details", [])
    if details:
        print("\n[ 🔎 문항 결과 ]")
        for d in details:
            mark = "O" if d.get("is_correct") else "X"
            print(f"  #{d.get('problem_number')} {mark} (user={d.get('user_answer')}, correct={d.get('correct_answer')})")

    if analysis:
        print("\n[ 🧠 분석 요약 ]")
        if "overall_assessment" in analysis:
            oa = analysis["overall_assessment"]
            for k, v in oa.items():
                if isinstance(v, (str, int, float)):
                    print(f"  - {k}: {v}")
        if "detailed_analysis" in analysis:
            print("\n[ ❗ 오답 분석 ]")
            for item in analysis.get("detailed_analysis", []):
                print(f"  · 문제 {item.get('problem_number')}: {item.get('mistake_type')} / {item.get('recommendation')}")
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
