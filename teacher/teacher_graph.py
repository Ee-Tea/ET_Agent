# 에이전트 및 변수 명 정리 필요
# 에이전트 별 input / result 스테이트 토의 후 공유 스테이트 구조 확정 / 추가로 인풋 데이터도 확인해서 코드 수정해야함
# 그리고 답변 노드가 필요할까...?
import json
from pathlib import Path
import sys
from typing import Dict, Any
import os
from dotenv import load_dotenv
from langsmith import traceable
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableLambda

from agents.analisys.analysis_agent import AnalysisAgent, print_analysis_result, score_agent # score_agent는 바꿔야함
from agents.base_agent import BaseAgent
from teacher_nodes import user_intent
from ..common.short_term.redis_memory import *
from agents.retrieve.retrieve_agent import retrieve_agent  # 방금 만든 클래스 import
from TestGenerator.pdf_quiz_groq_class import generate_agent # 시험문제 생성 에이전트 (이거 아래로는 이름 바꿔야댐)
from solution.solution_agent import solution_agent  # 문제 풀이 에이전트

# 한 번만 생성해서 재사용 (비용 절약)
retriever = retrieve_agent()
generator = generate_agent()
solution = solution_agent()
score = score_agent()
analyst = AnalysisAgent()

from typing_extensions import TypedDict, NotRequired

class QuestionItem(TypedDict):
    id: str              # 고유 ID (문자열 권장)
    text: str            # 문제 본문
    type: NotRequired[str]           # "mcq" | "short" | "tf" 등
    tags: NotRequired[list[str]]     # 단원/유형 태그

class AnswerItem(TypedDict):
    qid: str
    answer: str          # "A" / "정답 텍스트" 등

class ExplanationItem(TypedDict):
    qid: str
    explanation: str

class NoteItem(TypedDict):
    qid: str
    note: str

class SharedState(TypedDict):
    # 요청하신 필드 이름을 그대로 맞췄습니다.
    question: NotRequired[list[QuestionItem]]          # 문제들
    # answer: NotRequired[list[AnswerItem]]              # 각 문제에 대한 정답
    # explanation: NotRequired[list[ExplanationItem]]    # 해설
    wrong_question: NotRequired[list[str]]             # 틀린 문제 ID 목록
    weak_type: NotRequired[list[str]]                  # 취약 유형(태그/단원명 등)
    retrieve_answer: NotRequired[str]                  # 검색 결과 요약/답변
    notes: NotRequired[list[NoteItem]]                 # 오답노트
class TeacherState(TypedDict):
    user_query: str
    intent: str
    # 공통/공유: 여러 노드가 읽는 값
    shared: NotRequired[SharedState]
    # 개별 그래프 영역
    retrieval: NotRequired[dict]
    generation: NotRequired[dict]
    solution: NotRequired[dict]
    score: NotRequired[dict]
    analysis: NotRequired[dict]


class Orchestrator:
    """
    전체 워크플로우를 관리하고, 사용자 요청에 따라 
    적절한 에이전트를 선택하고 실행하는 오케스트레이터 클래스입니다.
    """
    def __init__(self, user_id: str, service: str, chat_id: str):
    # .env 파일에서 환경 변수 로드
        load_dotenv()
        # LangSmith 추적 환경 변수 확인
        if not os.getenv("LANGCHAIN_API_KEY"):
            print("경고: LANGCHAIN_API_KEY 환경 변수가 설정되지 않았습니다.")
            print(".env 파일에 키를 추가하거나 직접 환경 변수를 설정해주세요.")
        self.memory = RedisLangGraphMemory(user_id=user_id, service=service, chat_id=chat_id)

    # load / save 래퍼 (RunnableLambda용)
    def load_state(self, state: TeacherState) -> TeacherState:
        return self.memory.load(state)

    def persist_state(self, state: TeacherState) -> TeacherState:
        self.memory.save(state, state)
        return state
    
    def intent_classifier(state: TeacherState) -> TeacherState:
        """
        사용자의 의도를 분류하는 노드입니다.
        
        Args:
            state (TeacherState): 현재 상태 정보.
        
        Returns:
            TeacherState: 업데이트된 상태 정보.
        """
        user_query = state["user_query"]
        # 간단한 키워드 기반 의도 분류 예시
        intent = user_intent(user_query)
        print(f"사용자 의도 분류: {intent}")
        return {"intent": intent, "user_query": user_query}
    
    def select_agent(state: TeacherState):
        """
        사용자의 의도에 따라 적절한 에이전트를 선택합니다.
        
        Args:
            state (TeacherState): 현재 상태 정보.
        
        Returns:
            TeacherState: 선택된 에이전트 정보가 포함된 상태 정보.
        """
        intent = state["intent"]
        if intent == "retrieve":
            agent_name = "retrieve"
        elif intent == "generate":
            agent_name = "problem_generation"
        elif intent == "analyze":
            agent_name = "analysis"
        elif intent == "solution":
            agent_name = "solution"
        elif intent == "score":
            agent_name = "score"
        else:
            raise ValueError(f"알 수 없는 의도: {intent}")
        
        print(f"선택된 에이전트: {agent_name}")
        return agent_name
    
    def generator(state: TeacherState) -> TeacherState:
        """
        문제 생성 노드입니다. 현재는 단순히 상태를 반환합니다.
        
        Args:
            state (TeacherState): 현재 상태 정보.
        
        Returns:
            TeacherState: 업데이트된 상태 정보.
        """
        # 문제 생성 로직을 여기에 추가할 수 있습니다.
        print("문제 생성 노드 실행")
        # 1. 에이전트 실행
        agent_input = {
            "query": state.get("user_query", "")
        }
        agent_result = generator.execute(agent_input)

        # 2. TeacherState에 결과 병합
        new_state = dict(state)
        new_state.setdefault("generation", {})
        new_state["generation"].update(agent_result)

        # 필요 시 공유 영역(shared)에도 승격, 문제, 답, 해설 저장
        # if "retrieve_answer" in agent_result:
        #     new_state.setdefault("shared", {})
        #     new_state["shared"]["retrieve_answer"] = agent_result["retrieve_answer"]

        return new_state
    
    def solution(state: TeacherState) -> TeacherState:
        """
        문제 풀이 노드입니다. 현재는 단순히 상태를 반환합니다.
        
        Args:
            state (TeacherState): 현재 상태 정보.
        
        Returns:
            TeacherState: 업데이트된 상태 정보.
        """
        # 문제 풀이 로직을 여기에 추가할 수 있습니다.
        print("문제 풀이 노드 실행")
        # 1. 에이전트 실행
        # agent_input = {
        #     "retrieval_question": state.get("user_query", "")
        # }
        agent_result = solution.execute()

        # 2. TeacherState에 결과 병합
        new_state = dict(state)
        new_state.setdefault("solution", {})
        new_state["solution"].update(agent_result)

        # 필요 시 공유 영역(shared)에도 승격, 문제, 답, 해설 저장
        # if "retrieve_answer" in agent_result:
        #     new_state.setdefault("shared", {})
        #     new_state["shared"]["retrieve_answer"] = agent_result["retrieve_answer"]
        return new_state
    
    def score(state: TeacherState) -> TeacherState:
        """
        채점 노드입니다. 현재는 단순히 상태를 반환합니다.
        
        Args:
            state (TeacherState): 현재 상태 정보.
        
        Returns:
            TeacherState: 업데이트된 상태 정보.
        """
        # 채점 로직을 여기에 추가할 수 있습니다.
        print("채점 노드 실행")
        # 1. 에이전트 실행
        # agent_input = {
        #     "retrieval_question": state.get("user_query", "")
        # }
        agent_result = score.execute()

        # 2. TeacherState에 결과 병합
        new_state = dict(state)
        new_state.setdefault("score", {})
        new_state["score"].update(agent_result)

        # 필요 시 공유 영역(shared)에도 승격, 문제, 답, 해설 저장
        # if "retrieve_answer" in agent_result:
        #     new_state.setdefault("shared", {})
        #     new_state["shared"]["retrieve_answer"] = agent_result["retrieve_answer"]
        return new_state
    
    def analysis(state: TeacherState) -> TeacherState:
        """
        오답 분석 노드입니다. 현재는 단순히 상태를 반환합니다.
        
        Args:
            state (TeacherState): 현재 상태 정보.
        
        Returns:
            TeacherState: 업데이트된 상태 정보.
        """
        # 오답 분석 로직을 여기에 추가할 수 있습니다.
        print("오답 분석 노드 실행")
        # 1. 에이전트 실행
        # agent_input = {
        #     "retrieval_question": state.get("user_query", "")
        # }
        agent_result = analyst.execute()

        # 2. TeacherState에 결과 병합
        new_state = dict(state)
        new_state.setdefault("analysis", {})
        new_state["analysis"].update(agent_result)

        # 필요 시 공유 영역(shared)에도 승격, 문제, 답, 해설 저장
        # if "retrieve_answer" in agent_result:
        #     new_state.setdefault("shared", {})
        #     new_state["shared"]["retrieve_answer"] = agent_result["retrieve_answer"]
        return new_state
    
    def retrieve(state: TeacherState) -> TeacherState:
        """
        정보 검색 노드입니다. 현재는 단순히 상태를 반환합니다.
        
        Args:
            state (TeacherState): 현재 상태 정보.
        
        Returns:
            TeacherState: 업데이트된 상태 정보.
        """
        # 정보 검색 로직을 여기에 추가할 수 있습니다.
        print("정보 검색 노드 실행")
            # 1. 에이전트 실행
        agent_input = {
            "retrieval_question": state.get("user_query", "")
        }
        agent_result = retriever.execute(agent_input)

        # 2. TeacherState에 결과 병합
        new_state = dict(state)
        new_state.setdefault("retrieval", {})
        new_state["retrieval"].update(agent_result)

        # 필요 시 공유 영역(shared)에도 승격
        if "retrieve_answer" in agent_result:
            new_state.setdefault("shared", {})
            new_state["shared"]["retrieve_answer"] = agent_result["retrieve_answer"]

        return new_state
    
    def build_teacher_graph(self):
        builder = StateGraph(TeacherState)
            # 2) 시작점 → 로드 → 분기
        builder.add_edge(START, "load_state")
        builder.add_edge("load_state", "intent_classifier")
        
        builder.add_node("load_state", RunnableLambda(self.load_state))
        builder.add_node("persist_state", RunnableLambda(self.persist_state))
        builder.add_node("intent_classifier", RunnableLambda(self.intent_classifier))
        builder.add_node("generator", RunnableLambda(self.generator))
        builder.add_node("solution", RunnableLambda(self.solution))
        builder.add_node("score", RunnableLambda(self.score))
        builder.add_node("analysis", RunnableLambda(self.analysis))
        builder.add_node("retrieve", RunnableLambda(self.retrieve))
        
        builder.add_conditional_edges(
            "intent_classifier",
            self.select_agent,
            {
                "retrieve": "retrieve",
                "generate": "generator",
                "analyze": "analysis",
                "solution": "solution",
                "score": "score"
            }
        )
        for leaf in ["retrieve", "generator", "solution", "score", "analysis"]:
            builder.add_edge(leaf, "persist_state")
        
        return builder.compile()
        
    
    #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    def get_available_agents(self) -> Dict[str, str]:
        """등록된 에이전트들의 이름과 설명을 반환합니다."""
        return {agent_key: agent.description for agent_key, agent in self.agents.items()}

    @traceable(name="Orchestrator Run")
    def run(self, agent_name: str, input_file_path: str):
        """
        지정된 에이전트를 실행하고, 파일 입출력을 처리합니다.
        
        Args:
            agent_name (str): 실행할 에이전트의 이름입니다.
            input_file_path (str): 에이전트에 전달할 입력 데이터 파일 경로입니다.
        """
        # 1. 에이전트 선택
        agent = self.agents.get(agent_name)
        if not agent:
            print(f"오류: '{agent_name}'이라는 이름의 에이전트를 찾을 수 없습니다.")
            sys.exit(1)
            
        # 2. 입력 파일 로드
        input_file = Path(input_file_path)
        if not input_file.exists():
            print(f"오류: 입력 파일을 찾을 수 없습니다. ({input_file})")
            sys.exit(1)
            
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            print(f"파일 '{input_file}' 로드 성공")
        except json.JSONDecodeError:
            print(f"오류: '{input_file}'이 올바른 JSON 형식이 아닙니다.")
            sys.exit(1)
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
            sys.exit(1)

        # 3. 에이전트 실행
        try:
            print(f"🚀 '{agent_name}' 에이전트 실행 시작...")
            result = agent.execute(input_data)
            
            # 결과 검증
            if not isinstance(result, dict):
                raise ValueError("에이전트가 올바른 형식의 결과를 반환하지 않았습니다.")
                
            if result.get("status") == "error":
                print(f"⚠️  에이전트 실행 중 내부 오류 발생: {result.get('error_message', '알 수 없는 오류')}")
            else:
                print(f"✅ '{agent_name}' 에이전트 실행 완료.")
                
        except Exception as e:
            print(f"❌ 에이전트 실행 중 오류 발생: {e}")
            # 오류 발생 시에도 기본 구조로 결과 생성
            result = {
                "status": "error",
                "error_message": str(e),
                "metadata": {
                    "total_problems": 0,
                    "correct_count": 0,
                    "score": 0
                },
                "grading": {"results": [], "details": []},
                "analysis": {},
                "raw_data": {}
            }
            
        # 4. 결과 처리 및 저장
        self.handle_result(result, agent_name, input_file)

    def handle_result(self, result: Dict[str, Any], agent_name: str, input_file: Path):
        """
        에이전트 실행 결과를 처리하고 저장합니다.
        
        Args:
            result (Dict[str, Any]): 에이전트 실행 결과.
            agent_name (str): 실행된 에이전트의 이름.
            input_file (Path): 원본 입력 파일의 경로.
        """
        # 결과 출력
        if agent_name == "analysis":
            print_analysis_result(result)
        else:
            # 다른 에이전트들의 결과 출력 로직
            print("\n--- 실행 결과 ---")
            if result.get("status") == "success":
                print("✅ 성공적으로 완료되었습니다.")
                if "metadata" in result:
                    print("\n📊 메타데이터:")
                    for key, value in result["metadata"].items():
                        print(f"  - {key}: {value}")
            else:
                print("❌ 오류가 발생했습니다.")
                if "error_message" in result:
                    print(f"오류 메시지: {result['error_message']}")
            
            print("\n전체 결과:")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        # 결과 파일 저장
        output_file = input_file.with_name(f"{input_file.stem}_{agent_name}_result.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 결과가 '{output_file}'에 저장되었습니다.")
            
            # 요약 정보 출력
            if result.get("status") == "success" and result.get("metadata"):
                metadata = result["metadata"]
                print(f"\n📋 저장된 결과 요약:")
                print(f"  - 상태: {result.get('status', '알 수 없음')}")
                if agent_name == "analysis":
                    print(f"  - 총 문제 수: {metadata.get('total_problems', 0)}")
                    print(f"  - 정답률: {metadata.get('score', 0)}%")
                    print(f"  - 오답 여부: {'있음' if metadata.get('has_mistakes', False) else '없음'}")
                    
        except Exception as e:
            print(f"❌ 결과 저장 중 오류 발생: {e}")
            print("결과가 저장되지 않았지만, 분석은 완료되었습니다.")


if __name__ == "__main__":
    # 명령줄 인자 파싱
    # 예: python teacher.py analysis "path/to/your/input.json"
    if len(sys.argv) < 3:
        print("🔧 사용법: python teacher.py [agent_name] [input_file_path]")
        
        # 오케스트레이터 인스턴스 생성하여 등록된 에이전트 정보 가져오기
        orchestrator = Orchestrator()
        available_agents = orchestrator.get_available_agents()
        
        print("\n📋 사용 가능한 에이전트:")
        for agent_name, description in available_agents.items():
            print(f"  - {agent_name}: {description}")
        
        print("\n💡 예시:")
        if "analysis" in available_agents:
            print("  python teacher.py analysis ./test_sample/analysis_sample.json")
            print("  python teacher.py analysis C:/path/to/student_answers.json")
        else:
            # 첫 번째 등록된 에이전트를 예시로 사용
            first_agent = next(iter(available_agents.keys())) if available_agents else "agent_name"
            print(f"  python teacher.py {first_agent} ./path/to/input.json")
        sys.exit(1)
        
    agent_to_run = sys.argv[1]
    file_path = sys.argv[2]
    
    print("🎓 ET_Agent Teacher System")
    print("=" * 40)
    
    orchestrator = Orchestrator()
    orchestrator.run(agent_name=agent_to_run, input_file_path=file_path)
