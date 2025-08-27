import os
from typing import TypedDict, List, Dict, Literal, Optional, Tuple, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import connections
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
import json, re
from langchain_openai import ChatOpenAI
from ..base_agent import BaseAgent
from datetime import datetime

# 검색 에이전트 import (retrieve_agent 연동)
try:
    from ..retrieve.retrieve_agent import retrieve_agent
    SEARCH_AGENT_AVAILABLE = True
except ImportError:
    SEARCH_AGENT_AVAILABLE = False
    print("⚠️ 검색 에이전트를 import할 수 없습니다. 검색 기능은 비활성화됩니다.")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ✅ 상태 정의 (HITL 기능 추가)
class SolutionState(TypedDict):
    # 사용자 입력
    user_input_txt: str

    # 문제리스트, 문제, 보기
    user_problem: str
    user_problem_options: List[str]
    
    vectorstore: Milvus

    retrieved_docs: List[Document]
    similar_questions_text : str

    # 문제 해답/풀이/과목 생성
    generated_answer: str         # 해답
    generated_explanation: str   # 풀이
    generated_subject: str

    # HITL 관련 상태
    user_feedback: str           # 사용자 피드백
    feedback_type: str           # 피드백 유형 (comprehension, clarification, improvement)
    search_results: str          # 검색 에이전트 결과
    improved_explanation: str    # 개선된 풀이
    interaction_count: int       # 상호작용 횟수
    max_interactions: int        # 최대 상호작용 횟수
    
    results: List[Dict]
    validated: bool
    retry_count: int             # 검증 실패 시 재시도 횟수

    chat_history: List[str]
    
class SolutionAgent(BaseAgent):
    """Human-in-the-Loop가 포함된 문제 해답/풀이 생성 에이전트"""

    def __init__(self, max_interactions: int = 5):
        self.max_interactions = max_interactions
        self.graph = self._create_graph()
        
    @property
    def name(self) -> str:
        return "SolutionAgent"

    @property
    def description(self) -> str:
        return "사용자와의 상호작용을 통해 풀이를 개선하는 문제 해답/풀이 생성 에이전트입니다."

    def _llm(self, temperature: float = 0):
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            model=OPENAI_LLM_MODEL,
            temperature=temperature,
        )

    def _create_graph(self) -> StateGraph:
        """HITL 워크플로우 그래프 생성"""

        print("📚 HITL LangGraph 흐름 구성 중...")
        
        graph = StateGraph(SolutionState)

        # 기본 노드들
        graph.add_node("search_similarity", self._search_similar_questions)
        graph.add_node("generate_solution", self._generate_solution)
        graph.add_node("validate", self._validate_solution)
        
        # HITL 노드들
        graph.add_node("collect_user_feedback", self._collect_user_feedback)
        graph.add_node("process_feedback", self._process_feedback)
        graph.add_node("search_additional_info", self._search_additional_info)
        graph.add_node("improve_explanation", self._improve_explanation)
        graph.add_node("store", self._store_to_vector_db)

        # 워크플로우 설정
        graph.set_entry_point("search_similarity")
        graph.add_edge("search_similarity", "generate_solution")
        graph.add_edge("generate_solution", "validate")
        
        # HITL 조건부 엣지
        graph.add_conditional_edges(
            "validate", 
            self._route_after_validation,
            {"ok": "store", "feedback_needed": "collect_user_feedback", "retry": "generate_solution"}
        )
        
        graph.add_conditional_edges(
            "collect_user_feedback",
            self._route_after_feedback,
            {"continue": "store", "improve": "process_feedback", "search": "search_additional_info"}
        )
        
        graph.add_edge("process_feedback", "improve_explanation")
        graph.add_edge("search_additional_info", "improve_explanation")
        graph.add_edge("improve_explanation", "collect_user_feedback")

        return graph.compile()
    
    def _route_after_validation(self, state: SolutionState) -> str:
        """검증 후 라우팅 결정"""
        if state["validated"]:
            return "ok"
        elif state.get("retry_count", 0) < 3:
            return "retry"
        else:
            return "feedback_needed"
    
    def _route_after_feedback(self, state: SolutionState) -> str:
        """사용자 피드백 후 라우팅 결정"""
        feedback = state.get("user_feedback", "").lower()
        
        if "만족" in feedback or "좋음" in feedback or "이해" in feedback:
            return "continue"
        elif "검색" in feedback or "찾아" in feedback or "설명" in feedback:
            return "search"
        else:
            return "improve"
    
    #----------------------------------------nodes------------------------------------------------------

    def _search_similar_questions(self, state: SolutionState) -> SolutionState:
        print("\n🔍 [1단계] 유사 문제 검색 시작")
        print(state["user_problem"], state["user_problem_options"])
        
        vectorstore = state.get("vectorstore")
        if vectorstore is None:
            print("⚠️ 벡터스토어가 없어 유사 문제 검색을 건너뜁니다.")
            state["retrieved_docs"] = []
            state["similar_questions_text"] = ""
            print("🔍 [1단계] 유사 문제 검색 함수 종료 (건너뜀)")
            return state
        
        try:
            results = vectorstore.similarity_search(state["user_problem"], k=3)
        except Exception as e:
            print(f"⚠️ 유사 문제 검색 실패: {e}")
            results = []
        
        similar_questions = []
        for i, doc in enumerate(results):
            metadata = doc.metadata
            options = json.loads(metadata.get("options", "[]"))
            answer = metadata.get("answer", "")
            explanation = metadata.get("explanation", "")
            subject = metadata.get("subject", "기타")

            formatted = f"""[유사문제 {i+1}]
                문제: {doc.page_content}
                보기:
                """ + "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)]) + f"""
                정답: {answer}
                풀이: {explanation}
                과목: {subject}
                """
            similar_questions.append(formatted)
        
        state["retrieved_docs"] = results
        state["similar_questions_text"] = "\n\n".join(similar_questions) 

        print(f"유사 문제 {len(results)}개 검색 완료.")
        print("🔍 [1단계] 유사 문제 검색 함수 종료")
        return state

    def _generate_solution(self, state: SolutionState) -> SolutionState:
        print("\n✏️ [2단계] 해답 및 풀이 생성 시작")

        llm_gen = self._llm(0.5)  

        similar_problems = state.get("similar_questions_text", "")
        print("유사 문제들:\n", similar_problems[:100])

        prompt = f"""
            사용자가 입력한 질문:
            {state['user_input_txt']}

            다음은 사용자가 입력한 문제:
            {state['user_problem']}
            {state['user_problem_options']}

            아래는 이 문제와 유사한 문제들:
            {similar_problems}

            1. 사용자가 입력한 문제의 **정답**의 보기 번호를 정답으로 작성해 주세요.
            2. 이어서 그 정답인 근거를 담은 **풀이 과정**을 상세히 설명해 주세요.
            3. 이 문제의 과목을 정보처리기사 과목 5개 중에서 가장 적합한 것으로 지정해 주세요. 유사 문제들의 과목을 참고해도 좋습니다. [소프트웨어설계, 소프트웨어개발, 데이터베이스구축, 프로그래밍언어활용, 정보시스템구축관리]

            출력 형식:
            정답: ...
            풀이: ...
            과목: ...
        """

        response = llm_gen.invoke(prompt)
        result = response.content.strip()
        print("🧠 LLM 응답 완료")

        answer_match = re.search(r"정답:\s*(.+)", result)
        explanation_match = re.search(r"풀이:\s*(.+)", result, re.DOTALL)
        subject_match = re.search(r"과목:\s*(.+)", result)
        
        state["generated_answer"] = answer_match.group(1).strip() if answer_match else ""
        state["generated_explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        state["generated_subject"] = subject_match.group(1).strip() if subject_match else "기타"

        state["chat_history"].append(f"Q: {state['user_input_txt']}\nP: {state['user_problem']}\nA: {state['generated_answer']}\nE: {state['generated_explanation']}")

        return state

    # ✅ 정합성 검증 (간단히 길이 기준 사용)

    def _validate_solution(self, state: SolutionState) -> SolutionState:
        print("\n🧐 [3단계] 정합성 검증 시작")
        
        llm = self._llm(0)

        validation_prompt = f"""
        사용자 요구사항: {state['user_input_txt']}

        문제 질문: {state['user_problem']}
        문제 보기: {state['user_problem_options']}

        생성된 정답: {state['generated_answer']}
        생성된 풀이: {state['generated_explanation']}
        생성된 과목: {state['generated_subject']}

        생성된 해답과 풀이, 과목이 문제와 사용자 요구사항에 맞고, 논리적 오류나 잘못된 정보가 없습니까?
        적절하다면 '네', 그렇지 않다면 '아니오'로만 답변하세요.
        """

        validation_response = llm.invoke(validation_prompt)
        result_text = validation_response.content.strip().lower()

        # ✅ '네'가 포함된 응답일 경우에만 유효한 풀이로 판단
        print("📌 검증 응답:", result_text)
        state["validated"] = "네" in result_text
        
        if not state["validated"]:
            state["retry_count"] = state.get("retry_count", 0) + 1
            print(f"⚠️ 검증 실패 (재시도 {state['retry_count']}/5)")
        else:
            print("✅ 검증 결과: 통과")
            
        return state

    def _collect_user_feedback(self, state: SolutionState) -> SolutionState:
        """사용자로부터 풀이에 대한 피드백 수집"""
        print("\n💬 [HITL] 사용자 피드백 수집")
        
        # 실제 환경에서는 사용자 입력을 받아야 하지만, 여기서는 시뮬레이션
        print(f"\n📝 현재 풀이:")
        print(f"정답: {state['generated_answer']}")
        print(f"풀이: {state['generated_explanation']}")
        print(f"과목: {state['generated_subject']}")
        
        # 사용자 피드백 시뮬레이션 (실제로는 input() 사용)
        feedback_options = [
            "이 풀이가 이해가 됩니다. 만족합니다.",
            "풀이를 더 쉽게 설명해주세요.",
            "특정 용어에 대해 더 자세히 설명해주세요.",
            "검색해서 관련 정보를 찾아주세요."
        ]
        
        print("\n💭 피드백 옵션:")
        for i, option in enumerate(feedback_options, 1):
            print(f"{i}. {option}")
        
        # 실제 환경에서는 사용자 입력을 받아야 함
        # feedback_choice = input("\n어떤 피드백을 주시겠습니까? (1-4): ").strip()
        
        # 시뮬레이션을 위해 자동 선택 (실제로는 사용자 입력)
        feedback_choice = "2"  # 풀이를 더 쉽게 설명해주세요
        
        try:
            choice_idx = int(feedback_choice) - 1
            if 0 <= choice_idx < len(feedback_options):
                state["user_feedback"] = feedback_options[choice_idx]
                if choice_idx == 0:  # 만족
                    state["feedback_type"] = "comprehension"
                elif choice_idx == 1:  # 더 쉽게
                    state["feedback_type"] = "improvement"
                elif choice_idx == 2:  # 용어 설명
                    state["feedback_type"] = "clarification"
                else:  # 검색
                    state["feedback_type"] = "search"
            else:
                state["user_feedback"] = "풀이를 더 쉽게 설명해주세요."
                state["feedback_type"] = "improvement"
        except ValueError:
            state["user_feedback"] = "풀이를 더 쉽게 설명해주세요."
            state["feedback_type"] = "improvement"
        
        state["interaction_count"] = state.get("interaction_count", 0) + 1
        print(f"✅ 피드백 수집 완료: {state['user_feedback']}")
        
        return state

    def _process_feedback(self, state: SolutionState) -> SolutionState:
        """사용자 피드백을 분석하고 처리 방향 결정"""
        print(f"\n🔄 [HITL] 피드백 처리: {state['feedback_type']}")
        
        feedback_type = state.get("feedback_type", "improvement")
        
        if feedback_type == "improvement":
            print("📝 풀이를 더 쉽게 개선하겠습니다.")
        elif feedback_type == "clarification":
            print("🔍 특정 용어에 대해 더 자세히 설명하겠습니다.")
        elif feedback_type == "search":
            print("🔎 관련 정보를 검색하겠습니다.")
        
        return state

    def _search_additional_info(self, state: SolutionState) -> SolutionState:
        """검색 에이전트를 사용하여 추가 정보 검색"""
        print(f"\n🔎 [HITL] 추가 정보 검색 시작")
        
        if not SEARCH_AGENT_AVAILABLE:
            print("⚠️ 검색 에이전트를 사용할 수 없습니다.")
            state["search_results"] = "검색 에이전트를 사용할 수 없습니다."
            return state
        
        try:
            # 검색 에이전트 실행
            search_query = f"{state['user_problem']} {state['generated_explanation']}"
            search_results = retrieve_agent().execute({
                "query": search_query,
                "max_results": 3
            })
            
            if search_results and "results" in search_results:
                # 검색 결과를 텍스트로 변환
                results_text = []
                for i, result in enumerate(search_results["results"][:3]):
                    results_text.append(f"[검색결과 {i+1}]\n{result.get('content', '')}")
                
                state["search_results"] = "\n\n".join(results_text)
                print(f"✅ 검색 완료: {len(search_results['results'])}개 결과")
            else:
                state["search_results"] = "검색 결과를 찾을 수 없습니다."
                print("⚠️ 검색 결과가 없습니다.")
                
        except Exception as e:
            print(f"⚠️ 검색 중 오류 발생: {e}")
            state["search_results"] = f"검색 중 오류가 발생했습니다: {e}"
        
        return state

    def _improve_explanation(self, state: SolutionState) -> SolutionState:
        """사용자 피드백과 검색 결과를 바탕으로 풀이 개선"""
        print(f"\n✨ [HITL] 풀이 개선 시작")
        
        llm = self._llm(0.3)
        
        # 개선 프롬프트 구성
        improvement_prompt = f"""
        원본 문제: {state['user_problem']}
        원본 보기: {state['user_problem_options']}
        원본 정답: {state['generated_answer']}
        원본 풀이: {state['generated_explanation']}
        
        사용자 피드백: {state['user_feedback']}
        피드백 유형: {state['feedback_type']}
        
        추가 검색 결과:
        {state.get('search_results', '검색 결과 없음')}
        
        위 정보를 바탕으로 풀이를 개선해주세요:
        
        1. 피드백 유형에 따라 적절히 개선:
           - improvement: 더 쉽고 이해하기 쉽게 설명
           - clarification: 특정 용어나 개념을 더 자세히 설명
           - search: 검색 결과를 활용하여 풀이를 보강
        
        2. 출력 형식:
        정답: [개선된 정답]
        풀이: [개선된 풀이]
        과목: [과목]
        개선사항: [어떤 부분을 어떻게 개선했는지 설명]
        """
        
        try:
            response = llm.invoke(improvement_prompt)
            result = response.content.strip()
            
            # 개선된 결과 파싱
            answer_match = re.search(r"정답:\s*(.+)", result)
            explanation_match = re.search(r"풀이:\s*(.+)", result, re.DOTALL)
            subject_match = re.search(r"과목:\s*(.+)", result)
            improvement_match = re.search(r"개선사항:\s*(.+)", result, re.DOTALL)
            
            if answer_match:
                state["generated_answer"] = answer_match.group(1).strip()
            if explanation_match:
                state["improved_explanation"] = explanation_match.group(1).strip()
            if subject_match:
                state["generated_subject"] = subject_match.group(1).strip()
            if improvement_match:
                improvement_note = improvement_match.group(1).strip()
            else:
                improvement_note = "사용자 피드백에 따라 풀이를 개선했습니다."
            
            # 개선된 풀이를 메인 풀이로 설정
            if state.get("improved_explanation"):
                state["generated_explanation"] = state["improved_explanation"]
            
            # 채팅 히스토리에 개선 과정 추가
            state["chat_history"].append(f"개선: {improvement_note}")
            
            print(f"✅ 풀이 개선 완료: {improvement_note}")
            
        except Exception as e:
            print(f"⚠️ 풀이 개선 중 오류 발생: {e}")
            state["improved_explanation"] = state["generated_explanation"]
        
        return state

    # ✅ 임베딩 후 벡터 DB 저장
    def _store_to_vector_db(self, state: SolutionState) -> SolutionState:  
        print("\n🧩 [4단계] 임베딩 및 벡터 DB 저장 시작")

    
        vectorstore = state["vectorstore"] 

        # 중복 문제 확인
        similar = vectorstore.similarity_search(state["user_problem"], k=1)
        if similar and state["user_problem"].strip() in similar[0].page_content:
            print("⚠️ 동일한 문제가 존재하여 저장 생략")
        else:
            # 문제, 해답, 풀이를 각각 metadata로 저장
            doc = Document(
                page_content=state["user_problem"],
                metadata={
                    "options": json.dumps(state.get("user_problem_options", [])), 
                    "answer": state["generated_answer"],
                    "explanation": state["generated_explanation"],
                    "subject": state["generated_subject"],
                }
            )
            vectorstore.add_documents([doc])
            print("✅ 문제+해답+풀이 저장 완료")

        # 결과를 state에 저장 (항상 실행)
        print(f"\n📝 결과 저장 시작:")
        print(f"   - 현재 문제: {state['user_problem'][:50]}...")
        print(f"   - 생성된 정답: {state['generated_answer'][:30]}...")
        print(f"   - 검증 상태: {state['validated']}")
        print(f"   - 상호작용 횟수: {state.get('interaction_count', 0)}")
        
        item = {
            "user_problem": state["user_problem"],
            "user_problem_options": state["user_problem_options"],
            "generated_answer": state["generated_answer"],
            "generated_explanation": state["generated_explanation"],
            "generated_subject": state["generated_subject"],
            "validated": state["validated"],
            "interaction_count": state.get("interaction_count", 0),
            "user_feedback": state.get("user_feedback", ""),
            "chat_history": state.get("chat_history", [])
        }
        
        state["results"].append(item)
        print(f"✅ 결과 저장 완료: {len(state['results'])}개")
        
        return state

    def invoke(
            self, 
            user_input_txt: str,
            user_problem: str,
            user_problem_options: List[str],
            vectorstore: Optional[Milvus] = None,
            recursion_limit: int = 1000,
        ) -> Dict:

        # ✅ Milvus 연결 및 벡터스토어 생성
        if vectorstore is None:
            try:
                embedding_model = HuggingFaceEmbeddings(
                    model_name="jhgan/ko-sroberta-multitask",
                    model_kwargs={"device": "cpu"}
                )

                if "default" in connections.list_connections():
                    connections.disconnect("default")
                connections.connect(alias="default", host="localhost", port="19530")

                vectorstore = Milvus(
                    embedding_function=embedding_model,
                    collection_name="problems",
                    connection_args={"host": "localhost", "port": "19530"}
                )
                print("✅ Milvus 벡터스토어 연결 성공")
            except Exception as e:
                print(f"⚠️ Milvus 연결 실패: {e}")
                print("   - 벡터스토어 없이 실행을 계속합니다.")
                vectorstore = None
        
        initial_state: SolutionState = {
            "user_input_txt": user_input_txt,
            "user_problem": user_problem,
            "user_problem_options": user_problem_options,
            "vectorstore": vectorstore,
            "retrieved_docs": [],
            "similar_questions_text": "",
            "generated_answer": "",
            "generated_explanation": "",
            "generated_subject": "",
            "validated": False,
            "retry_count": 0,
            "user_feedback": "",
            "feedback_type": "",
            "search_results": "",
            "improved_explanation": "",
            "interaction_count": 0,
            "max_interactions": self.max_interactions,
            "results": [],
            "chat_history": []
        }
        
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})

        # 결과 확인 및 디버깅
        results = final_state.get("results", [])
        print(f"   - 총 결과 수: {len(results)}")
        
        if results:
            for i, result in enumerate(results):
                print(f"   - 결과 {i+1}: {result.get('user_problem', '')[:30]}...")
                print(f"     상호작용 횟수: {result.get('interaction_count', 0)}")
        else:
            print("   ⚠️ results가 비어있습니다!")
            print(f"   - final_state 내용: {final_state}")
        
        return final_state


if __name__ == "__main__":
    # ✅ Milvus 연결 및 벡터스토어 생성
    embedding_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"}
    )

    if "default" in connections.list_connections():
        connections.disconnect("default")
    connections.connect(alias="default", host="localhost", port="19530")

    vectorstore = Milvus(
        embedding_function=embedding_model,
        collection_name="problems",
        connection_args={"host": "localhost", "port":"19530"}
    )

    agent = SolutionAgent(max_interactions=5)

    user_input_txt = input("\n❓ 사용자 질문: ").strip()
    user_problem = input("\n❓ 사용자 문제: ").strip()
    user_problem_options_raw = input("\n❓ 사용자 보기 (쉼표로 구분): ").strip()
    user_problem_options = [opt.strip() for opt in user_problem_options_raw.split(",") if opt.strip()]

    final_state = agent.invoke(
        user_input_txt=user_input_txt,
        user_problem=user_problem,
        user_problem_options=user_problem_options,
        vectorstore=vectorstore,
    )

    print(f"\n🎯 최종 결과:")
    print(f"문제: {final_state.get('user_problem', '')}")
    print(f"정답: {final_state.get('generated_answer', '')}")
    print(f"풀이: {final_state.get('generated_explanation', '')}")
    print(f"과목: {final_state.get('generated_subject', '')}")
    print(f"상호작용 횟수: {final_state.get('interaction_count', 0)}")
    print(f"사용자 피드백: {final_state.get('user_feedback', '')}")
