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


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ✅ 상태 정의
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

    results: List[Dict]
    validated: bool
    retry_count: int             # 검증 실패 시 재시도 횟수

    chat_history: List[str]
    
class SolutionAgent(BaseAgent):
    """문제 해답/풀이 생성 에이전트"""

    def __init__(self):
        self.graph = self._create_graph()
        
    @property
    def name(self) -> str:
        return "SolutionAgent"

    @property
    def description(self) -> str:
        return "시험문제를 인식하여 답과 풀이, 해설을 제공하는 에이전트입니다."

    def _llm(self, temperature: float = 0):
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            model=OPENAI_LLM_MODEL,  # ✅ 환경변수에서 가져온 모델
            temperature=temperature,
        )

    def _create_graph(self) -> StateGraph:
        """워크플로우 그래프 생성"""

        # ✅ LangGraph 구성
        print("📚 LangGraph 흐름 구성 중...")
        
        graph = StateGraph(SolutionState)

        # 공통 처리
        graph.add_node("search_similarity", self._search_similar_questions)
        graph.add_node("generate_solution", self._generate_solution)
        graph.add_node("validate", self._validate_solution)
        graph.add_node("store", self._store_to_vector_db)

        graph.set_entry_point("search_similarity")
        graph.add_edge("search_similarity", "generate_solution")
        graph.add_edge("generate_solution", "validate")
        graph.add_edge("store", END)

        graph.add_conditional_edges(
            "validate", 
            lambda s: "ok" if s["validated"] else ("back" if s.get("retry_count", 0) < 5 else END),
            {"ok": "store", "back": "generate_solution"}
        )

        return graph.compile()
    
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

            1. 사용가자 입력한 문제의 **정답**의 보기 번호를 정답으로 작성해 주세요.
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
                    "subject": state["generated_explanation"],
                }
            )
            vectorstore.add_documents([doc])
            print("✅ 문제+해답+풀이 저장 완료")

        # 결과를 state에 저장 (항상 실행)
        print(f"\n📝 결과 저장 시작:")
        print(f"   - 현재 문제: {state['user_problem'][:50]}...")
        print(f"   - 생성된 정답: {state['generated_answer'][:30]}...")
        print(f"   - 검증 상태: {state['validated']}")
        
        item = {
            "user_problem": state["user_problem"],
            "user_problem_options": state["user_problem_options"],
            "generated_answer": state["generated_answer"],
            "generated_explanation": state["generated_explanation"],
            "generated_subject": state["generated_subject"],
            "validated": state["validated"],
            "chat_history": state.get("chat_history", [])
        }
        
        
        state["results"].append(item)
        print(f"✅ 결과 저장 완료: {len(state['results'])}개")
        for key, value in item.items():
            print(f"{key}: {value}")
        
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

            "results": [],
            
            "chat_history": []
        }
        
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
        
        # 그래프 시각화
        # try:
        #     graph_image_path = "solution_agent_workflow.png"
        #     with open(graph_image_path, "wb") as f:
        #         f.write(self.graph.get_graph().draw_mermaid_png())
        #     print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
        # except Exception as e:
        #     print(f"그래프 시각화 중 오류 발생: {e}")
        #     print("워크플로우는 정상적으로 작동합니다.")

        # 결과 확인 및 디버깅
        results = final_state.get("results", [])
        print(f"   - 총 결과 수: {len(results)}")
        
        if results:
            for i, result in enumerate(results):
                print(f"   - 결과 {i+1}: {result.get('question', '')[:30]}...")
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

    agent = SolutionAgent()

    # 그래프 시각화 (선택)
    # try:
    #     graph_image_path = "solution_agent_workflow.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(agent.graph.get_graph().draw_mermaid_png())
    #     print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류 발생: {e}")
    #     print("워크플로우는 정상적으로 작동합니다.")

    user_input_txt = input("\n❓ 사용자 질문: ").strip()
    user_problem = input("\n❓ 사용자 문제: ").strip()
    user_problem_options_raw = input("\n❓ 사용자 보기 (쉼표로 구분): ").strip()
    user_problem_options = [opt.strip() for opt in user_problem_options_raw.split(",") if opt.strip()]

    final_state = agent.execute(
        user_input_txt=user_input_txt,
        user_problem=user_problem,
        user_problem_options=user_problem_options,
    )

    # # 결과를 JSON 파일로 저장
    # results = final_state.get("results", [])
    # results_data = {
    #     "timestamp": datetime.now().isoformat(),
    #     "user_input_txt": final_state.get("user_input_txt",""),
    #     "total_results": len(results),
    #     "results": results,
    # }

    # results_filename = os.path.join(f"solution_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    # os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    # with open(results_filename, "w", encoding="utf-8") as f:
    #     json.dump(results_data, f, ensure_ascii=False, indent=2)
    # print(f"✅ 해답 결과가 JSON 파일로 저장되었습니다: {results_filename}")
