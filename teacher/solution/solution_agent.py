import os
from typing import TypedDict, List, Dict, Literal, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import connections
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
import re
import json
from langchain_openai import ChatOpenAI
from teacher.base_agent import BaseAgent
from docling.document_converter import DocumentConverter

load_dotenv()
GROQ_API_KEY=REDACTED("GROQ_API_KEY=REDACTED SolutionState(TypedDict):
    user_question: str
    user_problems: List[Dict]
    user_problem: str
    user_problem_options: List[str]

    source_type: Literal["internal", "external"]
    # 내부/외부 원천
    short_term_memory: List[Dict]
    external_file_paths: List[str] 

    vectorstore: Milvus
    retrieved_docs: List[Document]
    similar_questions_text : str

    generated_answer: str         # 해답
    generated_explanation: str   # 풀이
    results: List[Dict]
    validated: bool

    exam_title: str
    difficulty: str
    subject: str

    chat_history: List[str]
class SolutionAgent(BaseAgent):
    """문제 해답/풀이 생성 에이전트"""

    def __init__(self):
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """워크플로우 그래프 생성"""

        # ✅ LangGraph 구성
        print("📚 LangGraph 흐름 구성 중...")
        
        graph = StateGraph(SolutionState)

        # 분기 & 로딩
        graph.add_node("route", self._route)
        graph.add_node("load_from_short_term_memory", self._load_from_stm)
        graph.add_node("load_from_external_docs", self._load_from_external)

        # 공통 처리
        graph.add_node("search_similarity", self._search_similar_questions)
        graph.add_node("generate_solution", self._generate_solution)
        graph.add_node("validate", self._validate_solution)
        graph.add_node("store", self._store_to_vector_db)
        graph.add_node("next_problem", self._next_problem)

        graph.set_entry_point("route")
        graph.add_conditional_edges("route", lambda s: s["source_type"],
                                {"internal": "load_from_short_term_memory",
                                "external": "load_from_external_docs"})
        graph.add_edge("load_from_short_term_memory", "next_problem")
        graph.add_edge("load_from_external_docs", "next_problem")

        graph.add_edge("next_problem", "search_similarity")
        graph.add_edge("search_similarity", "generate_solution")
        graph.add_edge("generate_solution", "validate")

        graph.add_conditional_edges(
            "validate", 
            lambda s: "ok" if s["validated"] else "stop",
            {"ok": "store", "stop": END}
        )

        # 저장 후 남은 문제가 있으면 next_problem로 루프
        g.add_conditional_edges(
            "store",
            lambda s: "more" if len(s.get("user_problems", [])) > 0 else "done",
            {"more": "next_problem", "done": END}
        )

        return graph.compile()
    
    # --------- 분기 ----------
    def _route(self, state: SolutionState) -> SolutionState:
        # 오케스트레이터가 채워준 source_type을 그대로 사용
        st = state["source_type"]
        print(f"🧭 분기: {st}")
        return state

    # --------- 내부: STM에서 문제 1개 꺼내와 state에 세팅 ----------
    def _load_from_stm(self, state: SolutionState) -> SolutionState:
        stm = state.get("short_term_memory", [])
        state["user_problems"] = [{"question": x.get("question",""),
                                "options": x.get("options",[])} for x in stm]
        state["short_term_memory"] = []  # 큐로 이관
        return state
    
    # --------- 외부: Docling으로 문서 → 텍스트 → JSON(문제/옵션) → state에 세팅 ----------
    def _load_from_external(self, state: SolutionState) -> SolutionState:
        print("📄 [외부] 첨부 문서 로드 및 Docling 변환")
        paths = state.get("external_file_paths", [])
        converter = DocumentConverter()
        extracted_pairs: List[Dict[str, object]] = []

        for p in paths:
            doc = converter.convert(p)  # Docling Document
            text = doc.export_to_text()
            # 간단한 규칙: '보기' 또는 선택지 패턴이 있는 블록을 문제/옵션으로 분리
            chunks = self._split_by_questions(text)
            for qtext, opts in chunks:
                # 요구: JSON 키는 "문제", "옵션" 만 사용
                extracted_pairs.append({"문제": qtext.strip(), "옵션": [o.strip() for o in opts]})

        if not extracted_pairs:
            raise ValueError("Docling으로부터 문제를 추출하지 못했습니다. 문서 포맷을 확인하세요.")

        # 일단 첫 문제만 이번 state에 적재 (한 번에 한 문제 흐름 유지)
        state["user_problems"] = [{"question": p["문제"], "options": p["옵션"]} for p in extracted_pairs]

        # 필요시, 이후 문제는 다음 실행 사이클에서 처리하도록 별도 보관 로직을 추가해도 됨.
        print(f"✅ Docling 추출(문제/옵션) 예시: {first}")
        return state
    
    # 간단한 문제/보기 파서 (문서 포맷에 맞게 조정 가능)
    def _split_by_questions(self, text: str) -> List[tuple]:
        blocks = re.split(r"\n\s*\n", text)  # 빈 줄 기준 거칠게 분할
        results = []
        for b in blocks:
            lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
            if not lines:
                continue
            # 옵션 라인 감지 (숫자. 또는 숫자) 패턴)
            opts = [ln for ln in lines if re.match(r"^\(?\d+\)?[).]\s*", ln)]
            if opts:
                # 문제문은 옵션 라인 제외 첫 줄 위주로 사용
                question_lines = [ln for ln in lines if ln not in opts]
                qtext = " ".join(question_lines) if question_lines else lines[0]
                # 옵션 텍스트 정제: "1) ..." → "..." 로
                clean_opts = [re.sub(r"^\(?\d+\)?[).]\s*", "", o) for o in opts]
                results.append((qtext, clean_opts))
        return results


    def _search_similar_questions(self, state: SolutionState) -> SolutionState:
        print("\n🔍 [1단계] 유사 문제 검색 시작")
        results = state["vectorstore"].similarity_search(state["user_problem"], k=3)

        similar_questions = []
        for i, doc in enumerate(results):
            metadata = doc.metadata
            options = json.loads(metadata.get("options", "[]"))
            answer = metadata.get("answer", "")
            explanation = metadata.get("explanation", "")

            formatted = f"""[유사문제 {i+1}]
                문제: {doc.page_content}
                보기:
                """ + "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)]) + f"""
                정답: {answer}
                풀이: {explanation}
                """
            similar_questions.append(formatted)
        
        state["retrieved_docs"] = results
        state["similar_questions_text"] = "\n\n".join(similar_questions) 

        print(f"유사 문제 {len(results)}개 검색 완료.")
        print("🔍 [1단계] 유사 문제 검색 함수 종료")
        return state

    def _generate_solution(self, state: SolutionState) -> SolutionState:

        print("\n✏️ [2단계] 해답 및 풀이 생성 시작")

        llm = ChatOpenAI(
            api_key=GROQ_API_KEY=REDACTED="https://api.groq.com/openai/v1",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.5
        )

        similar_problems = state.get("similar_questions_text", "")
        print("유사 문제들:\n", similar_problems)

        prompt = f"""
            사용자가 입력한 질문:
            {state['user_question']}
            다음은 사용자가 입력한 문제:
            {state['user_problem']}
            {state['user_problem_options']}

            아래는 이 문제와 유사한 문제들:
            {similar_problems}

            1. 이 문제의 **정답**만 간결하게 한 문장으로 먼저 작성해 주세요.
            2. 이어서 그 정답인 근거를 담은 **풀이 과정**을 상세히 설명해 주세요.

            출력 형식:
            정답: ...
            풀이: ...
        """

        response = llm.invoke(prompt)
        result = response.content.strip()
        print("🧠 LLM 응답 완료")

        answer_match = re.search(r"정답:\s*(.+)", result)
        explanation_match = re.search(r"풀이:\s*(.+)", result, re.DOTALL)
        state["generated_answer"] = answer_match.group(1).strip() if answer_match else ""
        state["generated_explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        state["chat_history"].append(f"Q: {state['user_question']}\nP: {state['user_problem']}\nA: {state['generated_answer']}\nE: {state['generated_explanation']}")

        return state

    # ✅ 정합성 검증 (간단히 길이 기준 사용)

    def _validate_solution(self, state: SolutionState) -> SolutionState:
        print("\n🧐 [3단계] 정합성 검증 시작")
        
        llm = ChatOpenAI(
            api_key=GROQ_API_KEY=REDACTED="https://api.groq.com/openai/v1",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0
        )

        validation_prompt = f"""
        사용자 요구사항: {state['user_question']}

        질문: {state['user_problem']}
        정답: {state['generated_answer']}
        풀이: {state['generated_explanation']}

        위 해답과 풀이가 문제 사용자 요구사항에 맞고, 논리적 오류나 잘못된 정보가 없습니까?
        적절하다면 '네', 그렇지 않다면 '아니오'로만 답변하세요.
        """

        validation_response = llm.invoke(validation_prompt)
        result_text = validation_response.content.strip().lower()

        # ✅ '네'가 포함된 응답일 경우에만 유효한 풀이로 판단
        print("📌 검증 응답:", result_text)
        state["validated"] = "네" in result_text
        print(f"✅ 검증 결과: {'통과' if state['validated'] else '불통과'}")
        return state


    # ✅ 임베딩 후 벡터 DB 저장
    def _store_to_vector_db(self, state: SolutionState) -> SolutionState:  
        print("\n🧩 [4단계] 임베딩 및 벡터 DB 저장 시작")

        if state["source_type"] == "external":

            vectorstore = state["vectorstore"] 

            # 중복 문제 확인
            similar = vectorstore.similarity_search(state["user_problem"], k=1)
            if similar and state["user_problem"].strip() in similar[0].page_content:
                print("⚠️ 동일한 문제가 존재하여 저장 생략")
                return state

            # 문제, 해답, 풀이를 각각 metadata로 저장
            doc = Document(
                page_content=state["user_problem"],
                metadata={
                    "options": json.dumps(state.get("user_problem_options", [])), 
                    "answer": state["generated_answer"],
                    "explanation": state["generated_explanation"]
                }
            )
            vectorstore.add_documents([doc])
            print("✅ 문제+해답+풀이 저장 완료")

            return state
        else:
            print("⚠️ 내부 저장소는 벡터 DB 저장을 지원하지 않습니다. 내부 문제로만 저장합니다.")
            # 내부: 요구 스키마(JSON)로 파일 누적 저장
            store_path = "./internal_store.json"
            data = {
                "exam_title": state.get("exam_title", "내부 문제 모음"),
                "total_questions": 0,
                "difficulty": state.get("difficulty", "중급"),
                "subjects": {},  # subject명: {"requested_count":0,"actual_count":n,"questions":[...]}
            }

            if os.path.exists(store_path):
                try:
                    with open(store_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    pass

            subj = state.get("subject", "기타")
            subjects = data.setdefault("subjects", {})
            bucket = subjects.setdefault(subj, {"requested_count": 0, "actual_count": 0, "questions": []})

            bucket["questions"].append({
                "question": state["user_problem"],
                "options": [f"  {i+1}. {opt}" for i, opt in enumerate(state.get("user_problem_options", []))],
                "answer": state["generated_answer"],
                "explanation": state["generated_explanation"],
                "subject": subj,
            })
            bucket["actual_count"] = len(bucket["questions"])

            # 총 문항 수 재계산
            total = 0
            for v in subjects.values():
                total += len(v.get("questions", []))
            data["total_questions"] = total

            with open(store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ 내부 문제 저장(JSON 스키마) 완료 → {store_path}")

        item = {
            "question": state["user_problem"],
            "options": state["user_problem_options"],
            "generated_answer": state["generated_answer"],
            "generated_explanation": state["generated_explanation"],
            "validated": state["validated"],
            "chat_history": state.get("chat_history", []),
        }
        state.setdefault("results", []).append(item)
        return state
    
    def _next_problem(self, state: SolutionState) -> SolutionState:
        queue = state.get("user_problems", [])
        if not queue:
            raise ValueError("처리할 문제가 없습니다. user_problems가 비어있어요.")
        current = queue.pop(0)
        state["user_problem"] = current.get("question", "")
        state["user_problem_options"] = current.get("options", [])
        state["user_problems"] = queue
        return state

    def execute(
            self, 
            user_question: str, 
            source_type: Literal["internal", "external"],
            vectorstore: Optional[Milvus] = None,
            short_term_memory: Optional[List[Dict]] = None,
            external_file_paths: Optional[List[str]] = None,
            exam_title: str = "정보처리기사 모의고사 (Groq 순차 버전)",
            difficulty: str = "중급",
            subject: str = "기타",
        ) -> Dict:

        # ✅ Milvus 연결 및 벡터스토어 생성
        if vectorstore is None:

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
        
        initial_state: SolutionState = {
            "user_question": user_question,
            "user_problems": [], 
            "user_problem": "",
            "user_problem_options": [],

            "source_type": source_type,
            "short_term_memory": short_term_memory or [],
            "external_file_paths": external_file_paths or [],

            "vectorstore": vectorstore,
            "retrieved_docs": [],
            "similar_questions_text": "",

            "generated_answer": "",
            "generated_explanation": "",
            "validated": False,
            "results": [],
            
            "exam_title": exam_title,
            "difficulty": difficulty,
            "subject": subject,

            "chat_history": []
        }
        
        final_state = self.graph.invoke(initial_state)
        return final_state["results"]

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
        connection_args={"host": "localhost", "port": "19530"}
    )

    # ✅ JSON 파일 로딩
    with open("./sample_user.json", "r", encoding="utf-8") as f:
        user_problems = json.load(f)

    # ✅ 사용자 질문 입력
    user_question = input("\n❓ 사용자 질문을 입력하세요 : ").strip()

    agent = SolutionAgent()
    results = agent.execute(user_question, user_problems, vectorstore)
   
    for i, result in enumerate(results):
        print(f"\n==== 문제 {i + 1} ====")
        print("Q:", result["question"])
        print("A:", result["generated_answer"])
        print("E:", result["generated_explanation"])
        print("검증:", "통과" if result["validated"] else "불통과")
        print("히스토리:", result["chat_history"])

    # # 그래프 시각화
    # try:
    #     graph_image_path = "agent_workflow.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(graph.get_graph().draw_mermaid_png())
    #     print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류 발생: {e}")