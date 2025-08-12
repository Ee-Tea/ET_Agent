import os
from typing import TypedDict, List, Dict, Literal, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import connections
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
import json, re
from langchain_openai import ChatOpenAI
from teacher.base_agent import BaseAgent
from docling.document_converter import DocumentConverter

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")




# ✅ 상태 정의
class SolutionState(TypedDict):
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
        graph.add_conditional_edges(
            "store",
            lambda s: "more" if len(s.get("user_problems", [])) > 0 else "done",
            {"more": "next_problem", "done": END}
        )

        return graph.compile()
    
    def _llm_extract_qas(self, text: str, llm) -> List[tuple]:
        """
        LLM에게 페이지 텍스트를 주고
        [{"문제":"...","옵션":["...","..."]}, ...] 만 받는다.
        실패 시 [] 반환.
        """
        sys_prompt = (
            "너는 시험 문제 PDF에서 텍스트를 구조화하는 도우미다. "
            "다양한 번호/불릿(1., (1), ①, 가., -, • 등)을 이해하고, "
            "문항을 '문제'와 '옵션'으로만 묶어 JSON 배열로 출력한다. "
            "옵션은 보기 항목만 포함하고, 설명/해설/정답 등은 포함하지 않는다. "
            "응답은 반드시 JSON 배열만 출력한다. 다른 문장이나 코드는 절대 포함하지 말 것."
        )
        user_prompt = (
            "다음 텍스트에서 문항을 최대한 정확히 추출해 JSON 배열로 만들어줘.\n"
            "요구 스키마: [{\"문제\":\"...\",\"옵션\":[\"...\",\"...\"]}]\n"
            "규칙:\n"
            "- 질문 본문에서 번호(예: '1.', '(1)', '①', '가.' 등)와 불필요한 머리글은 제거.\n"
            "- 옵션에서도 마찬가지로 번호/불릿 제거 후 순수 텍스트만 남김.\n"
            "- 옵션은 2~6개가 일반적이며, 그보다 많으면 상위 6개까지만 사용.\n"
            "- 추출이 불가하면 빈 배열([])을 출력.\n\n"
            f"텍스트:\n{text}"
        )

        try:
            resp = llm.invoke([{"role":"system","content":sys_prompt},
                            {"role":"user","content":user_prompt}])
            content = (resp.content or "").strip()

            # JSON만 남기기 (혹시 모델이 불필요한 텍스트를 붙였을 때 대비)
            m = re.search(r"\[.*\]", content, re.S)
            if not m:
                return []
            arr = json.loads(m.group(0))

            results = []
            for item in arr:
                q = (item.get("문제") or "").strip()
                opts = [str(o).strip() for o in (item.get("옵션") or [])]
                if q:
                    results.append((q, opts))
            return results
        except Exception:
            return []

    def _clean_numbering(self, s: str) -> str:
        if not s:
            return s
        s = s.strip()
        # 선행 번호/불릿 패턴 제거
        s = re.sub(r"^\s*(?:\(?\d{1,3}\)?[.)]|[①-⑳]|[A-Za-z가-힣][.)]|[-•])\s*", "", s)
        # 내부 이중 공백 정리
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()
    
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
        if not paths:
            raise ValueError("external_file_paths 가 비어있습니다. 외부 분기에서는 파일 경로가 필요합니다.")

        converter = DocumentConverter()
        extracted_pairs: List[Dict[str, object]] = []

        # LLM (구조화 전용, temperature 0)
        llm = ChatOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0
        )

        for p in paths:
            result = converter.convert(p)
            doc = result.document

            # 페이지 단위로 텍스트 추출 (가능하면 페이지 경계 보존)
            if hasattr(doc, "export_to_markdown"):
                raw = doc.export_to_markdown(strict_text=True)
            elif hasattr(doc, "export_to_text"):
                raw = doc.export_to_text()
            else:
                raw = ""

            pages = [pg for pg in raw.split("\f")] if "\f" in raw else raw.split("\n\n\n")  # 간단한 페이지 분리 폴백

            for page_text in pages:
                page_text = page_text.strip()
                if not page_text:
                    continue

                # 1차: 자동 패턴 파서
                blocks = self._split_by_questions_auto(page_text)

                # 문항 수가 너무 적거나(예: 0~1개) 옵션 없는 항목이 많으면 LLM으로 재시도
                need_llm = (len(blocks) <= 1) or (sum(1 for q, opts in blocks if opts) <= 0)

                if need_llm:
                    llm_items = self._llm_extract_qas(page_text, llm)
                    blocks = llm_items if llm_items else blocks  # LLM 실패하면 1차 결과 유지

                for qtext, opts in blocks:
                    qtext = self._clean_numbering(qtext)
                    opts = [self._clean_numbering(o) for o in (opts or [])]
                    # 최소 품질 필터
                    if len(qtext) < 3:
                        continue
                    if opts and not (2 <= len(opts) <= 6):
                        opts = opts[:6]
                    extracted_pairs.append({"문제": qtext.strip(), "옵션": [o.strip() for o in opts]})

        if not extracted_pairs:
            raise ValueError("문서에서 문제/보기를 추출하지 못했습니다. PDF 포맷을 확인하세요.")

        # 중복 제거(질문 텍스트 기준)
        seen = set()
        deduped = []
        for it in extracted_pairs:
            key = it["문제"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)

        state["user_problems"] = [{"question": it["문제"], "options": it["옵션"]} for it in deduped]
        print(f"✅ 추출된 문항 수: {len(state['user_problems'])}")
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
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
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
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
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
            recursion_limit: int = 1000,
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
        
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
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

    # 그래프 시각화
    try:
        graph_image_path = "agent_workflow.png"
        with open(graph_image_path, "wb") as f:
            f.write(agent.get_graph().draw_mermaid_png())
        print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")
   
    for i, result in enumerate(results):
        print(f"\n==== 문제 {i + 1} ====")
        print("Q:", result["question"])
        print("A:", result["generated_answer"])
        print("E:", result["generated_explanation"])
        print("검증:", "통과" if result["validated"] else "불통과")
        print("히스토리:", result["chat_history"])
