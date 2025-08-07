import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import connections
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
import re
import json
from langchain_openai import ChatOpenAI



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ 상태 정의
class RAGState(TypedDict):
    user_question: str
    user_problem: str
    user_problem_options: List[str]
    chat_history: List[str]

    vectorstore: Milvus
    docs: List[Document]
    retrieved_docs: List[Document]
    similar_questions_text : str

    generated_answer: str         # 해답
    generated_explanation: str   # 풀이
    validated: bool

def search_similar_questions(state: RAGState) -> RAGState:
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

def generate_solution(state: RAGState) -> RAGState:

    print("\n✏️ [2단계] 해답 및 풀이 생성 시작")

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
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

def validate_solution(state: RAGState) -> RAGState:
    print("\n🧐 [3단계] 정합성 검증 시작")
    
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
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
def store_to_vector_db(state: RAGState) -> RAGState:  
    print("\n🧩 [4단계] 임베딩 및 벡터 DB 저장 시작")

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

# ✅ LangGraph 구성
print("📚 LangGraph 흐름 구성 중...")

graph = StateGraph(RAGState)

graph.add_node("search_similarity", RunnableLambda(search_similar_questions))
graph.add_node("generate_solution", RunnableLambda(generate_solution))
graph.add_node("validate", RunnableLambda(validate_solution))
graph.add_node("store", RunnableLambda(store_to_vector_db))

graph.set_entry_point("search_similarity")

graph.add_edge("search_similarity", "generate_solution")
graph.add_edge("generate_solution", "validate")
graph.add_conditional_edges(
    "validate",
    lambda s: "true" if s["validated"] else "false",
    {
        "true": "store",
        "false": END
    }
)
graph.set_finish_point("store")

graph = graph.compile()

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

    for i, problem in enumerate(user_problems):
        print(f"\n===== 문제 {i + 1} 처리 시작 =====")

        state: RAGState = {
            "user_question": user_question,
            "user_problem": problem["question"],
            "user_problem_options": problem["options"],
            "vectorstore": vectorstore,
            "docs": [],
            "retrieved_docs": [],
            "similar_questions_text": "",
            "generated_answer": "",
            "generated_explanation": "",
            "validated": False,
            "chat_history": []
        }

        # ✅ LangGraph 실행
        state = graph.invoke(state)

        # ✅ 결과 출력
        print("\n생성된 해답:", state["generated_answer"])
        print("\n생성된 풀이:", state["generated_explanation"])

        print("\n✅ 히스토리:")
        for entry in state["chat_history"]:
            print(entry)

    # # 그래프 시각화
    # try:
    #     graph_image_path = "agent_workflow.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(graph.get_graph().draw_mermaid_png())
    #     print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류 발생: {e}")