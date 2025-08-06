from nodes.extractor import extract_query_elements, query_rewrite
from nodes.merge_responder import merge_context, generate_answer
from nodes.search import wiki_tool, ddg_tool
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict

class RetrievalState(TypedDict):
    retrieval_question: str
    keywords: list[str]
    rewritten_question: str
    wiki: str
    ddg: str
    merged_context: str
    answer: str

def extract_fn(state):
    """키워드 추출 노드"""
    question = state["retrieval_question"]
    keywords = extract_query_elements(question)
    print(f"추출된 키워드: {keywords}")
    return {"retrieval_question": question,
            "keywords" : keywords}

def rewrite_fn(state):
    """질문 재작성 노드"""
    question = state["retrieval_question"]
    keywords = state["keywords"]
    rewritten_question = query_rewrite(question, keywords)
    print(f"재작성된 질문: {rewritten_question}")
    return {"rewritten_question": rewritten_question}

def search_wiki_fn(state):
    """위키백과 검색 노드"""
    question = state["rewritten_question"]
    wiki_result = wiki_tool.run(question)
    return {"wiki": wiki_result}

def search_ddg_fn(state):
    """DuckDuckGo 검색 노드"""
    question = state["rewritten_question"]
    ddg_result = ddg_tool.run(question)
    return {"ddg": ddg_result}

def merge_fn(state):
    """검색 결과 병합 노드"""
    wiki_result = state["wiki"]
    ddg_result = state["ddg"]
    merged_context = merge_context(wiki_result, ddg_result)
    print(f"병합된 컨텍스트: {merged_context}")
    return {"merged_context": merged_context}

def answer_fn(state):
    """답변 생성 노드"""
    question = state["retrieval_question"]
    context = state["merged_context"]
    answer = generate_answer(question, context)
    return {"answer": answer}

def build_retrieval_graph(extract_fn, rewrite_fn, search_wiki_fn, search_ddg_fn, merge_fn, answer_fn):
    """검색 그래프 빌드"""

    builder = StateGraph(RetrievalState)

    # 2️⃣ builder에 node/edge 추가
    builder.add_node("extract", RunnableLambda(extract_fn))
    builder.add_node("rewrite", RunnableLambda(rewrite_fn))
    builder.add_node("search_wiki", RunnableLambda(search_wiki_fn))
    builder.add_node("search_ddg", RunnableLambda(search_ddg_fn))
    builder.add_node("merge", RunnableLambda(merge_fn))
    builder.add_node("answer", RunnableLambda(answer_fn))

    builder.set_entry_point("extract")

    # ✅ parallel 메서드는 builder에만 존재
    builder.add_edge("extract", "rewrite")
    builder.add_edge("rewrite", "search_wiki")
    builder.add_edge("rewrite", "search_ddg")
    builder.add_edge("search_wiki", "merge")
    builder.add_edge("search_ddg", "merge")
    builder.add_edge("merge", "answer")

    # 3️⃣ 최종 컴파일된 graph 얻기
    graph = builder.compile()

    return graph

graph = build_retrieval_graph(
    extract_fn,
    rewrite_fn,
    search_wiki_fn,
    search_ddg_fn,
    merge_fn,
    answer_fn
)

initial_state = {
    "retrieval_question": "소프트웨어 생명 주기 (소프트웨어 수명 주기)의 정의와 종류에 대해 알려줘"
}
print(initial_state["retrieval_question"])
result = graph.invoke(initial_state)
print("답변 시작")
print(result["answer"])  # 그래프 실행 후 답변 출력