from .nodes.extractor import extract_query_elements, query_rewrite, query_reinforce
from .nodes.merge_responder import merge_context, generate_answer
from .nodes.search import wiki_tool, ddg_tool
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict
from .nodes.verifier import fact_check
from ..base_agent import BaseAgent
from typing import Dict, List, TypedDict, Annotated

class RetrievalState(TypedDict):
    retrieval_question: str
    keywords: list[str]
    rewritten_question: str
    wiki: str
    ddg: str
    merged_context: str
    answer: str
    fact_check_result: dict

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

def verify_fn(state):
    """답변 검증 노드"""
    question = state["retrieval_question"]
    context = state["merged_context"]
    answer = state["answer"]
    fact_check_result = fact_check(question, context, answer)
    print(f"검증 결과: {fact_check_result}")
    return {"fact_check_result": fact_check_result}

def reinforce_fn(state):
    """질문 보강 노드"""
    rewritten_question = query_reinforce(state)
    return {"rewritten_question": rewritten_question}

def check_verdict(state):
    """검증 결과에 따라 분기"""
    verdict = state.get("fact_check_result", {}).get("verdict", "NOT ENOUGH INFO")
    if verdict == "SUPPORTED":
        return "pass"
    else:
        return "fail"

def build_retrieval_graph(extract_fn, rewrite_fn, search_wiki_fn, search_ddg_fn, merge_fn, answer_fn):
    """검색 그래프 빌드"""

    builder = StateGraph(RetrievalState)

    # 2️⃣ builder에 node/edge 추가
    builder.add_node("extract", RunnableLambda(extract_fn))
    builder.add_node("rewrite", RunnableLambda(rewrite_fn))
    builder.add_node("search_wiki", RunnableLambda(search_wiki_fn))
    builder.add_node("search_ddg", RunnableLambda(search_ddg_fn))
    builder.add_node("merge", RunnableLambda(merge_fn))
    # NOTE: state key 'answer' 와 노드명이 동일하면 최신 langgraph에서 충돌(ValueError)
    #       노드명을 'generate_answer'로 변경
    builder.add_node("generate_answer", RunnableLambda(answer_fn))
    builder.add_node("reinforce", RunnableLambda(reinforce_fn))
    builder.add_node("verify", RunnableLambda(verify_fn))

    builder.set_entry_point("extract")

    # ✅ parallel 메서드는 builder에만 존재
    builder.add_edge("extract", "rewrite")
    # NOTE: 최신 langgraph 버전에서 동일 노드(rewrite)에서 다중 직접 엣지 추가 시
    # "Already found path for node" 오류 발생 → 순차 실행으로 전환
    builder.add_edge("rewrite", "search_wiki")
    builder.add_edge("search_wiki", "search_ddg")  # wiki 검색 후 ddg 검색
    builder.add_edge("search_ddg", "merge")  # 두 결과(state에 wiki, ddg 존재) 병합
    builder.add_edge("merge", "generate_answer")
    builder.add_edge("generate_answer", "verify")
    builder.add_conditional_edges("verify",check_verdict, {"pass": END, "fail" : "reinforce"})
    # NOTE: 'reinforce'에서 다중 엣지를 추가하면 최신 langgraph에서 ValueError("Already found path for node 'reinforce'") 발생
    #       순차 구조로 loop 시키기 위해 wiki -> ddg -> merge 체인을 그대로 재사용: reinforce -> search_wiki 하나만 둔다.
    builder.add_edge("reinforce", "search_wiki")  # 보강 loop 시작 (search_wiki -> search_ddg 순차 진행)

    # 3️⃣ 최종 컴파일된 graph 얻기
    graph = builder.compile()

    return graph

class retrieve_agent(BaseAgent):
    """
    검색 에이전트 클래스입니다.
    위키백과와 DuckDuckGo를 사용하여 질문에 대한 답변을 생성합니다.
    """
    def __init__(self):
        self.graph = build_retrieval_graph(
            extract_fn, rewrite_fn, search_wiki_fn, search_ddg_fn, merge_fn, answer_fn
        )

    @property
    def name(self) -> str:
        return "RetrievalAgent"

    @property
    def description(self) -> str:
        return "위키백과와 DuckDuckGo를 사용하여 질문에 대한 답변을 생성하는 에이전트입니다."

    def invoke(self, input_data: Dict) -> Dict:
        """검색 그래프 실행 (간단 래퍼)"""
        initial_state: Dict[str, str] = {
            "retrieval_question": input_data.get("retrieval_question", "")
        }
        return self.graph.invoke(initial_state)  # type: ignore[arg-type]

# initial_state = {
#     "retrieval_question": "소프트웨어 생명 주기 (소프트웨어 수명 주기)의 정의와 종류에 대해 알려줘"
# }
# print(initial_state["retrieval_question"])
# result = graph.invoke(initial_state)
# print("답변 시작")
# print(result["answer"])  # 그래프 실행 후 답변 출력