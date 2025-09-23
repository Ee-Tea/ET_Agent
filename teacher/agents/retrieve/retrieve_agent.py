from .nodes.extractor import extract_query_elements, query_rewrite, query_reinforce
from .nodes.merge_responder import merge_context, generate_answer
# ⛔ wiki_tool 제거
from .nodes.search import ddg_tool
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict, NotRequired
from .nodes.verifier import fact_check
from ..base_agent import BaseAgent
from typing import Dict, List, Annotated

class RetrievalState(TypedDict):
    retrieval_question: str
    keywords: List[str]
    rewritten_question: str
    # ⛔ wiki는 더 이상 필수 아님
    wiki: NotRequired[str]
    ddg: NotRequired[str]
    milvus: NotRequired[str]
    merged_context: NotRequired[str]
    answer: NotRequired[str]
    fact_check_result: NotRequired[dict]
    milvus_data: Dict  # MilvusDB 연결 정보

def extract_fn(state):
    """키워드 추출 노드"""
    question = state["retrieval_question"]
    keywords = extract_query_elements(question)
    print(f"추출된 키워드: {keywords}")
    return {
        "retrieval_question": question,
        "keywords": keywords
    }

def rewrite_fn(state):
    """질문 재작성 노드"""
    question = state["retrieval_question"]
    keywords = state["keywords"]
    rewritten_question = query_rewrite(question, keywords)
    print(f"재작성된 질문: {rewritten_question}")
    return {"rewritten_question": rewritten_question}

# ⛔ wiki 검색 노드/함수 제거
# def search_wiki_fn(state): ...

def search_ddg_fn(state):
    """DuckDuckGo 검색 노드"""
    question = state["rewritten_question"]
    ddg_result = ddg_tool.run(question)
    return {"ddg": ddg_result}

def search_milvus_fn(state):
    """MilvusDB 벡터 유사도 검색 노드"""
    question = state["rewritten_question"]
    milvus_data = state.get("milvus_data", {})

    if not milvus_data:
        print("⚠️ milvus_data 없음 → MilvusDB 검색 건너뜀")
        return {"milvus": ""}

    try:
        from common.milvus_helpers import search_milvus_documents

        collection_name = milvus_data.get("collection_name", "concepts")
        k = int(milvus_data.get("top_k", 10))
        results = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name=collection_name,
            query=question,
            k=k
        )

        if not results:
            print("⚠️ MilvusDB에서 관련 문서를 찾을 수 없음")
            return {"milvus": ""}

        # 검색 결과를 텍스트로 변환 (Document 객체 사용)
        milvus_text = "\n\n".join([
            f"[문서 {i+1}] {result.page_content}"
            for i, result in enumerate(results)
        ])

        print(f"✅ MilvusDB에서 {len(results)}개 문서 검색 완료")
        return {"milvus": milvus_text}

    except Exception as e:
        print(f"❌ MilvusDB 검색 실패: {e}")
        return {"milvus": ""}

def merge_fn(state):
    ddg_result = state.get("ddg", "")
    milvus_result = state.get("milvus", "")
    merged_context = merge_context(ddg_result, milvus_result)  # ← 2-인자 버전
    return {"merged_context": merged_context}

def answer_fn(state):
    """답변 생성 노드"""
    question = state["retrieval_question"]
    context = state.get("merged_context", "")
    answer = generate_answer(question, context)
    return {"answer": answer}

def verify_fn(state):
    """답변 검증 노드"""
    fact_check_result = fact_check(state)
    print(f"검증 결과: {fact_check_result}")
    return {"fact_check_result": fact_check_result}

def reinforce_fn(state):
    """질문 보강 노드"""
    rewritten_question = query_reinforce(state)
    return {"rewritten_question": rewritten_question}

def check_verdict(state):
    """검증 결과에 따라 분기"""
    verdict = state.get("fact_check_result", {}).get("verdict", "NOT ENOUGH INFO")
    if verdict == "SUPPORTS":
        return "pass"
    else:
        return "fail"

def build_retrieval_graph(
    extract_fn,
    rewrite_fn,
    # ⛔ search_wiki_fn 제거
    search_ddg_fn,
    search_milvus_fn,
    merge_fn,
    answer_fn
):
    """검색 그래프 빌드"""
    builder = StateGraph(RetrievalState)

    builder.add_node("extract", RunnableLambda(extract_fn))
    builder.add_node("rewrite", RunnableLambda(rewrite_fn))
    builder.add_node("search_ddg", RunnableLambda(search_ddg_fn))
    builder.add_node("search_milvus", RunnableLambda(search_milvus_fn))
    builder.add_node("merge", RunnableLambda(merge_fn))
    builder.add_node("answer", RunnableLambda(answer_fn))
    builder.add_node("reinforce", RunnableLambda(reinforce_fn))
    builder.add_node("verify", RunnableLambda(verify_fn))

    builder.set_entry_point("extract")

    builder.add_edge("extract", "rewrite")
    builder.add_edge("rewrite", "search_ddg")
    builder.add_edge("rewrite", "search_milvus")
    builder.add_edge("search_ddg", "merge")
    builder.add_edge("search_milvus", "merge")
    builder.add_edge("merge", "answer")
    builder.add_edge("answer", "verify")
    builder.add_conditional_edges("verify", check_verdict, {"pass": END, "fail": "reinforce"})
    builder.add_edge("reinforce", "search_ddg")
    builder.add_edge("reinforce", "search_milvus")

    graph = builder.compile()
    return graph

class retrieve_agent(BaseAgent):
    """
    검색 에이전트 클래스입니다.
    DuckDuckGo와 Milvus를 사용하여 질문에 대한 답변을 생성합니다.
    """
    def __init__(self):
        self.graph = build_retrieval_graph(
            extract_fn,
            rewrite_fn,
            # ⛔ search_wiki_fn 인자 제거
            search_ddg_fn,
            search_milvus_fn,
            merge_fn,
            answer_fn
        )

    @property
    def name(self) -> str:
        return "RetrievalAgent"

    @property
    def description(self) -> str:
        return "DuckDuckGo와 Milvus를 사용하여 질문에 대한 답변을 생성하는 에이전트입니다."

    def invoke(self, input_data: Dict) -> Dict:
        """
        입력 데이터를 기반으로 검색 에이전트를 실행합니다.
        """
        initial_state: RetrievalState = {
            "retrieval_question": input_data.get("retrieval_question", ""),
            "milvus_data": input_data.get("milvus_data", {}),
            # ⛔ wiki는 사용하지 않지만 병합 함수가 3-인자면 빈값으로 둬도 안전
            "wiki": ""
        }
        result = self.graph.invoke(initial_state)

        # teacher graph에서 기대하는 형태로 변환
        return {
            "retrieve_answer": result.get("answer", ""),
            "retrieval": result  # 전체 결과도 포함
        }
