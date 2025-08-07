import os
import pickle
import numpy as np
import pandas as pd
import faiss
import torch
from tqdm import tqdm
from typing import TypedDict, List, Optional
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ===== 1. 상태 정의 =====
class RAGState(TypedDict):
    query: str
    documents: Optional[List[str]]
    answer: Optional[str]

# ===== 2. 문서 검색기 =====
text_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

index = faiss.read_index("multimodal_rag.index")
with open("multimodal_rag.pkl", "rb") as f:
    documents = pickle.load(f)

def retrieve_docs(state: RAGState) -> RAGState:
    query = state["query"]
    text_vec = text_model.encode([query])[0]
    text_vec = l2_normalize(text_vec)
    D, I = index.search(np.array([text_vec]).astype("float32"), k=5)
    scored_docs = []
    for idx, dist in zip(I[0], D[0]):
        similarity = 1 - dist
        doc = documents[idx]
        scored_docs.append(f"[유사도: {similarity:.4f}] {doc}")
    return {**state, "documents": scored_docs}

# ===== 3. LLM 체인 정의 =====
prompt = PromptTemplate.from_template("""
너는 농작물 재해 대응 전문가야.
다음 정보를 참고해서 질문에 정확히 답변해줘.
이걸 사용하는 사람은 농사에 대해 잘 모르고 정보가 필요한 사람들이야.
태풍, 폭염 같은 재해로부터 작물을 어떻게 보호하고 관리해야 할지,
시기별로 어떤 작업이 중요한지 설명해줘.

⚠️ 문서는 표 형식에서 이미지로 추출된 내용이라 줄바꿈 없이 단순 나열된 경우가 많아.
⚠️ 중요하지 않은 반복(1월~12월 등)은 생략하고 핵심 작업 위주로 요약해줘.

❓ 질문: {question}

📄 문서:
{context}

📢 답변:
""")

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7
)

rag_chain = prompt | llm | StrOutputParser()

def generate_answer(state: RAGState) -> RAGState:
    question = state.get("query", "")
    documents = state.get("documents", [])
    context = "\n\n".join(documents) if documents else "관련 문서를 찾을 수 없습니다."
    answer = rag_chain.invoke({"question": question, "context": context})
    return {**state, "answer": answer}

# ===== 4. LangGraph 구성 =====
graph = StateGraph(RAGState)
graph.add_node("RetrieveDocs", retrieve_docs)
graph.add_node("GenerateAnswer", generate_answer)
graph.set_entry_point("RetrieveDocs")
graph.add_edge("RetrieveDocs", "GenerateAnswer")
graph.add_edge("GenerateAnswer", END)
graph = graph.compile()

try:
    graph_image_path = "agent_workflow.png"
    with open(graph_image_path, "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
except Exception as e:
    print(f"그래프 시각화 중 오류 발생: {e}")

# ===== 5. 평가 코드 =====
def evaluate_on_golden_set(csv_path: str):
    df = pd.read_csv(csv_path)  # question, answer 컬럼 필수
    results = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        query = row["question"]
        ground_truth = row["answer"]

        input_state = {
            "query": query,
            "documents": None,
            "answer": None
        }

        try:
            result = graph.invoke(input_state)
            rag_answer = result["answer"]

            emb1 = text_model.encode(ground_truth, convert_to_tensor=True)
            emb2 = text_model.encode(rag_answer, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(emb1, emb2).item()

            results.append({
                "question": query,
                "답변(정답)": ground_truth,
                "LLM 생성 답변": rag_answer,
                "유사도": round(similarity, 4)
            })

            # ✅ 실시간 출력
            print(f"\n🟡 질문: {query}")
            print(f"✅ 정답: {ground_truth}")
            print(f"🤖 LLM 답변: {rag_answer}")
            print(f"📈 유사도: {round(similarity, 4)}")

        except Exception as e:
            print(f"❌ 오류 발생: {type(e).__name__} - {e}")
            results.append({
                "question": query,
                "답변(정답)": ground_truth,
                "LLM 생성 답변": f"[오류 발생] {e}",
                "유사도": -1
            })

    # ✅ CSV 저장
    pd.DataFrame(results).to_csv("rag_eval_results.csv", index=False)
    print("\n✅ rag_eval_results.csv 파일로 저장 완료!")

# ===== 6. 단일 실행 =====
if __name__ == "__main__":
    csv_path = "골든셋.csv"
    print("🚀 골든셋 기반 RAG 평가를 시작합니다...\n")
    evaluate_on_golden_set(csv_path)
