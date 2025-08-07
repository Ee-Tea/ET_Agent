import streamlit as st
from solution_agent import graph, RAGState, HuggingFaceEmbeddings, Milvus, connections
from pdf_generation import generate_pdf
import os
import json
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(layout="wide")
st.title("🧠 문제 자동 해답 생성기")

# ✅ Milvus 연결 및 벡터스토어 준비
@st.cache_resource
def init_vectorstore():
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
    return vectorstore

vectorstore = init_vectorstore()

# ✅ 사용자 질문 입력
user_question = st.text_input("💬 사용자 요구사항", value="이 문제의 정답과 자세한 풀이를 알려줘.")

# ✅ JSON 업로드
uploaded_file = st.file_uploader("📂 JSON 파일 업로드 (question, options만 포함)", type="json")

all_results = []

if uploaded_file:
    user_problems = json.load(uploaded_file)

    for i, problem in enumerate(user_problems):
        st.markdown(f"---\n### 📘 문제 {i+1}")
        st.markdown(f"**문제:** {problem['question']}")
        st.markdown(f"**보기:** {', '.join(problem['options'])}")

        with st.spinner(f"🧠 문제 {i+1}에 대한 해답 생성 중..."):
            try:
                # ✅ 상태 설정
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
                result = graph.invoke(state)

                # ✅ 결과 누적
                all_results.append({
                    "index": i + 1,
                    "question": problem["question"],
                    "options": problem["options"],
                    "answer": result["generated_answer"],
                    "explanation": result["generated_explanation"]
                })


                # ✅ 해답 보기 버튼 (Expander 형태로 숨김)
                with st.expander("📝 해답 보기"):
                    st.markdown(f"**✅ 정답:** {result['generated_answer']}")
                    st.markdown(f"**📖 풀이:**\n{result['generated_explanation']}")

                # ✅ 유사 문제
                with st.expander("📚 유사 문제 보기"):
                    st.markdown(f"```\n{result['similar_questions_text']}\n```")

                # ✅ 히스토리
                with st.expander("📜 전체 히스토리"):
                    for item in result["chat_history"]:
                        st.text(item)

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

if all_results:
    st.markdown("---")
    st.subheader("📄 전체 결과 PDF 다운로드")
    pdf_buffer = generate_pdf(all_results)
    st.download_button(
        label="📥 PDF 다운로드",
        data=pdf_buffer,
        file_name="generated_solutions.pdf",
        mime="application/pdf"
    )