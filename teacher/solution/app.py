import streamlit as st
from solution_agent import SolutionAgent, HuggingFaceEmbeddings, Milvus, connections
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

    # ✅ SolutionAgent 초기화
    agent = SolutionAgent()

    with st.spinner("🧠 모든 문제에 대한 해답 생성 중..."):
        try:
            # ✅ SolutionAgent를 사용하여 모든 문제 처리
            results = agent.execute(user_question, user_problems, vectorstore)
            
            for i, result in enumerate(results):
                st.markdown(f"---\n### 📘 문제 {i+1}")
                st.markdown(f"**문제:** {result['question']}")
                st.markdown("**보기:**")
                for j, option in enumerate(result['options'], 1):
                    st.markdown(f"{j}. {option}")

                # ✅ 해답 보기 버튼 (Expander 형태로 숨김)
                with st.expander("📝 해답 보기"):
                    # 정답 번호 찾기
                    answer_text = result['generated_answer']
                    answer_number = None
                    
                    # 정답이 "1", "2", "3", "4" 중 하나인지 확인
                    if answer_text.strip() in ["1", "2", "3", "4"]:
                        answer_number = int(answer_text.strip())
                        st.markdown(f"**✅ 정답:** {answer_number}. {result['options'][answer_number-1]}")
                    else:
                        # 정답이 번호가 아닌 경우, 보기 옵션에서 일치하는 것을 찾기
                        answer_found = False
                        for j, option in enumerate(result['options'], 1):
                            if answer_text.strip() in option or option in answer_text.strip():
                                st.markdown(f"**✅ 정답:** {j}. {option}")
                                answer_found = True
                                break
                        
                        if not answer_found:
                            st.markdown(f"**✅ 정답:** {result['generated_answer']}")
                    
                    st.markdown(f"**📖 풀이:**\n{result['generated_explanation']}")
                    st.markdown(f"**🔍 검증:** {'통과' if result['validated'] else '불통과'}")

                # ✅ 히스토리
                with st.expander("📜 전체 히스토리"):
                    for item in result["chat_history"]:
                        st.text(item)

                # ✅ 결과 누적
                all_results.append({
                    "index": i + 1,
                    "question": result["question"],
                    "options": result["options"],
                    "answer": result["generated_answer"],
                    "explanation": result["generated_explanation"]
                })

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