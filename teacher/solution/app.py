import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv

# 에이전트
from solution_agent import SolutionAgent

# 벡터스토어 준비
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections

# PDF 생성 유틸 (정답/풀이 포함 출력)
from result2pdf_generation import generate_pdf

load_dotenv()
st.set_page_config(layout="wide")
st.title("🧠 문제 자동 해답 생성기")

# -----------------------------
# Milvus 연결 및 벡터스토어
# -----------------------------
@st.cache_resource
def init_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
    )
    if "default" in connections.list_connections():
        connections.disconnect("default")
    connections.connect(alias="default", host="localhost", port="19530")

    vectorstore = Milvus(
        embedding_function=embedding_model,
        collection_name="problems",
        connection_args={"host": "localhost", "port": "19530"},
    )
    return vectorstore

vectorstore = init_vectorstore()

# -----------------------------
# 사이드바 옵션
# -----------------------------
with st.sidebar:
    st.subheader("저장 메타데이터")
    exam_title = st.text_input("시험 제목", value="정보처리기사 모의고사 (Groq 순차 버전)")
    difficulty = st.selectbox("난이도", ["초급", "중급", "고급"], index=1)
    subject = st.text_input("과목(내부 저장용)", value="기타")

# -----------------------------
# 입력 섹션 (챗봇 UX)
# -----------------------------
user_question = st.text_input(
    "💬 사용자 요구사항",
    value="정답을 먼저 한 문장으로, 이어서 자세한 풀이를 작성해줘."
)

uploaded_pdfs = st.file_uploader(
    "📚 PDF 업로드 (여러 파일 가능, Docling이 처리 가능한 포맷)",
    type=["pdf"],
    accept_multiple_files=True,
)

run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run = st.button("🧠 해답 생성 시작", type="primary", use_container_width=True)

# 결과 저장용 변수 초기화
all_results = []

if run:
    if not uploaded_pdfs:
        st.warning("PDF 파일을 하나 이상 업로드해 주세요.")
    else:
        # 임시 파일로 저장 후 경로 전달
        temp_paths = []
        try:
            for uf in uploaded_pdfs:
                suffix = os.path.splitext(uf.name)[1] or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    temp_paths.append(tmp.name)

            agent = SolutionAgent()
            
            # 디버깅 정보 표시
            st.info(f"📁 업로드된 PDF 파일: {len(temp_paths)}개")
            for i, path in enumerate(temp_paths):
                st.info(f"   {i+1}. {os.path.basename(path)}")
            
            with st.spinner("🧠 문서에서 문제 추출 및 해답 생성 중..."):
                try:
                    results = agent.execute(
                        user_question=user_question,
                        source_type="external",          # PDF → Docling → 외부 분기
                        vectorstore=vectorstore,
                        external_file_paths=temp_paths,  # 방금 저장한 임시 경로들
                        exam_title=exam_title,
                        difficulty=difficulty,
                        subject=subject,
                    )
                    st.success("✅ 에이전트 실행 완료!")
                except Exception as e:
                    st.error(f"❌ 에이전트 실행 중 오류: {str(e)}")
                    st.error("🔍 PDF 파일 형식이나 내용을 확인해 주세요.")
                    results = []

            if not results or len(results) == 0:
                st.info("문제를 추출하지 못했습니다. PDF 포맷을 확인해 주세요.")
            else:
                st.success(f"총 {len(results)}개 문항 처리 완료!")
                st.info(f"처리 통계: 검증 통과 {sum(1 for r in results if r.get('validated'))}개, 실패 {sum(1 for r in results if not r.get('validated'))}개")

                # 문항 렌더링
                for i, result in enumerate(results, start=1):
                    st.markdown(f"---\n### 📘 문제 {i}")
                    st.markdown(f"**문제:** {result.get('question','').strip() or '(비어있음)'}")
                    st.markdown("**보기:**")
                    opts = result.get("options", []) or []
                    if not opts:
                        st.write("- (보기 없음)")
                    else:
                        for j, option in enumerate(opts, 1):
                            # 옵션에 번호가 이미 포함되어 있으면 그대로 보여줌
                            st.markdown(f"{option}" if option.strip().startswith(("1.", "1)")) else f"{j}. {option}")

                    # 해설/정답 보기
                    with st.expander("📝 해답 · 풀이 보기"):
                        answer_text = str(result.get("generated_answer", "")).strip()
                        explanation = result.get("generated_explanation", "")
                        st.markdown(f"**✅ 정답:** {answer_text or '(생성되지 않음)'}")
                        st.markdown(f"**📖 풀이:**\n{explanation or '(생성되지 않음)'}")
                        # 수정 필요: 재시도 횟수 정보도 표시
                        validation_status = "통과" if result.get('validated') else "불통과"
                        retry_count = result.get('retry_count', 0)
                        st.markdown(f"**🔍 검증:** {validation_status}")
                        if retry_count > 0:
                            st.markdown(f"**⚠️ 재시도:** {retry_count}회")

                        # 히스토리(있을 경우)
                        history = result.get("chat_history", [])
                        if history:
                            with st.expander("📜 전체 히스토리"):
                                for item in history:
                                    st.text(item)

                    # PDF 저장용 누적
                    all_results.append({
                        "index": i,
                        "question": result.get("question", ""),
                        "options": opts,
                        "answer": result.get("generated_answer", ""),
                        "explanation": result.get("generated_explanation", ""),
                        "validated": result.get("validated", False),
                        "retry_count": result.get("retry_count", 0)
                    })

        except Exception as e:
            st.error(f"❌ 처리 중 오류: {e}")
        finally:
            # 임시 파일 정리
            for p in temp_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass

# -----------------------------
# PDF 다운로드 (정답/풀이 포함)
# -----------------------------
if all_results:
    st.markdown("---")
    st.subheader("📄 전체 결과 PDF 다운로드")
    try:
        pdf_buffer = generate_pdf(all_results)  # 정답/풀이 포함 PDF
        st.download_button(
            label="📥 PDF 다운로드",
            data=pdf_buffer,
            file_name="generated_solutions.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"❌ PDF 생성 중 오류: {str(e)}")
        st.info("📋 결과는 여전히 화면에서 확인할 수 있습니다.")
