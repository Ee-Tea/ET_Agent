import streamlit as st
from retrieve import graph  # 기존 LangGraph 그래프 import

# 1️⃣ 사용자 입력 받기
st.title("📚 LLM 기반 검증형 검색 봇")
user_question = st.text_input("질문을 입력하세요:", placeholder="예: 소프트웨어 생명 주기의 정의와 종류는?")

# 2️⃣ 질문이 입력되면 LangGraph 실행
if user_question:
    with st.spinner("검색 및 검증 중입니다... ⏳"):
        initial_state = {"retrieval_question": user_question}

        # LangGraph 실행
        result = graph.invoke(initial_state)

        # 상태 표시용 정보들 출력
        st.subheader("📌 전체 과정 요약")
        st.markdown(f"**초기 질문**: `{result.get('retrieval_question', '')}`")
        st.markdown(f"**추출된 키워드**: `{result.get('keywords', [])}`")
        st.markdown(f"**재작성된 질문**: `{result.get('rewritten_question', '')}`")

        with st.expander("🔍 검색 결과 병합 컨텍스트 보기"):
            st.code(result.get("merged_context", "(없음)"))

        st.subheader("📥 LLM 응답")
        st.success(result.get("answer", "응답이 없습니다."))

        if "fact_check_result" in result:
            st.subheader("🧪 응답 검증 결과")
            verdict = result['fact_check_result'].get("verdict", "UNKNOWN")
            confidence = result['fact_check_result'].get("confidence", 0.0)
            evidence = result['fact_check_result'].get("evidence", [])

            verdict_display = f"✅ 검증 통과 ({verdict})" if verdict == "SUPPORTED" else f"❌ 검증 실패 ({verdict})"
            st.markdown(f"**검증 결과**: {verdict_display}")
            st.markdown(f"**신뢰도(confidence)**: `{confidence}`")

            if evidence:
                st.markdown("**검증 근거:**")
                for i, ev in enumerate(evidence, 1):
                    st.markdown(f"- {i}. {ev}")
