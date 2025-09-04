import streamlit as st
import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
import traceback

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# teacher_graph 모듈 임포트
from .teacher_graph import create_app, TeacherState

# 페이지 설정
st.set_page_config(
    page_title="Teacher AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .result-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # 헤더
    st.markdown('<div class="main-header">🎓 Teacher AI Assistant</div>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 환경 변수 설정
        st.subheader("API 설정")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="OpenAI API 키를 입력하세요"
        )
        
        openai_base_url = st.text_input(
            "OpenAI Base URL",
            value=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            help="API 베이스 URL (기본값: Groq)"
        )
        
        openai_model = st.text_input(
            "LLM Model",
            value=os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct"),
            help="사용할 LLM 모델"
        )
        
        # 환경 변수 업데이트
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if openai_base_url:
            os.environ["OPENAI_BASE_URL"] = openai_base_url
        if openai_model:
            os.environ["OPENAI_LLM_MODEL"] = openai_model
        
        st.divider()
        
        # 사용 예시
        st.subheader("💡 사용 예시")
        st.markdown("""
        **채점 및 분석:**
        - "내 답은 1 2 3 4 5야"
        - "답안: 1번, 2번, 3번, 4번, 5번"
        
        **문제 생성:**
        - "소프트웨어설계 5문제 만들어줘"
        - "데이터베이스구축 10문제 출제해줘"
        
        **문제 검색:**
        - "운영체제 관련 문제 찾아줘"
        - "SQL 문제 검색해줘"
        """)
    
    # 메인 컨텐츠 - 통합 입력창
    st.header("🎯 AI 교사와 대화하기")
    
    # 사용자 입력
    user_query = st.text_area(
        "무엇을 도와드릴까요?",
        placeholder="예: 내 답은 1 2 3 4 5야, 소프트웨어설계 5문제 만들어줘, 운영체제 관련 문제 찾아줘...",
        height=120
    )
    
    if st.button("🚀 실행", type="primary", use_container_width=True):
        if not user_query.strip():
            st.error("질문을 입력해주세요!")
            return
        
        if not openai_api_key:
            st.error("OpenAI API 키를 설정해주세요!")
            return
        
        # 진행 상황 표시
        with st.spinner("AI가 요청을 처리하고 있습니다..."):
            try:
                # teacher_graph 앱 생성 및 실행
                app = create_app()
                
                # 초기 상태 설정
                init_state: TeacherState = {
                    "user_query": user_query.strip(),
                    "intent": "",
                    "artifacts": {},
                }
                
                # 그래프 실행
                result = app.invoke(init_state)
                
                # 결과 표시
                st.success("✅ 처리가 완료되었습니다!")
                
                # 결과 요약
                intent = result.get("intent", "(분류실패)")
                shared = result.get("shared", {})
                
                # 의도에 따른 결과 표시
                if "채점" in intent or "분석" in intent:
                    st.subheader("📝 채점 및 분석 결과")
                    
                    score = result.get("score", {})
                    analysis = result.get("analysis", {})
                    
                    # 채점 결과
                    if score and score.get("status") == "success":
                        results = score.get("results", [])
                        correct_count = sum(results)
                        total_count = len(results)
                        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("총 문제 수", total_count)
                        with col2:
                            st.metric("정답 수", correct_count)
                        with col3:
                            st.metric("정답률", f"{accuracy:.1f}%")
                        
                        # 문제별 결과
                        st.subheader("📊 문제별 결과")
                        for i, result in enumerate(results, 1):
                            status = "✅ 정답" if result == 1 else "❌ 오답"
                            st.write(f"문제 {i}: {status}")
                    
                    # 분석 결과
                    if analysis and analysis.get("status") == "success":
                        analysis_data = analysis.get("analysis", {})
                        
                        st.subheader("🧠 AI 분석 결과")
                        
                        # 종합 평가
                        overall_assessment = analysis_data.get("overall_assessment", {})
                        if overall_assessment:
                            if "title" in overall_assessment:
                                st.markdown(f"**{overall_assessment['title']}**")
                            
                            if "strengths" in overall_assessment:
                                st.markdown("**💪 강점**")
                                st.write(overall_assessment["strengths"])
                            
                            if "weaknesses" in overall_assessment:
                                st.markdown("**🔧 보완점**")
                                st.write(overall_assessment["weaknesses"])
                            
                            if "action_plan" in overall_assessment:
                                action_plan = overall_assessment["action_plan"]
                                st.markdown("**📈 학습 계획**")
                                if "short_term_goal" in action_plan:
                                    st.write(f"**단기 목표:** {action_plan['short_term_goal']}")
                                if "long_term_goal" in action_plan:
                                    st.write(f"**장기 목표:** {action_plan['long_term_goal']}")
                                if "recommended_strategies" in action_plan:
                                    st.write("**권장 전략:**")
                                    for strategy in action_plan["recommended_strategies"]:
                                        st.write(f"• {strategy}")
                            
                            if "final_message" in overall_assessment:
                                st.markdown("**💌 격려 메시지**")
                                st.write(overall_assessment["final_message"])
                        
                        # 상세 분석
                        detailed_analysis = analysis_data.get("detailed_analysis", [])
                        if detailed_analysis:
                            st.subheader("📋 상세 분석")
                            for item in detailed_analysis:
                                with st.expander(f"문제 {item.get('problem_number', 'N/A')} - {item.get('subject', 'N/A')}"):
                                    st.write(f"**실수 유형:** {item.get('mistake_type', 'N/A')}")
                                    st.write(f"**원인 분석:** {item.get('analysis', 'N/A')}")
                
                elif "생성" in intent:
                    st.subheader("🎲 생성된 문제")
                    
                    if result.get("generation", {}).get("status") == "success":
                        # 생성된 문제 표시
                        questions = shared.get("question", [])
                        options = shared.get("options", [])
                        answers = shared.get("answer", [])
                        explanations = shared.get("explanation", [])
                        
                        for i, (question, option_list, answer, explanation) in enumerate(zip(questions, options, answers, explanations), 1):
                            with st.expander(f"문제 {i}", expanded=True):
                                st.markdown(f"**문제:** {question}")
                                
                                if option_list:
                                    st.markdown("**보기:**")
                                    for j, option in enumerate(option_list, 1):
                                        st.write(f"{j}. {option}")
                                
                                st.markdown(f"**정답:** {answer}")
                                st.markdown(f"**해설:** {explanation}")
                    else:
                        st.error("문제 생성에 실패했습니다.")
                
                elif "검색" in intent:
                    st.subheader("🔍 검색 결과")
                    
                    if result.get("retrieval", {}).get("status") == "success":
                        # 검색 결과 표시
                        retrieve_answer = shared.get("retrieve_answer", "")
                        
                        if retrieve_answer:
                            st.markdown("**🔍 검색 결과:**")
                            st.write(retrieve_answer)
                        else:
                            st.info("검색 결과가 없습니다.")
                    else:
                        st.error("검색에 실패했습니다.")
                
                else:
                    st.info(f"요청 유형: {intent}")
                    st.write("처리 결과:", result)
                
                # PDF 다운로드 (모든 PDF 생성 시)
                pdf_dir = Path("agents/solution/pdf_outputs")
                if pdf_dir.exists():
                    pdf_files = list(pdf_dir.glob("*.pdf"))
                    if pdf_files:
                        # 가장 최근 PDF 파일 찾기
                        latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
                        
                        # 의도에 따른 PDF 제목 설정
                        if "채점" in intent or "분석" in intent:
                            pdf_title = "📄 분석 리포트 다운로드"
                            filename_prefix = "분석리포트"
                        elif "생성" in intent:
                            pdf_title = "📄 문제집 다운로드"
                            filename_prefix = "문제집"
                        elif "검색" in intent:
                            pdf_title = "📄 검색 결과 다운로드"
                            filename_prefix = "검색결과"
                        else:
                            pdf_title = "📄 결과 다운로드"
                            filename_prefix = "결과"
                        
                        st.subheader(pdf_title)
                        
                        with open(latest_pdf, "rb") as f:
                            pdf_bytes = f.read()
                        
                        # 파일명 생성
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{filename_prefix}_{timestamp}.pdf"
                        
                        st.download_button(
                            label="📥 PDF 다운로드",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.info(f"📁 파일 위치: {latest_pdf}")
                
            except Exception as e:
                st.error(f"❌ 오류가 발생했습니다: {str(e)}")
                st.code(traceback.format_exc())
    
    # 추가 예정 사항 섹션
    st.divider()
    st.subheader("🚀 추가 예정 사항")
    
    planned_features = [
        "과목 입력 안했을 때 전과목 문제 만들기(과목별 전체 /5)",
        "OCR 붙이기",
        "HITL로 해설에 Retrieve 내용 추가 붙이기",
        "숏텀 메모리 저장 방식 검토(중복검사 등등)",
        "  ㄴ 현재 append-only",
        "전체 오케스트레이터 구현(상관없는 질문 처리)",
        "챗봇 답변 생성 추가",
        "shared에 과목 저장되는 거 고치기"
    ]
    
    for feature in planned_features:
        if feature.startswith("  "):
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{feature}")
        else:
            st.markdown(f"• {feature}")

if __name__ == "__main__":
    main()
