#!/usr/bin/env python3
"""빠른 PDF 파싱 테스트"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'teacher'))

def test_pdf_parsing():
    from teacher_graph import Orchestrator
    
    # 필수 파라미터들 제공
    orchestrator = Orchestrator(
        user_id="test_user",
        service="test_service", 
        chat_id="test_chat",
        init_agents=False  # 에이전트 초기화 스킵해서 빠르게 테스트
    )
    
    # PDF 파일 경로
    pdf_path = "teacher/agents/solution/pdf_outputs/과목당5문제씩만들어줘_문제집.pdf"
    
    print(f"🧪 PDF 파싱 테스트: {pdf_path}")
    
    try:
        problems = orchestrator._extract_problems_from_pdf([pdf_path])
        print(f"\n🎯 최종 결과: {len(problems)}개 문제 추출")
        
        for i, problem in enumerate(problems[:3], 1):
            print(f"\n📝 문제 {i}:")
            print(f"   질문: {problem.get('question', '')[:100]}...")
            print(f"   보기 수: {len(problem.get('options', []))}")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_parsing()