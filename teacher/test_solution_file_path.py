#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# uv run teacher/test_solution_file_path.py
"""
Solution Agent 파일 경로 테스트 스크립트
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from agents.solution.solution_agent import SolutionAgent

def test_solution_agent_with_file():
    """파일 경로를 포함하여 solution agent를 테스트합니다."""
    
    print("🧪 Solution Agent 파일 경로 테스트 시작")
    
    # Solution Agent 생성
    agent = SolutionAgent()
    
    # 테스트용 파일 경로 (사용자가 직접 지정하거나 환경변수에서 가져옴)
    test_file_paths = []
    
    # 환경변수에서 테스트 파일 경로 가져오기
    test_file_env = os.getenv("TEST_PDF_FILE")
    if test_file_env and os.path.exists(test_file_env):
        test_file_paths.append(test_file_env)
        print(f"✅ 환경변수에서 테스트 파일 발견: {test_file_env}")
    
    # 기본 테스트 파일이 없으면 안내
    if not test_file_paths:
        print("⚠️ 테스트할 PDF 파일이 없습니다.")
        print("   환경변수 TEST_PDF_FILE을 설정하거나 직접 파일 경로를 지정해주세요.")
        print("   예시: export TEST_PDF_FILE='path/to/your/test.pdf'")
        return
    
    # 파일 존재 여부 확인
    existing_files = []
    for file_path in test_file_paths:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ 파일 발견: {file_path}")
        else:
            print(f"❌ 파일 없음: {file_path}")
    
    if not existing_files:
        print("⚠️ 테스트할 파일이 없습니다. 파일 경로를 확인해주세요.")
        return
    
    # Solution Agent 실행 테스트
    try:
        print(f"\n🚀 Solution Agent 실행 (파일: {existing_files[0]})")
        
        # Milvus 연결 없이 실행 시도
        print(f"⚠️ Milvus 연결 없이 실행 시도 (오류 예상)")
        result = agent.execute(
            user_question="이 PDF 파일의 문제들을 풀어주세요",
            source_type="external",
            external_file_paths=existing_files,
            exam_title="정보처리기사 모의고사 테스트",
            difficulty="중급",
            subject="기타"
        )
        
        print(f"\n📊 실행 결과:")
        print(f"   - 결과 타입: {type(result)}")
        print(f"   - 결과 개수: {len(result) if isinstance(result, list) else 'N/A'}")
        
        if isinstance(result, list) and len(result) > 0:
            print(f"   - 첫 번째 결과: {result[0]}")
        
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        print(f"   - 오류 타입: {type(e).__name__}")
        if "Milvus" in str(e):
            print(f"   - 해결방법: Milvus 서버가 실행되지 않았습니다.")
            print(f"   - 또는 vectorstore=None으로 설정하여 테스트할 수 있습니다.")
        import traceback
        traceback.print_exc()

def test_solution_agent_internal():
    """내부 메모리 모드로 solution agent를 테스트합니다."""
    
    print("\n🧪 Solution Agent 내부 모드 테스트")
    
    # Milvus 연결 없이 테스트하기 위해 간단한 문제 데이터만 사용
    test_problems = [
        {
            "question": "정보처리기사 시험에서 가장 중요한 과목은?",
            "options": ["소프트웨어 설계", "소프트웨어 개발", "데이터베이스 구축", "정보시스템 구축"]
        }
    ]
    
    try:
        # 간단한 테스트를 위해 SolutionAgent의 기본 구조만 확인
        from agents.solution.solution_agent import SolutionAgent
        agent = SolutionAgent()
        
        print(f"✅ SolutionAgent 생성 성공")
        print(f"   - 에이전트 이름: {agent.name}")
        print(f"   - 에이전트 설명: {agent.description}")
        print(f"   - 그래프 생성: {'성공' if agent.graph else '실패'}")
        
        # Milvus 연결 없이 실행 시도
        print(f"\n⚠️ Milvus 연결 없이 실행 시도 (오류 예상)")
        result = agent.execute(
            user_question="이 문제를 풀어주세요",
            source_type="internal",
            short_term_memory=test_problems,
            exam_title="내부 테스트",
            difficulty="초급",
            subject="테스트"
        )
        
        print(f"📊 내부 모드 결과:")
        print(f"   - 결과: {result}")
        
    except Exception as e:
        print(f"❌ 내부 모드 실행 중 오류: {e}")
        print(f"   - 오류 타입: {type(e).__name__}")
        if "Milvus" in str(e):
            print(f"   - 해결방법: Milvus 서버가 실행되지 않았습니다.")
            print(f"   - 또는 vectorstore=None으로 설정하여 테스트할 수 있습니다.")

if __name__ == "__main__":
    print("=" * 60)
    print("Solution Agent 파일 경로 테스트")
    print("=" * 60)
    
    # 외부 파일 모드 테스트
    test_solution_agent_with_file()
    
    # 내부 모드 테스트
    test_solution_agent_internal()
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
