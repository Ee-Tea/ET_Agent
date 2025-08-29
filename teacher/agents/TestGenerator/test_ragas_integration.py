#!/usr/bin/env python3
"""
RAGAS 통합 테스트 스크립트
generator.py에 통합된 RAGAS 검증 방식을 테스트합니다.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 경로 설정
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# 환경 변수 로드
load_dotenv()

from generator import InfoProcessingExamAgent

def test_ragas_integration():
    """RAGAS 통합 테스트"""
    print("🧪 RAGAS 통합 테스트 시작")
    print("=" * 60)
    
    # 환경 변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    
    # TestGenerator 에이전트 초기화
    try:
        agent = InfoProcessingExamAgent()
        print("✅ TestGenerator 에이전트 초기화 완료")
    except Exception as e:
        print(f"❌ 에이전트 초기화 실패: {e}")
        return
    
    # 테스트 파라미터
    test_params = {
        "subject_area": "소프트웨어설계",
        "target_count": 3,  # 빠른 테스트를 위해 3문제만
        "difficulty": "중급"
    }
    
    print(f"\n📋 테스트 파라미터:")
    print(f"  - 과목: {test_params['subject_area']}")
    print(f"  - 문제 수: {test_params['target_count']}")
    print(f"  - 난이도: {test_params['difficulty']}")
    
    # RAGAS 설정 확인
    print(f"\n🔍 RAGAS 설정:")
    print(f"  - RAGAS_ENABLED: {os.getenv('RAGAS_ENABLED', 'true')}")
    print(f"  - RAGAS_QUALITY_THRESHOLD: {os.getenv('RAGAS_QUALITY_THRESHOLD', '0.5')}")
    print(f"  - RAGAS_MAX_ATTEMPTS: {os.getenv('RAGAS_MAX_ATTEMPTS', '3')}")
    
    # 문제 생성 및 검증 테스트
    print(f"\n🚀 RAGAS 기반 문제 생성 시작...")
    try:
        result = agent._generate_subject_quiz(
            subject_area=test_params["subject_area"],
            target_count=test_params["target_count"],
            difficulty=test_params["difficulty"]
        )
        
        if result.get("error"):
            print(f"❌ 문제 생성 실패: {result['error']}")
            return
        
        questions = result.get("questions", [])
        print(f"✅ 문제 생성 완료: {len(questions)}개")
        
        # 생성된 문제 표시
        print(f"\n📝 생성된 문제들:")
        print("-" * 60)
        
        for i, q in enumerate(questions, 1):
            print(f"\n문제 {i}: {q.get('question', 'N/A')}")
            options = q.get('options', [])
            for j, opt in enumerate(options, 1):
                print(f"  {j}. {opt}")
            print(f"정답: {q.get('answer', 'N/A')}")
            print(f"해설: {q.get('explanation', 'N/A')}")
        
        print(f"\n🎯 RAGAS 검증 결과:")
        if 'ragas_score' in result:
            print(f"  - 전체 품질 점수: {result['ragas_score']:.4f}")
        if 'ragas_metrics' in result:
            print(f"  - 세부 메트릭: {result['ragas_metrics']}")
        
        print(f"\n✅ RAGAS 통합 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ragas_integration()
