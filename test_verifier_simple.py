#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# verifier.py가 있는 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'teacher', 'agents', 'retrieve', 'nodes'))

try:
    from verifier import FixedResponseSystem
    print("✅ FixedResponseSystem 임포트 성공!")
    
    # FixedResponseSystem 인스턴스 생성
    fixed_system = FixedResponseSystem()
    print("✅ FixedResponseSystem 인스턴스 생성 성공!")
    
    # 테스트 쿼리들
    test_queries = [
        "오늘 날씨 어때?",  # 주제 외 거절
        "안녕하세요",       # 인사
        "감사합니다",       # 감사
        "정보처리기사 시험에 대해 알려주세요",  # 주제 관련
        "파이썬 프로그래밍이 뭔가요?",        # 주제 관련
        "시간이 몇시야?",   # 주제 외 거절
        "맛집 추천해줘"     # 주제 외 거절
    ]
    
    print("\n🧪 FixedResponseSystem 테스트:")
    print("=" * 60)
    
    for query in test_queries:
        result = fixed_system.generate_response(query)
        print(f"\n질문: {query}")
        print(f"결과: {result}")
        
        # 주제 외 거절 메시지인지 확인
        if result["type"] == "rejection" and result["category"] == "주제_외_거절":
            print("✅ 주제 외 거절 메시지 정상 작동!")
        elif result["type"] == "quick_response":
            print("✅ 빠른 응답 정상 작동!")
        elif result["type"] == "topic_related":
            print("✅ 주제 관련 질문으로 인식!")
        
    print("\n✅ FixedResponseSystem 테스트 완료!")
    
except ImportError as e:
    print(f"❌ FixedResponseSystem 임포트 실패: {e}")
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
