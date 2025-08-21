#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'teacher', 'agents', 'retrieve', 'nodes'))

try:
    from verifier import FixedResponseSystem
    print("✅ FixedResponseSystem 임포트 성공!")
    
    # FixedResponseSystem 인스턴스 생성
    fixed_system = FixedResponseSystem()
    print("✅ FixedResponseSystem 인스턴스 생성 성공!")
    
    # 테스트 쿼리들
    test_queries = [
        "오늘 날씨 어때요?",
        "안녕하세요",
        "감사합니다",
        "정보처리기사 시험에 대해 알려주세요",
        "파이썬 프로그래밍이 뭔가요?"
    ]
    
    print("\n🧪 FixedResponseSystem 테스트:")
    print("=" * 50)
    
    for query in test_queries:
        result = fixed_system.generate_response(query)
        print(f"\n질문: {query}")
        print(f"결과: {result}")
        
    print("\n✅ FixedResponseSystem 테스트 완료!")
    
except ImportError as e:
    print(f"❌ FixedResponseSystem 임포트 실패: {e}")
except Exception as e:
    print(f"❌ 오류 발생: {e}")
