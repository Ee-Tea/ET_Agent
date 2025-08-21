#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'teacher', 'agents', 'TestGenerator'))

try:
    print("1️⃣ TestGenerator 모듈 임포트 시도...")
    from generator import InfoProcessingExamAgent
    print("✅ TestGenerator 모듈 임포트 성공!")
    
    print("\n2️⃣ InfoProcessingExamAgent 인스턴스 생성 시도...")
    # GROQ_API_KEY 없이 테스트
    agent = InfoProcessingExamAgent()
    print("✅ InfoProcessingExamAgent 인스턴스 생성 성공!")
    
    print("\n3️⃣ FixedResponseSystem 통합 확인...")
    if hasattr(agent, 'fixed_response_system') and agent.fixed_response_system:
        print("✅ FixedResponseSystem이 정상적으로 통합되었습니다!")
        
        # 테스트 쿼리들
        test_queries = [
            "오늘 날씨 어때요?",
            "안녕하세요",
            "감사합니다",
            "정보처리기사 시험에 대해 알려주세요"
        ]
        
        print("\n🧪 FixedResponseSystem 테스트:")
        print("=" * 50)
        
        for query in test_queries:
            result = agent.check_off_topic_query(query)
            print(f"\n질문: {query}")
            print(f"결과: {result}")
            
    else:
        print("❌ FixedResponseSystem이 통합되지 않았습니다.")
        print(f"fixed_response_system 속성: {getattr(agent, 'fixed_response_system', 'None')}")
    
    print("\n4️⃣ Milvus 통합 확인...")
    if hasattr(agent, 'vectorstore'):
        print("✅ vectorstore 속성이 존재합니다.")
        print(f"vectorstore 타입: {type(agent.vectorstore)}")
    else:
        print("❌ vectorstore 속성이 없습니다.")
    
    print("\n✅ 모든 테스트 완료!")
    
except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
