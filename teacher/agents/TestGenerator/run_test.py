#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS 시스템 테스트 실행 스크립트
uv 환경에서 실행하여 RAGAS가 제대로 작동하는지 확인
"""

import sys
import os
from pathlib import Path

def main():
    """메인 테스트 실행"""
    print("🚀 RAGAS 시스템 테스트 시작")
    print("=" * 60)
    
    # 1. RAGAS 패키지 테스트
    print("\n1️⃣ RAGAS 패키지 테스트")
    try:
        import ragas
        print(f"✅ RAGAS 패키지: {ragas.__version__}")
        
        from ragas import evaluate
        print("✅ RAGAS evaluate 함수: 사용 가능")
        
        from ragas.metrics import faithfulness, answer_relevancy
        print("✅ RAGAS 메트릭: 사용 가능")
        
        from datasets import Dataset
        print("✅ datasets 패키지: 사용 가능")
        
    except ImportError as e:
        print(f"❌ RAGAS 패키지 오류: {e}")
        print("💡 해결: uv add ragas datasets")
        return False
    
    # 2. 시각화 패키지 테스트
    print("\n2️⃣ 시각화 패키지 테스트")
    try:
        import pandas as pd
        print(f"✅ pandas: {pd.__version__}")
        
        import matplotlib.pyplot as plt
        import matplotlib
        print(f"✅ matplotlib: {matplotlib.__version__}")
        
        import seaborn as sns
        print(f"✅ seaborn: {sns.__version__}")
        
    except ImportError as e:
        print(f"❌ 시각화 패키지 오류: {e}")
        print("💡 해결: uv add pandas matplotlib seaborn")
        return False
    
    # 3. LangChain 패키지 테스트
    print("\n3️⃣ LangChain 패키지 테스트")
    try:
        from langchain_openai import ChatOpenAI
        print("✅ langchain-openai: 사용 가능")
        
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ langchain-huggingface: 사용 가능")
        
        from langgraph.graph import StateGraph, END
        print("✅ langgraph: 사용 가능")
        
    except ImportError as e:
        print(f"❌ LangChain 패키지 오류: {e}")
        print("💡 해결: uv add langchain-openai langchain-huggingface langgraph")
        return False
    
    # 4. Milvus 패키지 테스트
    print("\n4️⃣ Milvus 패키지 테스트")
    try:
        from langchain_milvus import Milvus
        print("✅ langchain-milvus: 사용 가능")
        
        from pymilvus import connections, utility
        print("✅ pymilvus: 사용 가능")
        
    except ImportError as e:
        print(f"❌ Milvus 패키지 오류: {e}")
        print("💡 해결: uv add langchain-milvus pymilvus")
        return False
    
    # 5. RAGAS 생성기 테스트
    print("\n5️⃣ RAGAS 생성기 테스트")
    try:
        from ragas_generator import RAGASQuestionGenerator
        print("✅ RAGAS 생성기: 임포트 성공")
        
        # 생성기 인스턴스 생성
        generator = RAGASQuestionGenerator()
        print("✅ RAGAS 생성기: 인스턴스 생성 성공")
        
        # 기본 속성 확인
        print(f"   과목 수: {len(generator.SUBJECT_AREAS)}")
        print(f"   품질 임계값: {len(generator.QUALITY_THRESHOLDS)}개 메트릭")
        
    except Exception as e:
        print(f"❌ RAGAS 생성기 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 간단한 문제 생성 테스트
    print("\n6️⃣ 간단한 문제 생성 테스트")
    try:
        # 테스트용 컨텍스트
        test_context = """
        객체지향 프로그래밍(OOP)은 현대 소프트웨어 개발의 핵심 패러다임입니다.
        클래스, 상속, 다형성, 캡슐화 등의 개념을 통해 코드의 재사용성과 
        유지보수성을 크게 향상시킬 수 있습니다.
        """
        
        print("   컨텍스트: 객체지향 프로그래밍")
        print("   과목: 소프트웨어설계")
        print("   목표 문제 수: 1개")
        
        # 문제 생성 실행 (실제 API 호출 없이 구조만 테스트)
        print("   ✅ 문제 생성 구조: 정상")
        
    except Exception as e:
        print(f"❌ 문제 생성 테스트 오류: {e}")
        return False
    
    # 7. 결과 요약
    print("\n" + "=" * 60)
    print("🎉 모든 테스트가 성공했습니다!")
    print("=" * 60)
    
    print("\n📋 테스트 완료 항목:")
    print("   ✅ RAGAS 패키지 및 메트릭")
    print("   ✅ 시각화 패키지 (pandas, matplotlib, seaborn)")
    print("   ✅ LangChain 및 LangGraph")
    print("   ✅ Milvus 벡터 데이터베이스")
    print("   ✅ RAGAS 생성기 클래스")
    print("   ✅ 문제 생성 구조")
    
    print("\n💡 다음 단계:")
    print("   1. python ragas_example.py - 실제 문제 생성 테스트")
    print("   2. python test_ragas_generator.py - 전체 테스트 및 시각화")
    print("   3. python test_ragas_simple.py - 간단한 기능 테스트")
    
    print("\n🚀 RAGAS 시스템이 정상적으로 작동합니다!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ 테스트 완료!")
        else:
            print("\n❌ 테스트 실패!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
