#!/usr/bin/env python3
"""
설정 테스트 스크립트
Ragas 및 관련 의존성이 제대로 설치되어 있는지 확인합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def test_imports():
    """필요한 모듈들이 제대로 import되는지 테스트"""
    print("🔍 모듈 import 테스트 중...")
    
    try:
        # 기본 모듈들
        import json
        import pandas as pd
        import asyncio
        print("✅ 기본 모듈들 import 성공")
        
        # 환경변수
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ dotenv 로드 성공")
        
        # OpenAI
        import openai
        print("✅ openai import 성공")
        
        # LangChain
        from langchain_openai import ChatOpenAI
        from langchain_community.document_loaders import DirectoryLoader
        from langchain_core.documents import Document
        print("✅ langchain 모듈들 import 성공")
        
        # Ragas
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.testset import TestsetGenerator
        from ragas.testset.synthesizers import default_query_distribution
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.dataset import Dataset
        print("✅ ragas 모듈들 import 성공")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 기타 오류: {e}")
        return False

def test_environment():
    """환경 변수 설정 확인"""
    print("\n🔧 환경 변수 테스트 중...")
    
    # OpenAI API 키 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ OPENAI_API_KEY 설정됨")
        print(f"   키 길이: {len(openai_key)}자")
        print(f"   키 시작: {openai_key[:10]}...")
    else:
        print("❌ OPENAI_API_KEY가 설정되지 않음")
        return False
    
    return True

def test_data_files():
    """데이터 파일 존재 확인"""
    print("\n📁 데이터 파일 테스트 중...")
    
    exam_dir = Path(__file__).parent.parent / "exam"
    
    # JSON 파일들 확인
    json_files = [
        "2024년3회_기사필기_전체문제.json",
        "2025년1회_기사필기_전체문제.json", 
        "2025년2회_기사필기_전체문제.json"
    ]
    
    found_files = []
    for json_file in json_files:
        file_path = exam_dir / json_file
        if file_path.exists():
            print(f"✅ {json_file} 존재")
            found_files.append(file_path)
        else:
            print(f"❌ {json_file} 없음")
    
    # 파싱된 파일들 확인
    parsed_dir = exam_dir / "parsed_exam_json"
    if parsed_dir.exists():
        parsed_files = list(parsed_dir.glob("*.json"))
        print(f"✅ 파싱된 파일 {len(parsed_files)}개 발견")
        for f in parsed_files[:3]:  # 처음 3개만 표시
            print(f"   - {f.name}")
    else:
        print("❌ 파싱된 파일 디렉토리 없음")
    
    return len(found_files) > 0

def test_openai_connection():
    """OpenAI API 연결 테스트"""
    print("\n🌐 OpenAI API 연결 테스트 중...")
    
    try:
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        
        client = openai.OpenAI()
        
        # 간단한 API 호출 테스트
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print("✅ OpenAI API 연결 성공")
        print(f"   응답: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API 연결 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🧪 설정 테스트 시작")
    print("=" * 50)
    
    tests = [
        ("모듈 Import", test_imports),
        ("환경 변수", test_environment),
        ("데이터 파일", test_data_files),
        ("OpenAI 연결", test_openai_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} 테스트")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 오류: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n전체: {passed}/{len(results)} 테스트 통과")
    
    if passed == len(results):
        print("🎉 모든 테스트 통과! 골든 데이터셋 생성을 시작할 수 있습니다.")
    else:
        print("⚠️ 일부 테스트 실패. 문제를 해결한 후 다시 시도해주세요.")

if __name__ == "__main__":
    main()
