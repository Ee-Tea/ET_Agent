#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdfplumber를 사용한 PDF 문제 추출 테스트 스크립트
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from teacher.pdf_preprocessor_ai import PDFPreprocessor

def test_pdfplumber_extraction():
    """pdfplumber를 사용한 문제 추출 테스트"""
    
    # PDF 파일 경로 (테스트용)
    pdf_files = [
        "1. 2024년3회_정보처리기사필기기출문제_cut.pdf",
        "1. 2024년3회_정보처리기사필기기출문제.pdf"
    ]
    
    # 실제 존재하는 파일만 필터링
    existing_files = [f for f in pdf_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ 테스트할 PDF 파일을 찾을 수 없습니다.")
        print("사용 가능한 PDF 파일:")
        for f in os.listdir("."):
            if f.endswith(".pdf"):
                print(f"  - {f}")
        return
    
    print(f"🧪 pdfplumber 테스트 시작")
    print(f"📁 테스트 파일: {existing_files}")
    
    # PDFPreprocessor 초기화
    preprocessor = PDFPreprocessor()
    
    try:
        # pdfplumber를 사용한 문제 추출
        print("\n🔧 pdfplumber를 사용한 문제 추출 시작...")
        problems = preprocessor.extract_problems_with_pdfplumber(existing_files)
        
        if problems:
            print(f"\n✅ 성공적으로 {len(problems)}개 문제 추출!")
            
            # 결과 저장
            output_file = "pdfplumber_extracted_problems.json"
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(problems, f, ensure_ascii=False, indent=2)
            
            print(f"💾 결과가 {output_file}에 저장되었습니다.")
            
            # 처음 3개 문제 미리보기
            print(f"\n📋 추출된 문제 미리보기 (처음 3개):")
            for i, problem in enumerate(problems[:3], 1):
                print(f"\n{i}. 문제:")
                print(f"   질문: {problem.get('question', '')[:100]}...")
                print(f"   보기: {len(problem.get('options', []))}개")
                for j, option in enumerate(problem.get('options', [])[:3], 1):
                    print(f"     {j}) {option[:50]}...")
                if len(problem.get('options', [])) > 3:
                    print(f"     ... 외 {len(problem.get('options', [])) - 3}개")
        else:
            print("❌ 문제 추출에 실패했습니다.")
            
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def test_column_extraction():
    """컬럼 분리 기능만 테스트"""
    pdf_files = [
        "1. 2024년3회_정보처리기사필기기출문제_cut.pdf"
    ]
    
    existing_files = [f for f in pdf_files if os.path.exists(f)]
    if not existing_files:
        print("❌ 테스트할 PDF 파일을 찾을 수 없습니다.")
        return
    
    print(f"🧪 컬럼 분리 테스트 시작")
    
    preprocessor = PDFPreprocessor()
    
    try:
        with pdfplumber.open(existing_files[0]) as pdf:
            page = pdf.pages[0]  # 첫 번째 페이지만 테스트
            
            print(f"📄 페이지 크기: {page.width} x {page.height}")
            
            # 컬럼 분리 테스트
            left_col, right_col = preprocessor._split_page_into_columns(page)
            
            print(f"\n📋 왼쪽 컬럼 (길이: {len(left_col)}):")
            print(left_col[:200] + "..." if len(left_col) > 200 else left_col)
            
            print(f"\n📋 오른쪽 컬럼 (길이: {len(right_col)}):")
            print(right_col[:200] + "..." if len(right_col) > 200 else right_col)
            
    except Exception as e:
        print(f"❌ 컬럼 분리 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 PDF 문제 추출 테스트 시작")
    print("=" * 50)
    
    # 1. 컬럼 분리 테스트
    test_column_extraction()
    
    print("\n" + "=" * 50)
    
    # 2. 전체 문제 추출 테스트
    test_pdfplumber_extraction()
    
    print("\n✅ 테스트 완료!")
