#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdfplumber를 사용한 2단(좌/우 열) PDF 파싱 테스트 스크립트
- pdfplumber로 왼쪽/오른쪽 컬럼을 구분하여 문제 추출
- LLM을 사용하여 문제와 보기를 정확하게 묶어서 추출
- 문제 번호 순으로 자동 정렬
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def preview_problems(problems: List[Dict], n: int = 5):
    """문제 미리보기 출력"""
    print(f"\n📝 미리보기(상위 {min(n, len(problems))}문항):")
    for i, p in enumerate(problems[:n], 1):
        q = (p.get("question") or "").strip().replace("\n", " ")
        opts = p.get("options") or []
        print(f"  [{i}] {q[:120]}{'...' if len(q) > 120 else ''}")
        for j, o in enumerate(opts[:4], 1):
            print(f"      {j}. {o[:80]}{'...' if len(o) > 80 else ''}")

def save_problems_to_json(problems: List[Dict], filename: str):
    """문제를 JSON 파일로 저장"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(problems, f, ensure_ascii=False, indent=2)
        print(f"💾 결과를 {filename}에 저장했습니다.")
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")

def main():
    """메인 함수"""
    print("🚀 pdfplumber를 사용한 PDF 문제 추출 테스트")
    print("=" * 60)
    
    # PDF 파일 경로 설정
    pdf_file = "1. 2024년3회_정보처리기사필기기출문제_cut.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_file}")
        print("📁 현재 디렉토리의 PDF 파일들:")
        for f in os.listdir("."):
            if f.endswith(".pdf"):
                print(f"  - {f}")
        return
    
    try:
        # PDFPreprocessor 임포트 및 초기화
        from teacher.pdf_preprocessor_ai import PDFPreprocessor
        
        print(f"📖 PDF 파일 처리 중: {pdf_file}")
        preprocessor = PDFPreprocessor()
        
        # pdfplumber를 사용하여 문제 추출
        print("\n🔧 pdfplumber로 문제 추출 시작...")
        problems = preprocessor.extract_problems_with_pdfplumber([pdf_file])
        
        if not problems:
            print("❌ 문제 추출에 실패했습니다.")
            return
        
        print(f"\n✅ 총 {len(problems)}개 문제 추출 완료!")
        
        # 결과 미리보기
        preview_problems(problems)
        
        # JSON 파일로 저장
        output_filename = f"pdfplumber_{pdf_file.replace('.pdf', '_problems.json')}"
        save_problems_to_json(problems, output_filename)
        
        # 문제 번호별 요약
        print(f"\n📊 문제 번호별 요약:")
        number_counts = {}
        for problem in problems:
            question = problem.get('question', '')
            # 문제 번호 추출
            import re
            number_match = re.search(r'^(\d+)\s*\.', question)
            if number_match:
                number = int(number_match.group(1))
                number_counts[number] = number_counts.get(number, 0) + 1
        
        # 번호 순으로 정렬하여 출력
        for number in sorted(number_counts.keys()):
            print(f"  {number:2d}번: {number_counts[number]}개")
        
        print(f"\n🎯 테스트 완료! {len(problems)}개 문제를 성공적으로 추출했습니다.")
        
    except ImportError as e:
        print(f"❌ 모듈 임포트 실패: {e}")
        print("📦 필요한 패키지가 설치되어 있는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
