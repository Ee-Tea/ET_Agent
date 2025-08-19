#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2단 PDF 파싱 테스트 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_2column_pdf_parsing():
    """2단 PDF 파싱 기능을 테스트합니다."""
    
    print("🔍 2단 PDF 파싱 테스트 시작")
    print("=" * 50)
    
    try:
        # PDF 전처리기 import
        from teacher.pdf_preprocessor import PDFPreprocessor
        
        # 전처리기 초기화
        print("📚 PDF 전처리기 초기화 중...")
        preprocessor = PDFPreprocessor()
        print("✅ PDF 전처리기 초기화 완료")
        
        # 테스트할 PDF 파일 경로 (사용자가 수정 가능)
        test_pdf_path = "teacher/agents/solution/pdf_outputs/1. 2024년3회_정보처리기사필기기출문제_cut.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"⚠️ 테스트 PDF 파일을 찾을 수 없습니다: {test_pdf_path}")
            print("📝 다른 PDF 파일 경로를 입력하세요:")
            test_pdf_path = input("PDF 파일 경로: ").strip()
            
            if not test_pdf_path or not os.path.exists(test_pdf_path):
                print("❌ 유효하지 않은 파일 경로입니다.")
                return
        
        print(f"📄 테스트 파일: {test_pdf_path}")
        print("=" * 50)
        
        # 1. 기본 PDF 텍스트 추출 테스트
        print("\n🔧 1단계: 기본 PDF 텍스트 추출")
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            
            # Docling으로 마크다운 추출
            doc_md = converter.convert(test_pdf_path).document.export_to_markdown()
            doc_md = preprocessor.normalize_docling_markdown(doc_md)
            
            print(f"✅ 기본 텍스트 추출 완료 (길이: {len(doc_md)} 문자)")
            print(f"📝 텍스트 미리보기 (처음 200자):")
            print(f"   {doc_md[:200]}...")
            
        except Exception as e:
            print(f"❌ 기본 텍스트 추출 실패: {e}")
            return
        
        # 2. 문제 블록 분할 테스트
        print("\n🔧 2단계: 문제 블록 분할")
        try:
            blocks = preprocessor._split_problem_blocks(doc_md)
            print(f"✅ 문제 블록 분할 완료: {len(blocks)}개 블록")
            
            if blocks:
                print(f"📝 첫 번째 블록 미리보기:")
                print(f"   {blocks[0][:150]}...")
            
        except Exception as e:
            print(f"❌ 문제 블록 분할 실패: {e}")
            return
        
        # 3. 2단 컬럼 처리 테스트
        print("\n🔧 3단계: 2단 컬럼 처리")
        try:
            # 2단 컬럼 재정렬
            reordered_text = preprocessor._reorder_two_columns_with_pdfminer(test_pdf_path)
            print(f"✅ 2단 컬럼 재정렬 완료 (길이: {len(reordered_text)} 문자)")
            
            # 재정렬된 텍스트로 문제 블록 분할
            reordered_blocks = preprocessor._split_problem_blocks(reordered_text)
            print(f"✅ 재정렬 후 문제 블록 분할: {len(reordered_blocks)}개 블록")
            
            if reordered_blocks:
                print(f"📝 재정렬 후 첫 번째 블록 미리보기:")
                print(f"   {reordered_blocks[0][:150]}...")
            
        except Exception as e:
            print(f"❌ 2단 컬럼 처리 실패: {e}")
            return
        
        # 4. 통합 처리 테스트
        print("\n🔧 4단계: 통합 처리 테스트")
        try:
            # _process_pdf_text 메서드로 통합 처리
            processed_blocks = preprocessor._process_pdf_text(doc_md, test_pdf_path)
            print(f"✅ 통합 처리 완료: {len(processed_blocks)}개 블록")
            
            if processed_blocks:
                print(f"📝 통합 처리 후 첫 번째 블록 미리보기:")
                print(f"   {processed_blocks[0][:150]}...")
            
        except Exception as e:
            print(f"❌ 통합 처리 실패: {e}")
            return
        
        # 5. 최종 문제 추출 테스트
        print("\n🔧 5단계: 최종 문제 추출")
        try:
            problems = preprocessor.extract_problems_from_pdf([test_pdf_path])
            print(f"✅ 최종 문제 추출 완료: {len(problems)}개 문제")
            
            if problems:
                print(f"📝 첫 번째 문제 미리보기:")
                problem = problems[0]
                print(f"   문제: {problem.get('question', '')[:100]}...")
                print(f"   보기: {problem.get('options', [])[:3]}...")
            
        except Exception as e:
            print(f"❌ 최종 문제 추출 실패: {e}")
            return
        
        print("\n" + "=" * 50)
        print("🎉 모든 테스트 완료!")
        print("📊 결과 요약:")
        print(f"   - 기본 텍스트: {len(doc_md)} 문자")
        print(f"   - 기본 블록: {len(blocks)}개")
        print(f"   - 2단 재정렬 후 블록: {len(reordered_blocks)}개")
        print(f"   - 통합 처리 후 블록: {len(processed_blocks)}개")
        print(f"   - 최종 문제: {len(problems)}개")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def test_specific_pdf():
    """특정 PDF 파일의 2단 처리만 테스트합니다."""
    
    print("🎯 특정 PDF 2단 처리 테스트")
    print("=" * 50)
    
    # PDF 파일 경로 입력
    pdf_path = input("테스트할 PDF 파일 경로를 입력하세요: ").strip()
    
    if not pdf_path or not os.path.exists(pdf_path):
        print("❌ 유효하지 않은 파일 경로입니다.")
        return
    
    try:
        from teacher.pdf_preprocessor import PDFPreprocessor
        preprocessor = PDFPreprocessor()
        
        print(f"📄 테스트 파일: {pdf_path}")
        
        # 2단 컬럼 재정렬 테스트
        print("\n🔄 2단 컬럼 재정렬 중...")
        reordered = preprocessor._reorder_two_columns_with_pdfminer(pdf_path)
        
        print(f"✅ 재정렬 완료 (길이: {len(reordered)} 문자)")
        print(f"📝 재정렬된 텍스트 미리보기:")
        print(f"   {reordered[:300]}...")
        
        # 문제 블록 분할 테스트
        print("\n🔄 문제 블록 분할 중...")
        blocks = preprocessor._split_problem_blocks(reordered)
        
        print(f"✅ 블록 분할 완료: {len(blocks)}개 블록")
        if blocks:
            print(f"📝 첫 번째 블록:")
            print(f"   {blocks[0][:200]}...")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 2단 PDF 파싱 테스트 도구")
    print("=" * 50)
    print("1. 전체 테스트 실행")
    print("2. 특정 PDF 2단 처리 테스트")
    print("3. 종료")
    
    while True:
        try:
            choice = input("\n선택하세요 (1-3): ").strip()
            
            if choice == "1":
                test_2column_pdf_parsing()
                break
            elif choice == "2":
                test_specific_pdf()
                break
            elif choice == "3":
                print("👋 테스트를 종료합니다.")
                break
            else:
                print("⚠️ 1, 2, 3 중에서 선택해주세요.")
                
        except KeyboardInterrupt:
            print("\n👋 테스트를 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            break
