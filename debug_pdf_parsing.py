#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 파싱 디버깅 도구 - 2단 PDF 문제 해결
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def debug_pdf_structure(pdf_path: str):
    """PDF 구조를 자세히 분석합니다."""
    
    print(f"🔍 PDF 구조 상세 분석: {pdf_path}")
    print("=" * 60)
    
    try:
        # 1. 파일 기본 정보
        file_size = os.path.getsize(pdf_path)
        print(f"📁 파일 크기: {file_size:,} bytes")
        
        # 2. PDFMiner로 직접 분석
        print("\n🔧 PDFMiner 직접 분석")
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer, LTChar, LTLine, LTImage
            
            pages = list(extract_pages(pdf_path))
            print(f"📄 총 페이지 수: {len(pages)}")
            
            for i, page in enumerate(pages[:3]):  # 처음 3페이지만
                print(f"\n📖 페이지 {i+1}:")
                
                # 텍스트 요소들
                text_elements = [obj for obj in page if isinstance(obj, LTTextContainer)]
                print(f"   텍스트 요소: {len(text_elements)}개")
                
                # 좌표 정보
                for j, text_obj in enumerate(text_elements[:5]):  # 처음 5개만
                    bbox = text_obj.bbox
                    text_content = text_obj.get_text().strip()
                    if text_content:
                        print(f"     {j+1}. 좌표: ({bbox[0]:.1f}, {bbox[1]:.1f}) → ({bbox[2]:.1f}, {bbox[3]:.1f})")
                        print(f"        내용: {text_content[:50]}...")
                
                # 선/이미지 요소들
                lines = [obj for obj in page if isinstance(obj, LTLine)]
                images = [obj for obj in page if isinstance(obj, LTImage)]
                print(f"   선 요소: {len(lines)}개, 이미지: {len(images)}개")
                
        except Exception as e:
            print(f"❌ PDFMiner 분석 실패: {e}")
        
        # 3. Docling으로 텍스트 추출
        print("\n🔧 Docling 텍스트 추출")
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            
            doc = converter.convert(pdf_path)
            md_content = doc.document.export_to_markdown()
            
            print(f"📝 마크다운 길이: {len(md_content)} 문자")
            print(f"📝 마크다운 미리보기:")
            print(f"   {md_content[:300]}...")
            
            # 줄별로 분석
            lines = md_content.split('\n')
            print(f"\n📊 줄별 분석 (처음 10줄):")
            for i, line in enumerate(lines[:10]):
                if line.strip():
                    print(f"   {i+1:2d}: {line[:80]}")
            
        except Exception as e:
            print(f"❌ Docling 분석 실패: {e}")
        
        # 4. 2단 구조 감지 시도
        print("\n🔧 2단 구조 감지")
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer
            
            pages = list(extract_pages(pdf_path))
            if pages:
                page = pages[0]  # 첫 페이지
                text_elements = [obj for obj in page if isinstance(obj, LTTextContainer)]
                
                if text_elements:
                    # x 좌표로 좌우 분류
                    left_elements = []
                    right_elements = []
                    page_width = page.width
                    mid_x = page_width / 2
                    
                    for obj in text_elements:
                        bbox = obj.bbox
                        center_x = (bbox[0] + bbox[2]) / 2
                        if center_x < mid_x:
                            left_elements.append(obj)
                        else:
                            right_elements.append(obj)
                    
                    print(f"   페이지 너비: {page_width:.1f}")
                    print(f"   중앙점: {mid_x:.1f}")
                    print(f"   좌측 요소: {len(left_elements)}개")
                    print(f"   우측 요소: {len(right_elements)}개")
                    
                    # 좌우 텍스트 내용 비교
                    if left_elements and right_elements:
                        left_text = " ".join([obj.get_text().strip() for obj in left_elements[:3]])
                        right_text = " ".join([obj.get_text().strip() for obj in right_elements[:3]])
                        
                        print(f"   좌측 텍스트: {left_text[:50]}...")
                        print(f"   우측 텍스트: {right_text[:50]}...")
                        
                        # 2단 구조 여부 판단
                        if len(left_elements) > 5 and len(right_elements) > 5:
                            print("   ✅ 2단 구조로 판단됨")
                        else:
                            print("   ⚠️ 2단 구조가 아닐 수 있음")
                    else:
                        print("   ❌ 좌우 요소가 부족함")
                        
        except Exception as e:
            print(f"❌ 2단 구조 감지 실패: {e}")
        
    except Exception as e:
        print(f"❌ 전체 분석 실패: {e}")
        import traceback
        traceback.print_exc()

def test_pdf_preprocessor_methods(pdf_path: str):
    """PDF 전처리기의 각 메서드를 개별적으로 테스트합니다."""
    
    print(f"\n🔧 PDF 전처리기 메서드 개별 테스트")
    print("=" * 60)
    
    try:
        from teacher.pdf_preprocessor import PDFPreprocessor
        preprocessor = PDFPreprocessor()
        
        # 1. _reorder_two_columns_with_pdfminer 테스트
        print("\n🔄 1. 2단 컬럼 재정렬 메서드 테스트")
        try:
            result = preprocessor._reorder_two_columns_with_pdfminer(pdf_path)
            print(f"   결과 길이: {len(result)} 문자")
            print(f"   결과 미리보기: {result[:200]}...")
            
            if len(result) < 100:
                print("   ⚠️ 결과가 너무 짧음 - PDFMiner가 제대로 작동하지 않을 수 있음")
                
        except Exception as e:
            print(f"   ❌ 실패: {e}")
        
        # 2. _split_problem_blocks 테스트
        print("\n🔄 2. 문제 블록 분할 메서드 테스트")
        try:
            # 간단한 테스트 텍스트
            test_text = "1. 첫 번째 문제입니다.\n① 보기1\n② 보기2\n2. 두 번째 문제입니다.\n① 보기1\n② 보기2"
            blocks = preprocessor._split_problem_blocks(test_text)
            print(f"   테스트 텍스트 결과: {len(blocks)}개 블록")
            
            # 실제 PDF 텍스트로 테스트
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            doc = converter.convert(pdf_path)
            pdf_text = doc.document.export_to_markdown()
            
            blocks = preprocessor._split_problem_blocks(pdf_text)
            print(f"   실제 PDF 결과: {len(blocks)}개 블록")
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
        
        # 3. _process_pdf_text 테스트
        print("\n🔄 3. 통합 처리 메서드 테스트")
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            doc = converter.convert(pdf_path)
            pdf_text = doc.document.export_to_markdown()
            
            result = preprocessor._process_pdf_text(pdf_text, pdf_path)
            print(f"   결과: {len(result)}개 블록")
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
            
    except Exception as e:
        print(f"❌ 전처리기 테스트 실패: {e}")

def interactive_debug():
    """대화형 디버깅 모드"""
    
    print("🚀 PDF 파싱 디버깅 도구")
    print("=" * 60)
    
    # PDF 파일 경로 입력
    pdf_path = input("디버깅할 PDF 파일 경로를 입력하세요: ").strip()
    
    if not pdf_path or not os.path.exists(pdf_path):
        print("❌ 유효하지 않은 파일 경로입니다.")
        return
    
    print(f"\n📄 선택된 파일: {pdf_path}")
    
    while True:
        print("\n🔧 디버깅 옵션:")
        print("1. PDF 구조 상세 분석")
        print("2. 전처리기 메서드 개별 테스트")
        print("3. 전체 테스트 실행")
        print("4. 종료")
        
        try:
            choice = input("\n선택하세요 (1-4): ").strip()
            
            if choice == "1":
                debug_pdf_structure(pdf_path)
            elif choice == "2":
                test_pdf_preprocessor_methods(pdf_path)
            elif choice == "3":
                debug_pdf_structure(pdf_path)
                test_pdf_preprocessor_methods(pdf_path)
            elif choice == "4":
                print("👋 디버깅을 종료합니다.")
                break
            else:
                print("⚠️ 1-4 중에서 선택해주세요.")
                
        except KeyboardInterrupt:
            print("\n👋 디버깅을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    interactive_debug()
