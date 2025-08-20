#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 텍스트 추출 및 저장 스크립트
- 전체 텍스트를 txt로 저장
- 왼쪽/오른쪽 컬럼을 분리해서 저장
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def save_pdf_text(pdf_path: str):
    """PDF 텍스트를 여러 형태로 저장"""
    
    print(f"📄 PDF 파일: {pdf_path}")
    
    # 1. 전체 텍스트 저장 (Docling)
    print("\n🔧 1단계: Docling으로 전체 텍스트 추출")
    try:
        from docling.document_converter import DocumentConverter
        from teacher.pdf_preprocessor_ai import PDFPreprocessor
        
        pre = PDFPreprocessor()
        converter = DocumentConverter()
        doc_result = converter.convert(pdf_path)
        
        # 전체 마크다운 추출
        raw_text = doc_result.document.export_to_markdown()
        raw_text = pre.normalize_docling_markdown(raw_text)
        raw_text = pre._strip_headers_for_llm(raw_text)
        if hasattr(pre, "_fix_korean_spacing_noise"):
            raw_text = pre._fix_korean_spacing_noise(raw_text)
        
        # 전체 텍스트 저장
        with open("extracted_full_text.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"✅ 전체 텍스트 저장 완료: extracted_full_text.txt ({len(raw_text)} chars)")
        
    except Exception as e:
        print(f"❌ Docling 추출 실패: {e}")
        return
    
    # 2. 컬럼별 텍스트 저장 (pdfminer)
    print("\n🔧 2단계: pdfminer로 좌/우 컬럼 분리")
    try:
        pages_cols = pre._extract_columns_with_pdfminer(pdf_path)
        
        if pages_cols:
            # 왼쪽 컬럼만 모으기
            left_text = "\n\n".join([p.get("left", "") for p in pages_cols if p.get("left")])
            with open("extracted_left_column.txt", "w", encoding="utf-8") as f:
                f.write(left_text)
            print(f"✅ 왼쪽 컬럼 저장 완료: extracted_left_column.txt ({len(left_text)} chars)")
            
            # 오른쪽 컬럼만 모으기
            right_text = "\n\n".join([p.get("right", "") for p in pages_cols if p.get("right")])
            with open("extracted_right_column.txt", "w", encoding="utf-8") as f:
                f.write(right_text)
            print(f"✅ 오른쪽 컬럼 저장 완료: extracted_right_column.txt ({len(right_text)} chars)")
            
            # 페이지별 컬럼 정보도 저장
            with open("extracted_columns_info.txt", "w", encoding="utf-8") as f:
                f.write(f"총 {len(pages_cols)}페이지\n")
                f.write("=" * 50 + "\n\n")
                for i, page in enumerate(pages_cols, 1):
                    f.write(f"=== 페이지 {i} ===\n")
                    f.write(f"LEFT 길이: {len(page.get('left', ''))} chars\n")
                    f.write(f"RIGHT 길이: {len(page.get('right', ''))} chars\n")
                    f.write("-" * 30 + "\n")
                    f.write("LEFT 미리보기:\n")
                    f.write((page.get('left', '')[:500] + '...') if len(page.get('left', '')) > 500 else page.get('left', ''))
                    f.write("\n\nRIGHT 미리보기:\n")
                    f.write((page.get('right', '')[:500] + '...') if len(page.get('right', '')) > 500 else page.get('right', ''))
                    f.write("\n\n")
            
            print(f"✅ 컬럼 정보 저장 완료: extracted_columns_info.txt")
            
        else:
            print("⚠️ 컬럼 추출 실패")
            
    except Exception as e:
        print(f"❌ 컬럼 추출 실패: {e}")
    
    # 3. 청크별 분할 결과도 저장
    print("\n🔧 3단계: 청크별 분할 결과 저장")
    try:
        chunks = pre._chunk_by_paragraph(raw_text, max_chars=16000)
        
        with open("extracted_chunks.txt", "w", encoding="utf-8") as f:
            f.write(f"총 {len(chunks)}개 청크\n")
            f.write("=" * 50 + "\n\n")
            for i, chunk in enumerate(chunks, 1):
                f.write(f"=== 청크 {i} ===\n")
                f.write(f"길이: {len(chunk)} chars\n")
                f.write("-" * 30 + "\n")
                f.write(chunk)
                f.write("\n\n")
        
        print(f"✅ 청크 분할 결과 저장 완료: extracted_chunks.txt")
        
    except Exception as e:
        print(f"❌ 청크 분할 실패: {e}")
    
    print("\n🎯 모든 텍스트 추출 완료!")
    print("생성된 파일들:")
    print("  - extracted_full_text.txt (전체 텍스트)")
    print("  - extracted_left_column.txt (왼쪽 컬럼)")
    print("  - extracted_right_column.txt (오른쪽 컬럼)")
    print("  - extracted_columns_info.txt (컬럼 정보)")
    print("  - extracted_chunks.txt (청크 분할)")

def main():
    pdf_path = "1. 2024년3회_정보처리기사필기기출문제_cut.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)
    
    save_pdf_text(pdf_path)

if __name__ == "__main__":
    print("🚀 PDF 텍스트 추출 및 저장 도구")
    print(" - 전체 텍스트, 좌/우 컬럼, 청크 분할 결과를 txt로 저장")
    main()
