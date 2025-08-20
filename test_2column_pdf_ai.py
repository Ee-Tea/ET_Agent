#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2단(좌/우 열) PDF 파싱 테스트 스크립트 - 정리판
 - 헤더/번호 분할 사용 안 함
 - 좌/우 컬럼 추출 → 열 단위 LLM 일괄 추출 흐름만 검증
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트 경로 추가 (필요 시 수정)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def preview_problems(problems: List[Dict], n: int = 5):
    print(f"\n📝 미리보기(상위 {min(n, len(problems))}문항):")
    for i, p in enumerate(problems[:n], 1):
        q = (p.get("question") or "").strip().replace("\n", " ")
        opts = p.get("options") or []
        print(f"  [{i}] {q[:120]}{'...' if len(q) > 120 else ''}")
        for j, o in enumerate(opts[:4], 1):
            print(f"     - {j}) {str(o).strip()[:100]}{'...' if len(str(o))>100 else ''}")
    print()

def run_quick_all(pdf_path: str):
    from teacher.pdf_preprocessor_ai import PDFPreprocessor

    print("📚 PDF 전처리기 초기화 중...")
    pre = PDFPreprocessor()
    print("✅ 준비 완료")

    if not os.path.exists(pdf_path):
        print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)

    print(f"\n📄 테스트 대상: {pdf_path}")
    print("=" * 60)

    # 1) Docling 원문 추출 프리뷰(폴백용 텍스트 품질 확인)
    try:
        from docling.document_converter import DocumentConverter
        print("\n🔧 1단계: Docling 텍스트 추출")
        conv = DocumentConverter()
        doc_md = conv.convert(pdf_path).document.export_to_markdown()
        doc_md = pre.normalize_docling_markdown(doc_md)
        doc_md = pre._strip_headers_for_llm(doc_md)
        if hasattr(pre, "_fix_korean_spacing_noise"):
            doc_md = pre._fix_korean_spacing_noise(doc_md)
        print(f"✅ 텍스트 길이: {len(doc_md)}")
        print("📝 미리보기(처음 200자):")
        print(doc_md[:200] + ("..." if len(doc_md) > 200 else ""))
    except Exception as e:
        print(f"⚠️ Docling 추출 프리뷰 실패(계속 진행): {e}")

    # 2) 컬럼(좌/우) 추출 프리뷰
    print("\n🔧 2단계: 좌/우 컬럼 추출 프리뷰(pdfminer)")
    try:
        pages = pre._extract_columns_with_pdfminer(pdf_path)
        print(f"✅ 페이지 수: {len(pages)}")
        if pages:
            l0 = (pages[0].get("left") or "")
            r0 = (pages[0].get("right") or "")
            print(f"   - p1 LEFT  : {len(l0)} chars  | preview: {(l0[:120] + '...') if len(l0)>120 else l0}")
            print(f"   - p1 RIGHT : {len(r0)} chars  | preview: {(r0[:120] + '...') if len(r0)>120 else r0}")
    except Exception as e:
        print(f"⚠️ 컬럼 추출 프리뷰 실패(계속 진행): {e}")

    # 3) 열 기반 전체 추출(end-to-end)
    print("\n🔧 3단계: 열 기반 LLM 추출(엔드투엔드)")
    try:
        problems = pre.extract_problems_from_pdf([pdf_path])
        print(f"✅ 최종 문제 추출 완료: {len(problems)}문항")
        preview_problems(problems, n=8)
    except Exception as e:
        print(f"❌ 최종 추출 실패: {e}")
        raise

def run_columns_only(pdf_path: str):
    """열 추출만 보고 싶을 때"""
    from teacher.pdf_preprocessor_ai import PDFPreprocessor
    pre = PDFPreprocessor()
    pages = pre._extract_columns_with_pdfminer(pdf_path)
    print(f"✅ 페이지 수: {len(pages)}")
    total_left = sum(len(p.get("left") or "") for p in pages)
    total_right = sum(len(p.get("right") or "") for p in pages)
    print(f"   - LEFT 총 길이 : {total_left}")
    print(f"   - RIGHT 총 길이: {total_right}")
    if pages:
        l0 = (pages[0].get("left") or "")
        r0 = (pages[0].get("right") or "")
        print("📝 1페이지 미리보기:")
        print("   [LEFT ]", (l0[:5000] + "...") if len(l0) > 200 else l0)
        print("   [RIGHT]", (r0[:5000] + "...") if len(r0) > 200 else r0)

def main():
    parser = argparse.ArgumentParser(description="2단 PDF 열 기반 파싱 테스트")
    parser.add_argument("pdf", nargs="?", default="1. 2024년3회_정보처리기사필기기출문제_cut.pdf", help="테스트할 PDF 파일 경로")
    parser.add_argument("--mode", choices=["all", "cols"], default="all",
                        help="all: 전체 테스트 / cols: 컬럼 추출만")
    args = parser.parse_args()

    pdf_path = args.pdf
    if args.mode == "cols":
        run_columns_only(pdf_path)
    else:
        run_quick_all(pdf_path)

if __name__ == "__main__":
    print("🚀 2단 PDF 파싱 테스트 도구 (열 기반)")
    print(" - 헤더/번호 분할 미사용, 열 단위 LLM 추출만 검증")
    print(" - OPENAI_API_KEY=REDACTED 환경변수 필요(모델에 따라)")
    main()
