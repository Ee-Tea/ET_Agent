# -*- coding: utf-8 -*-
"""
정보처리기사 개념서 PDF -> Docling -> 과목별 JSON 저장
실행:  python run

- 단일 PDF(INPUT_PATH)를 Docling으로 추출
- 12페이지 이후에 나오는 과목 헤더(소프트웨어 설계/개발/데이터베이스 구축/프로그래밍 언어 활용/정보시스템 구축관리) 기준으로
  본문을 과목별로 분리
- 의미 필터링 없음(이미지 토큰·표·불릿 모두 유지), 문단/표 단위로 쪼개어 items화
- 각 과목별로 1개 JSON 파일 저장 (스키마 유지)
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import re
import json
import warnings
import tempfile
from pathlib import Path
from typing import List, Dict

# --------------------
# 설정
# --------------------
INPUT_PATH      = os.getenv("PDF_INPUT_PATH", "./teacher/eduwill.pdf")   # 입력 PDF 1개
OUTPUT_DIR      = os.getenv("PDF_OUTPUT_DIR", "./teacher/concepts")      # 출력 디렉토리
TITLE_MAX       = int(os.getenv("PDF_TITLE_MAX", "120"))                 # item_title 길이

# 과목 키워드 (순서 유지!)
SUBJECT_ORDER = [
    "소프트웨어 설계",
    "소프트웨어 개발",
    "데이터베이스 구축",
    "프로그래밍 언어 활용",
    "정보시스템 구축관리",
]

# HF 캐시: 사용자 홈 경로로 지정(권한 문제 회피)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    cache_dir = Path(tempfile.gettempdir()) / "huggingface_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
except Exception:
    pass

# Docling
try:
    from docling.document_converter import DocumentConverter
except ImportError as e:
    raise SystemExit(f"❌ Docling 미설치: {e}\n-> pip install docling")

# --------------------
# 유틸
# --------------------
def norm_spaces(s: str) -> str:
    if not s: return ""
    s = s.replace("\r", "")
    # 하이픈 줄바꿈 보정: 데이-터 → 데이터
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

H1 = re.compile(r"(?m)^\s*#\s+(.+?)\s*$")
H2 = re.compile(r"(?m)^\s*##\s+(.+?)\s*$")
H3 = re.compile(r"(?m)^\s*###\s+(.+?)\s*$")
MD_TABLE_SEP = re.compile(r"(?m)^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,  # 또는 TesseractOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

def docling_convert_to_md(pdf_path: str) -> str:
    ocr_opts = RapidOcrOptions(lang="korean,english")
    pipe_opts = PdfPipelineOptions(
        do_ocr=True,
        force_full_page_ocr=True,
        ocr_options=ocr_opts,
        do_table_structure=True,
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe_opts)}
    )
    res = converter.convert(pdf_path)
    return res.document.export_to_markdown()



def slug(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^0-9A-Za-z가-힣_]+", "", s)
    return s

def pick_title(text: str, limit: int = TITLE_MAX) -> str:
    first_line = (text or "").split("\n",1)[0].strip()
    # 불릿/번호 프리픽스 제거
    first_line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|[가-하]\)\s+|[A-Z]\)\s+|[①-⑩]\s*)", "", first_line)
    title = first_line[:limit]
    return title + ("…" if len(first_line) > limit else "")

# --------------------
# 과목 분리
# --------------------
def find_subject_anchors(md: str) -> List[Dict]:
    """
    마크다운에서 과목별 시작 위치(라인 인덱스)를 찾는다.
    - 헤더(#/##/###)에 과목명이 포함되면 우선
    - 일반 라인에도 과목명이 '포함'되면 앵커로 인정(예: '제1과목 소프트웨어 설계')
    """
    lines = md.split("\n")
    anchors = []
    for idx, ln in enumerate(lines):
        raw = ln.strip()
        header_text = None
        m1 = H1.match(ln); m2 = H2.match(ln); m3 = H3.match(ln)
        if m1: header_text = m1.group(1).strip()
        elif m2: header_text = m2.group(1).strip()
        elif m3: header_text = m3.group(1).strip()
        target_text = header_text if header_text else raw

        for subj in SUBJECT_ORDER:
            if subj in target_text:
                anchors.append({"subject": subj, "line": idx})
                break

    # 중복 제거(같은 과목의 복수 매칭 중 첫 등장만)
    first_seen = {}
    filtered = []
    for a in anchors:
        s = a["subject"]
        if s not in first_seen:
            first_seen[s] = a["line"]
            filtered.append(a)

    # SUBJECT_ORDER 순서대로 정렬
    filtered.sort(key=lambda x: SUBJECT_ORDER.index(x["subject"]))
    return filtered

def split_by_subject(md: str) -> Dict[str, str]:
    """
    과목 시작 앵커를 기준으로 md를 과목별 본문으로 분할한다.
    12페이지 이전 서문은 무시(=첫 과목 앵커 이전은 버림).
    """
    md = norm_spaces(md)
    lines = md.split("\n")
    anchors = find_subject_anchors(md)

    if not anchors:
        # 과목 키워드를 못 찾았으면 전체를 "정보처리기사(통합)"로 저장
        return {"정보처리기사": md}

    # 구간 자르기
    parts: Dict[str, str] = {}
    for i, a in enumerate(anchors):
        subj = a["subject"]
        start = a["line"]
        end = anchors[i+1]["line"] if i+1 < len(anchors) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        parts[subj] = body

    return parts

# --------------------
# 문단/표/불릿 단위 아이템화(의미 필터 X)
# --------------------
def block_items_from_subject_text(text: str) -> List[str]:
    """
    의미 필터링 없이:
    - 빈 줄로 문단 분리
    - 표/불릿/일반 문단 그대로 보존
    - 너무 빈번한 빈 줄 정리만 수행
    """
    text = norm_spaces(text)
    if not text:
        return []
    blocks = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return blocks

# --------------------
# 처리 & 저장
# --------------------
def save_subject_json(subject: str, blocks: List[str], stem: str, out_dir: Path):
    items = []
    for i, blk in enumerate(blocks, 1):
        items.append({
            "subject": subject,
            "item_id": f"{i:03d}",
            "item_title": pick_title(blk),
            "content": blk,
            "chunk_size": len(blk)
        })
    data = {
        "subject": subject,
        "total_items": len(items),
        "items": items
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}__{slug(subject)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 저장: {out_path} (items={len(items)})")

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    pdf = Path(INPUT_PATH)
    if not pdf.exists():
        print(f"❌ PDF 없음: {pdf}")
        return

    print(f"📖 Docling 변환: {pdf}")
    md = docling_convert_to_md(str(pdf))
    if not md.strip():
        print("⚠️ 텍스트를 추출하지 못했습니다.")
        return

    # 과목별 분할
    subject_texts = split_by_subject(md)

    # 각 과목을 블록으로 나눠 저장
    for subj, subj_md in subject_texts.items():
        # 요구: 12페이지부터 과목들이 시작 → 첫 과목 이전은 split_by_subject에서 이미 버림
        blocks = block_items_from_subject_text(subj_md)
        save_subject_json(subj, blocks, pdf.stem, Path(OUTPUT_DIR))

if __name__ == "__main__":
    main()
