"""
텍스트(.txt) 파일들을 섹션/단락 단위로 분할 후 JSONL로 변환합니다.

기본 동작:
- 입력 파일 경로(복수) 또는 디렉터리+패턴으로 .txt 파일을 수집합니다.
- 간단한 헤더(제목) 패턴과 공백 라인 단위로 섹션을 구분합니다.
- 섹션을 지정한 글자 수 기준으로 겹침(chunk_overlap) 포함해 청크로 나눕니다.
- JSONL로 출력합니다. 각 레코드에는 id, title, source, chunk_index, text, metadata가 포함됩니다.

사용 예시:
  uv run python teacher/agents/retrieve/convert_texts_to_jsonl.py \
    --inputs teacher/agents/retrieve/data/에듀윌/데이터베이스 구축.txt \
             teacher/agents/retrieve/data/에듀윌/소프트웨어 개발.txt \
             teacher/agents/retrieve/data/에듀윌/소프트웨어 설계.txt \
             teacher/agents/retrieve/data/에듀윌/정보시스템 구축 관리.txt \
             teacher/agents/retrieve/data/에듀윌/프로그래밍 언어 활용.txt \
    --output_dir teacher/agents/retrieve/data/에듀윌/jsonl \
    --chunk_size 1000 --chunk_overlap 100

또는 폴더 패턴으로 수집:
  uv run python teacher/agents/retrieve/convert_texts_to_jsonl.py \
    --input_dir teacher/agents/retrieve/data/에듀윌 --glob "*.txt" \
    --output_dir teacher/agents/retrieve/data/에듀윌/jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def discover_input_files(
    inputs: Sequence[str], input_dir: Optional[str], glob_pattern: Optional[str]
) -> List[Path]:
    file_paths: List[Path] = []

    # 직접 지정된 inputs 우선
    for item in inputs:
        p = Path(item)
        if p.is_file():
            file_paths.append(p)
        elif p.is_dir():
            file_paths.extend(sorted(p.rglob("*.txt")))
        else:
            # 와일드카드 글롭 허용
            base = Path(".")
            file_paths.extend(sorted(base.glob(item)))

    # 디렉터리 + 글롭 패턴
    if input_dir:
        base = Path(input_dir)
        if not base.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        pattern = glob_pattern or "*.txt"
        file_paths.extend(sorted(base.rglob(pattern)))

    # 중복 제거 및 .txt만 유지
    unique: List[Path] = []
    seen = set()
    for p in file_paths:
        if p.suffix.lower() == ".txt":
            if p.resolve() not in seen:
                seen.add(p.resolve())
                unique.append(p)
    return unique


def read_text(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding, errors="ignore")


def normalize_text(text: str) -> str:
    # 줄 끝 공백 제거, 윈도우 개행 정규화
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # 가끔 붙는 인라인 라인넘버(L123:) 제거 시도
    text = re.sub(r"^L\d+:\s?", "", text, flags=re.MULTILINE)
    return text


HEADER_PATTERNS: List[re.Pattern[str]] = [
    # 1. 숫자 기반 헤더: "1.", "1)" , "1. 제목" 등
    re.compile(r"^\s*\d+(?:[.)]|(?:\s+))"),
    # 2. 다단계 번호: "1.2", "2.3.4" 등
    re.compile(r"^\s*\d+(?:\.\d+)+\s+"),
    # 3. 한글 카테고리 키워드로 시작하는 큰 단락들(대략적 휴리스틱)
    re.compile(r"^\s*(개요|정의|특징|종류|구성|목적|개념|장점|단점|방법|절차)\s*[:：]\s*"),
]


def is_header_line(line: str) -> bool:
    if not line:
        return False
    # 너무 짧은 라인이나 구분선 라인은 헤더로 보지 않음
    if set(line) <= {"-", "=", "_", "*", " ", "\t"}:
        return False
    for pat in HEADER_PATTERNS:
        if pat.match(line):
            return True
    # 콜론으로 끝나는 굵은 문장도 헤더 후보
    if len(line) <= 60 and line.strip().endswith(":"):
        return True
    return False


def split_into_sections(text: str) -> List[Tuple[str, List[str]]]:
    """
    텍스트를 섹션 단위로 분리합니다.
    반환: [(section_title, section_lines)]
    """
    lines = text.split("\n")
    sections: List[Tuple[str, List[str]]] = []

    current_title: str = ""
    current_lines: List[str] = []

    def flush():
        nonlocal current_title, current_lines
        if current_lines:
            # 내용 앞/뒤 공백 라인 제거
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_title, content.split("\n")))
        current_title = ""
        current_lines = []

    for raw in lines:
        line = raw.strip()
        if is_header_line(line):
            # 이전 섹션 마감
            flush()
            current_title = line
        else:
            current_lines.append(raw)

    flush()

    # 헤더를 전혀 찾지 못한 경우: 하나의 섹션으로 전체 반환
    if not sections:
        all_lines = [ln for ln in lines if ln.strip()]
        if all_lines:
            sections = [("", all_lines)]
    return sections


def split_into_paragraphs(text: str) -> List[List[str]]:
    """
    빈 줄(연속 포함) 기준으로 단락 분리. 각 단락은 라인 리스트로 반환.
    """
    # 정규화된 텍스트를 공백 라인 1개 이상으로 분할
    paragraphs = re.split(r"\n\s*\n+", text)
    result: List[List[str]] = []
    for para in paragraphs:
        p = para.strip()
        if not p:
            continue
        lines = [ln for ln in p.split("\n")]
        # 완전히 공백인 라인만 있는 경우 제외
        if any(ln.strip() for ln in lines):
            result.append(lines)
    return result


def chunk_text(
    lines: Sequence[str], chunk_size: int, chunk_overlap: int
) -> List[str]:
    """
    라인들을 하나의 문자열로 합친 후 글자 수 기준 청크 분할.
    """
    text = "\n".join(line for line in lines).strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        # 다음 시작은 오버랩을 고려하여 이동
        start = max(0, end - chunk_overlap)
        if start >= n:
            break
        # 무한 루프 방지
        if len(chunks) > 1 and chunks[-1] == chunks[-2]:
            break
    return chunks


def to_records_for_file(
    path: Path,
    encoding: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[dict]:
    raw = read_text(path, encoding)
    normalized = normalize_text(raw)
    # sections는 기본 섹션 모드에서만 사용
    sections = split_into_sections(normalized)

    records: List[dict] = []
    file_title = path.stem
    source_rel = str(path.as_posix())

    chunk_counter = 0
    for section_title, section_lines in sections:
        chunks = chunk_text(section_lines, chunk_size, chunk_overlap)
        for chunk in chunks:
            record = {
                "id": str(uuid.uuid4()),
                "title": file_title,
                "source": source_rel,
                "chunk_index": chunk_counter,
                "text": chunk,
                "metadata": {
                    "section_title": section_title,
                },
            }
            records.append(record)
            chunk_counter += 1
    return records


def to_records_for_file_paragraph(
    path: Path,
    encoding: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[dict]:
    raw = read_text(path, encoding)
    normalized = normalize_text(raw)
    paragraphs = split_into_paragraphs(normalized)

    records: List[dict] = []
    file_title = path.stem
    source_rel = str(path.as_posix())

    chunk_counter = 0
    for para_idx, para_lines in enumerate(paragraphs):
        # 단락 모드에서도 너무 긴 단락은 안전하게 분할
        chunks = chunk_text(para_lines, chunk_size if chunk_size > 0 else 100000000, chunk_overlap)
        for chunk in chunks:
            first_line = para_lines[0].strip() if para_lines else ""
            record = {
                "id": str(uuid.uuid4()),
                "title": file_title,
                "source": source_rel,
                "chunk_index": chunk_counter,
                "text": chunk,
                "metadata": {
                    "paragraph_index": para_idx,
                    "section_title": first_line[:120],
                },
            }
            records.append(record)
            chunk_counter += 1
    return records


def write_jsonl(records: Iterable[dict], output_path: Path, ensure_ascii: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=ensure_ascii, separators=(",", ":")))
            f.write("\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TXT -> JSONL 변환기")
    p.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="입력 파일 경로 또는 글롭 패턴(복수). 디렉터리를 주면 재귀적으로 .txt 수집",
    )
    p.add_argument(
        "--input_dir",
        default=None,
        help="기본 수집 디렉터리(선택). --glob와 함께 사용",
    )
    p.add_argument(
        "--glob",
        default=None,
        help="--input_dir 기준의 글롭 패턴(기본 *.txt). 예: *.txt",
    )
    p.add_argument(
        "--output",
        required=False,
        help="단일 JSONL 파일 경로(모든 입력을 합쳐서 저장)",
    )
    p.add_argument(
        "--output_dir",
        required=False,
        help="입력 파일별로 개별 JSONL을 저장할 디렉터리",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="청크 글자 수(기본 1000). 0이면 통청크",
    )
    p.add_argument(
        "--chunk_overlap",
        type=int,
        default=100,
        help="청크 간 겹침 글자 수(기본 100)",
    )
    p.add_argument(
        "--split_mode",
        choices=["section", "paragraph"],
        default="paragraph",
        help="분할 모드: section(헤더 기반) | paragraph(빈 줄 단락 기반)",
    )
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="입력 파일 인코딩(기본 utf-8)",
    )
    p.add_argument(
        "--no_ensure_ascii",
        action="store_true",
        help="JSON 직렬화 시 ensure_ascii=False 설정(유니코드 그대로 저장)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    files = discover_input_files(args.inputs, args.input_dir, args.glob)
    if not files:
        print("입력 .txt 파일을 찾지 못했습니다. --inputs 또는 --input_dir/--glob를 확인하세요.", file=sys.stderr)
        return 2

    if not args.output and not args.output_dir:
        print("--output 또는 --output_dir 중 하나는 반드시 지정해야 합니다.", file=sys.stderr)
        return 2

    ensure_ascii = not args.no_ensure_ascii

    # 파일별 저장 모드
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for fp in files:
            if args.split_mode == "paragraph":
                recs = to_records_for_file_paragraph(
                    fp,
                    encoding=args.encoding,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
            else:
                recs = to_records_for_file(
                    fp,
                    encoding=args.encoding,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
            out_file = out_dir / f"{fp.stem}.jsonl"
            write_jsonl(recs, out_file, ensure_ascii=ensure_ascii)
            print(f"작성: {len(recs)}개 → {out_file}")
            total += len(recs)
        print(f"완료: 총 {total}개 레코드, {len(files)}개 파일로 저장 → {out_dir}")
        return 0

    # 단일 파일로 합치기
    all_records: List[dict] = []
    for fp in files:
        if args.split_mode == "paragraph":
            file_records = to_records_for_file_paragraph(
                fp,
                encoding=args.encoding,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        else:
            file_records = to_records_for_file(
                fp,
                encoding=args.encoding,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        all_records.extend(file_records)

    out_path = Path(args.output)
    write_jsonl(all_records, out_path, ensure_ascii=ensure_ascii)
    print(f"완료: {len(all_records)}개 레코드 → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


