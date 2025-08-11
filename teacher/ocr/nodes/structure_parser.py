# structure_parser.py
import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import easyocr

from image_loader import convert_pdf_to_images, load_image

# ================== EasyOCR 초기화 (GPU 사용) ==================
# GPU가 없으면 내부적으로 CPU로 폴백됩니다.
reader = easyocr.Reader(['ko', 'en'], gpu=True)

# ================== 정규식/상수 ==================
_q_header_re = re.compile(r"^\s*(\d{1,3})(?:[.)]|번)?(\s|$)")  # 문항 헤더
HEADER_PATTERNS = [
    r"^정보처리기사.*$",
    r"^.*기출문제.*$",
    r"^제\d+과목.*$",
    r"^\d+회.*$",
]
CIRCLED = {str(i): chr(0x2460 + (i - 1)) for i in range(1, 11)}  # 1->① ... 10->⑩

# ================== OCR with boxes (EasyOCR) ==================
def ocr_with_boxes(image: Image.Image) -> List[Dict[str, Any]]:
    """
    EasyOCR 결과(detail=1)를 줄(라인) 단위 dict 리스트로 변환.
    return: [{text, conf, bbox[xmin,ymin,xmax,ymax], cx, cy}, ...]
    """
    arr = np.array(image)
    results = reader.readtext(arr, detail=1, paragraph=False)

    lines: List[Dict[str, Any]] = []
    for bbox, txt, conf in results:
        txt = str(txt).strip()
        if not txt:
            continue
        # 가벼운 정규화
        txt = (txt.replace("\t", " ")
                  .replace("：", ":").replace("﹕", ":")
                  .replace("（", "(").replace("）", ")"))
        txt = re.sub(r"[ ]{2,}", " ", txt)

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min, x_max = float(min(xs)), float(max(xs))
        y_min, y_max = float(min(ys)), float(max(ys))
        lines.append({
            "text": txt,
            "conf": float(conf),
            "bbox": [x_min, y_min, x_max, y_max],
            "cx": (x_min + x_max) / 2.0,
            "cy": (y_min + y_max) / 2.0,
        })

    # 위→아래, 좌→우 정렬
    lines.sort(key=lambda r: (round(r["cy"]/8), r["cx"]))
    return lines

# ================== 문항 블록 ==================
def is_q_header(text: str) -> bool:
    return bool(_q_header_re.match(text))

def extract_qno(text: str) -> str:
    m = _q_header_re.match(text)
    return m.group(1) if m else ""

def split_question_blocks(ocr_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    OCR 라인들을 문항 블록으로 분할.
    반환: [{qno, lines, y_min, y_max}, ...]
    """
    blocks: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []
    qno: Optional[str] = None

    for r in ocr_lines:
        t = r["text"]
        if is_q_header(t):
            if cur:
                y_min = min(l["bbox"][1] for l in cur)
                y_max = max(l["bbox"][3] for l in cur)
                blocks.append({"qno": qno or "", "lines": cur, "y_min": y_min, "y_max": y_max})
            cur = [r]
            qno = extract_qno(t)
        else:
            if cur:
                cur.append(r)

    if cur:
        y_min = min(l["bbox"][1] for l in cur)
        y_max = max(l["bbox"][3] for l in cur)
        blocks.append({"qno": qno or "", "lines": cur, "y_min": y_min, "y_max": y_max})

    # 머리말 제거
    filtered = []
    for b in blocks:
        kept = [l for l in b["lines"] if not any(re.match(p, l["text"]) for p in HEADER_PATTERNS)]
        if kept:
            b["lines"] = kept
            filtered.append(b)
    return filtered

# ================== 행 군집/유틸 ==================
def estimate_line_height(lines: List[Dict[str, Any]]) -> int:
    hs = [l["bbox"][3] - l["bbox"][1] for l in lines]
    if not hs:
        return 14
    h = float(np.median(hs))
    return int(max(12, min(24, round(h))))  # 12~24 범위로 클램프

def group_lines_into_rows(lines: List[Dict[str, Any]], y_tol: int | None = None) -> List[List[Dict[str, Any]]]:
    # y_tol 자동 추정
    if y_tol is None:
        y_tol = estimate_line_height(lines)
    rows: List[List[Dict[str, Any]]] = []
    for r in sorted(lines, key=lambda x: x["cy"]):
        placed = False
        for row in rows:
            cy_mean = sum(li["cy"] for li in row) / len(row)
            if abs(r["cy"] - cy_mean) <= y_tol:
                row.append(r); placed = True; break
        if not placed:
            rows.append([r])
    return rows

def row_to_text(row: List[Dict[str, Any]]) -> str:
    row_sorted = sorted(row, key=lambda x: x["cx"])
    s = " ".join(r["text"] for r in row_sorted)
    return re.sub(r"\s+", " ", s).strip()

def normalize_choice_marker(text: str) -> Tuple[bool, str, str]:
    """
    행 앞의 마커 표준화: ①~⑩ / (1) / 1) / 1. / 1 :
    반환: (is_choice, norm_marker, body)
    """
    s = text.lstrip()

    # ①~⑩
    m = re.match(r"^([①-⑩])\s+(.*)$", s)
    if m:
        return True, m.group(1), m.group(2).strip()

    # (1) / 1) / 1. / 1 : / 1-
    m = re.match(r"^[(]?\s*([1-9]|10)\s*[\).:\-]?\s+(.*)$", s)
    if m:
        num = m.group(1)
        body = m.group(2).strip()
        circ = CIRCLED.get(num, num)
        return True, circ, body

    # ①ABC (공백 없는 변형)
    m = re.match(r"^([①-⑩])(.*)$", s)
    if m:
        return True, m.group(1), m.group(2).strip()

    # 행 앞 잡음 제거
    s = re.sub(r"^[@>\-•\u2022]\s*", "", s)
    return False, "", s.strip()

def split_row_into_cells(row: List[Dict[str, Any]]) -> List[str]:
    """
    한 행을 x 간격으로 2~3칸 자동 분할(두 단/표 형태 보기를 위해).
    """
    toks = sorted(row, key=lambda x: x["cx"])
    if not toks:
        return []
    gaps = [b["cx"] - a["cx"] for a, b in zip(toks, toks[1:])]
    if not gaps:
        return [row_to_text(row)]
    median_gap = float(np.median(gaps))
    cells = []
    cur = [toks[0]]
    for a, b, g in zip(toks, toks[1:], gaps):
        if g > max(40.0, 2.0 * median_gap):  # 절대/상대 기준
            cells.append(cur)
            cur = [b]
        else:
            cur.append(b)
    cells.append(cur)

    out = []
    for c in cells:
        txt = row_to_text(c)
        if txt:
            out.append(txt)
    return out

# ================== 개선: 다음 문항 시작 y 찾기 ==================
def find_next_question_y(lines: List[Dict[str, Any]], start_y: float) -> Optional[float]:
    """
    start_y 이후에서 '숫자.)'/'숫자)'/'숫자번' 패턴을 가진 라인의 최상단 y를 찾아 반환.
    없으면 None.
    """
    cand = []
    for r in lines:
        if r["cy"] <= start_y:
            continue
        t = r["text"]
        if re.match(r"^\s*\d{1,3}(?:[.)]|번)\b", t):
            cand.append(r["bbox"][1])  # top y
    return min(cand) if cand else None

# ================== 개선: 토큰 x-분포로 2컬럼 분할 ==================
def split_tokens_into_columns(tokens: List[Dict[str, Any]], bins: int = 48) -> tuple[list, list]:
    if not tokens:
        return tokens, []
    xs = [t["cx"] for t in tokens]
    x_min, x_max = min(xs), max(xs)
    if x_max - x_min < 60:
        return tokens, []
    hist, edges = np.histogram(xs, bins=bins, range=(x_min, x_max))
    mid = hist[1:-1]
    if len(mid) == 0:
        return tokens, []
    idx = int(np.argmin(mid)) + 1
    cut = (edges[idx] + edges[idx+1]) / 2.0
    left = [t for t in tokens if t["cx"] <= cut]
    right = [t for t in tokens if t["cx"] > cut]
    # 좌우 불균형이 심하면 실패로 간주
    total = len(tokens)
    if len(left) < 2 or len(right) < 2 or min(len(left), len(right)) < total * 0.2:
        return tokens, []
    left.sort(key=lambda r: (round(r["cy"]/8), r["cx"]))
    right.sort(key=lambda r: (round(r["cy"]/8), r["cx"]))
    return left, right

# ================== 개선: 파편 병합/중복 제거 ==================
def merge_fragments(items: List[str], min_len=12) -> List[str]:
    """짧은 파편(문자수 < min_len)을 앞 선택지와 자동 병합"""
    out = []
    for s in items:
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue
        if out and len(s) < min_len:
            out[-1] = (out[-1] + " " + s).strip()
        else:
            out.append(s)
    return out

def dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set(); out = []
    for s in items:
        key = re.sub(r"\s+", " ", s).strip()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

# ================== 문제/보기 추출 ==================
def extract_question_text_from_block(lines: List[Dict[str, Any]], qno: Optional[str] = None) -> str:
    full = " ".join(l["text"] for l in lines)
    # 맨 앞 문항번호 제거
    full = re.sub(r"^\s*\d{1,3}(?:[.)]|번)?\s*", "", full)
    # 첫 '?'까지
    qpos = full.find("?")
    if qpos != -1:
        full = full[:qpos + 1]
    full = re.sub(r"\s+", " ", full).strip()
    return f"{qno}. {full}" if qno and full else full

def extract_choices_from_block(lines: List[Dict[str, Any]]) -> List[str]:
    if not lines:
        return []

    # 1) 문제 끝 y
    q_end_y = None
    for r in lines:
        if "?" in r["text"]:
            q_end_y = r["bbox"][3]; break
    if q_end_y is None:
        ys = [r["cy"] for r in lines]
        q_end_y = float(np.percentile(ys, 40))

    # 2) 다음 문항 시작 y
    next_q_y = find_next_question_y(lines, q_end_y)

    # 3) 후보 영역 필터 (넓게 잡고 → 이후 폴백에서 더 넓힘)
    def in_band(r, low, high=None):
        y_top, y_bot = r["bbox"][1], r["bbox"][3]
        if high is not None:
            return y_top >= low - 2 and y_bot <= high - 1
        return y_top >= low - 2

    cand = [r for r in lines if in_band(r, q_end_y, next_q_y)]
    if not cand:
        cand = [r for r in lines if in_band(r, q_end_y, None)]  # 아래쪽 전체 폴백
        if not cand:
            return []

    # 4) 빠른 경로: 마커 기반(줄 연결) → 충분하면 반환
    rows = group_lines_into_rows(cand)  # y_tol 자동
    choices: list[str] = []
    carry = None
    for row in rows:
        rt = row_to_text(row)
        is_choice, marker, body = normalize_choice_marker(rt)
        if is_choice:
            if carry is not None:
                choices.append(f"{carry[0]} {carry[1]}".strip())
            carry = (marker, body)
        else:
            if carry is not None and body:
                carry = (carry[0], (carry[1] + " " + body).strip())
        if len(choices) >= 4:
            break
    if carry is not None and len(choices) < 4:
        choices.append(f"{carry[0]} {carry[1]}".strip())

    # 마커로 3~4개 확보되면 정리해서 반환
    if len(choices) >= 3:
        choices = merge_fragments(choices)
        choices = dedup_preserve_order(choices)
        cleaned = []
        for i, c in enumerate(choices[:4], start=1):
            c = re.sub(r"\s*:\s*", ": ", c)
            c = re.sub(r"\s+", " ", c).strip()
            cleaned.append(f"{CIRCLED[str(i)]} {c}")
        return cleaned

    # 5) x-분포 2컬럼 분할 → 각 컬럼에서 위→아래 한 줄=한 보기 후보
    left, right = split_tokens_into_columns(cand)
    tmp = []
    if right:
        for col in (left, right):
            col_rows = group_lines_into_rows(col)
            for row in col_rows:
                rt = row_to_text(row)
                ok, m, b = normalize_choice_marker(rt)
                tmp.append(b if ok else rt)
    else:
        # 분할 실패: 행을 칸(cell)으로 더 쪼개서 후보 늘리기
        for row in rows:
            for cell in split_row_into_cells(row):
                ok, m, b = normalize_choice_marker(cell)
                tmp.append(b if ok else cell)

    # 파편 병합/중복 제거/잡음 제거
    tmp = [t for t in tmp if len(t) > 1 and not re.fullmatch(r"[>\-{}[\]]+", t)]
    tmp = merge_fragments(tmp)
    tmp = dedup_preserve_order(tmp)

    if not choices:
        choices = tmp[:4]
    else:
        need = 4 - len(choices)
        choices.extend(tmp[:need])

    # 여전히 모자라면 후보 영역 자체에서 남는 줄 채우기
    if len(choices) < 4:
        leftovers = [row_to_text(r) for r in rows if len(row_to_text(r)) > 1]
        leftovers = merge_fragments(dedup_preserve_order(leftovers))
        need = 4 - len(choices)
        if need > 0:
            choices.extend([s for s in leftovers if s not in choices][:need])

    # 최종 정리 + ①~④
    choices = merge_fragments(choices)
    choices = dedup_preserve_order(choices)
    cleaned = []
    for i, c in enumerate(choices[:4], start=1):
        c = re.sub(r"\s*:\s*", ": ", c)
        c = re.sub(r"\s+", " ", c).strip()
        cleaned.append(f"{CIRCLED[str(i)]} {c}")
    return cleaned


# ================== 페이지 → 문항 ==================
def parse_question_blocks_from_page(image: Image.Image) -> List[Dict[str, Any]]:
    ocr_lines = ocr_with_boxes(image)
    blocks = split_question_blocks(ocr_lines)

    results = []
    for b in blocks:
        question = extract_question_text_from_block(b["lines"], b["qno"])
        choices = extract_choices_from_block(b["lines"])
        results.append({
            "qno": b["qno"],
            "question": question,
            "choices": choices,
        })
    return results

# ================== PDF 전체/CLI ==================
def parse_structure_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    image_paths = convert_pdf_to_images(pdf_path)
    all_results = []
    for img_path in image_paths:
        image = load_image(img_path)
        all_results.extend({"page": img_path, **r} for r in parse_question_blocks_from_page(image))

    def _key(x):
        try:
            return int(x.get("qno") or 0)
        except:
            return 0

    all_results.sort(key=lambda x: (_key(x), x["page"]))
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--max-pages", type=int, default=0)
    args = parser.parse_args()

    # 페이지 변환
    image_paths = convert_pdf_to_images(args.pdf)
    if args.max_pages and args.max_pages > 0:
        image_paths = image_paths[:args.max_pages]

    for i, p in enumerate(image_paths, 1):
        print(f"[PAGE {i}/{len(image_paths)}] {p}")
        page_results = parse_question_blocks_from_page(load_image(p))
        for r in page_results:
            print(json.dumps(r, ensure_ascii=False, indent=2))
