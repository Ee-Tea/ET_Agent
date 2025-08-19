# -*- coding: utf-8 -*-
"""
초간단 OCR 유틸 (Tesseract 전용) + MCQ(문항+4보기) 파서

필수 패키지:
  pip install pillow pytesseract

Windows:
  - Tesseract 설치 후 kor.traineddata 설치 (Korean)
  - 예) C:\\Program Files\\Tesseract-OCR\\tesseract.exe
  - 환경변수 TESSERACT_CMD 또는 set_tesseract_cmd()로 경로 지정 가능
"""

from __future__ import annotations
import os
import io
import re
import json
from typing import List, Dict, Optional, Union

from PIL import Image, ImageOps
import pytesseract

ImageLike = Union[str, bytes, bytearray, io.BytesIO, Image.Image]

# 보기 접두 패턴: 1) 2. ③ 가. A) 등
_OPTION_PREFIX = r"(?:\(?[1-4①-④A-Da-d가-라]\)?[)\].]?)"
OPTION_LINE_RE = re.compile(rf"^\s*{_OPTION_PREFIX}\s+")
OPTION_SPLIT_RE = re.compile(rf"(?=\s*{_OPTION_PREFIX}\s+)")  # 한 줄에 보기가 여러 개 섞인 경우

def set_tesseract_cmd(path: Optional[str] = None) -> None:
    """Tesseract 실행 경로 설정 (인자 > env:TESSERACT_CMD)."""
    cmd = path or os.getenv("TESSERACT_CMD") or ""
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd  # type: ignore

def _load_image(img: ImageLike) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (bytes, bytearray, io.BytesIO)):
        return Image.open(io.BytesIO(img) if not isinstance(img, io.BytesIO) else img).convert("RGB")
    return Image.open(img).convert("RGB")

def _simple_preprocess(im: Image.Image, scale: float = 1.5, binarize: bool = True) -> Image.Image:
    """Pillow만 사용한 소박한 전처리: 리사이즈 + 그레이스케일 + 이진화."""
    if scale and scale != 1.0:
        w, h = im.size
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    gray = ImageOps.grayscale(im)
    if binarize:
        # 간단 임계값: 180 전후가 무난
        gray = gray.point(lambda x: 255 if x >= 180 else 0)
    return gray

def ocr_image_to_text(
    image: ImageLike,
    *,
    lang: str = "kor+eng",
    psm: str = "6",
    oem: str = "3",
    tesseract_cmd: Optional[str] = None,
    scale: float = 1.5,
    binarize: bool = True,
) -> str:
    """이미지 → Tesseract OCR → 텍스트(후처리 포함)."""
    set_tesseract_cmd(tesseract_cmd)
    im = _load_image(image)
    im = _simple_preprocess(im, scale=scale, binarize=binarize)

    cfg = f"--oem {oem} --psm {psm}"
    text = pytesseract.image_to_string(im, lang=lang, config=cfg)

    # 가벼운 후처리
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=\w)", "", text)     # 줄끝 하이픈 연결
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _clean_question_prefix(q: str) -> str:
    # "문제 1.", "1)", "1." 등 선행 번호 제거
    q = q.strip()
    q = re.sub(r"^\s*(?:문제\s*)?\d{1,3}\s*[\).:]\s*", "", q)
    return q.strip()

def _strip_option_prefix(opt: str) -> str:
    opt = re.sub(rf"^\s*{_OPTION_PREFIX}\s*", "", opt).strip()
    return opt

def _explode_option_runs(line: str) -> List[str]:
    """한 줄에 ①②③④가 연달아 있는 경우를 분해."""
    line = line.strip()
    parts = OPTION_SPLIT_RE.split(line)
    return [p for p in (x.strip() for x in parts) if p]

def parse_mcq(text: str) -> List[Dict[str, List[str]]]:
    """
    OCR 텍스트에서 '문제질문 + 보기 4개' 묶음을 순차 파싱.
    - 첫 번째 '보기 라인'이 나오기 전까지는 모두 질문으로 누적
    - 보기 4개를 모으면 한 묶음 완성
    - 라인 하나에 여러 보기가 붙은 경우도 분해해서 사용
    """
    lines: List[str] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        # '정답:', '해설:' 같은 라인은 버림(있다면)
        if re.search(r"(정답|해설|답안|풀이)\s*[:：]", raw):
            continue
        # 한 줄에 여러 보기 접두가 있으면 분해
        chunked = _explode_option_runs(raw) if OPTION_SPLIT_RE.search(raw) else [raw]
        lines.extend(chunked)

    results: List[Dict[str, List[str]]] = []
    qbuf: List[str] = []
    opts: List[str] = []

    def flush():
        nonlocal qbuf, opts
        if qbuf and len(opts) == 4:
            q = _clean_question_prefix(" ".join(qbuf))
            results.append({
                "question": q,
                "options": [_strip_option_prefix(o) for o in opts]
            })
        qbuf, opts = [], []

    for ln in lines:
        if OPTION_LINE_RE.match(ln):  # 보기
            opts.append(ln)
            if len(opts) == 4:
                flush()
        else:  # 질문
            # 이전 문항이 완성되지 않았는데 새 질문이 시작되면, 이전 것은 폐기/리셋
            if qbuf and opts and len(opts) < 4:
                qbuf, opts = [], []  # 덜 채워진 묶음 리셋
            qbuf.append(ln)

    # 마지막에도 혹시 완성되었으면 flush
    if qbuf and len(opts) == 4:
        flush()

    return results

def extract_mcq_from_image(
    image: ImageLike,
    *,
    lang: str = "kor+eng",
    psm: str = "6",
    oem: str = "3",
    tesseract_cmd: Optional[str] = None,
    scale: float = 1.5,
    binarize: bool = True,
) -> List[Dict[str, List[str]]]:
    """이미지 한 장에서 바로 MCQ 리스트 추출."""
    text = ocr_image_to_text(
        image, lang=lang, psm=psm, oem=oem,
        tesseract_cmd=tesseract_cmd, scale=scale, binarize=binarize
    )
    return parse_mcq(text)

# 빠른 수동 테스트
if __name__ == "__main__":
    import sys
    img_path = "./teacher/agents/solution/user_problems.png"
    data = extract_mcq_from_image(img_path)
    if "--json" in sys.argv:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        for i, item in enumerate(data, 1):
            print(f"[문항 {i}] {item['question']}")
            for j, opt in enumerate(item["options"], 1):
                print(f"  {j}. {opt}")
