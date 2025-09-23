"""
Unified PDF Preprocessor (robust import)
---------------------------------------
요구사항
1) PDF가 1열/2열인지 자동 판별
2) 1열이면 Docling 파이프라인으로 텍스트 파싱
3) 2열이면 pdfplumber 파이프라인으로 텍스트 파싱
4) [{"question": str, "options": [str, ...]}] 형태로 반환

이 버전은 경로 의존성을 제거했습니다.
- 환경변수/여러 후보 경로/모듈 임포트 순으로 `pdf_preprocessor.py`와 `pdf_preprocessor_ai.py`를 찾습니다.
- 기본 후보:
  - (동일 폴더) teacher/pdf_preprocessor.py
  - (프로젝트 루트) ./teacher/pdf_preprocessor.py, ./pdf_preprocessor.py
  - 모듈 임포트: teacher.pdf_preprocessor, pdf_preprocessor
- 필요 시 환경변수로 명시:
  - PDF_PREPROCESSOR_DOC_PATH
  - PDF_PREPROCESSOR_AI_PATH

사용 예시
---------
from unified_pdf_preprocessor import UnifiedPDFPreprocessor
pre = UnifiedPDFPreprocessor()
problems = pre.extract("your_pdf_file.pdf")
print(problems[0]["question"], problems[0]["question"])    
"""

from __future__ import annotations
import importlib
import importlib.util
import os
from typing import List, Dict, Optional

# ---------------
# 안전한 동적 import
# ---------------

def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈 로딩 실패: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _try_import_module(names: List[str]) -> Optional[object]:
    """여러 모듈 이름 후보로 importlib.import_module 시도."""
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# ---------------
# 1/2열 판별 (pdfplumber 휴리스틱)
# ---------------

def _is_two_column_pdf(pdf_path: str, sample_pages: int = 3) -> bool:
    """간단한 휴리스틱으로 2열 여부 판정.
    규칙(페이지별 스코어 평균):
    - 페이지 중앙(mid_x)을 기준으로 좌/우 텍스트 비율이 모두 0.3 이상이면 2열 점수 1, 아니면 0
    - sample_pages(앞쪽 페이지) 평균>=0.5면 2열로 간주
    """
    try:
        import pdfplumber  # type: ignore
    except Exception:
        # pdfplumber가 없으면 판별 불가 → 1열로 가정(=Docling 사용)
        return False

    try:
        with pdfplumber.open(pdf_path) as pdf:
            n = len(pdf.pages)
            if n == 0:
                return False
            pages_to_check = min(sample_pages, n)
            score_sum = 0.0
            for i in range(pages_to_check):
                page = pdf.pages[i]
                words = page.extract_words(use_text_flow=True) or []
                if not words:
                    continue
                width = float(page.width or 0) or 0.0
                if width <= 0:
                    continue
                mid_x = width / 2.0
                left_cnt = sum(1 for w in words if float(w.get("x0", 0)) < mid_x)
                right_cnt = sum(1 for w in words if float(w.get("x0", 0)) >= mid_x)
                total = left_cnt + right_cnt
                if total == 0:
                    continue
                left_ratio = left_cnt / total
                right_ratio = right_cnt / total
                page_score = 1.0 if (left_ratio >= 0.3 and right_ratio >= 0.3) else 0.0
                score_sum += page_score
            if pages_to_check == 0:
                return False
            avg = score_sum / pages_to_check
            return avg >= 0.5
    except Exception:
        # 판정 중 에러가 나면 보수적으로 1열로 간주
        return False


# ---------------
# Unified Wrapper
# ---------------

class UnifiedPDFPreprocessor:
    def __init__(self,
                 docling_impl_path: Optional[str] = None,
                 plumber_impl_path: Optional[str] = None) -> None:
        """두 구현을 견고하게 로드합니다.
        1) 명시 경로(인자) → 2) 환경변수 → 3) 후보 경로 → 4) 모듈 import 순서.
        """
        here = os.path.abspath(os.path.dirname(__file__))
        cwd = os.path.abspath(os.getcwd())

        # ----- Docling 구현 찾기 -----
        env_doc = os.getenv("PDF_PREPROCESSOR_DOC_PATH")
        doc_candidates = [
            docling_impl_path,
            env_doc,
            os.path.join(here, "pdf_preprocessor.py"),
            os.path.join(cwd, "teacher", "pdf_preprocessor.py"),
            os.path.join(cwd, "pdf_preprocessor.py"),
        ]
        doc_path = _first_existing([p for p in doc_candidates if p])
        if doc_path:
            self._docling_mod = _import_from_path("pdf_preprocessor_docling", doc_path)
        else:
            mod = _try_import_module(["teacher.pdf_preprocessor", "pdf_preprocessor"])  # type: ignore
            if mod is None:
                tried = "\n - " + "\n - ".join([str(p) for p in doc_candidates if p])
                raise FileNotFoundError(
                    "Docling 구현(pdf_preprocessor.py)을 찾을 수 없습니다. 시도한 위치:" + tried +
                    "\n또는 모듈 import(teacher.pdf_preprocessor / pdf_preprocessor)도 실패했습니다."
                )
            self._docling_mod = mod

        # ----- pdfplumber 구현 찾기 -----
        env_ai = os.getenv("PDF_PREPROCESSOR_AI_PATH")
        ai_candidates = [
            plumber_impl_path,
            env_ai,
            os.path.join(here, "pdf_preprocessor_ai.py"),
            os.path.join(cwd, "teacher", "pdf_preprocessor_ai.py"),
            os.path.join(cwd, "pdf_preprocessor_ai.py"),
        ]
        ai_path = _first_existing([p for p in ai_candidates if p])
        if ai_path:
            self._plumber_mod = _import_from_path("pdf_preprocessor_plumber", ai_path)
        else:
            mod = _try_import_module(["teacher.pdf_preprocessor_ai", "pdf_preprocessor_ai"])  # type: ignore
            if mod is None:
                tried = "\n - " + "\n - ".join([str(p) for p in ai_candidates if p])
                raise FileNotFoundError(
                    "pdfplumber 구현(pdf_preprocessor_ai.py)을 찾을 수 없습니다. 시도한 위치:" + tried +
                    "\n또는 모듈 import(teacher.pdf_preprocessor_ai / pdf_preprocessor_ai)도 실패했습니다."
                )
            self._plumber_mod = mod

        # 각 구현의 클래스 핸들러 준비 (두 파일 모두 PDFPreprocessor 클래스를 가진다고 가정)
        try:
            self._docling_pre = getattr(self._docling_mod, "PDFPreprocessor")()  # type: ignore[attr-defined]
        except Exception as e:
            raise ImportError(f"Docling 구현에서 PDFPreprocessor 클래스를 인스턴스화할 수 없습니다: {e}")
        try:
            self._plumber_pre = getattr(self._plumber_mod, "PDFPreprocessor")()  # type: ignore[attr-defined]
        except Exception as e:
            raise ImportError(f"pdfplumber 구현에서 PDFPreprocessor 클래스를 인스턴스화할 수 없습니다: {e}")

    def extract(self, pdf_path: str) -> List[Dict]:
        """PDF에서 문제/보기를 추출하여 리스트로 반환합니다.
        반환 스키마: [{"question": str, "options": [str, ...]}, ...]
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        # 1) 1열/2열 판별
        two_col = _is_two_column_pdf(pdf_path)

        # 2) 분기 실행 (두 구현 모두 파일 리스트 입력 인터페이스 가정)
        if two_col:
            problems = self._plumber_pre.extract_problems_with_pdfplumber([pdf_path])
        else:
            problems = self._docling_pre.extract_problems_from_pdf([pdf_path])

        # 3) 후처리: 스키마 정규화 (question/options만 보장)
        normalized: List[Dict] = []
        for p in problems or []:
            q = str(p.get("question", "")).strip()
            opts = p.get("options", [])
            if not isinstance(opts, list):
                opts = []
            opts = [str(o).strip() for o in opts if str(o).strip()]
            if q and opts:
                normalized.append({"question": q, "options": opts})
        return normalized


# ---------------
# CLI 테스트
# ---------------
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="입력 PDF 경로")
    ap.add_argument("--max", type=int, default=5, help="앞에서부터 N개만 미리보기")
    args = ap.parse_args()

    pre = UnifiedPDFPreprocessor()
    items = pre.extract(args.pdf)

    print(f"\n🎯 총 {len(items)}개 문항 추출")
    for i, it in enumerate(items[: max(0, args.max) ], 1):
        print(f"\n[{i}] {it['question']}")
        for j, opt in enumerate(it.get('options', []), 1):
            print(f"  - {j}) {opt}")

    # 전체를 JSON으로 저장
    out_json = os.path.splitext(os.path.basename(args.pdf))[0] + "_qa.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_json}")
