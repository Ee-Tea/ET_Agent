# folder_to_pdf_cli.py
# 폴더 내 파일을 0~ 번호로 선택해서 하나의 PDF로 변환/병합
# 실행:
#   python folder_to_pdf_cli.py [--font "C:/Windows/Fonts/NanumGothic.ttf"]
#   python folder_to_pdf_cli.py --excel-native --fit-width 1 --fit-height 0
# - 지원: .txt, .csv, .xlsx, .png, .jpg, .jpeg (+ .pdf는 병합용)
# - 출력: 같은 폴더에 PDF 생성

import re
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, LongTable, Table, TableStyle, Paragraph, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet

# (선택) PDF 병합: pypdf 설치되어 있으면 사용
try:
    from pypdf import PdfReader, PdfWriter
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

# (선택) Excel 네이티브 PDF 내보내기: Windows + Excel + pywin32 필요
try:
    import win32com.client as win32  # type: ignore
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False

# ===================== 기본 레이아웃 =====================
PAGE_W, PAGE_H = A4
MARGIN_L = 50
MARGIN_R = 50
MARGIN_T = 50
MARGIN_B = 50
LINE_SPACING = 16

DEFAULT_FONT = "Helvetica"
DEFAULT_FONT_SIZE = 11
KOREAN_FALLBACK_FONT_SIZE = 10

SUPPORTED = (".txt", ".csv", ".xlsx", ".png", ".jpg", ".jpeg", ".pdf")

# ===== 전역 옵션(메인에서 설정) =====
EXCEL_NATIVE = False
EXCEL_FIT_WIDTH: int | None = 1
EXCEL_FIT_HEIGHT: int | None = 0
EXCEL_GRIDLINES = False
EXCEL_HEADINGS = False
EXCEL_QUALITY = "standard"  # or "minimum"
TOP_KEEP_ROWS = 0  # 텍스트 변환 모드에서만 사용

# ===== Excel 네이티브(원본 서식 유지) PDF 내보내기 =====
XL_TYPE_PDF = 0
XL_QUALITY_STANDARD = 0
XL_QUALITY_MINIMUM = 1

def export_workbook_to_pdf_native(xlsx_path: Path, out_pdf: Path) -> None:
    """Excel 애플리케이션의 ExportAsFixedFormat으로 통짜 PDF(모든 시트) 생성."""
    if not HAS_WIN32:
        raise RuntimeError("pywin32가 없어 Excel 네이티브 내보내기를 사용할 수 없습니다.")
    excel = win32.DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False
    try:
        wb = excel.Workbooks.Open(str(xlsx_path))
        try:
            for sh in wb.Worksheets:
                ps = sh.PageSetup
                if EXCEL_FIT_WIDTH is not None:
                    ps.FitToPagesWide = EXCEL_FIT_WIDTH
                if EXCEL_FIT_HEIGHT is not None:
                    ps.FitToPagesTall = EXCEL_FIT_HEIGHT
                ps.PrintGridlines = bool(EXCEL_GRIDLINES)
                ps.PrintHeadings = bool(EXCEL_HEADINGS)

            qual = XL_QUALITY_STANDARD if EXCEL_QUALITY == "standard" else XL_QUALITY_MINIMUM
            wb.ExportAsFixedFormat(
                Type=XL_TYPE_PDF,
                Filename=str(out_pdf),
                Quality=qual,
                IncludeDocProperties=True,
                IgnorePrintAreas=False,
                From=None, To=None, OpenAfterPublish=False
            )
        finally:
            wb.Close(SaveChanges=False)
    finally:
        excel.Quit()
        time.sleep(0.3)

# ===================== 폰트 =====================
def register_font_auto() -> tuple[str, int]:
    """Windows 한글 폰트를 자동 등록(가능하면). 실패 시 기본 폰트."""
    candidates = [
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\Arial.ttf",  # 최소 영문 폴백
    ]
    for p in candidates:
        fp = Path(p)
        if fp.exists():
            try:
                name = fp.stem
                pdfmetrics.registerFont(TTFont(name, str(fp)))
                return name, (KOREAN_FALLBACK_FONT_SIZE if ("nanum" in name.lower() or "malgun" in name.lower()) else DEFAULT_FONT_SIZE)
            except Exception:
                continue
    return DEFAULT_FONT, DEFAULT_FONT_SIZE

def register_font_override(ttf: str | None) -> tuple[str, int]:
    """--font로 TTF가 들어오면 우선 적용, 아니면 자동 탐색."""
    if ttf:
        fp = Path(ttf)
        if fp.exists():
            try:
                name = fp.stem
                pdfmetrics.registerFont(TTFont(name, str(fp)))
                return name, KOREAN_FALLBACK_FONT_SIZE
            except Exception as e:
                print(f"[경고] 지정 폰트 등록 실패({e}) → 자동 탐색으로 전환")
    return register_font_auto()

# ===================== 텍스트/이미지 그리기 =====================
def wrap_lines(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
    words = (text or "").replace("\r", "").split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        width = pdfmetrics.stringWidth(test, font_name, font_size)
        if width <= max_width or cur == "":
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    out = []
    for raw_line in "\n".join(lines).split("\n"):
        out.append(raw_line)
    return out

def new_canvas(path: str):
    return canvas.Canvas(path, pagesize=A4)

def draw_text_block(c: canvas.Canvas, text: str, font_name: str, font_size: int):
    usable_w = PAGE_W - MARGIN_L - MARGIN_R
    y = PAGE_H - MARGIN_T
    c.setFont(font_name, font_size)
    for para in (text or "").split("\n"):
        if not para.strip():
            y -= LINE_SPACING
            if y < MARGIN_B:
                c.showPage()
                c.setFont(font_name, font_size)
                y = PAGE_H - MARGIN_T
            continue
        wrapped = wrap_lines(para, font_name, font_size, usable_w)
        for line in wrapped:
            c.drawString(MARGIN_L, y, line)
            y -= LINE_SPACING
            if y < MARGIN_B:
                c.showPage()
                c.setFont(font_name, font_size)
                y = PAGE_H - MARGIN_T

def draw_image(c: canvas.Canvas, img_path: str):
    usable_w = PAGE_W - MARGIN_L - MARGIN_R
    usable_h = PAGE_H - MARGIN_T - MARGIN_B
    with Image.open(img_path) as im:
        w, h = im.size
        ratio = min(usable_w / w, usable_h / h)
        new_w, new_h = w * ratio, h * ratio
        x = MARGIN_L + (usable_w - new_w) / 2
        y = MARGIN_B + (usable_h - new_h) / 2
        c.drawImage(ImageReader(im), x, y, width=new_w, height=new_h)

# ===================== CSV/XLSX 텍스트 변환(백업 경로) =====================
def _clean_table_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    if len(df.columns) > 0 and (df.columns.astype(str).str.startswith("Unnamed").mean() >= 0.5):
        header_idx = None
        for i, row in df.iterrows():
            if row.notna().mean() >= 0.6:
                header_idx = i
                break
        if header_idx is not None:
            df.columns = [str(x).strip() for x in df.iloc[header_idx].fillna("")]
            df = df.iloc[header_idx+1:]
    df = df.fillna("")
    return df

def csv_to_text(path: str, max_rows: int = 1000) -> str:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    df = _clean_table_df(df)
    if len(df) > max_rows:
        head = df.head(max_rows)
        note = f"\n\n[주의] 행이 {len(df)}개라 너무 커서 {max_rows}행까지만 표시했습니다."
        return head.to_string(index=False) + note
    return df.to_string(index=False)

def _is_integer_like_series(s: pd.Series) -> bool:
    vals = [str(x).strip() for x in s.tolist()]
    vals = [v for v in vals if v != "" and v.lower() != "nan" and v.lower() != "none"]
    if not vals:
        return False
    for v in vals:
        if not re.fullmatch(r"-?\d+", v):
            return False
    return True

def _looks_like_index_col(s: pd.Series) -> bool:
    if not _is_integer_like_series(s):
        return False
    vals = []
    for v in s.tolist():
        v = str(v).strip()
        if v == "" or v.lower() in ("nan", "none"):
            continue
        try:
            vals.append(int(v))
        except Exception:
            return False
    if len(vals) < 3:
        return False
    diffs = np.diff(vals)
    nonpos = (diffs <= 0).sum()
    return nonpos <= 1  # 거의 증가

def xlsx_to_text(path: str, max_rows: int = 1000) -> str:
    df0 = pd.read_excel(path, header=None)
    preamble_lines = []
    if TOP_KEEP_ROWS > 0 and len(df0) >= TOP_KEEP_ROWS:
        pre_df = df0.iloc[:TOP_KEEP_ROWS].fillna("")
        for _, row in pre_df.iterrows():
            cells = [str(v).strip() for v in row.tolist()]
            non_empty_cells = [c for c in cells if c and c.lower() not in ("nan", "none")]
            if non_empty_cells:
                preamble_lines.append(" | ".join(non_empty_cells))
        df_work = df0.iloc[TOP_KEEP_ROWS:].copy()
    else:
        df_work = df0.copy()
    df_work = df_work.replace({np.nan: ""})
    df_work = df_work.applymap(lambda x: "" if str(x).strip().lower() in ("nan", "none") else str(x))
    df_work = df_work.applymap(lambda x: re.sub(r"^\s+$", "", x))
    df_work = df_work.loc[~(df_work.apply(lambda r: all(str(x) == "" for x in r), axis=1))]
    df_work = df_work.loc[:, ~(df_work.apply(lambda c: all(str(x) == "" for x in c)))]
    if df_work.empty:
        return "\n".join(preamble_lines) if preamble_lines else ""
    header_idx = None
    for i, row in df_work.iterrows():
        vals = [str(x).strip() for x in row.tolist()]
        has_text = any(re.search(r"[A-Za-z가-힣]", v) for v in vals if v)
        non_empty_ratio = (np.array([v != "" for v in vals]).mean() if vals else 0.0)
        if has_text or non_empty_ratio >= 0.6:
            header_idx = i
            break
    if header_idx is not None:
        headers = [(str(x).strip() if str(x).strip() else f"col_{j}") for j, x in enumerate(df_work.iloc[header_idx])]
        df_tbl = df_work.iloc[header_idx + 1:].copy()
        df_tbl.columns = headers
    else:
        df_tbl = df_work.copy()
        df_tbl.columns = [f"col_{j}" for j in range(df_tbl.shape[1])]
    if df_tbl.shape[1] >= 1:
        first_header = str(df_tbl.columns[0]).strip().lower()
        protected = {"no", "번호", "id", "index"}
        first_col = df_tbl.iloc[:, 0]
        if first_header not in protected and _looks_like_index_col(first_col):
            df_tbl = df_tbl.iloc[:, 1:]
    df_tbl = df_tbl.replace({np.nan: ""})
    df_tbl = df_tbl.applymap(lambda x: "" if str(x).strip().lower() in ("nan", "none") else str(x).strip())
    note = ""
    if len(df_tbl) > max_rows:
        df_out = df_tbl.head(max_rows)
        note = f"\n\n[주의] 행이 {len(df_tbl)}개라 너무 커서 {max_rows}행까지만 표시했습니다."
    else:
        df_out = df_tbl
    preamble_text = "\n".join(preamble_lines)
    table_text = df_out.to_string(index=False)
    return preamble_text + ("\n\n" if preamble_text else "") + table_text + note

# ===================== ★ 핵심: Excel → PDF 직접 테이블 변환 (한글+폭자동+다쪽) =====================
def _measure_col_widths(data, font_name, font_size, cell_pad=6):
    cols = len(data[0]) if data else 0
    widths = [0.0] * cols
    for row in data:
        for j, val in enumerate(row):
            s = str(val)
            w = pdfmetrics.stringWidth(s, font_name, font_size) + cell_pad * 2
            if w > widths[j]:
                widths[j] = w
    return widths

def _fit_to_page(col_widths, max_width):
    total = sum(col_widths)
    if total <= max_width:
        return col_widths
    scale = max_width / total if total > 0 else 1.0
    return [w * scale for w in col_widths]

def xlsx_to_pdf_direct(xlsx_path: Path, out_pdf: Path, font_name: str = None, font_size: int = None):
    """
    Excel 미설치 환경에서도:
    - 모든 시트/모든 셀을 문자열로 렌더링
    - 한글 폰트 적용(□/■ 방지)
    - 열 너비를 A4 가용폭에 자동 맞춤
    - 표가 길면 LongTable이 자동 페이지 분할
    - 시트마다 제목 표시
    """
    if not font_name or not font_size:
        font_name, font_size = register_font_auto()

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=MARGIN_T, bottomMargin=MARGIN_B
    )
    styles = getSampleStyleSheet()
    elements = []

    xls = pd.ExcelFile(xlsx_path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None).fillna("")
        data = df.astype(str).values.tolist()

        # 시트 제목
        elements.append(Paragraph(f"<b>{sheet}</b>", styles["Heading4"]))

        if not data or (len(data) == 1 and len(data[0]) == 0):
            elements.append(Paragraph("(빈 시트)", styles["Normal"]))
            elements.append(PageBreak())
            continue

        # 열 너비 계산 → 가용 폭에 맞게 축소
        usable_w = PAGE_W - MARGIN_L - MARGIN_R
        col_widths = _measure_col_widths(data, font_name, font_size)
        col_widths = _fit_to_page(col_widths, usable_w)

        # LongTable: 자동 페이지 나눔
        table = LongTable(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),          # ★ 한글 폰트 적용
            ('FONTSIZE', (0, 0), (-1, -1), font_size),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),  # 첫 행 헤더 느낌
            ('LINEBELOW', (0, 0), (-1, 0), 0.75, colors.black),
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        elements.append(table)
        elements.append(PageBreak())

    if elements and isinstance(elements[-1], PageBreak):
        elements.pop()

    doc.build(elements)

# ===================== 변환 본체 =====================
def convert_files_to_pdf(inputs: list[Path], output_pdf: Path, font_name: str, font_size: int):
    tmp_out = output_pdf.with_name(output_pdf.stem + "_tmp_content.pdf")
    c = new_canvas(str(tmp_out))

    # 표지
    c.setFont(font_name, font_size)
    c.drawString(MARGIN_L, PAGE_H - MARGIN_T, f"[파일 변환 결과]")
    c.drawString(MARGIN_L, PAGE_H - MARGIN_T - LINE_SPACING, f"생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.showPage()

    extra_pdfs: list[Path] = []

    for p in inputs:
        suffix = p.suffix.lower()
        title = f"[{p.name}]"
        c.setFont(font_name, font_size)
        c.drawString(MARGIN_L, PAGE_H - MARGIN_T, title)
        y_start = PAGE_H - MARGIN_T - LINE_SPACING * 2

        if suffix == ".txt":
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            c.drawString(MARGIN_L, y_start, "(텍스트)")
            c.showPage()
            draw_text_block(c, text, font_name, font_size)

        elif suffix == ".csv":
            text = csv_to_text(str(p))
            c.drawString(MARGIN_L, y_start, "(CSV 표)")
            c.showPage()
            draw_text_block(c, text, font_name, font_size)

        elif suffix == ".xlsx":
            if EXCEL_NATIVE and HAS_WIN32:
                c.drawString(MARGIN_L, y_start, "(엑셀 원본 레이아웃으로 변환)")
                c.showPage()
                tmp_excel_pdf = p.with_name(f"{p.stem}__native_tmp.pdf")
                try:
                    export_workbook_to_pdf_native(p, tmp_excel_pdf)
                    extra_pdfs.append(tmp_excel_pdf)
                except Exception as e:
                    c.drawString(MARGIN_L, PAGE_H - MARGIN_T, f"(!) 네이티브 변환 실패 → 텍스트 방식: {e}")
                    c.showPage()
                    text = xlsx_to_text(str(p))
                    draw_text_block(c, text, font_name, font_size)
            else:
                # ★ 엑셀 설치 없이: 직접 표 PDF 생성
                tmp_excel_pdf = p.with_name(f"{p.stem}__direct_tmp.pdf")
                try:
                    xlsx_to_pdf_direct(p, tmp_excel_pdf, font_name=font_name, font_size=font_size)
                    extra_pdfs.append(tmp_excel_pdf)
                    c.drawString(MARGIN_L, y_start, "(엑셀 표: 직접 PDF 변환)")
                    c.showPage()
                except Exception as e:
                    text = xlsx_to_text(str(p))
                    c.drawString(MARGIN_L, y_start, f"(텍스트 방식 대체: {e})")
                    c.showPage()
                    draw_text_block(c, text, font_name, font_size)

        elif suffix in (".png", ".jpg", ".jpeg"):
            c.drawString(MARGIN_L, y_start, "(이미지)")
            c.showPage()
            draw_image(c, str(p))
            c.showPage()

        elif suffix == ".pdf":
            c.drawString(MARGIN_L, y_start, "(PDF 병합 예정)")
            c.showPage()

        else:
            c.drawString(MARGIN_L, y_start, f"(지원하지 않는 형식: {suffix})")
            c.showPage()

    c.save()

    # PDF 병합
    input_has_pdf = any(p.suffix.lower() == ".pdf" for p in inputs)
    if input_has_pdf or extra_pdfs:
        if not HAS_PYPDF:
            print("[안내] 입력/추가 PDF가 있으나 'pypdf'가 없어 병합하지 않습니다.")
            tmp_out.replace(output_pdf)
            return

        writer = PdfWriter()
        try:
            reader = PdfReader(str(tmp_out))
            for page in reader.pages:
                writer.add_page(page)
        except Exception as e:
            print(f"[경고] 내부 PDF 추가 실패: {e}")

        for p in extra_pdfs:
            try:
                r = PdfReader(str(p))
                for page in r.pages:
                    writer.add_page(page)
            except Exception as e:
                print(f"[경고] 엑셀 직접변환 PDF 병합 실패({p.name}): {e}")

        for p in inputs:
            if p.suffix.lower() != ".pdf":
                continue
            try:
                r = PdfReader(str(p))
                for page in r.pages:
                    writer.add_page(page)
            except Exception as e:
                print(f"[경고] {p.name} 병합 실패: {e}")

        with open(output_pdf, "wb") as out_f:
            writer.write(out_f)

        tmp_out.unlink(missing_ok=True)
        for p in extra_pdfs:
            p.unlink(missing_ok=True)
    else:
        tmp_out.replace(output_pdf)

# ===================== 선택 번호 파싱 =====================
def parse_selection(inp: str, n: int) -> list[int]:
    inp = (inp or "").strip().lower()
    if inp == "all":
        return list(range(n))
    out = set()
    for part in inp.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a); b = int(b)
            if a > b: a, b = b, a
            for i in range(a, b+1):
                if 0 <= i < n:
                    out.add(i)
        else:
            i = int(part)
            if 0 <= i < n:
                out.add(i)
    return sorted(out)

# ===================== 메인 =====================
def main():
    global EXCEL_NATIVE, EXCEL_FIT_WIDTH, EXCEL_FIT_HEIGHT, EXCEL_GRIDLINES, EXCEL_HEADINGS, EXCEL_QUALITY, TOP_KEEP_ROWS

    parser = argparse.ArgumentParser(description="폴더 파일을 선택해 하나의 PDF로 변환/병합")
    parser.add_argument("--font", "-f", default=None, help="한글 TTF 폰트 경로(예: C:/Windows/Fonts/NanumGothic.ttf)")
    parser.add_argument("--excel-native", action="store_true", help="엑셀을 원본 서식 그대로 PDF로 내보내기(Windows+Excel+pywin32 필요)")
    parser.add_argument("--fit-width", type=int, default=1, help="엑셀 네이티브 모드: 가로 페이지 수(예: 1)")
    parser.add_argument("--fit-height", type=int, default=0, help="엑셀 네이티브 모드: 세로 페이지 수(0=자동)")
    parser.add_argument("--gridlines", action="store_true", help="엑셀 네이티브 모드: 눈금선 출력")
    parser.add_argument("--headings", action="store_true", help="엑셀 네이티브 모드: 행/열 머리글 출력")
    parser.add_argument("--quality", choices=["standard", "minimum"], default="standard", help="엑셀 네이티브 모드: PDF 품질")
    parser.add_argument("--top-keep-rows", type=int, default=3, help="엑셀 텍스트 모드에서 상단 보존할 행 수")
    args = parser.parse_args()

    EXCEL_NATIVE = bool(args.excel_native)
    EXCEL_FIT_WIDTH = args.fit_width
    EXCEL_FIT_HEIGHT = args.fit_height
    EXCEL_GRIDLINES = bool(args.gridlines)
    EXCEL_HEADINGS = bool(args.headings)
    EXCEL_QUALITY = args.quality
    TOP_KEEP_ROWS = int(args.top_keep_rows)

    if EXCEL_NATIVE and not HAS_WIN32:
        print("[경고] pywin32(Excel 자동화)가 없어 --excel-native 옵션을 사용할 수 없습니다. 텍스트/직접변환으로 진행합니다.")
        EXCEL_NATIVE = False

    BASE_DIR = Path(__file__).resolve().parent
    TARGET_DIR = BASE_DIR
    print(f"[폴더] {TARGET_DIR}")

    files = [p for p in sorted(TARGET_DIR.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not files:
        print("지원되는 파일이 없습니다.")
        return

    print("\n[파일 목록]")
    for i, p in enumerate(files):
        print(f"  {i:>3} : {p.name}")

    sel = input("\n변환할 번호 선택 (예: 0,2-4  또는 all): ").strip()
    try:
        idx = parse_selection(sel, len(files))
    except Exception as e:
        print(f"[입력 오류] {e}")
        return

    if not idx:
        print("선택된 파일이 없습니다.")
        return

    selected = [files[i] for i in idx]
    print("\n[선택된 파일]")
    for p in selected:
        print(" -", p.name)

    # 폰트 결정
    font_name, font_size = register_font_override(args.font)
    print(f"[폰트] 사용: {font_name} (size={font_size})")

    # 출력 파일명
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"merged_{ts}.pdf"
    out_name = input(f"\n출력 PDF 파일명 입력 (기본: {default_name}): ").strip()
    if not out_name:
        out_name = default_name
    if not out_name.lower().endswith(".pdf"):
        out_name += ".pdf"
    out_path = TARGET_DIR / out_name

    convert_files_to_pdf(selected, out_path, font_name, font_size)
    print(f"\n[완료] PDF 저장: {out_path}")

if __name__ == "__main__":
    main()
