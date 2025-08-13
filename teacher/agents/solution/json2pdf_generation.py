# make_booklet_simple.py
# 사용법: python make_booklet_simple.py input.json output.pdf
import sys, json
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import enums

# ❗ 프로젝트에 fonts/NanumGothic.ttf 를 넣어두세요.
FONT_PATH = "../fonts/NanumGothic.ttf"
FONT_NAME = "NanumGothic"
pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))

def _extract_questions(data):
    # data가 리스트인 경우 직접 반환
    if isinstance(data, list):
        return data
    
    # all_questions 우선 사용, 없으면 subjects.*.questions 합침
    if isinstance(data, dict) and isinstance(data.get("all_questions"), list):
        return data["all_questions"]
    
    out = []
    subs = data.get("subjects", {}) if isinstance(data, dict) else {}
    if isinstance(subs, dict):
        for v in subs.values():
            arr = v.get("questions", [])
            if isinstance(arr, list):
                out.extend(arr)
    return out

def build_pdf(input_json_path, output_pdf_path, limit=None):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data가 리스트인지 딕셔너리인지 확인
    if isinstance(data, list):
        # data가 리스트인 경우 (예: solution_results.json)
        title = "문제집"
        questions = data[:limit] if limit else data
    else:
        # data가 딕셔너리인 경우 (예: user_problems_json.json)
        title = (data.get("exam_title") or "문제집").strip()
        questions = _extract_questions(data)[:limit] if limit else _extract_questions(data)

    # 스타일 (폰트 하나만 사용)
    title_style = ParagraphStyle(
        name="title", fontName=FONT_NAME, fontSize=18, leading=22,
        alignment=enums.TA_CENTER, spaceAfter=10,
    )
    q_style = ParagraphStyle(
        name="question", fontName=FONT_NAME, fontSize=12.5, leading=17,
        spaceAfter=6,
    )
    opt_style = ParagraphStyle(
        name="option", fontName=FONT_NAME, fontSize=11.5, leading=15,
        leftIndent=8*mm,
    )

    doc = SimpleDocTemplate(
        output_pdf_path, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm
    )

    story = []
    story.append(Paragraph(f"{title} - 문제(1~{len(questions)})", title_style))
    story.append(Spacer(1, 6*mm))

    for idx, q in enumerate(questions, start=1):
        question_text = (q.get("question") or "").strip()
        story.append(Paragraph(f"문제 {idx}. {question_text}", q_style))

        for i, opt in enumerate(q.get("options") or [], start=1):
            # 옵션에 번호 붙이기 (1, 2, 3, 4)
            # option_text = f"{i}. {str(opt).strip()}"
            option_text = f"{str(opt).strip()}"
            story.append(Paragraph(option_text, opt_style))

        story.append(Spacer(1, 5*mm))

        # 가독성을 위해 10문항마다 페이지 나눔
        if idx % 10 == 0 and idx != len(questions):
            story.append(PageBreak())

    doc.build(story)
    print(f"✅ 생성 완료: {output_pdf_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python make_booklet_simple.py input.json output.pdf")
        sys.exit(1)
    
    # limit 파라미터가 있으면 사용, 없으면 모든 문제 생성
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    build_pdf(sys.argv[1], sys.argv[2], limit=limit)
