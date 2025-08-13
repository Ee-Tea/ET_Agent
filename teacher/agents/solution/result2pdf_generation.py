from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
import io

pdfmetrics.registerFont(TTFont("NanumGothic", "../fonts/NanumGothic.ttf"))

def generate_pdf(results):


    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    # 📌 한글 지원 스타일 정의
    styles = {
        "title": ParagraphStyle(name="title", fontName="NanumGothic", fontSize=16, spaceAfter=10),
        "normal": ParagraphStyle(name="normal", fontName="NanumGothic", fontSize=11, spaceAfter=6),
    }

    story = []

    for item in results:
        story.append(Paragraph(f"문제 {item['index']}", styles["title"]))
        story.append(Paragraph(f"문제: {item['question']}", styles["normal"]))
        story.append(Paragraph("보기:", styles["normal"]))
        for idx, opt in enumerate(item["options"], 1):
            story.append(Paragraph(f"{idx}. {opt}", styles["normal"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"정답: {item['answer']}", styles["normal"]))
        story.append(Paragraph(f"풀이: {item['explanation']}", styles["normal"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer