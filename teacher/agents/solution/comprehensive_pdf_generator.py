#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
종합 시험 PDF 생성기
- 문제집 (문제 + 보기)
- 답안집 (문제 + 보기 + 정답 + 풀이)
- 오답 분석 (틀린 문제 + 정답 + 풀이 + 오답 원인)
"""

import json
import os
from datetime import datetime
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib import enums
from reportlab.lib.colors import black, white, red, blue, green, gray

# 한글 폰트 등록
_HERE = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.abspath(os.path.join(_HERE, "..", "..", "fonts", "NanumGothic.ttf"))
FONT_NAME = "NanumGothic"

# 폰트 파일 존재 확인 및 경로 출력
print(f"[DEBUG] 폰트 경로: {FONT_PATH}")
print(f"[DEBUG] 폰트 파일 존재: {os.path.exists(FONT_PATH)}")

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
    print(f"[DEBUG] 폰트 등록 성공: {FONT_NAME}")
else:
    print(f"[WARNING] 폰트 파일을 찾을 수 없음: {FONT_PATH}")
    # 대체 폰트 시도
    try:
        # 기본 폰트 사용
        pdfmetrics.registerFont(TTFont("Helvetica", "Helvetica"))
        FONT_NAME = "Helvetica"
        print(f"[DEBUG] 대체 폰트 사용: {FONT_NAME}")
    except Exception as e:
        print(f"[ERROR] 대체 폰트 등록 실패: {e}")
        FONT_NAME = "Helvetica"  # 기본값

class ComprehensivePDFGenerator:
    """종합 시험 PDF 생성기"""
    
    def __init__(self):
        self.styles = self._create_styles()
    
    def _create_styles(self):
        """PDF 스타일 정의"""
        styles = {
            "title": ParagraphStyle(
                name="title", 
                fontName=FONT_NAME, 
                fontSize=18, 
                leading=22,
                alignment=enums.TA_CENTER, 
                spaceAfter=10,
                textColor=black
            ),
            "subtitle": ParagraphStyle(
                name="subtitle", 
                fontName=FONT_NAME, 
                fontSize=14, 
                leading=18,
                alignment=enums.TA_CENTER, 
                spaceAfter=8,
                textColor=blue
            ),
            "question": ParagraphStyle(
                name="question", 
                fontName=FONT_NAME, 
                fontSize=12.5, 
                leading=17,
                spaceAfter=6,
                textColor=black
            ),
            "option": ParagraphStyle(
                name="option", 
                fontName=FONT_NAME, 
                fontSize=11.5, 
                leading=15,
                leftIndent=8*mm,
                textColor=black
            ),
            "answer": ParagraphStyle(
                name="answer", 
                fontName=FONT_NAME, 
                fontSize=12, 
                leading=16,
                spaceAfter=4,
                textColor=green
            ),
            "explanation": ParagraphStyle(
                name="explanation", 
                fontName=FONT_NAME, 
                fontSize=11, 
                leading=15,
                leftIndent=4*mm,
                textColor=black
            ),
            "header": ParagraphStyle(
                name="header", 
                fontName=FONT_NAME, 
                fontSize=10, 
                leading=12,
                textColor=gray
            )
        }
        return styles
    
    def generate_problem_booklet(self, problems, output_path, title="시험 문제집"):
        """문제집 생성 (문제 + 보기만)"""
        print(f"[DEBUG] generate_problem_booklet 시작")
        print(f"[DEBUG] problems 개수: {len(problems)}")
        print(f"[DEBUG] output_path: {output_path}")
        print(f"[DEBUG] title: {title}")
        
        try:
            # 문제 데이터 검증
            if not problems:
                print("[ERROR] problems가 비어있음")
                return None
            
            print(f"[DEBUG] 첫 번째 문제 샘플: {problems[0] if problems else 'None'}")
            
            doc = SimpleDocTemplate(
                output_path, 
                pagesize=A4,
                leftMargin=15*mm, 
                rightMargin=15*mm,
                topMargin=15*mm, 
                bottomMargin=15*mm
            )
            print(f"[DEBUG] SimpleDocTemplate 생성 완료")
            
            story = []
            
            # 헤더
            story.append(Paragraph(f"{title}", self.styles["title"]))
            story.append(Paragraph(f"생성일: {datetime.now().strftime('%Y-%m-%d')}", self.styles["header"]))
            story.append(Paragraph(f"총 문항 수: {len(problems)}개", self.styles["header"]))
            story.append(Spacer(1, 8*mm))
            print(f"[DEBUG] 헤더 추가 완료")
            
            # 문제들
            for idx, problem in enumerate(problems, 1):
                question_text = problem.get("question", "").strip()
                print(f"[DEBUG] 문제 {idx} 처리: {question_text[:50]}...")
                
                story.append(Paragraph(f"문제 {idx}. {question_text}", self.styles["question"]))
                
                # 보기
                options = problem.get("options", [])
                print(f"[DEBUG] 보기 {idx}: {options}")
                
                for i, option in enumerate(options, 1):
                    option_text = f"{i}) {str(option).strip()}"
                    story.append(Paragraph(option_text, self.styles["option"]))
                
                story.append(Spacer(1, 5*mm))
                
                # 10문항마다 페이지 나눔
                if idx % 10 == 0 and idx != len(problems):
                    story.append(PageBreak())
                    print(f"[DEBUG] 페이지 나눔 추가 (문제 {idx})")
            
            print(f"[DEBUG] story 구성 완료, 총 {len(story)}개 요소")
            print(f"[DEBUG] PDF 빌드 시작...")
            
            doc.build(story)
            print(f"✅ 문제집 생성 완료: {output_path}")
            
            # 파일 생성 확인
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"[DEBUG] PDF 파일 생성 확인: {output_path} (크기: {file_size:,} bytes)")
                return output_path
            else:
                print(f"[ERROR] PDF 파일이 생성되지 않음: {output_path}")
                return None
                
        except Exception as e:
            print(f"[ERROR] generate_problem_booklet 실행 중 오류: {e}")
            import traceback
            print(f"[DEBUG] 상세 오류: {traceback.format_exc()}")
            return None
    
    def generate_answer_booklet(self, problems, output_path, title="시험 답안집"):
        """답안집 생성 (문제 + 보기 + 정답 + 풀이)"""
        doc = SimpleDocTemplate(
            output_path, 
            pagesize=A4,
            leftMargin=15*mm, 
            rightMargin=15*mm,
            topMargin=15*mm, 
            bottomMargin=15*mm
        )
        
        story = []
        
        # 헤더
        story.append(Paragraph(f"{title}", self.styles["title"]))
        story.append(Paragraph(f"생성일: {datetime.now().strftime('%Y-%m-%d')}", self.styles["header"]))
        story.append(Paragraph(f"총 문항 수: {len(problems)}개", self.styles["header"]))
        story.append(Spacer(1, 8*mm))
        
        # 문제들과 답안
        for idx, problem in enumerate(problems, 1):
            question_text = problem.get("question", "").strip()
            story.append(Paragraph(f"문제 {idx}. {question_text}", self.styles["question"]))
            
            # 보기
            options = problem.get("options", [])
            for i, option in enumerate(options, 1):
                option_text = f"{i}. {str(option).strip()}"
                story.append(Paragraph(option_text, self.styles["option"]))
            
            story.append(Spacer(1, 3*mm))
            
            # 정답과 풀이
            answer = problem.get("generated_answer", "정답 없음")
            explanation = problem.get("generated_explanation", "풀이 없음")
            
            story.append(Paragraph(f"<b>정답:</b> {answer}", self.styles["answer"]))
            story.append(Paragraph(f"<b>풀이:</b> {explanation}", self.styles["explanation"]))
            
            story.append(Spacer(1, 8*mm))
            
            # 8문항마다 페이지 나눔 (답안이 길어서)
            if idx % 8 == 0 and idx != len(problems):
                story.append(PageBreak())
        
        doc.build(story)
        print(f"✅ 답안집 생성 완료: {output_path}")
    
    def generate_analysis_report(self, problems, output_path, title="오답 분석 리포트"):
        """오답 분석 리포트 생성"""
        doc = SimpleDocTemplate(
            output_path, 
            pagesize=A4,
            leftMargin=15*mm, 
            rightMargin=15*mm,
            topMargin=15*mm, 
            bottomMargin=15*mm
        )
        
        story = []
        
        # 헤더
        story.append(Paragraph(f"{title}", self.styles["title"]))
        story.append(Paragraph(f"생성일: {datetime.now().strftime('%Y-%m-%d')}", self.styles["header"]))
        story.append(Paragraph(f"총 문항 수: {len(problems)}개", self.styles["header"]))
        story.append(Spacer(1, 8*mm))
        
        # 통계 요약
        total_questions = len(problems)
        questions_with_answers = sum(1 for p in problems if p.get("generated_answer"))
        questions_with_explanations = sum(1 for p in problems if p.get("generated_explanation"))
        
        summary_data = [
            ["항목", "수치"],
            ["총 문항 수", str(total_questions)],
            ["정답 생성된 문항", str(questions_with_answers)],
            ["풀이 생성된 문항", str(questions_with_explanations)],
            ["완성도", f"{questions_with_answers/total_questions*100:.1f}%"]
        ]
        
        summary_table = Table(summary_data, colWidths=[80*mm, 40*mm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), gray),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        story.append(Paragraph("📊 시험 요약", self.styles["subtitle"]))
        story.append(summary_table)
        story.append(Spacer(1, 8*mm))
        
        # 문제별 상세 분석
        story.append(Paragraph("🔍 문제별 상세 분석", self.styles["subtitle"]))
        story.append(Spacer(1, 5*mm))
        
        for idx, problem in enumerate(problems, 1):
            question_text = problem.get("question", "").strip()
            story.append(Paragraph(f"문제 {idx}. {question_text}", self.styles["question"]))
            
            # 보기
            options = problem.get("options", [])
            for i, option in enumerate(options, 1):
                option_text = f"{i}. {str(option).strip()}"
                story.append(Paragraph(option_text, self.styles["option"]))
            
            story.append(Spacer(1, 3*mm))
            
            # 정답과 풀이
            answer = problem.get("generated_answer", "")
            explanation = problem.get("generated_explanation", "")
            
            if answer:
                story.append(Paragraph(f"<b>정답:</b> {answer}", self.styles["answer"]))
            else:
                story.append(Paragraph("<b>정답:</b> <font color='red'>생성되지 않음</font>", self.styles["answer"]))
            
            if explanation:
                story.append(Paragraph(f"<b>풀이:</b> {explanation}", self.styles["explanation"]))
            else:
                story.append(Paragraph("<b>풀이:</b> <font color='red'>생성되지 않음</font>", self.styles["explanation"]))
            
            story.append(Spacer(1, 8*mm))
            
            # 6문항마다 페이지 나눔 (분석 내용이 길어서)
            if idx % 6 == 0 and idx != len(problems):
                story.append(PageBreak())
        
        doc.build(story)
        print(f"✅ 오답 분석 리포트 생성 완료: {output_path}")
    
    def generate_all_pdfs(self, problems, base_filename="comprehensive_exam"):
        """모든 PDF 생성 (문제집, 답안집, 분석 리포트)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 문제집
        problem_pdf = f"{base_filename}_문제집_{timestamp}.pdf"
        self.generate_problem_booklet(problems, problem_pdf)
        
        # 답안집
        answer_pdf = f"{base_filename}_답안집_{timestamp}.pdf"
        self.generate_answer_booklet(problems, answer_pdf)
        
        # 분석 리포트
        analysis_pdf = f"{base_filename}_분석리포트_{timestamp}.pdf"
        self.generate_analysis_report(problems, analysis_pdf)
        
        print(f"\n🎉 모든 PDF 생성 완료!")
        print(f"   📚 문제집: {problem_pdf}")
        print(f"   📝 답안집: {answer_pdf}")
        print(f"   📊 분석 리포트: {analysis_pdf}")
        
        return {
            "problem_pdf": problem_pdf,
            "answer_pdf": answer_pdf,
            "analysis_pdf": analysis_pdf
        }

def main():
    """메인 실행 함수"""
    # JSON 파일에서 문제 데이터 로드
    json_file = "user_problems_json.json"
    
    if not os.path.exists(json_file):
        print(f"❌ JSON 파일을 찾을 수 없습니다: {json_file}")
        return
    
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 문제 데이터 추출
        if isinstance(data, dict) and "problems" in data:
            problems = data["problems"]
        elif isinstance(data, list):
            problems = data
        else:
            print("❌ 올바른 문제 데이터 형식이 아닙니다.")
            return
        
        print(f"📚 로드된 문제 수: {len(problems)}개")
        
        # PDF 생성기 초기화
        generator = ComprehensivePDFGenerator()
        
        # 모든 PDF 생성
        result_files = generator.generate_all_pdfs(problems, "정보처리기사_시험")
        
        print(f"\n✅ PDF 생성 완료! 다음 파일들을 확인하세요:")
        for file_type, file_path in result_files.items():
            print(f"   - {file_type}: {file_path}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
