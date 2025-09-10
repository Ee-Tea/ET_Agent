#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
종합 시험 PDF 생성기
- 문제집 (문제 + 보기)
- 답안집 (문제 + 보기 + 정답 + 풀이)
- 오답 분석 (틀린 문제 + 정답 + 풀이 + 오답 원인)
"""

import json
import os, re
from datetime import datetime
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import black, gray, white, HexColor
from reportlab.lib.units import mm
from reportlab.lib import enums

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
        # pdfmetrics.registerFont(TTFont("Helvetica", "Helvetica"))
        FONT_NAME = "Helvetica"
        print(f"[DEBUG] 대체 폰트 사용: {FONT_NAME}")
    except Exception as e:
        print(f"[ERROR] 대체 폰트 등록 실패: {e}")
        FONT_NAME = "Helvetica"  # 기본값

class ComprehensivePDFGenerator:
    """종합 시험 PDF 생성기"""
    
    def __init__(self):
        self.styles = self._make_styles()
    
    def _make_styles(self):
        base = getSampleStyleSheet()

        styles = {
            "title": ParagraphStyle(
                name="rpt_title", parent=base["Title"],
                fontName=FONT_NAME, fontSize=24, leading=28, alignment=1
            ),
            "subtitle": ParagraphStyle(
                name="rpt_subtitle", parent=base["Heading2"],
                fontName=FONT_NAME, fontSize=16, leading=22, alignment=1,
                textColor=HexColor("#3b5bdb")
            ),
            "header": ParagraphStyle(
                name="rpt_header", parent=base["Normal"],
                fontName=FONT_NAME, fontSize=10, leading=14,
                textColor=HexColor("#555555")
            ),
            "question": ParagraphStyle(
                name="rpt_question", parent=base["Normal"],
                fontName=FONT_NAME, fontSize=11, leading=16
            ),
            "option": ParagraphStyle(
                name="rpt_option", parent=base["Normal"],
                fontName=FONT_NAME, fontSize=10, leading=14, leftIndent=10
            ),
            "answer": ParagraphStyle(
                name="rpt_answer", parent=base["Normal"],
                fontName=FONT_NAME, fontSize=10, leading=14
            ),
            "explanation": ParagraphStyle(
                name="rpt_explanation", parent=base["Normal"],
                fontName=FONT_NAME, fontSize=10, leading=14
            ),
            "section": ParagraphStyle(
                name="rpt_section", parent=base["Heading2"],
                fontName=FONT_NAME, fontSize=14, leading=20
            ),
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
                    # 기존 번호 제거 (1), 2), 3), 4) 또는 1. 2. 3. 4. 패턴)
                    clean_option = str(option).strip()
                    # 정규식으로 번호 패턴 제거
                    import re
                    clean_option = re.sub(r'^[①②③④⑤⑥⑦⑧⑨⑩]?\s*[0-9]+[\.\)]\s*', '', clean_option)
                    clean_option = re.sub(r'^[0-9]+[\.\)]\s*', '', clean_option)
                    
                    option_text = f"{i}) {clean_option}"
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
                # 기존 번호 제거 (1), 2), 3), 4) 또는 1. 2. 3. 4. 패턴)
                clean_option = str(option).strip()
                # 정규식으로 번호 패턴 제거
                import re
                clean_option = re.sub(r'^[①②③④⑤⑥⑦⑧⑨⑩]?\s*[0-9]+[\.\)]\s*', '', clean_option)
                clean_option = re.sub(r'^[0-9]+[\.\)]\s*', '', clean_option)
                
                option_text = f"{i}) {clean_option}"
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
    
    def generate_analysis_report(self, data, output_path, title="오답 분석 리포트"):
        """
        data(dict) 또는 problems(list[dict]) 모두 허용
        dict 예시:
         {
           "problems":[{question, options, user_answer, generated_answer, generated_explanation}, ...],
           "score_result": {"correct_count":.., "total_count":.., "accuracy":..},
           "weak_types": ["데이터베이스구축", ...] 또는 [{"label":..,"count":..}],
           "analysis": {"detailed_analysis":[{..},..], "overall_assessment":{...}}
         }
        """
        # ---- 입력 표준화 ----
        if isinstance(data, dict):
            problems      = data.get("problems") or []
            score_result  = data.get("score_result") or {}
            weak_types    = data.get("weak_types") or []
            analysis_blk  = data.get("analysis") or {}
        else:
            problems      = data or []
            score_result  = {}
            weak_types    = []
            analysis_blk  = {}

        # 정확도/정답수 폴백
        def _num(x):
            m = re.search(r"\d+", str(x))
            return m.group(0) if m else None

        if isinstance(score_result, dict) and score_result.get("total_count"):
            correct = int(score_result.get("correct_count") or 0)
            total   = int(score_result.get("total_count") or len(problems))
            accuracy = float(score_result.get("accuracy") or (correct / total if total else 0.0))
        else:
            auto = [1 if (_num(p.get("user_answer")) == _num(p.get("generated_answer"))) else 0
                    for p in problems if isinstance(p, dict)]
            correct = sum(auto)
            total   = len(auto) if auto else len(problems)
            accuracy = (correct / total) if total else 0.0

        # 약점 유형 표준화
        if isinstance(weak_types, dict):
            weak_types = [weak_types]
        weak_labels = []
        for w in weak_types:
            if isinstance(w, dict):
                weak_labels.append(str(w.get("label") or w.get("type") or w))
            else:
                weak_labels.append(str(w))

        # 상세/총평 표준화
        detailed = (analysis_blk.get("detailed_analysis") 
                    or analysis_blk.get("details")
                    or (analysis_blk if isinstance(analysis_blk, list) else []))
        overall  = analysis_blk.get("overall_assessment") or {}

        # ---- PDF 빌드 ----
        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=15*mm, rightMargin=15*mm,
            topMargin=15*mm, bottomMargin=15*mm
        )
        story = []

        # 헤더
        story.append(Paragraph(title, self.styles["title"]))
        story.append(Paragraph(f"생성일: {datetime.now().strftime('%Y-%m-%d')}", self.styles["header"]))
        story.append(Paragraph(f"총 문항 수: {len(problems)}개", self.styles["header"]))
        story.append(Spacer(1, 8*mm))

        # 요약
        summary_data = [
            ["항목", "수치"],
            ["총 문항 수", str(total)],
            ["정답 수", str(correct)],
            ["오답 수", str(max(0, total - correct))],
            ["정확도", f"{accuracy*100:.1f}%"],
            ["약점 유형 수", str(len(weak_labels))]
        ]
        t = Table(summary_data, colWidths=[80*mm, 40*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), gray),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        story.append(Paragraph("📊 시험 요약", self.styles["subtitle"]))
        story.append(t)
        story.append(Spacer(1, 6*mm))

        # 약점 유형
        if weak_labels:
            story.append(Paragraph("⚠️ 약점 유형", self.styles["section"]))
            for lab in weak_labels:
                story.append(Paragraph(f"- {lab}", self.styles["answer"]))
            story.append(Spacer(1, 6*mm))

        # 문제별 상세
        story.append(Paragraph("🔍 문제별 상세 분석", self.styles["section"]))
        story.append(Spacer(1, 3*mm))
        for idx, p in enumerate(problems, 1):
            if not isinstance(p, dict): 
                continue
            q = str(p.get("question", "")).strip()
            story.append(Paragraph(f"문제 {idx}. {q}", self.styles["question"]))
            for i, opt in enumerate(p.get("options", []) or [], 1):
                story.append(Paragraph(f"{i}. {str(opt).strip()}", self.styles["option"]))
            ua = p.get("user_answer", "")
            ca = p.get("generated_answer", "")
            ex = p.get("generated_explanation", "")
            story.append(Paragraph(f"<b>사용자 답:</b> {ua}", self.styles["answer"]))
            story.append(Paragraph(f"<b>정답:</b> {ca}", self.styles["answer"]))
            if ex:
                story.append(Paragraph(f"<b>풀이:</b> {ex}", self.styles["explanation"]))
            story.append(Spacer(1, 6*mm))
            if idx % 6 == 0 and idx != len(problems):
                story.append(PageBreak())

        # 상세 분석(LLM 산출) 섹션
        if isinstance(detailed, list) and detailed:
            story.append(Paragraph("🧠 LLM 상세 분석", self.styles["section"]))
            story.append(Spacer(1, 3*mm))
            rows = [["문항", "유형", "과목", "분석"]]
            for i, item in enumerate(detailed, 1):
                if not isinstance(item, dict): 
                    continue
                analysis_txt = str(item.get("analysis") or item.get("Analysis") or "")
                mistake      = str(item.get("mistake_type") or item.get("Mistake Type") or "")
                subject      = str(item.get("subject") or item.get("Subject") or "")
                pnum         = str(item.get("problem_number") or item.get("Problem Number") or i)
                
                # 긴 텍스트를 Paragraph로 감싸서 자동 줄바꿈 처리
                analysis_para = Paragraph(analysis_txt, self.styles["explanation"])
                mistake_para = Paragraph(mistake, self.styles["answer"])
                subject_para = Paragraph(subject, self.styles["answer"])
                
                rows.append([pnum, mistake_para, subject_para, analysis_para])
            
            # 열 너비 조정: 분석 열을 더 넓게, 전체 너비를 A4에 맞게 조정
            dt = Table(rows, colWidths=[20*mm, 40*mm, 40*mm, 100*mm])
            dt.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), gray),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('WORDWRAP', (0, 0), (-1, -1), 'CJK'),  # 한글 줄바꿈 활성화
                ('LEFTPADDING', (0, 0), (-1, -1), 3),
                ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3)
            ]))
            story.append(dt)
            story.append(Spacer(1, 6*mm))

        # 종합 평가(LLM 산출) - 개선된 버전
        if isinstance(overall, dict) and overall:
            story.append(PageBreak())
            story.append(Paragraph("📝 종합 평가", self.styles["section"]))
            story.append(Spacer(1, 6*mm))
            
            # 제목 (title)
            title = overall.get("title") or overall.get("Title") or overall.get("제목")
            if title:
                story.append(Paragraph(f"📋 {title}", self.styles["subtitle"]))
                story.append(Spacer(1, 6*mm))
            
            # 총평 (final_message)
            final_message = overall.get("final_message") or overall.get("Final Message") or overall.get("총평")
            if final_message:
                story.append(Paragraph("🎯 총평", self.styles["subtitle"]))
                story.append(Paragraph(str(final_message), self.styles["explanation"]))
                story.append(Spacer(1, 8*mm))
            
            # 강점 분석 (strengths_analysis)
            strengths_analysis = overall.get("strengths_analysis") or overall.get("Strengths Analysis") or overall.get("강점 분석")
            if strengths_analysis:
                story.append(Paragraph("💪 강점 분석", self.styles["subtitle"]))
                story.append(Paragraph(str(strengths_analysis), self.styles["explanation"]))
                story.append(Spacer(1, 6*mm))
            
            # 강점과 약점을 나란히 표시
            strengths = overall.get("strengths") or overall.get("Strengths") or overall.get("강점")
            weaknesses = overall.get("weaknesses") or overall.get("Weaknesses") or overall.get("약점")
            
            if strengths or weaknesses:
                story.append(Paragraph("📊 강점과 약점", self.styles["subtitle"]))
                story.append(Spacer(1, 4*mm))
                
                # 강점과 약점을 2열 테이블로 표시
                strength_weakness_data = []
                if strengths:
                    strength_weakness_data.append(["✅ 강점", str(strengths)])
                if weaknesses:
                    strength_weakness_data.append(["❌ 약점", str(weaknesses)])
                
                if strength_weakness_data:
                    # 텍스트를 Paragraph로 감싸서 자동 줄바꿈 처리
                    formatted_data = []
                    for row in strength_weakness_data:
                        formatted_row = [
                            Paragraph(row[0], self.styles["answer"]),  # 강점/약점 라벨
                            Paragraph(row[1], self.styles["explanation"])  # 내용
                        ]
                        formatted_data.append(formatted_row)
                    
                    sw_table = Table(formatted_data, colWidths=[25*mm, 155*mm])
                    sw_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), HexColor('#E8F5E8')),
                        ('BACKGROUND', (1, 0), (1, -1), HexColor('#F8F8F8')),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 1, black),
                        ('WORDWRAP', (0, 0), (-1, -1), 'CJK'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 4),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
                    ]))
                    story.append(sw_table)
                    story.append(Spacer(1, 8*mm))
            
            # 맞춤 학습 계획 (action_plan)
            action_plan = overall.get("action_plan") or overall.get("Action Plan") or overall.get("맞춤 학습 계획")
            if action_plan:
                story.append(Paragraph("📚 맞춤 학습 계획", self.styles["subtitle"]))
                if isinstance(action_plan, dict):
                    plan_title = action_plan.get("title") or action_plan.get("Title") or "학습 계획"
                    story.append(Paragraph(f"<b>{plan_title}</b>", self.styles["explanation"]))
                    
                    short_term = action_plan.get("short_term_goal") or action_plan.get("Short Term Goal")
                    if short_term:
                        story.append(Paragraph(f"🎯 단기 목표: {str(short_term)}", self.styles["explanation"]))
                    
                    long_term = action_plan.get("long_term_goal") or action_plan.get("Long Term Goal")
                    if long_term:
                        story.append(Paragraph(f"🚀 장기 목표: {str(long_term)}", self.styles["explanation"]))
                    
                    # 추천 전략 (recommended_strategies)
                    recommend_strategies = action_plan.get("recommended_strategies") or action_plan.get("Recommended Strategies")
                    if recommend_strategies and isinstance(recommend_strategies, list):
                        story.append(Paragraph("🎯 추천 전략:", self.styles["explanation"]))
                        for i, strategy in enumerate(recommend_strategies, 1):
                            story.append(Paragraph(f"  {i}. {str(strategy)}", self.styles["explanation"]))
                        story.append(Spacer(1, 4*mm))
                    
                    recommendations = action_plan.get("recommendations") or action_plan.get("Recommendations")
                    if recommendations and isinstance(recommendations, list):
                        story.append(Paragraph("💡 권장사항:", self.styles["explanation"]))
                        for i, rec in enumerate(recommendations, 1):
                            story.append(Paragraph(f"  {i}. {str(rec)}", self.styles["explanation"]))
                else:
                    story.append(Paragraph(str(action_plan), self.styles["explanation"]))
                story.append(Spacer(1, 6*mm))
            
            # 심화 학습 계획 (deepen_learning_plan)
            deepen_plan = overall.get("deepen_learning_plan") or overall.get("Deepen Learning Plan") or overall.get("심화 학습 계획")
            if deepen_plan:
                story.append(Paragraph("🔥 심화 학습 계획", self.styles["subtitle"]))
                if isinstance(deepen_plan, dict):
                    plan_title = deepen_plan.get("title") or deepen_plan.get("Title") or "심화 학습"
                    story.append(Paragraph(f"<b>{plan_title}</b>", self.styles["explanation"]))
                    
                    recommendations = deepen_plan.get("recommendations") or deepen_plan.get("Recommendations")
                    if recommendations and isinstance(recommendations, list):
                        for i, rec in enumerate(recommendations, 1):
                            story.append(Paragraph(f"  {i}. {str(rec)}", self.styles["explanation"]))
                else:
                    story.append(Paragraph(str(deepen_plan), self.styles["explanation"]))
                story.append(Spacer(1, 6*mm))
            
            # 추천 전략 (recommend_strategies)
            recommend_strategies = overall.get("recommend_strategies") or overall.get("Recommend Strategies") or overall.get("추천 전략")
            if recommend_strategies:
                story.append(Paragraph("🎯 추천 전략", self.styles["subtitle"]))
                if isinstance(recommend_strategies, list):
                    for i, strategy in enumerate(recommend_strategies, 1):
                        story.append(Paragraph(f"{i}. {str(strategy)}", self.styles["explanation"]))
                else:
                    story.append(Paragraph(str(recommend_strategies), self.styles["explanation"]))
                story.append(Spacer(1, 6*mm))
            
            # 팁과 다음 단계
            tips = overall.get("tips") or overall.get("Tips") or overall.get("팁")
            next_steps = overall.get("next_steps") or overall.get("Next Steps") or overall.get("다음 단계")
            
            if tips:
                story.append(Paragraph("💡 학습 팁", self.styles["subtitle"]))
                story.append(Paragraph(str(tips), self.styles["explanation"]))
                story.append(Spacer(1, 6*mm))
            
            if next_steps:
                story.append(Paragraph("🚀 다음 단계", self.styles["subtitle"]))
                story.append(Paragraph(str(next_steps), self.styles["explanation"]))
                story.append(Spacer(1, 6*mm))

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
