#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exam 폴더의 PDF들을 파싱하고 과목별로 구분하여 txt 파일로 저장하는 스크립트
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from pdf_preprocessor_ai import PDFPreprocessor

# 과목별 문제 번호 범위 정의
SUBJECT_RANGES = {
    "소프트웨어 설계": (1, 20),
    "소프트웨어 개발": (21, 40),
    "데이터베이스 구축": (41, 60),
    "프로그래밍 언어 활용": (61, 80),
    "정보시스템 구축 관리": (81, 100)
}

def get_subject_by_problem_number(problem_number: int) -> str:
    """문제 번호에 따른 과목 반환"""
    for subject, (start, end) in SUBJECT_RANGES.items():
        if start <= problem_number <= end:
            return subject
    return "기타"

def extract_problem_number(question: str) -> Optional[int]:
    """질문에서 문제 번호 추출"""
    # "1.", "1)", "(1)" 등의 패턴에서 번호 추출
    patterns = [
        r'^(\d+)\s*\.',
        r'^(\d+)\s*\)',
        r'^\((\d+)\)',
        r'(\d+)번\s*문제',
        r'(\d+)번',
        r'문제\s*(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return int(match.group(1))
    
    return None

def sort_problems_by_number(problems: List[Dict]) -> List[Dict]:
    """문제를 번호 순으로 정렬"""
    def extract_number(problem):
        question = problem.get('question', '')
        number = extract_problem_number(question)
        return number if number is not None else 999999
    
    return sorted(problems, key=extract_number)

def group_problems_by_subject(problems: List[Dict]) -> Dict[str, List[Dict]]:
    """문제를 과목별로 그룹화"""
    subject_groups = {subject: [] for subject in SUBJECT_RANGES.keys()}
    subject_groups["기타"] = []
    
    for problem in problems:
        question = problem.get('question', '')
        number = extract_problem_number(question)
        
        if number is not None:
            subject = get_subject_by_problem_number(number)
            subject_groups[subject].append(problem)
        else:
            subject_groups["기타"].append(problem)
    
    return subject_groups

def format_problem_text(problem: Dict, index: int) -> str:
    """개별 문제를 텍스트로 포맷팅"""
    question = problem.get('question', '')
    options = problem.get('options', [])
    
    # 문제 번호 추출
    number = extract_problem_number(question)
    number_str = f"[{number:2d}번]" if number is not None else f"[{index:2d}]"
    
    # 보기 번호 매핑
    option_marks = ['①', '②', '③', '④']
    
    text = f"{number_str} {question}\n"
    
    for i, option in enumerate(options):
        if i < len(option_marks):
            text += f"  {option_marks[i]} {option}\n"
    
    text += "\n"
    return text

def save_subject_file(subject: str, problems: List[Dict], output_dir: Path, filename_prefix: str):
    """과목별 문제를 txt 파일로 저장"""
    if not problems:
        return
    
    # 과목별 파일명 생성
    subject_filename = f"{filename_prefix}_{subject}.txt"
    output_path = output_dir / subject_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {subject} ===\n")
        f.write(f"문제 수: {len(problems)}개\n")
        f.write("=" * 50 + "\n\n")
        
        # 문제 번호 순으로 정렬
        sorted_problems = sort_problems_by_number(problems)
        
        for i, problem in enumerate(sorted_problems, 1):
            f.write(format_problem_text(problem, i))
    
    print(f"✅ {subject}: {len(problems)}개 문제 → {subject_filename}")

def save_all_problems_file(problems: List[Dict], output_dir: Path, filename_prefix: str):
    """모든 문제를 하나의 파일로 저장"""
    if not problems:
        return
    
    # 전체 문제 파일명
    all_filename = f"{filename_prefix}_전체문제.txt"
    output_path = output_dir / all_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 전체 문제 ===\n")
        f.write(f"총 문제 수: {len(problems)}개\n")
        f.write("=" * 50 + "\n\n")
        
        # 문제 번호 순으로 정렬
        sorted_problems = sort_problems_by_number(problems)
        
        # 과목별로 구분하여 저장
        subject_groups = group_problems_by_subject(sorted_problems)
        
        for subject in SUBJECT_RANGES.keys():
            if subject_groups[subject]:
                f.write(f"\n{'='*20} {subject} {'='*20}\n")
                f.write(f"문제 수: {len(subject_groups[subject])}개\n\n")
                
                for i, problem in enumerate(subject_groups[subject], 1):
                    f.write(format_problem_text(problem, i))
    
    print(f"✅ 전체 문제: {len(problems)}개 → {all_filename}")

def process_pdf_file(pdf_path: Path, output_dir: Path):
    """개별 PDF 파일 처리"""
    print(f"\n📖 PDF 처리 시작: {pdf_path.name}")
    
    try:
        # PDF 전처리기 초기화
        preprocessor = PDFPreprocessor()
        
        # PDF에서 문제 추출
        problems = preprocessor.extract_problems_with_pdfplumber([str(pdf_path)])
        
        if not problems:
            print(f"⚠️ {pdf_path.name}에서 문제를 추출할 수 없습니다.")
            return
        
        print(f"✅ {len(problems)}개 문제 추출 완료")
        
        # 파일명에서 연도와 회차 추출
        filename = pdf_path.stem
        filename_prefix = filename.replace("cut_", "").replace(".", "_")
        
        # 과목별로 그룹화
        subject_groups = group_problems_by_subject(problems)
        
        # 과목별 파일 저장
        for subject, subject_problems in subject_groups.items():
            if subject_problems:
                save_subject_file(subject, subject_problems, output_dir, filename_prefix)
        
        # 전체 문제 파일 저장
        save_all_problems_file(problems, output_dir, filename_prefix)
        
        # 통계 출력
        print(f"\n📊 {filename} 처리 완료:")
        for subject, subject_problems in subject_groups.items():
            if subject_problems:
                print(f"  - {subject}: {len(subject_problems)}개")
        
    except Exception as e:
        print(f"❌ {pdf_path.name} 처리 실패: {e}")

def main():
    """메인 함수"""
    # exam 폴더 경로
    exam_dir = Path(__file__).parent / "exam"
    
    if not exam_dir.exists():
        print(f"❌ exam 폴더를 찾을 수 없습니다: {exam_dir}")
        return
    
    # PDF 파일 목록
    pdf_files = list(exam_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ exam 폴더에 PDF 파일이 없습니다: {exam_dir}")
        return
    
    print(f"📁 exam 폴더에서 {len(pdf_files)}개의 PDF 파일을 발견했습니다.")
    
    # 출력 디렉토리 (exam 폴더 내에 생성)
    output_dir = exam_dir / "parsed_results"
    output_dir.mkdir(exist_ok=True)
    
    print(f"📂 결과 저장 위치: {output_dir}")
    
    # 각 PDF 파일 처리
    for pdf_file in pdf_files:
        process_pdf_file(pdf_file, output_dir)
    
    print(f"\n🎉 모든 PDF 파일 처리 완료!")
    print(f"📁 결과 파일들은 {output_dir} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()

