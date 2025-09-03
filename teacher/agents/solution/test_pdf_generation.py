#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 생성 기능 테스트 스크립트
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from comprehensive_pdf_generator import ComprehensivePDFGenerator

def test_pdf_generation():
    """PDF 생성 기능을 테스트합니다."""
    
    print("🧪 PDF 생성 기능 테스트 시작")
    
    # 테스트용 문제 데이터 생성
    test_problems = [
        {
            "question": "정보처리기사 시험에서 가장 중요한 과목은?",
            "options": ["소프트웨어 설계", "소프트웨어 개발", "데이터베이스 구축", "정보시스템 구축"],
            "generated_answer": "소프트웨어 설계",
            "generated_explanation": "소프트웨어 설계는 전체 시스템의 구조와 동작을 정의하는 핵심 단계로, 이후 모든 개발 과정의 기초가 됩니다."
        },
        {
            "question": "UML 다이어그램 중 시스템의 정적 구조를 나타내는 것은?",
            "options": ["시퀀스 다이어그램", "클래스 다이어그램", "액티비티 다이어그램", "상태 다이어그램"],
            "generated_answer": "클래스 다이어그램",
            "generated_explanation": "클래스 다이어그램은 시스템의 정적 구조를 보여주며, 클래스, 속성, 메서드, 관계 등을 표현합니다."
        },
        {
            "question": "데이터베이스 정규화의 목적은?",
            "options": ["데이터 중복 제거", "데이터 크기 증가", "쿼리 성능 저하", "데이터 일관성 저하"],
            "generated_answer": "데이터 중복 제거",
            "generated_explanation": "정규화는 데이터 중복을 제거하고 데이터 무결성을 향상시키는 것이 주요 목적입니다."
        }
    ]
    
    try:
        # PDF 생성기 초기화
        generator = ComprehensivePDFGenerator()
        
        print(f"📚 테스트 문제 수: {len(test_problems)}개")
        
        # 개별 PDF 생성 테스트
        print("\n1️⃣ 문제집 생성 테스트")
        generator.generate_problem_booklet(test_problems, "test_문제집.pdf", "테스트 문제집")
        
        print("\n2️⃣ 답안집 생성 테스트")
        generator.generate_answer_booklet(test_problems, "test_답안집.pdf", "테스트 답안집")
        
        print("\n3️⃣ 분석 리포트 생성 테스트")
        generator.generate_analysis_report(test_problems, "test_분석리포트.pdf", "테스트 분석 리포트")
        
        # 전체 PDF 생성 테스트
        print("\n4️⃣ 전체 PDF 생성 테스트")
        result_files = generator.generate_all_pdfs(test_problems, "test_종합시험")
        
        print(f"\n✅ 모든 테스트 완료!")
        print(f"생성된 파일들:")
        for file_type, file_path in result_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"   - {file_type}: {file_path} ({file_size:,} bytes)")
            else:
                print(f"   - {file_type}: {file_path} (생성 실패)")
        
        # 테스트 파일 정리
        print(f"\n🧹 테스트 파일 정리 중...")
        test_files = [
            "test_문제집.pdf", "test_답안집.pdf", "test_분석리포트.pdf"
        ]
        for file_path in result_files.values():
            test_files.append(file_path)
        
        for file_path in test_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"   - 삭제됨: {file_path}")
                except Exception as e:
                    print(f"   - 삭제 실패: {file_path} - {e}")
        
        print(f"\n🎉 PDF 생성 기능 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_generation()
