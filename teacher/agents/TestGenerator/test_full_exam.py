#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정보처리기사 전체 문제 생성 테스트 스크립트
5과목 × 20문제 = 총 100문제 자동 생성 테스트
"""

import os
import sys
from pathlib import Path

# 현재 파일의 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from generator import InfoProcessingExamAgent

def test_full_exam_generation():
    """전체 문제 생성 테스트"""
    print("=" * 60)
    print("정보처리기사 전체 문제 생성 테스트 시작")
    print("=" * 60)
    
    try:
        # 에이전트 초기화
        print("1. 에이전트 초기화 중...")
        agent = InfoProcessingExamAgent()
        print(f"   ✓ 에이전트 초기화 완료: {agent.name}")
        print(f"   ✓ 설명: {agent.description}")
        
        # PDF 파일 확인
        pdf_files = agent.get_pdf_files()
        print(f"\n2. PDF 파일 확인: {len(pdf_files)}개 발견")
        for pdf in pdf_files[:5]:  # 처음 5개만 표시
            print(f"   - {os.path.basename(pdf)}")
        if len(pdf_files) > 5:
            print(f"   ... 및 {len(pdf_files) - 5}개 더")
        
        # 전체 문제 생성 테스트 (병렬 처리 2개)
        print(f"\n3. 전체 문제 생성 시작 (병렬 처리: 2개)")
        print("   - 소프트웨어설계: 20문제")
        print("   - 소프트웨어개발: 20문제") 
        print("   - 데이터베이스구축: 20문제")
        print("   - 프로그래밍언어활용: 20문제")
        print("   - 정보시스템구축관리: 20문제")
        print("   - 총 목표: 100문제")
        
        result = agent.execute({
            "mode": "full_exam",
            "difficulty": "중급",
            "parallel_agents": 2,
            "save_to_file": True,
            "filename": "전체문제생성_테스트결과.json"
        })
        
        if result.get("success"):
            print("\n4. ✓ 전체 문제 생성 성공!")
            exam_result = result["result"]
            
            print(f"\n   📊 생성 결과 요약:")
            print(f"   - 총 생성된 문제: {exam_result.get('total_questions', 0)}개")
            print(f"   - 성공률: {exam_result.get('generation_summary', {}).get('success_rate', 'N/A')}")
            print(f"   - 생성 시간: {exam_result.get('generation_summary', {}).get('generation_time', 'N/A')}")
            
            print(f"\n   📚 과목별 결과:")
            for subject, info in exam_result.get("subjects", {}).items():
                status = info.get("status", "UNKNOWN")
                actual = info.get("actual_count", 0)
                requested = info.get("requested_count", 0)
                print(f"   - {subject}: {actual}/{requested} ({status})")
            
            if "file_path" in result:
                print(f"\n   💾 결과 파일 저장됨: {result['file_path']}")
            
            # 실패한 과목이 있다면 표시
            failed_subjects = exam_result.get("generation_summary", {}).get("failed_subjects", [])
            if failed_subjects > 0:
                print(f"\n   ⚠️  실패한 과목: {failed_subjects}개")
                for failed in exam_result.get("failed_subjects", []):
                    print(f"      - {failed.get('subject', 'Unknown')}: {failed.get('error', 'Unknown error')}")
            
        else:
            print(f"\n❌ 전체 문제 생성 실패: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("전체 문제 생성 테스트 완료")
    print("=" * 60)
    return True

def test_subject_quiz_generation():
    """단일 과목 문제 생성 테스트 (참고용)"""
    print("\n" + "=" * 40)
    print("단일 과목 문제 생성 테스트 (참고용)")
    print("=" * 40)
    
    try:
        agent = InfoProcessingExamAgent()
        
        # 소프트웨어설계 과목으로 5문제 테스트
        print("소프트웨어설계 과목 5문제 생성 테스트...")
        result = agent.execute({
            "mode": "subject_quiz",
            "subject_area": "소프트웨어설계",
            "target_count": 5,
            "difficulty": "중급",
            "save_to_file": True,
            "filename": "소프트웨어설계_5문제_테스트.json"
        })
        
        if result.get("success"):
            quiz_result = result["result"]
            print(f"   ✓ 성공: {quiz_result.get('quiz_count', 0)}문제 생성")
            print(f"   - 과목: {quiz_result.get('subject_area', 'N/A')}")
            print(f"   - 난이도: {quiz_result.get('difficulty', 'N/A')}")
            if "file_path" in result:
                print(f"   - 파일: {result['file_path']}")
        else:
            print(f"   ❌ 실패: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"   ❌ 오류: {str(e)}")

if __name__ == "__main__":
    print("정보처리기사 문제 생성 테스트 시작")
    print("현재 작업 디렉토리:", os.getcwd())
    
    # 단일 과목 테스트 먼저 실행 (참고용)
    test_subject_quiz_generation()
    
    # 전체 문제 생성 테스트 실행
    success = test_full_exam_generation()
    
    if success:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n💥 일부 테스트가 실패했습니다.")
    
    print("\n테스트 완료. 결과 파일을 확인해보세요.")
