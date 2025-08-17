#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정보처리기사 선택과목 문제 생성 테스트 스크립트
특정 3과목 × 10문제 = 총 30문제 자동 생성 테스트
"""

import os
import sys
from pathlib import Path

# 현재 파일의 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from generator import InfoProcessingExamAgent

def test_partial_exam_generation():
    """선택과목 문제 생성 테스트"""
    print("=" * 60)
    print("정보처리기사 선택과목 문제 생성 테스트 시작")
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
        for pdf in pdf_files[:3]:  # 처음 3개만 표시
            print(f"   - {os.path.basename(pdf)}")
        if len(pdf_files) > 3:
            print(f"   ... 및 {len(pdf_files) - 3}개 더")
        
        # 선택할 3과목 정의
        selected_subjects = ["소프트웨어설계", "데이터베이스구축", "프로그래밍언어활용"]
        questions_per_subject = 10
        
        print(f"\n3. 선택과목 문제 생성 시작 (병렬 처리: 2개)")
        print(f"   - 선택된 과목: {', '.join(selected_subjects)}")
        print(f"   - 과목당 문제 수: {questions_per_subject}문제")
        print(f"   - 총 목표: {len(selected_subjects) * questions_per_subject}문제")
        
        result = agent.execute({
            "mode": "partial_exam",
            "selected_subjects": selected_subjects,
            "questions_per_subject": questions_per_subject,
            "difficulty": "중급",
            "parallel_agents": 2,
            "save_to_file": True,
            "filename": "선택과목_3과목_30문제_테스트결과.json"
        })
        
        if result.get("success"):
            print("\n4. ✓ 선택과목 문제 생성 성공!")
            exam_result = result["result"]
            
            print(f"\n   📊 생성 결과 요약:")
            print(f"   - 총 생성된 문제: {exam_result.get('total_questions', 0)}개")
            print(f"   - 성공률: {exam_result.get('generation_summary', {}).get('success_rate', 'N/A')}")
            print(f"   - 생성 시간: {exam_result.get('generation_summary', {}).get('generation_time', 'N/A')}")
            print(f"   - 성공한 과목: {exam_result.get('generation_summary', {}).get('successful_subjects', 0)}개")
            print(f"   - 실패한 과목: {exam_result.get('generation_summary', {}).get('failed_subjects', 0)}개")
            
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
            print(f"\n❌ 선택과목 문제 생성 실패: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("선택과목 문제 생성 테스트 완료")
    print("=" * 60)
    return True

def test_different_combinations():
    """다양한 과목 조합으로 테스트"""
    print("\n" + "=" * 50)
    print("다양한 과목 조합 테스트")
    print("=" * 50)
    
    try:
        agent = InfoProcessingExamAgent()
        
        # 테스트할 다양한 조합들
        test_combinations = [
            {
                "name": "핵심 과목 2개",
                "subjects": ["소프트웨어설계", "데이터베이스구축"],
                "questions": 15
            },
            {
                "name": "개발 관련 3개",
                "subjects": ["소프트웨어개발", "프로그래밍언어활용", "정보시스템구축관리"],
                "questions": 8
            },
            {
                "name": "설계 중심 2개",
                "subjects": ["소프트웨어설계", "소프트웨어개발"],
                "questions": 12
            }
        ]
        
        for i, combo in enumerate(test_combinations, 1):
            print(f"\n{i}. {combo['name']} 테스트:")
            print(f"   과목: {', '.join(combo['subjects'])}")
            print(f"   과목당 문제: {combo['questions']}개")
            
            result = agent.execute({
                "mode": "partial_exam",
                "selected_subjects": combo['subjects'],
                "questions_per_subject": combo['questions'],
                "difficulty": "중급",
                "parallel_agents": 2,
                "save_to_file": True,
                "filename": f"{combo['name']}_{len(combo['subjects'])}과목_{combo['questions']*len(combo['subjects'])}문제.json"
            })
            
            if result.get("success"):
                exam_result = result["result"]
                total_generated = exam_result.get('total_questions', 0)
                success_rate = exam_result.get('generation_summary', {}).get('success_rate', 'N/A')
                print(f"   ✓ 성공: {total_generated}문제 생성 ({success_rate})")
                
                if "file_path" in result:
                    print(f"   파일: {os.path.basename(result['file_path'])}")
            else:
                print(f"   ❌ 실패: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"   ❌ 오류: {str(e)}")

if __name__ == "__main__":
    print("정보처리기사 선택과목 문제 생성 테스트 시작")
    print("현재 작업 디렉토리:", os.getcwd())
    
    # 메인 테스트: 3과목 × 10문제
    success = test_partial_exam_generation()
    
    if success:
        print("\n🎉 메인 테스트가 성공적으로 완료되었습니다!")
        
        # 추가 조합 테스트 실행
        test_different_combinations()
        
        print("\n🎉 모든 테스트가 완료되었습니다!")
    else:
        print("\n💥 메인 테스트가 실패했습니다.")
    
    print("\n테스트 완료. 결과 파일들을 확인해보세요.")
