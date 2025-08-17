#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generator.py 테스트용 스크립트
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "teacher")))

from agents.TestGenerator.generator import InfoProcessingExamAgent

def test_generator():
    """generator.py의 기본 기능을 테스트합니다."""
    print("=== InfoProcessingExamAgent 테스트 시작 ===")
    
    try:
        # 에이전트 초기화
        print("1. 에이전트 초기화 중...")
        agent = InfoProcessingExamAgent()
        print("   ✓ 에이전트 초기화 성공")
        
        # subject_quiz 모드 테스트
        print("\n2. subject_quiz 모드 테스트...")
        test_input = {
            "mode": "subject_quiz",
            "subject_area": "소프트웨어설계",
            "target_count": 3,
            "difficulty": "중급",
            "save_to_file": False
        }
        
        print(f"   입력: {test_input}")
        result = agent.execute(test_input)
        print(f"   결과: {result}")
        
        if result.get("success"):
            print("   ✓ subject_quiz 모드 성공")
        else:
            print(f"   ✗ subject_quiz 모드 실패: {result.get('error')}")
            
    except Exception as e:
        print(f"   ✗ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generator()
