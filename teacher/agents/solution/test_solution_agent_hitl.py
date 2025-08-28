#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Agent HITL 테스트 파일

이 파일은 solution_agent_hitl.py의 풀이 평가 및 개선 워크플로우를 테스트합니다.
"""

import os
import sys
import json
from typing import Dict, Any

from langchain_milvus import Milvus

# 현재 디렉토리에서 solution_agent_hitl.py를 직접 import
from .solution_agent_hitl import SolutionAgent, SolutionState
from langchain.schema import Document

def create_test_state() -> SolutionState:
    """테스트용 상태를 생성합니다."""
    
    # 테스트 문제 (사용자가 제공한 문제)
    test_problem = "소프트웨어 설계에서 사용되는 대표적인 추상화 기법이 아닌 것은?"
    test_options = ["자료 추상화", "제어 추상화", "과정 추상화", "강도 추상화"]
    
    # 테스트 모드 설정
    test_mode = True
    test_score = 35  # 낮은 점수로 설정하여 개선이 필요하도록 함
    test_feedback_type = "term_explanation"  # 용어 설명이 필요한 것으로 설정
    
    return {
        "user_input_txt": f"{test_problem}\n{chr(10).join([f'{i+1}. {opt}' for i, opt in enumerate(test_options)])}",
        "user_problem": test_problem,
        "user_problem_options": test_options,
        "vectorstore_p": Milvus,  # 테스트용으로는 벡터스토어 사용 안함
        "vectorstore_c": Milvus,
        "retrieved_docs": [],
        "problems_contexts_text": "",
        "concept_contexts": [],
        "concept_contexts_text": "",
        "generated_answer": "",  # LLM이 생성할 예정
        "generated_explanation": "",  # LLM이 생성할 예정
        "generated_subject": "",  # LLM이 생성할 예정
        "results": [],
        "validated": True,  # 검증은 통과했다고 가정
        "retry_count": 0,
        "chat_history": [],
        "solution_score": 0.0,
        "feedback_analysis": "",
        "needs_improvement": False,
        "improved_solution": "",
        "search_results": "",
        "test_mode": test_mode,
        "test_score": test_score,
        "test_feedback_type": test_feedback_type
    }

def generate_solution_with_llm(agent: SolutionAgent, state: SolutionState) -> SolutionState:
    """LLM을 사용하여 실제 풀이를 생성합니다."""
    print("🤖 [LLM] 풀이 생성을 시작합니다...")
    
    try:
        # LLM을 사용하여 풀이 생성
        llm = agent._llm(0.7)
        
        # 풀이 생성 프롬프트
        prompt = f"""
        다음 문제에 대한 답과 풀이를 생성해주세요.

        문제: {state['user_problem']}
        보기:
        {chr(10).join([f'{i+1}. {opt}' for i, opt in enumerate(state['user_problem_options'])])}

        다음 형식으로 응답해주세요:
        {{
            "answer": "정답 번호 (1, 2, 3, 4 중 하나)",
            "explanation": "상세한 풀이 설명",
            "subject": "과목명"
        }}

        풀이는 단계별로 명확하게 설명하고, 각 보기에 대한 분석을 포함해주세요.
        """
        
        response = llm.invoke(prompt)
        
        # JSON 파싱
        import json
        try:
            result = json.loads(response.content)
            state["generated_answer"] = result.get("answer", "4")
            state["generated_explanation"] = result.get("explanation", "")
            state["generated_subject"] = result.get("subject", "소프트웨어 설계")
            
            print(f"✅ 풀이 생성 완료:")
            print(f"  - 정답: {state['generated_answer']}")
            print(f"  - 과목: {state['generated_subject']}")
            print(f"  - 풀이 길이: {len(state['generated_explanation'])} 문자")
            
        except json.JSONDecodeError:
            print("⚠️ LLM 응답을 JSON으로 파싱할 수 없습니다. 기본값을 사용합니다.")
            state["generated_answer"] = "4"
            state["generated_explanation"] = response.content
            state["generated_subject"] = "소프트웨어 설계"
            
    except Exception as e:
        print(f"❌ LLM 풀이 생성 중 오류 발생: {e}")
        # 오류 발생 시 기본 풀이 사용
        state["generated_answer"] = "4"
        state["generated_explanation"] = """
        추상화는 복잡한 시스템을 단순화하는 기법입니다.
        자료 추상화는 데이터의 구조를 추상화하는 것이고,
        제어 추상화는 제어 흐름을 추상화하는 것입니다.
        과정 추상화는 프로세스를 추상화하는 것입니다.
        강도 추상화는 존재하지 않는 추상화 기법입니다.
        """
        state["generated_subject"] = "소프트웨어 설계"
    
    return state

def test_solution_evaluation():
    """풀이 평가 기능을 테스트합니다."""
    print("🧪 === 풀이 평가 테스트 시작 ===")
    
    # SolutionAgent 인스턴스 생성
    agent = SolutionAgent()
    
    # 테스트 상태 생성
    state = create_test_state()
    
    print(f"📝 테스트 문제: {state['user_problem']}")
    print(f"📝 테스트 보기: {state['user_problem_options']}")
    
    # LLM을 사용하여 실제 풀이 생성
    print("\n🤖 LLM을 사용하여 풀이를 생성합니다...")
    state = generate_solution_with_llm(agent, state)
    
    print(f"\n📝 생성된 풀이: {state['generated_explanation'][:200]}...")
    print(f"🧪 테스트 모드: {state['test_mode']}")
    print(f"🧪 강제 점수: {state['test_score']}")
    print(f"🧪 강제 피드백 타입: {state['test_feedback_type']}")
    
    # 풀이 평가 실행
    print("\n" + "="*50)
    state = agent._evaluate_solution(state)
    
    print(f"\n📊 평가 결과:")
    print(f"  - 점수: {state['solution_score']}/100")
    print(f"  - 개선 필요: {state['needs_improvement']}")
    print(f"  - 피드백 분석: {state['feedback_analysis']}")
    
    return state

def test_user_feedback_collection(state: SolutionState):
    """사용자로부터 피드백을 수집하고 분석합니다."""
    print("\n🧪 === 사용자 피드백 수집 테스트 시작 ===")
    
    agent = SolutionAgent()
    
    # 실제 interrupt 실행 (에러가 날 것을 예상하고 테스트)
    print("💬 [실제 실행] 사용자 피드백을 입력해주세요.")
    print("   (interrupt가 실행되어 사용자 입력을 기다립니다)")
    
    try:
        # 실제 user_feedback 도구 호출하여 interrupt 실행
        feedback_result = agent.user_feedback("풀이에 대한 의견을 자유롭게 입력해주세요.")
        
        # 피드백 분석 결과 저장
        state["feedback_analysis"] = feedback_result
        
        print(f"💬 사용자 피드백 수집 완료: {feedback_result}")
        
    except Exception as e:
        print(f"⚠️ 피드백 수집 중 오류 발생: {e}")
        # 오류 발생 시 테스트용 피드백 설정
        state["feedback_analysis"] = "term_explanation"
        print(f"💬 테스트용 피드백으로 설정: {state['feedback_analysis']}")
    
    return state

def test_solution_improvement(state: SolutionState):
    """풀이 개선 기능을 테스트합니다."""
    print("\n🧪 === 풀이 개선 테스트 시작 ===")
    
    agent = SolutionAgent()
    
    # 풀이 개선 실행
    state = agent._improve_solution(state)
    
    print(f"\n🔧 개선 결과:")
    if state.get("improved_solution"):
        print(f"  - 개선된 풀이: {state['improved_solution'][:200]}...")
    else:
        print("  - 개선된 풀이가 생성되지 않았습니다.")
    
    return state

def test_additional_info_search(state: SolutionState):
    """추가 정보 검색 기능을 테스트합니다."""
    print("\n🧪 === 추가 정보 검색 테스트 시작 ===")
    
    agent = SolutionAgent()
    
    # 추가 정보 검색 실행
    state = agent._search_additional_info(state)
    
    print(f"\n🔍 검색 결과:")
    if state.get("search_results"):
        print(f"  - 검색된 정보: {state['search_results'][:200]}...")
    else:
        print("  - 검색된 정보가 없습니다.")
    
    return state

def test_solution_finalization(state: SolutionState):
    """최종 풀이 정리 기능을 테스트합니다."""
    print("\n🧪 === 최종 풀이 정리 테스트 시작 ===")
    
    agent = SolutionAgent()
    
    # 최종 풀이 정리 실행
    state = agent._finalize_solution(state)
    
    print(f"\n✨ 최종 풀이:")
    print(f"  - 최종 풀이: {state['generated_explanation'][:300]}...")
    
    return state

def run_full_test():
    """전체 워크플로우를 테스트합니다."""
    print("🚀 === 전체 워크플로우 테스트 시작 ===")
    print("이 테스트는 풀이 평가부터 최종 풀이 정리까지의 전체 과정을 검증합니다.\n")
    
    try:
        # 1. 풀이 평가 테스트
        state = test_solution_evaluation()
        
        # 2. 사용자 피드백 수집 테스트
        state = test_user_feedback_collection(state)
        
        # 3. 풀이 개선 테스트
        state = test_solution_improvement(state)
        
        # 4. 추가 정보 검색 테스트
        state = test_additional_info_search(state)
        
        # 5. 최종 풀이 정리 테스트
        state = test_solution_finalization(state)
        
        print("\n🎉 === 전체 테스트 완료 ===")
        print("모든 단계가 성공적으로 실행되었습니다!")
        
        # 최종 결과 요약
        print(f"\n📋 최종 결과 요약:")
        print(f"  - 원본 풀이 길이: {len(state.get('generated_explanation', ''))} 문자")
        print(f"  - 개선된 풀이 길이: {len(state.get('improved_solution', ''))} 문자")
        print(f"  - 검색 결과 길이: {len(state.get('search_results', ''))} 문자")
        print(f"  - 최종 풀이 길이: {len(state.get('generated_explanation', ''))} 문자")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 전체 테스트 실행
    run_full_test()
