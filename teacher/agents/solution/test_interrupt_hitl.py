#!/usr/bin/env python3
"""
Interrupt 기반 HITL 피드백 시스템 테스트 스크립트
LangGraph의 interrupt와 Command를 사용하여 실행 일시 중단 및 재개를 테스트합니다.
"""

import os
import sys
from dotenv import load_dotenv

# relative import 사용
from .solution_agent_hitl import SolutionAgent
from langgraph.types import Command

def test_interrupt_hitl():
    """Interrupt 기반 HITL 시스템 테스트"""
    print("🚀 Interrupt 기반 HITL 시스템 테스트")
    print("=" * 60)
    
    # HITL 모드 설정 (manual로 설정하여 항상 피드백 수집)
    agent = SolutionAgent(max_interactions=5, hitl_mode="manual")
    
    # 테스트용 문제 설정
    user_input_txt = "프로세스와 스레드의 차이점을 이해하고 싶습니다."
    user_problem = "프로세스와 스레드의 차이점으로 올바른 것은?"
    user_problem_options = [
        "프로세스는 독립적인 메모리 공간을 가지며, 스레드는 프로세스 내에서 메모리를 공유한다",
        "프로세스와 스레드는 모두 독립적인 메모리 공간을 가진다",
        "프로세스는 메모리를 공유하고, 스레드는 독립적인 메모리 공간을 가진다",
        "프로세스와 스레드는 모두 메모리를 공유한다"
    ]
    
    print(f"📝 테스트 문제:")
    print(f"질문: {user_input_txt}")
    print(f"문제: {user_problem}")
    print(f"보기:")
    for i, option in enumerate(user_problem_options, 1):
        print(f"  {i}. {option}")
    print()
    
    try:
        # 1단계: 에이전트 실행 시작
        print("🔍 [1단계] 에이전트 실행 시작...")
        print("⚠️ interrupt가 호출되면 실행이 일시 중단됩니다.")
        
        # 에이전트 실행 (벡터스토어 없이)
        final_state = agent.invoke(
            user_input_txt=user_input_txt,
            user_problem=user_problem,
            user_problem_options=user_problem_options,
            vectorstore=None,  # 테스트를 위해 벡터스토어 없이 실행
        )
        
        print("\n✅ 에이전트 실행 완료!")
        
    except Exception as e:
        if "interrupt" in str(e).lower():
            print(f"\n⏸️ [HITL] 실행이 일시 중단되었습니다: {e}")
            print("사용자 피드백을 입력하여 실행을 재개할 수 있습니다.")
            
            # 2단계: 사용자 피드백 입력
            print("\n💬 [2단계] 사용자 피드백 입력")
            print("다음 중 하나를 선택하거나 자유롭게 입력하세요:")
            print("1. 이해됨 - '이해가 됩니다', '좋습니다', '만족합니다'")
            print("2. 더 쉬운 풀이 필요 - '더 쉽게 설명해주세요', '복잡해요'")
            print("3. 용어 설명 필요 - '이 용어가 뭔지 모르겠어요', '설명이 부족해요'")
            
            user_feedback = input("\n💬 피드백을 입력하세요: ").strip()
            
            if not user_feedback:
                user_feedback = "풀이를 더 쉽게 설명해주세요"
                print(f"⚠️ 입력이 없어 기본값을 사용합니다: {user_feedback}")
            
            # 3단계: Command 객체를 사용하여 실행 재개
            print(f"\n🔄 [3단계] 실행 재개 중...")
            print(f"사용자 피드백: {user_feedback}")
            
            # Command 객체 생성
            command = Command(resume={"data": user_feedback})
            
            # 실행 재개
            final_state = agent.invoke(
                user_input_txt=user_input_txt,
                user_problem=user_problem,
                user_problem_options=user_problem_options,
                vectorstore=None,
                command=command,  # Command 객체 전달
            )
            
            print("\n✅ HITL 실행 재개 완료!")
            
        else:
            print(f"❌ 예상치 못한 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 최종 결과 출력
    print("\n" + "=" * 60)
    print("🎯 최종 결과:")
    print(f"문제: {final_state.get('user_problem', '')}")
    print(f"정답: {final_state.get('generated_answer', '')}")
    print(f"풀이: {final_state.get('generated_explanation', '')}")
    print(f"과목: {final_state.get('generated_subject', '')}")
    print(f"상호작용 횟수: {final_state.get('interaction_count', 0)}")
    print(f"사용자 피드백: {final_state.get('user_feedback', '')}")
    print(f"피드백 유형: {final_state.get('feedback_type', '')}")
    
    # 품질 점수 출력
    quality_scores = final_state.get('quality_scores', {})
    if quality_scores:
        print(f"\n📊 품질 점수:")
        for key, score in quality_scores.items():
            print(f"  {key}: {score:.1f}/100")
        print(f"  총점: {final_state.get('total_quality_score', 0):.1f}/100")
    
    # 채팅 히스토리 출력
    chat_history = final_state.get('chat_history', [])
    if chat_history:
        print(f"\n💬 상호작용 히스토리:")
        for i, chat in enumerate(chat_history, 1):
            print(f"  {i}. {chat[:100]}...")

def test_command_resume():
    """Command 객체를 사용한 실행 재개 테스트"""
    print("\n🧪 Command 객체를 사용한 실행 재개 테스트")
    print("=" * 60)
    
    agent = SolutionAgent(max_interactions=3, hitl_mode="manual")
    
    # 테스트용 문제
    user_input_txt = "데이터베이스 정규화에 대해 알고 싶습니다."
    user_problem = "데이터베이스 정규화의 목적은?"
    user_problem_options = [
        "데이터 중복 제거 및 일관성 유지",
        "데이터 크기 증가",
        "쿼리 속도 저하",
        "복잡성 증가"
    ]
    
    print(f"📝 테스트 문제: {user_problem}")
    
    try:
        # 1단계: 실행 시작
        print("\n🔍 [1단계] 실행 시작...")
        final_state = agent.invoke(
            user_input_txt=user_input_txt,
            user_problem=user_problem,
            user_problem_options=user_problem_options,
            vectorstore=None,
        )
        
        print("✅ 실행 완료!")
        
    except Exception as e:
        if "interrupt" in str(e).lower():
            print(f"\n⏸️ 실행 중단됨: {e}")
            
            # 다양한 피드백으로 테스트
            test_feedbacks = [
                "이해가 됩니다. 만족합니다.",
                "풀이를 더 쉽게 설명해주세요.",
                "정규화라는 용어가 뭔지 모르겠어요."
            ]
            
            for i, feedback in enumerate(test_feedbacks, 1):
                print(f"\n📝 테스트 {i}: {feedback}")
                
                # Command 객체 생성 및 실행 재개
                command = Command(resume={"data": feedback})
                
                try:
                    final_state = agent.invoke(
                        user_input_txt=user_input_txt,
                        user_problem=user_problem,
                        user_problem_options=user_problem_options,
                        vectorstore=None,
                        command=command,
                    )
                    
                    print(f"✅ 테스트 {i} 완료")
                    print(f"  피드백 유형: {final_state.get('feedback_type', '')}")
                    
                except Exception as resume_error:
                    print(f"❌ 테스트 {i} 실패: {resume_error}")
        else:
            print(f"❌ 오류: {e}")

def test_search_with_feedback():
    """유저 피드백이 포함된 검색 기능 테스트"""
    print("\n🔍 유저 피드백이 포함된 검색 기능 테스트")
    print("=" * 60)
    
    agent = SolutionAgent(max_interactions=3, hitl_mode="manual")
    
    # 테스트용 문제
    user_input_txt = "데이터베이스 정규화에 대해 알고 싶습니다."
    user_problem = "데이터베이스 정규화의 목적은?"
    user_problem_options = [
        "데이터 중복 제거 및 일관성 유지",
        "데이터 크기 증가",
        "쿼리 속도 저하",
        "복잡성 증가"
    ]
    
    print(f"📝 테스트 문제: {user_problem}")
    
    # 용어 설명이 필요한 피드백으로 테스트
    test_feedback = "정규화라는 용어가 뭔지 모르겠어요. 더 자세히 설명해주세요."
    print(f"💬 테스트 피드백: {test_feedback}")
    
    try:
        # 1단계: 실행 시작
        print("\n🔍 [1단계] 실행 시작...")
        final_state = agent.invoke(
            user_input_txt=user_input_txt,
            user_problem=user_problem,
            user_problem_options=user_problem_options,
            vectorstore=None,
        )
        
        print("✅ 실행 완료!")
        
    except Exception as e:
        if "interrupt" in str(e).lower():
            print(f"\n⏸️ 실행 중단됨: {e}")
            
            # Command 객체 생성 및 실행 재개
            command = Command(resume={"data": test_feedback})
            
            try:
                final_state = agent.invoke(
                    user_input_txt=user_input_txt,
                    user_problem=user_problem,
                    user_problem_options=user_problem_options,
                    vectorstore=None,
                    command=command,
                )
                
                print(f"✅ 검색 테스트 완료")
                print(f"  피드백 유형: {final_state.get('feedback_type', '')}")
                print(f"  검색 결과: {final_state.get('search_results', '')[:200]}...")
                
            except Exception as resume_error:
                print(f"❌ 검색 테스트 실패: {resume_error}")
        else:
            print(f"❌ 오류: {e}")

def test_graph_state():
    """그래프 상태 확인 테스트"""
    print("\n🔍 그래프 상태 확인 테스트")
    print("=" * 60)
    
    agent = SolutionAgent(max_interactions=2, hitl_mode="manual")
    
    # 그래프 상태 확인
    print(f"📊 그래프 정보:")
    print(f"  노드 수: {len(agent.graph.nodes)}")
    print(f"  그래프 타입: {type(agent.graph)}")
    print(f"  체크포인터: {type(agent.memory).__name__}")
    
    # 메모리 상태 확인
    print(f"\n💾 메모리 상태:")
    print(f"  메모리 타입: {type(agent.memory)}")
    print(f"  메모리 사용 가능: {agent.memory is not None}")

if __name__ == "__main__":
    load_dotenv()
    
    print("🎯 Interrupt 기반 HITL 시스템 테스트")
    print("1. 기본 HITL 테스트")
    print("2. Command 재개 테스트")
    print("3. 그래프 상태 확인")
    print("4. 유저 피드백 검색 테스트")
    
    choice = input("\n어떤 테스트를 실행하시겠습니까? (1-4, 기본값: 1): ").strip()
    
    if choice == "2":
        test_command_resume()
    elif choice == "3":
        test_graph_state()
    elif choice == "4":
        test_search_with_feedback()
    else:
        test_interrupt_hitl()
    
    print("\n✅ 모든 테스트 완료!")
    print("\n💡 주요 특징:")
    print("- interrupt를 사용한 실행 일시 중단")
    print("- Command 객체를 통한 실행 재개")
    print("- 체크포인터를 통한 상태 지속성")
    print("- 3가지 피드백 카테고리 자동 분류")
    print("- 유저 피드백이 포함된 검색 쿼리")
    print("- retrieve_agent.invoke() 메서드 사용")
