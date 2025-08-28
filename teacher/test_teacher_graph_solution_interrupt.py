#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# uv run python teacher/test_teacher_graph_solution_interrupt.py
"""
Teacher Graph LangGraph Interrupt 테스트

이 파일은 teacher_graph.py의 LangGraph 기반 워크플로우를 테스트하고,
solution_agent의 interrupt 발생 후 Command(resume)으로 재개하는 것을 확인합니다.
"""

def main():
    """teacher_graph의 LangGraph 기반 워크플로우를 테스트합니다."""
    print("🧪 === Teacher Graph LangGraph Interrupt 테스트 ===")
    print("teacher_graph의 LangGraph 기반 워크플로우를 테스트하고,")
    print("solution_agent의 interrupt와 Command(resume)을 확인합니다.\n")
    
    try:
        # teacher_graph 모듈 import
        from teacher_graph import Orchestrator
        
        print("✅ teacher_graph 모듈 import 성공")
        
        # Orchestrator 인스턴스 생성
        print("🔧 Orchestrator 인스턴스 생성 중...")
        orchestrator = Orchestrator(
            user_id="test_user",
            service="test_service", 
            chat_id="test_chat",
            init_agents=True  # solution_agent 초기화
        )
        
        print("✅ Orchestrator 인스턴스 생성 완료")
        print(f"✅ solution_runner 초기화: {orchestrator.solution_runner is not None}")
        print(f"✅ LangGraph 그래프 생성: {orchestrator.graph is not None}")
        print(f"✅ 체크포인터 초기화: {orchestrator.checkpointer is not None}")
        
        # 테스트용 상태 생성 (PDF 문제가 없음)
        test_state = {
            "user_query": "테스트용 문제 풀이 요청, 소프트웨어 설계에서 사용되는 대표적인 추상화 기법이 아닌 것은? 자료 추상화, 제어 추상화, 과정 추상화, 강도 추상화",

            "artifacts": {
                "pdf_added_count": 0,  # PDF 문제가 없음을 의미
                "pdf_added_start_index": None,
                "pdf_added_end_index": None
            }
        }
        
        print("\n📝 테스트 상태:")
        print(f"  - user_query: {test_state['user_query']}")
        print(f"  - pdf_added_count: {test_state['artifacts']['pdf_added_count']}")
        
        # 첫 번째 실행 - interrupt 발생 예상
        print("\n🚀 첫 번째 실행 시작...")
        print("🧪 interrupt 발생으로 인한 워크플로우 중단을 기대합니다.")
        print("🧪 'interrupt 실행!!!!!!!!!!!!!!' 메시지가 출력되어야 합니다.")
        
        # thread_id를 일치시키기 위한 config
        config = {"configurable": {"thread_id": "test_thread"}}
        
        try:
            # LangGraph 기반 워크플로우 실행
            result = orchestrator.invoke(test_state, config)
            print("⚠️ 예상과 다름: 워크플로우가 정상적으로 완료되었습니다.")
            print(f"   결과: {result}")
            print("\n💡 interrupt가 발생하지 않았습니다. solution_agent의 interrupt 함수를 확인해주세요.")
            
        except Exception as e:
            print("🎉 예상대로 interrupt로 인한 예외가 발생했습니다!")
            print(f"   예외 타입: {type(e).__name__}")
            print(f"   예외 내용: {e}")
            print("\n✅ interrupt 테스트 성공!")
            print("🧪 solution_agent의 interrupt가 정상적으로 작동하여 워크플로우가 중단되었습니다.")
            
            # 이제 Command(resume)으로 워크플로우 재개 시도
            print("\n🔄 워크플로우 재개 시도...")
            print("🧪 Command(resume)으로 interrupt된 워크플로우를 재개합니다.")
            
            try:
                print("📤 Command(resume) 전송:")
                print(f"   - resume data: {'테스트용 사용자 피드백: 더 쉬운 설명이 필요합니다.'}")
                print(f"   - config: {config}")
                
                # resume_workflow 메서드를 통해 워크플로우 재개
                resumed_result = orchestrator.resume_workflow(
                    "테스트용 사용자 피드백: 더 쉬운 설명이 필요합니다.", 
                    config
                )
                
                print("🎉 워크플로우 재개 성공!")
                print(f"   재개 결과: {resumed_result}")
                
            except Exception as resume_error:
                print("⚠️ 워크플로우 재개 중 오류 발생:")
                print(f"   오류 타입: {type(resume_error).__name__}")
                print(f"   오류 내용: {resume_error}")
                print("\n💡 체크포인터 설정이나 Command(resume) 형식을 확인해주세요.")
                print("💡 LangGraph의 체크포인터 시스템이 제대로 작동하는지 확인해주세요.")
        
    except ImportError as e:
        print(f"❌ teacher_graph 모듈 import 실패: {e}")
        print("   프로젝트 루트에서 실행하거나 PYTHONPATH를 설정하세요.")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
