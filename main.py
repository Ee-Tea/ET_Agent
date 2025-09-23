import os
import argparse
import sys
from typing import Dict, Any

from dotenv import load_dotenv

from supervisor import MainOrchestrator


def interactive_loop(user_id: str, chat_id: str) -> None:
    """대화형 루프 실행"""
    try:
        orchestrator = MainOrchestrator(user_id=user_id, chat_id=chat_id)
        print("\n=== Main Orchestrator (LangGraph 기반) ===")
        print("질문을 입력하세요. (종료: exit/quit, 세션 초기화: clear)")
        print("지원 서비스: Teacher (IT 교육), Farmer (농업 재배)\n")
        
        while True:
            try:
                q = input("Q> ").strip()
            except EOFError:
                print("\n[EOF] 종료합니다.")
                break
            except KeyboardInterrupt:
                print("\n[Ctrl+C] 종료합니다.")
                break
                
            if not q:
                continue
                
            if q.lower() in {"exit", "quit", "종료"}:
                print("종료합니다.")
                break
                
            try:
                result = orchestrator.process_query(q)
                print("\n--- 결과 ---")
                print(result)
                print("--------------\n")
            except Exception as e:
                print(f"\n[ERROR] 워크플로우 실행 중 예외가 발생했습니다:")
                print(f"오류: {e}")
                print("다시 시도해주세요.\n")
                continue
                
    except Exception as e:
        print(f"[ERROR] 오케스트레이터 초기화 실패: {e}")
        sys.exit(1)


def main():
    """메인 함수"""
    load_dotenv()
    
    # 환경 변수 확인
    if not os.getenv("OPENAI_API_KEY=REDACTED("[ERROR] OPENAI_API_KEY=REDACTED("환경 변수를 설정하거나 .env 파일에 OPENAI_API_KEY=REDACTED(1)
    
    parser = argparse.ArgumentParser(
        description="Main Orchestrator - LangGraph 기반 서비스 라우터",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                                    # 대화형 모드
  python main.py --query "데이터베이스 문제 만들어줘"    # 단일 쿼리 실행
  python main.py --user-id user123 --chat-id chat456 # 사용자/채팅 ID 지정
        """
    )
    
    parser.add_argument(
        "--user-id", 
        dest="user_id", 
        default=os.getenv("MAIN_USER_ID", "main_user"),
        help="사용자 ID (기본값: main_user)"
    )
    parser.add_argument(
        "--chat-id", 
        dest="chat_id", 
        default=os.getenv("MAIN_CHAT_ID", "main_chat"),
        help="채팅 ID (기본값: main_chat)"
    )
    parser.add_argument(
        "--query", 
        dest="query", 
        default=None, 
        help="단일 쿼리 실행 모드 (대화형 모드 대신)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="상세 로그 출력"
    )
    
    args = parser.parse_args()

    if args.query:
        # 단일 쿼리 모드
        try:
            orchestrator = MainOrchestrator(user_id=args.user_id, chat_id=args.chat_id)
            result = orchestrator.process_query(args.query)
            print(result)
        except Exception as e:
            print(f"[ERROR] 워크플로우 실행 중 예외가 발생했습니다:")
            print(f"오류: {e}")
            sys.exit(1)
    else:
        # 대화형 모드
        interactive_loop(args.user_id, args.chat_id)


if __name__ == "__main__":
    main()


