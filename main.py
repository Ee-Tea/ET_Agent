import os
import argparse
from typing import Dict, Any

from dotenv import load_dotenv

from supervisor import MainOrchestrator


def interactive_loop(user_id: str, chat_id: str) -> None:
    orchestrator = MainOrchestrator(user_id=user_id, chat_id=chat_id)
    print("\n=== Main Orchestrator (이미지 워크플로우 기반) ===")
    print("질문을 입력하세요. (종료: exit/quit, 세션 초기화: clear)\n")
    while True:
        try:
            q = input("Q> ").strip()
        except EOFError:
            print("\n[EOF] 종료합니다.")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("종료합니다.")
            break
        try:
            result = orchestrator.process_query(q)
            print("\n--- 결과 ---")
            print(result)
            print("--------------\n")
        except Exception as e:
            print("\n[ERROR] 워크플로우 실행 중 예외가 발생했습니다:")
            print(e)
            continue


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Main Orchestrator runner")
    parser.add_argument("--user-id", dest="user_id", default=os.getenv("MAIN_USER_ID", "main_user"))
    parser.add_argument("--chat-id", dest="chat_id", default=os.getenv("MAIN_CHAT_ID", "main_chat"))
    parser.add_argument("--query", dest="query", default=None, help="Run a single query in one-shot mode")
    args = parser.parse_args()

    if args.query:
        orchestrator = MainOrchestrator(user_id=args.user_id, chat_id=args.chat_id)
        try:
            result = orchestrator.process_query(args.query)
            print(result)
        except Exception as e:
            print("[ERROR] 워크플로우 실행 중 예외가 발생했습니다:")
            print(e)
    else:
        interactive_loop(args.user_id, args.chat_id)


if __name__ == "__main__":
    main()


