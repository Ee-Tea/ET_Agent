import os
from ..supervisor import MainOrchestrator

# 오케스트레이터 & 그래프 컴파일
def graph():
    """
    LangGraph API가 호출하는 그래프 팩토리 함수.
    환경변수는 여기서 읽고, 오케스트레이터를 지연 초기화합니다.
    """
    user_id = os.getenv("TEST_USER_ID", "demo_user")
    chat_id = os.getenv("TEST_CHAT_ID", "local")

    orch = MainOrchestrator(user_id=user_id, chat_id=chat_id)
    return orch._create_graph()