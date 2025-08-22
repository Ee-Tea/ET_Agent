import os
from .teacher_graph import Orchestrator
# 테스트용 식별자 (환경변수로 바꿔도 됩니다)
USER_ID  = os.getenv("TEST_USER_ID", "demo_user")
SERVICE  = os.getenv("TEST_SERVICE", "teacher")
CHAT_ID  = os.getenv("TEST_CHAT_ID", "local")

# 오케스트레이터 & 그래프 컴파일
orch = Orchestrator(user_id=USER_ID, service=SERVICE, chat_id=CHAT_ID)
graph = orch.build_teacher_graph()