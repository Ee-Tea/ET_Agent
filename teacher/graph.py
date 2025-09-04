import os
from ..supervisor import MainOrchestrator

USER_ID  = os.getenv("TEST_USER_ID", "demo_user")
SERVICE  = os.getenv("TEST_SERVICE", "teacher")
CHAT_ID  = os.getenv("TEST_CHAT_ID", "local")

# 오케스트레이터 & 그래프 컴파일
orch = MainOrchestrator(user_id=USER_ID, service=SERVICE, chat_id=CHAT_ID)
graph = orch.build_teacher_graph()