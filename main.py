from teacher.teacher import Teacher, TeacherState
from farmer.farmer import Farmer, RouterState
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph
from langgraph.prebuilt import InMemorySaver
from langgraph.graph import END, START
from langgraph.graph.message import Message
from langgraph.prebuilt import MergeIntoDict
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
import os, sys
from openai import OpenAI

load_dotenv()
llm_api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("OPENAI_LLM_MODE","llama4-scout-17b-16e-instruct")
llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
llm_base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
class MainState(TypedDict):
    user_query: str
    chat_service: str
    llm_answer: str
    
    teacher_state: NotRequired[TeacherState]
    farmer_state: NotRequired[RouterState]
    
class Orchestrator:
    def __init__(self):
        self.teacher = Teacher(user_id="demo_user", service="teacher", chat_id="local")
        self.farmer = Farmer(user_id="demo_user", service="farmer", chat_id="local")
        self.graph = self._create_graph()
        self.graph.compile()
        
    def select_service(self, state: MainState) -> MainState:
        return state
        
    def run_teacher(self, state: MainState) -> MainState:
        teacher = Teacher(user_id="demo_user", service="teacher", chat_id="local")
        teacher.run()
        return self.teacher.run(state)
    
    def run_farmer(self, state: MainState) -> MainState:
        return self.farmer.run(state)
    
    def export_graph(self):
        return self.graph.get_graph().draw_mermaid_png()
    
    
    def _create_graph(self):
        builder = StateGraph(MainState)




def main():
    print("Hello from et-agent!")


if __name__ == "__main__":
    main()
