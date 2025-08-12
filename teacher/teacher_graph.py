# teacher_graph.py
import os
from typing import Dict, Any
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict, NotRequired
from copy import deepcopy
from typing import Union
from langsmith import traceable  # (11) LangSmith 추적용 데코레이터

# [주의] 폴더/모듈 경로는 실제 프로젝트 구조에 맞게 조정하세요.
from agents.analysis.analysis_agent import AnalysisAgent, print_analysis_result, score_agent
from agents.base_agent import BaseAgent
from teacher_nodes import user_intent
from ..common.short_term.redis_memory import RedisLangGraphMemory
from agents.retrieve.retrieve_agent import retrieve_agent
from TestGenerator.pdf_quiz_groq_class import generate_agent
from solution.solution_agent import solution_agent

class SupportsExecute:
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...

retriever_runner: SupportsExecute
generator_runner: SupportsExecute
solution_runner: SupportsExecute
score_runner: SupportsExecute
analyst_runner: SupportsExecute

# --- 에이전트 인스턴스: 1회 생성 후 재사용 ---
retriever_runner = retrieve_agent()
generator_runner = generate_agent()
solution_runner  = solution_agent()
score_runner     = score_agent()
analyst_runner   = AnalysisAgent()

# ---------- 타입 ----------
class SharedState(TypedDict):
    question: NotRequired[list[str]]
    options: NotRequired[list[list[str]]]
    answer: NotRequired[list[str]]
    explanation: NotRequired[list[str]]
    subject: NotRequired[list[str]]
    wrong_question: NotRequired[list[str]]
    weak_type: NotRequired[list[str]]
    retrieve_answer: NotRequired[str]
    notes: NotRequired[list[str]]
    user_answer: NotRequired[list[str]]

class TeacherState(TypedDict):
    user_query: str
    intent: str
    shared: NotRequired[SharedState]
    retrieval: NotRequired[dict]
    generation: NotRequired[dict]
    solution: NotRequired[dict]
    score: NotRequired[dict]
    analysis: NotRequired[dict]
    routing: NotRequired[dict]  # ← 복귀용 플래그 저장

# shared 기본값
SHARED_DEFAULTS = {
    "question": [],
    "options": [],
    "answer": [],
    "explanation": [],
    "subject": [],
    "wrong_question": [],
    "weak_type": [],
    "notes": [],
    "user_answer": [],
    "retrieve_answer": ""  # 문자열
}

def ensure_shared(state: TeacherState) -> TeacherState:
    ns = deepcopy(state) if state else {}
    ns.setdefault("shared", {})
    for key, default_val in SHARED_DEFAULTS.items():
        if key not in ns["shared"] or not isinstance(ns["shared"][key], type(default_val)):
            ns["shared"][key] = deepcopy(default_val)
    return ns

# 리스트 정합성 검사
def validate_qas(shared: SharedState) -> None:
    n = len(shared.get("question", []))
    if not all(len(shared.get(k, [])) == n for k in ("options","answer","explanation","subject")):
        raise ValueError(
            f"[QA 정합성 오류] 길이 불일치: "
            f"q={len(shared.get('question', []))}, "
            f"opt={len(shared.get('options', []))}, "
            f"ans={len(shared.get('answer', []))}, "
            f"exp={len(shared.get('explanation', []))}, "
            f"subj={len(shared.get('subject', []))}"
        )

def safe_execute(agent, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        out = agent.execute(payload)
        return out if isinstance(out, dict) else {}
    except Exception as e:
        print(f"[WARN] agent {getattr(agent, 'name', agent)} failed: {e}")
        return {}
    
# ---- 선행 조건 체크 helpers ----
def has_questions(state: TeacherState) -> bool:
    sh = (state.get("shared") or {})
    return bool(sh.get("question")) and bool(sh.get("options"))

def has_solution_answers(state: TeacherState) -> bool:
    # solution 단계 산출물(모델이 낸 답/해설)이 shared.answer/explanation에 누적된 경우로 간주
    sh = (state.get("shared") or {})
    return bool(sh.get("answer")) and bool(sh.get("explanation"))

def has_score(state: TeacherState) -> bool:
    # score 노드가 뭔가 산출한 상태(간단 체크)
    sc = state.get("score") or {}
    return bool(sc)  # 필요시 키 구체화

def has_files_to_preprocess(state: TeacherState) -> bool:
    # 파일 기반 입력을 나중에 쓰실 수 있도록 훅 제공(현재는 False 고정)
    # ex) state.get("files") or state.get("upload_ids")
    return False


# ---------- 오케스트레이터 ----------
class Orchestrator:
    def __init__(self, user_id: str, service: str, chat_id: str):
        load_dotenv()
        if not os.getenv("LANGCHAIN_API_KEY"):
            print("경고: LANGCHAIN_API_KEY 환경 변수가 설정되지 않았습니다.")
        self.memory = RedisLangGraphMemory(user_id=user_id, service=service, chat_id=chat_id)

    # memory 래퍼
    def load_state(self, state: TeacherState) -> TeacherState:
        return self.memory.load(state)

    def persist_state(self, state: TeacherState) -> TeacherState:
        self.memory.save(state, state)
        return state
    


    # -------- 노드들 --------
    @traceable(name="teacher.intent_classifier")  # (11)
    def intent_classifier(self, state: TeacherState) -> TeacherState:
        # (12) 입력 유효성 점검 + 트리밍
        user_query = (state.get("user_query") or "").strip()
        # 비어 있으면 안전 폴백 intent
        if not user_query:
            intent = "retrieve"
        else:
            intent = (user_intent(user_query) or "").lower()
        print(f"사용자 의도 분류: {intent}")
        return {**state, "user_query": user_query, "intent": intent}

    def select_agent(self, state: TeacherState) -> str:
        # (8) 의도 미인식 시 폴백(기본: retrieve)
        mapping = {
            "retrieve": "retrieve",
            "generate": "generator",
            "analyze": "analysis",
            "solution": "solution",
            "score": "score",
        }
        intent = (state.get("intent") or "").lower()
        return mapping.get(intent, "retrieve")

    @traceable(name="teacher.generator")  # (11)
    def generator(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("문제 생성 노드 실행")
        agent_input = {"query": state.get("user_query", "")}
        agent_result = safe_execute(generator_runner, agent_input)

        new_state: TeacherState = {**state}
        new_state.setdefault("generation", {})
        new_state["generation"].update(agent_result)

        # 공유부 누적
        if "validated_questions" in agent_result:
            new_state.setdefault("shared", {})
            shared = new_state["shared"]
            shared.setdefault("question", [])
            shared.setdefault("options", [])
            shared.setdefault("answer", [])
            shared.setdefault("explanation", [])
            shared.setdefault("subject", [])

            for item in agent_result["validated_questions"]:
                if "question" in item and "options" in item:
                    # (13) options 타입 정규화: str -> list[str]
                    opts = item.get("options", [])
                    if isinstance(opts, str):
                        # 줄바꿈/구분자로 들어온 경우 안전 분해
                        lines = [x.strip() for x in opts.splitlines() if x.strip()]
                        opts = lines if lines else [opts.strip()]
                    elif isinstance(opts, list):
                        # 내부 원소가 문자열이 아니라면 문자열로 캐스팅
                        opts = [str(x).strip() for x in opts if str(x).strip()]
                    else:
                        # 알 수 없는 타입이면 빈 리스트
                        opts = []

                    shared["question"].append(item["question"])
                    shared["options"].append(opts)
                    shared["answer"].append(item.get("answer", ""))
                    shared["explanation"].append(item.get("explanation", ""))
                    shared["subject"].append(item.get("subject", ""))

        validate_qas(new_state["shared"])
        return new_state

    @traceable(name="teacher.solution")  # (11)
    def solution(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("문제 풀이 노드 실행")
        new_state: TeacherState = {**state}
        new_state.setdefault("solution", {})
        new_state.setdefault("shared", {})
        shared = new_state["shared"]

        questions = shared.get("question", []) or []
        options_list = shared.get("options", []) or []

        generated_answers: list[str] = []
        generated_explanations: list[str] = []

        for question, options in zip(questions, options_list):
            # (13) 방어적 정규화: 혹시라도 문자열로 들어왔다면 리스트화
            if isinstance(options, str):
                options = [x.strip() for x in options.splitlines() if x.strip()]
                if not options:
                    options = [options.strip()]

            agent_input = {
                "user_question": state.get("user_query", ""),
                "user_problem": question,
                "user_problem_options": options,
            }
            agent_result = safe_execute(solution_runner, agent_input)
            new_state["solution"].update(agent_result or {})

            if agent_result:
                if "generated_answer" in agent_result:
                    generated_answers.append(agent_result["generated_answer"])
                if "generated_explanation" in agent_result:
                    generated_explanations.append(agent_result["generated_explanation"])

        if generated_answers:
            shared.setdefault("answer", [])
            shared["answer"].extend(generated_answers)
        if generated_explanations:
            shared.setdefault("explanation", [])
            shared["explanation"].extend(generated_explanations)

        validate_qas(shared)
        return new_state

    @traceable(name="teacher.score")  # (11)
    def score(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("채점 노드 실행")
        user_query = state.get("user_query", "")
        shared = (state.get("shared") or {})
        solution_answers = shared.get("answer", []) or []

        agent_input = {
            "user_query": user_query,
            "solution_answer": solution_answers,
        }
        agent_result = safe_execute(score_runner, agent_input)

        new_state: TeacherState = {**state}
        new_state.setdefault("score", {})
        new_state["score"].update(agent_result or {})
        return new_state

    @traceable(name="teacher.analysis")  # (11)
    def analysis(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("오답 분석 노드 실행")
        shared = (state.get("shared") or {})
        agent_input = {
            "problem": shared.get("question", []) or [],
            "user_answer": shared.get("user_answer", []) or [],
            "solution_answer": shared.get("answer", []) or [],
            "solution": shared.get("explanation", []) or [],
        }
        agent_result = safe_execute(analyst_runner, agent_input)

        new_state: TeacherState = {**state}
        new_state.setdefault("analysis", {})
        new_state["analysis"].update(agent_result or {})

        if agent_result and "mistake_summary" in agent_result:
            new_state.setdefault("shared", {})
            new_state["shared"]["weak_type"] = agent_result["mistake_summary"]
        return new_state

    @traceable(name="teacher.retrieve")  # (11)
    def retrieve(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("정보 검색 노드 실행")
        agent_input = {"retrieval_question": state.get("user_query", "")}
        agent_result = safe_execute(retriever_runner, agent_input)

        new_state: TeacherState = {**state}
        new_state.setdefault("retrieval", {})
        new_state["retrieval"].update(agent_result or {})

        if agent_result and "retrieve_answer" in agent_result:
            new_state.setdefault("shared", {})
            new_state["shared"]["retrieve_answer"] = agent_result["retrieve_answer"]
        return new_state
        
    # ---- 라우터: 의존성 없으면 선행 노드로 보내고, 있으면 목적 노드로 보냄 ----
    def route_solution(self, state: TeacherState) -> str:
        # (옵션) 파일 전처리 필요 시 먼저 preprocess
        if has_files_to_preprocess(state):
            return "preprocess"
        # solution 전, 문제/보기 없으면 generator 먼저
        return "solution" if has_questions(state) else "mark_after_generator_solution"

    def route_score(self, state: TeacherState) -> str:
        # score 전, 모델의 풀이 결과가 없다면 solution 먼저
        return "score" if has_solution_answers(state) else "mark_after_solution_score"

    def route_analysis(self, state: TeacherState) -> str:
        # analysis 전, score 결과가 없으면 score 먼저
        return "analysis" if has_score(state) else "mark_after_score_analysis"

    # ---- 라우팅 표식(복귀 노드 지정) ----
    def mark_after_generator_solution(self, state: TeacherState) -> TeacherState:
        ns = {**state}
        ns.setdefault("routing", {})
        ns["routing"]["after_generator"] = "solution"
        return ns

    def mark_after_solution_score(self, state: TeacherState) -> TeacherState:
        ns = {**state}
        ns.setdefault("routing", {})
        ns["routing"]["after_solution"] = "score"
        return ns

    def mark_after_score_analysis(self, state: TeacherState) -> TeacherState:
        ns = {**state}
        ns.setdefault("routing", {})
        ns["routing"]["after_score"] = "analysis"
        return ns

    # ---- 선행 노드 종료 후 갈 곳 결정 ----
    def post_generator_route(self, state: TeacherState) -> str:
        nxt = ((state.get("routing") or {}).get("after_generator") or "").strip()
        return nxt if nxt else "persist_state"

    def post_solution_route(self, state: TeacherState) -> str:
        nxt = ((state.get("routing") or {}).get("after_solution") or "").strip()
        return nxt if nxt else "persist_state"

    def post_score_route(self, state: TeacherState) -> str:
        nxt = ((state.get("routing") or {}).get("after_score") or "").strip()
        return nxt if nxt else "persist_state"


def build_teacher_graph(self):
    builder = StateGraph(TeacherState)

    # 기존 노드들
    builder.add_node("load_state", RunnableLambda(self.load_state))
    builder.add_node("persist_state", RunnableLambda(self.persist_state))
    builder.add_node("intent_classifier", RunnableLambda(self.intent_classifier))
    builder.add_node("generator", RunnableLambda(self.generator))
    builder.add_node("solution", RunnableLambda(self.solution))
    builder.add_node("score", RunnableLambda(self.score))
    builder.add_node("analysis", RunnableLambda(self.analysis))
    builder.add_node("retrieve", RunnableLambda(self.retrieve))

    # (선택) 파일 전처리 노드
    builder.add_node("preprocess", RunnableLambda(self.preprocess))

    # 라우터/표식/후행 분기 노드 등록
    builder.add_node("mark_after_generator_solution", RunnableLambda(self.mark_after_generator_solution))
    builder.add_node("mark_after_solution_score", RunnableLambda(self.mark_after_solution_score))
    builder.add_node("mark_after_score_analysis", RunnableLambda(self.mark_after_score_analysis))

    # 시작 → 로드 → 인텐트 분기
    builder.add_edge(START, "load_state")
    builder.add_edge("load_state", "intent_classifier")

    # (의존성 인지형) 라우터 가상 노드들 추가
    builder.add_node("route_solution", RunnableLambda(self.route_solution))
    builder.add_node("route_score", RunnableLambda(self.route_score))
    builder.add_node("route_analysis", RunnableLambda(self.route_analysis))

    # intent → (의존성 라우터 or 직접)
    builder.add_conditional_edges(
        "intent_classifier",
        self.select_agent,
        {
            "retrieve": "retrieve",
            "generate": "generator",     # generator는 단독 수행
            "analyze": "route_analysis", # ← analysis는 라우터 거침
            "solution": "route_solution",# ← solution도 라우터 거침
            "score": "route_score",      # ← score도 라우터 거침
        },
    )

    # ---- 라우터 분기 정의 ----
    # route_solution: (preprocess) → generator 표식 → generator → (post) → solution
    builder.add_conditional_edges(
        "route_solution",
        self.route_solution,
        {
            "solution": "solution",
            "preprocess": "preprocess",
            "mark_after_generator_solution": "mark_after_generator_solution",
        },
    )
    builder.add_edge("preprocess", "mark_after_generator_solution")  # 전처리 후 generator로
    builder.add_edge("mark_after_generator_solution", "generator")

    # route_score: solution 선행 필요 시 표식 → solution
    builder.add_conditional_edges(
        "route_score",
        self.route_score,
        {
            "score": "score",
            "mark_after_solution_score": "mark_after_solution_score",
        },
    )
    builder.add_edge("mark_after_solution_score", "solution")

    # route_analysis: score 선행 필요 시 표식 → score
    builder.add_conditional_edges(
        "route_analysis",
        self.route_analysis,
        {
            "analysis": "analysis",
            "mark_after_score_analysis": "mark_after_score_analysis",
        },
    )
    builder.add_edge("mark_after_score_analysis", "score")

    # ---- 선행 노드 종료 후 “복귀” 분기 ----
    # generator 수행 뒤: 플래그 있으면 solution으로, 없으면 persist
    builder.add_conditional_edges(
        "generator",
        self.post_generator_route,
        {
            "solution": "solution",
            "persist_state": "persist_state",
        },
    )

    # solution 수행 뒤: 플래그 있으면 score로, 없으면 persist
    builder.add_conditional_edges(
        "solution",
        self.post_solution_route,
        {
            "score": "score",
            "persist_state": "persist_state",
        },
    )

    # score 수행 뒤: 플래그 있으면 analysis로, 없으면 persist
    builder.add_conditional_edges(
        "score",
        self.post_score_route,
        {
            "analysis": "analysis",
            "persist_state": "persist_state",
        },
    )

    # retrieve/analysis는 종료로
    builder.add_edge("retrieve", "persist_state")
    builder.add_edge("analysis", "persist_state")
    builder.add_edge("persist_state", END)

    return builder.compile()

