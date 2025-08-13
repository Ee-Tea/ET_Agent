# teacher_graph.py
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional
from copy import deepcopy

from dotenv import load_dotenv
from langsmith import traceable
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict, NotRequired

# ──────────────────────────────────────────────────────────────────────────────
# 경로는 실제 프로젝트 구조에 맞게 하나만 활성화하세요.
# from ...common.short_term.redis_memory import RedisLangGraphMemory   # 상대 임포트(패키지 실행 전제)
from ..common.short_term.redis_memory import RedisLangGraphMemory   # 절대 임포트(권장)

from agents.analysis.analysis_agent import AnalysisAgent, score_agent
from agents.retrieve.retrieve_agent import retrieve_agent
from TestGenerator.pdf_quiz_groq_class import generate_agent
from solution.solution_agent import solution_agent
from teacher_nodes import get_user_answer
# ──────────────────────────────────────────────────────────────────────────────

# ========== 타입/프로토콜 ==========
class SupportsExecute:
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...

# Stateless 가정(스레드 세이프 보장 안 되면 Orchestrator 인스턴스 멤버로 옮기세요)
retriever_runner: SupportsExecute = retrieve_agent()
generator_runner: SupportsExecute = generate_agent()
solution_runner : SupportsExecute = solution_agent()
score_runner    : SupportsExecute = score_agent()
analyst_runner  : SupportsExecute = AnalysisAgent()

# ---------- Graph State ----------
class SharedState(TypedDict):
    question: NotRequired[List[str]]
    options: NotRequired[List[List[str]]]
    answer: NotRequired[List[str]]
    explanation: NotRequired[List[str]]
    subject: NotRequired[List[str]]
    weak_type: NotRequired[List[str]]
    retrieve_answer: NotRequired[str]
    user_answer: NotRequired[List[str]]  # 사용자가 실제 제출한 답

class TeacherState(TypedDict):
    user_query: str
    intent: str
    shared: NotRequired[SharedState]
    retrieval: NotRequired[dict]
    generation: NotRequired[dict]
    solution: NotRequired[dict]
    score: NotRequired[dict]
    analysis: NotRequired[dict]
    history: NotRequired[List[dict]]      # 채팅 히스토리(메모리에서 로드)
    session: NotRequired[dict]            # 실행 플래그(예: {"loaded": True})
    artifacts: NotRequired[dict]          # 파일/중간 산출물 메타
    routing: NotRequired[dict]            # 의존성-복귀 플래그

# ---------- Shared 기본값/유틸 ----------
SHARED_DEFAULTS: Dict[str, Any] = {
    "question": [],
    "options": [],
    "answer": [],
    "explanation": [],
    "subject": [],
    "wrong_question": [],
    "weak_type": [],
    "user_answer": [],
    "retrieve_answer": "",
}

def ensure_shared(state: TeacherState) -> TeacherState:
    """shared 키 및 타입을 보정하여 이후 노드에서 안정적으로 사용 가능하게 합니다."""
    ns = deepcopy(state) if state else {}
    ns.setdefault("shared", {})
    for key, default_val in SHARED_DEFAULTS.items():
        cur = ns["shared"].get(key, None)
        if not isinstance(cur, type(default_val)):
            ns["shared"][key] = deepcopy(default_val)
    return ns

def validate_qas(shared: SharedState) -> None:
    """문항/보기/정답/해설/과목 길이 일관성 검증."""
    n = len(shared.get("question", []))
    if not all(len(shared.get(k, [])) == n for k in ("options", "answer", "explanation", "subject")):
        raise ValueError(
            f"[QA 정합성 오류] 길이 불일치: "
            f"q={len(shared.get('question', []))}, "
            f"opt={len(shared.get('options', []))}, "
            f"ans={len(shared.get('answer', []))}, "
            f"exp={len(shared.get('explanation', []))}, "
            f"subj={len(shared.get('subject', []))}"
        )

def safe_execute(agent: SupportsExecute, payload: Dict[str, Any]) -> Dict[str, Any]:
    """에이전트 실행 예외 방지 래퍼."""
    try:
        out = agent.execute(payload)
        return out if isinstance(out, dict) else {}
    except Exception as e:
        print(f"[WARN] agent {getattr(agent, 'name', type(agent).__name__)} failed: {e}")
        return {}

# ---------- 의존성 체크 ----------
def has_questions(state: TeacherState) -> bool:
    sh = (state.get("shared") or {})
    return bool(sh.get("question")) and bool(sh.get("options"))

def has_solution_answers(state: TeacherState) -> bool:
    sh = (state.get("shared") or {})
    return bool(sh.get("answer")) and bool(sh.get("explanation"))

def has_score(state: TeacherState) -> bool:
    sc = state.get("score") or {}
    return bool(sc)

def has_files_to_preprocess(state: TeacherState) -> bool:
    # 파일 전처리 훅: 필요 시 사용자가 올린 파일/ID 기준으로 True 리턴
    art = state.get("artifacts") or {}
    return bool(art.get("pdf_ids") or art.get("image_ids"))

# ========== Orchestrator ==========
class Orchestrator:
    def __init__(self, user_id: str, service: str, chat_id: str):
        load_dotenv()
        if not os.getenv("LANGCHAIN_API_KEY"):
            print("경고: LANGCHAIN_API_KEY 환경 변수가 설정되지 않았습니다.")
        # TTL/길이 제한은 redis_memory.py에서 설정
        self.memory = RedisLangGraphMemory(user_id=user_id, service=service, chat_id=chat_id)

    # ── Memory IO ────────────────────────────────────────────────────────────
    def load_state(self, state: TeacherState) -> TeacherState:
        """그래프 시작 시 단 1번만 메모리에서 상태를 불러와 state에 병합."""
        if (state.get("session") or {}).get("loaded"):
            return state
        loaded = self.memory.load(state)
        loaded.setdefault("session", {})
        loaded["session"]["loaded"] = True
        return ensure_shared(loaded)

    def persist_state(self, state: TeacherState) -> TeacherState:
        """그래프 리프 종료 후 단 1곳에서 메모리에 반영."""
        self.memory.save(state, state)
        return state

    # ── Intent & Routing ────────────────────────────────────────────────────
    @traceable(name="teacher.intent_classifier")
    def intent_classifier(self, state: TeacherState) -> TeacherState:
        uq = (state.get("user_query") or "").strip()
        intent = "retrieve" if not uq else "retrieve"  # 기본값: 검색
        try:
            from teacher_nodes import user_intent
            intent = (user_intent(uq) or intent).lower() if uq else intent
        except Exception:
            pass
        print(f"사용자 의도 분류: {intent}")
        return {**state, "user_query": uq, "intent": intent}

    def select_agent(self, state: TeacherState) -> str:
        mapping = {
            "retrieve": "retrieve",
            "generate": "generator",
            "analyze": "route_analysis",
            "solution": "route_solution",
            "score": "route_score",
        }
        return mapping.get((state.get("intent") or "").lower(), "retrieve")

    # ── Router (의존성 자동 보장) ───────────────────────────────────────────
    def route_solution(self, state: TeacherState) -> str:
        if has_files_to_preprocess(state):
            return "preprocess"
        return "solution" if has_questions(state) else "mark_after_generator_solution"

    def route_score(self, state: TeacherState) -> str:
        return "score" if has_solution_answers(state) else "mark_after_solution_score"

    def route_analysis(self, state: TeacherState) -> str:
        return "analysis" if has_score(state) else "mark_after_score_analysis"

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

    def post_generator_route(self, state: TeacherState) -> str:
        nxt = ((state.get("routing") or {}).get("after_generator") or "").strip()
        return nxt if nxt else "persist_state"

    def post_solution_route(self, state: TeacherState) -> str:
        nxt = ((state.get("routing") or {}).get("after_solution") or "").strip()
        return nxt if nxt else "persist_state"

    def post_score_route(self, state: TeacherState) -> str:
        nxt = ((state.get("routing") or {}).get("after_score") or "").strip()
        return nxt if nxt else "persist_state"

    # ── Nodes ───────────────────────────────────────────────────────────────
    @traceable(name="teacher.preprocess")
    def preprocess(self, state: TeacherState) -> TeacherState:
        """
        파일 기반 입력 전처리 훅.
        - 예) PDF/이미지 → OCR/파싱 → shared.question/options 채우기
        """
        state = ensure_shared(state)
        # TODO: artifacts 정보를 이용해 shared에 문항/보기 생성
        return state

    @traceable(name="teacher.generator")
    def generator(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("문제 생성 노드 실행")
        agent_input = {"query": state.get("user_query", "")}
        agent_result = safe_execute(generator_runner, agent_input)

        new_state: TeacherState = {**state}
        new_state.setdefault("generation", {})
        new_state["generation"].update(agent_result)

        # 공유부 누적(Append)
        if "validated_questions" in agent_result:
            sh = new_state.setdefault("shared", {})
            sh.setdefault("question", [])
            sh.setdefault("options", [])
            sh.setdefault("answer", [])
            sh.setdefault("explanation", [])
            sh.setdefault("subject", [])

            for item in agent_result["validated_questions"]:
                if "question" in item and "options" in item:
                    # options 정규화
                    opts = item.get("options", [])
                    if isinstance(opts, str):
                        lines = [x.strip() for x in opts.splitlines() if x.strip()]
                        opts = lines if lines else [opts.strip()]
                    elif isinstance(opts, list):
                        opts = [str(x).strip() for x in opts if str(x).strip()]
                    else:
                        opts = []
                    sh["question"].append(item["question"])
                    sh["options"].append(opts)
                    sh["answer"].append(item.get("answer", ""))
                    sh["explanation"].append(item.get("explanation", ""))
                    sh["subject"].append(item.get("subject", ""))

        validate_qas(new_state["shared"])
        return new_state

    @traceable(name="teacher.solution")
    def solution(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("문제 풀이 노드 실행")
        new_state: TeacherState = {**state}
        new_state.setdefault("solution", {})
        sh = new_state.setdefault("shared", {})

        questions = sh.get("question", []) or []
        options_list = sh.get("options", []) or []

        generated_answers: List[str] = []
        generated_explanations: List[str] = []

        for question, options in zip(questions, options_list):
            if isinstance(options, str):
                options = [x.strip() for x in options.splitlines() if x.strip()] or [options.strip()]
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
            sh.setdefault("answer", [])
            sh["answer"].extend(generated_answers)
        if generated_explanations:
            sh.setdefault("explanation", [])
            sh["explanation"].extend(generated_explanations)

        validate_qas(sh)
        return new_state

    @traceable(name="teacher.score")
    def score(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("채점 노드 실행")
        user_query = state.get("user_query", "")
        user_answer = get_user_answer(user_query)
        sh = (state.get("shared") or {})
        solution_answers = sh.get("answer", []) or []

        agent_input = {
            "user_answer": user_answer,
            "solution_answer": solution_answers,
        }
        agent_result = safe_execute(score_runner, agent_input)

        new_state: TeacherState = {**state}
        new_state.setdefault("score", {})
        new_state["score"].update(agent_result or {})
        return new_state

    @traceable(name="teacher.analysis")
    def analysis(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("오답 분석 노드 실행")
        sh = (state.get("shared") or {})
        user_query = state.get("user_query", "")
        user_answer = get_user_answer(user_query)
        if not user_answer:
            user_answer = sh.get("user_answer", []) or []
        agent_input = {
            "problem": sh.get("question", []) or [],
            "user_answer": user_answer,
            "solution_answer": sh.get("answer", []) or [],
            "solution": sh.get("explanation", []) or [],
        }
        agent_result = safe_execute(analyst_runner, agent_input)

        new_state: TeacherState = {**state}
        new_state.setdefault("analysis", {})
        new_state["analysis"].update(agent_result or {})

        if agent_result and "mistake_summary" in agent_result:
            new_state.setdefault("shared", {})
            new_state["shared"]["weak_type"] = agent_result["mistake_summary"]
        return new_state

    @traceable(name="teacher.retrieve")
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

    # ── Graph Build ──────────────────────────────────────────────────────────
    def build_teacher_graph(self):
        builder = StateGraph(TeacherState)

        # Core nodes
        builder.add_node("load_state", RunnableLambda(self.load_state))
        builder.add_node("persist_state", RunnableLambda(self.persist_state))
        builder.add_node("intent_classifier", RunnableLambda(self.intent_classifier))

        builder.add_node("generator", RunnableLambda(self.generator))
        builder.add_node("solution", RunnableLambda(self.solution))
        builder.add_node("score", RunnableLambda(self.score))
        builder.add_node("analysis", RunnableLambda(self.analysis))
        builder.add_node("retrieve", RunnableLambda(self.retrieve))

        # Preprocess
        builder.add_node("preprocess", RunnableLambda(self.preprocess))

        # Routing markers
        builder.add_node("mark_after_generator_solution", RunnableLambda(self.mark_after_generator_solution))
        builder.add_node("mark_after_solution_score", RunnableLambda(self.mark_after_solution_score))
        builder.add_node("mark_after_score_analysis", RunnableLambda(self.mark_after_score_analysis))

        # Routers
        builder.add_node("route_solution", RunnableLambda(self.route_solution))
        builder.add_node("route_score", RunnableLambda(self.route_score))
        builder.add_node("route_analysis", RunnableLambda(self.route_analysis))

        # Start → load → intent
        builder.add_edge(START, "load_state")
        builder.add_edge("load_state", "intent_classifier")

        # intent branching (with routers)
        builder.add_conditional_edges(
            "intent_classifier",
            self.select_agent,
            {
                "retrieve": "retrieve",
                "generator": "generator",
                "route_analysis": "route_analysis",
                "route_solution": "route_solution",
                "route_score": "route_score",
            },
        )

        # route_solution
        builder.add_conditional_edges(
            "route_solution",
            self.route_solution,
            {
                "solution": "solution",
                "preprocess": "preprocess",
                "mark_after_generator_solution": "mark_after_generator_solution",
            },
        )
        builder.add_edge("preprocess", "mark_after_generator_solution")
        builder.add_edge("mark_after_generator_solution", "generator")

        # route_score
        builder.add_conditional_edges(
            "route_score",
            self.route_score,
            {
                "score": "score",
                "mark_after_solution_score": "mark_after_solution_score",
            },
        )
        builder.add_edge("mark_after_solution_score", "solution")

        # route_analysis
        builder.add_conditional_edges(
            "route_analysis",
            self.route_analysis,
            {
                "analysis": "analysis",
                "mark_after_score_analysis": "mark_after_score_analysis",
            },
        )
        builder.add_edge("mark_after_score_analysis", "score")

        # post dependencies
        builder.add_conditional_edges(
            "generator",
            self.post_generator_route,
            {
                "solution": "solution",
                "persist_state": "persist_state",
            },
        )
        builder.add_conditional_edges(
            "solution",
            self.post_solution_route,
            {
                "score": "score",
                "persist_state": "persist_state",
            },
        )
        builder.add_conditional_edges(
            "score",
            self.post_score_route,
            {
                "analysis": "analysis",
                "persist_state": "persist_state",
            },
        )

        # retrieve/analysis → persist → END
        builder.add_edge("retrieve", "persist_state")
        builder.add_edge("analysis", "persist_state")
        builder.add_edge("persist_state", END)

        return builder.compile()

if __name__ == "__main__":
    """
    간단 테스트 런너:
      - 콘솔에서 사용자 질의(Q>)를 입력
      - 그래프 한 턴 실행
      - 핵심 결과 요약 출력
      - 'exit' / 'quit' 입력 시 종료
    """
    import os
    import traceback

    # 테스트용 식별자 (환경변수로 바꿔도 됩니다)
    USER_ID  = os.getenv("TEST_USER_ID", "demo_user")
    SERVICE  = os.getenv("TEST_SERVICE", "teacher")
    CHAT_ID  = os.getenv("TEST_CHAT_ID", "local")

    # 오케스트레이터 & 그래프 컴파일
    orch = Orchestrator(user_id=USER_ID, service=SERVICE, chat_id=CHAT_ID)
    app = orch.build_teacher_graph()

    print("\n=== Teacher Graph 테스트 ===")
    print("질문을 입력하세요. (종료: exit/quit)\n")

    try:
        while True:
            try:
                user_query = input("Q> ").strip()
            except EOFError:
                # 파이프 입력 등에서 EOF 들어오면 종료
                print("\n[EOF] 종료합니다.")
                break

            if not user_query:
                continue
            if user_query.lower() in {"exit", "quit"}:
                print("종료합니다.")
                break

            # 그래프 입력 상태 (intent는 분류 노드가 채웁니다)
            init_state: Dict[str, Any] = {
                "user_query": user_query,
                "intent": "",
                # 파일 테스트 시 artifacts에 id 넣어두면 preprocess 라우팅이 작동합니다.
                # "artifacts": {"pdf_ids": ["file_123"]},
            }

            try:
                result: Dict[str, Any] = app.invoke(init_state)
            except Exception:
                print("[ERROR] 그래프 실행 중 예외가 발생했습니다:")
                traceback.print_exc()
                continue

            # ─── 결과 요약 출력 ───
            intent = result.get("intent", "(분류실패)")
            shared = (result.get("shared") or {})
            generation = (result.get("generation") or {})
            solution = (result.get("solution") or {})
            score = (result.get("score") or {})
            analysis = (result.get("analysis") or {})
            retrieval = (result.get("retrieval") or {})

            print("\n--- 실행 요약 ---")
            print(f"Intent: {intent}")

            # 검색 요약
            ra = shared.get("retrieve_answer")
            if ra:
                print(f"[Retrieve] {str(ra)[:200]}{'...' if len(str(ra))>200 else ''}")

            # 문항/보기/정답/해설 개수
            q_cnt = len(shared.get("question", []) or [])
            a_cnt = len(shared.get("answer", []) or [])
            e_cnt = len(shared.get("explanation", []) or [])
            print(f"[QA] question={q_cnt}, answer={a_cnt}, explanation={e_cnt}")

            # 최근 문항 미리보기(있으면)
            if q_cnt:
                latest_q = shared["question"][-1]
                latest_opts = (shared.get("options") or [[]])[-1] if shared.get("options") else []
                print(f"[Last Q] {str(latest_q)[:180]}{'...' if len(str(latest_q))>180 else ''}")
                if latest_opts:
                    print("  Options:", "; ".join(latest_opts[:6]) + ("..." if len(latest_opts) > 6 else ""))

            # 최근 모델 풀이/해설 미리보기(있으면)
            if a_cnt:
                last_ans = shared["answer"][-1]
                print(f"[Model Answer] {str(last_ans)[:120]}{'...' if len(str(last_ans))>120 else ''}")
            if e_cnt:
                last_exp = shared["explanation"][-1]
                print(f"[Explanation]  {str(last_exp)[:160]}{'...' if len(str(last_exp))>160 else ''}")

            # 분석/취약유형
            weak = shared.get("weak_type")
            if weak:
                if isinstance(weak, list):
                    print(f"[Weak Types] {', '.join(map(str, weak[:6]))}{'...' if len(weak)>6 else ''}")
                else:
                    print(f"[Weak Types] {weak}")

            # 채점 결과 덤프(간단 표시)
            if score:
                # 특정 키가 있다면 골라서 노출하세요 (여기선 크기만)
                print(f"[Score] keys={list(score.keys())}")

            print("-----------------\n")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 종료합니다.")
