# teacher_graph.py
# uv run teacher/teacher_graph.py
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys, os
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict, NotRequired

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ──────────────────────────────────────────────────────────────────────────────
# 경로는 실제 프로젝트 구조에 맞게 하나만 활성화하세요.
# from ...common.short_term.redis_memory import RedisLangGraphMemory   # 상대 임포트(패키지 실행 전제)
# from ..common.short_term.redis_memory import RedisLangGraphMemory   # 절대 임포트(권장)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from common.short_term.redis_memory import RedisLangGraphMemory

from agents.analysis.analysis_agent import AnalysisAgent
from agents.score.score_engine import ScoreEngine as score_agent
from agents.retrieve.retrieve_agent import retrieve_agent
# from agents.TestGenerator.pdf_quiz_groq_class import InfoProcessingExamAgent as generate_agent
from agents.TestGenerator.generator import InfoProcessingExamAgent as generate_agent
# from agents.TestGenerator.generator_backup import InfoProcessingExamAgent as generate_agent
# from agents.solution.solution_agent import SolutionAgent as solution_agent
from agents.solution.solution_agent_hitl import SolutionAgent as solution_agent
from teacher_nodes import (
    get_user_answer, parse_generator_input, user_intent,                                    
    route_solution, route_score, route_analysis,
    mark_after_generator_solution, mark_after_solution_score, mark_after_score_analysis,
    post_generator_route, post_solution_route, post_score_route, post_analysis_route,
    generate_user_response, extract_problem_and_options
)
from file_path_mapper import FilePathMapper
from datetime import datetime
# ──────────────────────────────────────────────────────────────────────────────
from teacher_util import (
    ensure_shared, validate_qas, safe_execute,
    has_questions, has_solution_answers, has_score, has_files_to_preprocess,
    extract_image_paths, extract_problems_from_images, SupportsExecute
)
from pdf_preprocessor import PDFPreprocessor

# ========== 타입/프로토콜 ==========

# Stateless 가정(스레드 세이프 보장 안 되면 Orchestrator 인스턴스 멤버로 옮기세요)
from typing import Optional

retriever_runner: Optional[SupportsExecute] = None
generator_runner: Optional[SupportsExecute] = None
solution_runner : Optional[SupportsExecute] = None
score_runner    : Optional[SupportsExecute] = None
analyst_runner  : Optional[SupportsExecute] = None

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
    score_result: NotRequired[dict]

class TeacherState(TypedDict):
    user_query: str
    intent: str
    shared: NotRequired[SharedState]
    work: NotRequired[dict]
    retrieval: NotRequired[dict]
    generation: NotRequired[dict]
    solution: NotRequired[dict]
    score: NotRequired[dict]
    analysis: NotRequired[dict]
    history: NotRequired[List[dict]]      # 채팅 히스토리(메모리에서 로드)
    session: NotRequired[dict]            # 실행 플래그(예: {"loaded": True})
    artifacts: NotRequired[dict]          # 파일/중간 산출물 메타
    routing: NotRequired[dict]            # 의존성-복귀 플래그
    llm_response: NotRequired[str]        # LLM이 생성한 사용자 친화적 답변


# ========== Orchestrator ==========
class Orchestrator:
    def __init__(self, user_id: str, service: str, chat_id: str, init_agents: bool = True):
        load_dotenv()
        if not os.getenv("LANGCHAIN_API_KEY"):
            print("경고: LANGCHAIN_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # 이미지 처리에 필요한 환경 변수 설정
        if not os.getenv("OPENAI_VISION_MODEL"):
            os.environ["OPENAI_VISION_MODEL"] = "o4-mini"  # 기본값 설정
        if not os.getenv("MAX_OUTPUT_TOKENS"):
            os.environ["MAX_OUTPUT_TOKENS"] = "1200"  # 기본값 설정
        
        # TTL/길이 제한은 redis_memory.py에서 설정
        try:
            # Redis 포트를 6380으로 설정 (Docker 컨테이너 포트)
            os.environ['REDIS_PORT'] = '6380'
            self.memory = RedisLangGraphMemory(user_id=user_id, service=service, chat_id=chat_id)
        except Exception as e:
            print(f"⚠️ Redis 연결 실패: {e}")
            print("📝 메모리 기반으로 실행합니다.")
            # 간단한 메모리 기반 메모리 클래스
            class SimpleMemory:
                def load(self, state): return state
                def save(self, state, _): return state
            self.memory = SimpleMemory()
        
        # PDF 전처리기 초기화
        from unified_pdf_preprocessor import UnifiedPDFPreprocessor
        self.pdf_preprocessor = UnifiedPDFPreprocessor()
        
        # ⬇️ 에이전트는 옵션으로 초기화 (시각화 때는 False로)
        if init_agents:
            self.retriever_runner = retrieve_agent()
            self.generator_runner = generate_agent()
            self.solution_runner  = solution_agent()   # 추상클래스 구현체면 여기서 생성
            self.score_runner     = score_agent()
            self.analyst_runner   = AnalysisAgent()
        else:
            self.retriever_runner = None
            self.generator_runner = None
            self.solution_runner  = None
            self.score_runner     = None
            self.analyst_runner   = None

        # LangGraph 기반 그래프 생성
        self.checkpointer = InMemorySaver()
        self.graph = self._create_graph()

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
        # 저장 직전 shared를 정리(중복 제거 및 정렬)한 뒤 저장
        try:
            cleaned = self._dedupe_aligned_shared(state.get("shared", {}) or {})
            state = {**state, "shared": cleaned}
        except Exception as _:
            pass
        self.memory.save(state, state)
        return state

    # ── Helpers: selection & dedupe ─────────────────────────────────────────
    def _normalize_text(self, text: Any) -> str:
        try:
            return " ".join(str(text or "").split()).strip().lower()
        except Exception:
            return str(text or "").strip().lower()

    def _dedupe_aligned_shared(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        question/options/answer/explanation/subject/user_answer 리스트를
        question+options 조합 기준으로 중복 제거하여 정렬을 보존합니다.
        """
        if not isinstance(shared, dict):
            return shared
        questions = list(shared.get("question", []) or [])
        options_l = list(shared.get("options", []) or [])
        answers = list(shared.get("answer", []) or [])
        expls = list(shared.get("explanation", []) or [])
        subjects = list(shared.get("subject", []) or [])
        user_ans = list(shared.get("user_answer", []) or [])

        keep_q, keep_o, keep_a, keep_e, keep_s, keep_u = [], [], [], [], [], []
        seen = set()
        total = len(questions)
        for i in range(total):
            q = questions[i]
            opts_raw = options_l[i] if i < len(options_l) else []
            opts_list = []
            if isinstance(opts_raw, list):
                opts_list = [self._normalize_text(x) for x in opts_raw if str(x).strip()]
            elif isinstance(opts_raw, str):
                opts_list = [self._normalize_text(x) for x in opts_raw.splitlines() if x.strip()]
            key = (self._normalize_text(q), tuple(opts_list))
            if key in seen:
                continue
            seen.add(key)
            keep_q.append(q)
            keep_o.append(options_l[i] if i < len(options_l) else [])
            keep_a.append(answers[i] if i < len(answers) else "")
            keep_e.append(expls[i] if i < len(expls) else "")
            keep_s.append(subjects[i] if i < len(subjects) else "")
            keep_u.append(user_ans[i] if i < len(user_ans) else "")

        cleaned = dict(shared)
        cleaned["question"] = keep_q
        cleaned["options"] = keep_o
        cleaned["answer"] = keep_a
        cleaned["explanation"] = keep_e
        cleaned["subject"] = keep_s
        if user_ans:
            cleaned["user_answer"] = keep_u
        return cleaned

    def _ensure_work_selection(self, state: TeacherState) -> TeacherState:
        """사용자 입력으로부터 선택 인덱스/개수를 추출하여 work에 반영."""
        import re
        work = dict((state.get("work") or {}))
        if work.get("_sealed"):
            return {**state, "work": work}

        uq = state.get("user_query", "") or ""
        selected_indices: List[int] = []
        select_count: int = 0

        # 패턴 1: "1-3번", "2~5문제" 등 범위
        for m in re.finditer(r"(\d+)\s*[-~]\s*(\d+)", uq):
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b:
                selected_indices.extend(list(range(a - 1, b)))

        # 패턴 2: "3번", "12번" 등 단일 번호들
        for m in re.finditer(r"(\d+)\s*번", uq):
            idx = int(m.group(1)) - 1
            if idx >= 0:
                selected_indices.append(idx)

        # 패턴 3: "3개", "5 문제" 등 개수 지정
        m = re.search(r"(\d+)\s*(개|문제)", uq)
        if m:
            try:
                select_count = int(m.group(1))
            except Exception:
                select_count = 0

        # 중복 정리 및 정렬
        selected_indices = sorted(set([i for i in selected_indices if i >= 0]))
        work["selected_indices"] = selected_indices
        if select_count > 0:
            work["select_count"] = select_count
        work["_sealed"] = True  # 동일 턴 다중 호출 방지
        return {**state, "work": work}

    def _create_graph(self) -> StateGraph:
        """LangGraph 기반의 워크플로우 그래프를 생성합니다."""
        print("🔧 LangGraph 기반 워크플로우 그래프 생성 중...")
        
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

        # PDF Generation nodes
        builder.add_node("generate_pdfs", RunnableLambda(self.generate_pdfs))
        builder.add_node("generate_problem_pdf", RunnableLambda(self.generate_problem_pdf))
        builder.add_node("generate_answer_pdf", RunnableLambda(self.generate_answer_pdf))
        builder.add_node("generate_analysis_pdf", RunnableLambda(self.generate_analysis_pdf))

        # Routing markers
        builder.add_node("mark_after_generator_solution", RunnableLambda(mark_after_generator_solution))
        builder.add_node("mark_after_solution_score", RunnableLambda(mark_after_solution_score))
        builder.add_node("mark_after_score_analysis", RunnableLambda(mark_after_score_analysis))

        # Routers
        builder.add_node("route_solution", RunnableLambda(route_solution))
        builder.add_node("route_score", RunnableLambda(route_score))
        builder.add_node("route_analysis", RunnableLambda(route_analysis))

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
                "preprocess": "preprocess",  # solution 의도일 때 preprocess로
                "route_score": "route_score",
            },
        )

        # route_solution
        builder.add_conditional_edges(
            "route_solution",
            lambda state: state.get("routing", {}).get("solution_next", "mark_after_generator_solution"),
            {
                "solution": "solution",
                "mark_after_generator_solution": "mark_after_generator_solution",
            },
        )
        builder.add_conditional_edges(
            "preprocess",
            lambda state: "solution" if state.get("artifacts", {}).get("extracted_problem_count", 0) > 0 or state.get("artifacts", {}).get("pdf_added_count", 0) > 0 else "mark_after_generator_solution",
            {
                "solution": "solution",
                "mark_after_generator_solution": "mark_after_generator_solution",
            },
        )
        builder.add_edge("mark_after_generator_solution", "generator")

        # route_score
        builder.add_conditional_edges(
            "route_score",
            lambda state: state.get("routing", {}).get("score_next", "mark_after_solution_score"),
            {
                "score": "score",
                "mark_after_solution_score": "mark_after_solution_score",
            },
        )
        builder.add_edge("mark_after_solution_score", "solution")

        # route_analysis
        builder.add_conditional_edges(
            "route_analysis",
            lambda state: state.get("routing", {}).get("analysis_next", "mark_after_score_analysis"),
            {
                "analysis": "analysis",
                "mark_after_score_analysis": "mark_after_score_analysis",
            },
        )
        builder.add_edge("mark_after_score_analysis", "score")

        # post dependencies - 자동 PDF 생성 강화
        builder.add_conditional_edges(
            "generator",
            post_generator_route,
            {
                "solution": "solution",
                "generate_problem_pdf": "generate_problem_pdf",
            },
        )
        builder.add_conditional_edges(
            "solution",
            post_solution_route,
            {
                "score": "score",
                "generate_answer_pdf": "generate_answer_pdf",
            },
        )
        builder.add_edge("score","analysis")

        # retrieve → persist, analysis → generate_analysis_pdf → persist → END
        builder.add_edge("retrieve", "persist_state")
        builder.add_edge("analysis", "generate_analysis_pdf")
        builder.add_edge("generate_analysis_pdf", "persist_state")
        builder.add_edge("generate_problem_pdf", "persist_state")
        builder.add_edge("generate_answer_pdf", "persist_state")
        builder.add_edge("persist_state", END)

        print("✅ LangGraph 워크플로우 그래프 생성 완료")
        return builder.compile(checkpointer=self.checkpointer)

    def invoke(self, state: TeacherState, config: Optional[Dict] = None) -> TeacherState:
        """LangGraph 기반으로 워크플로우를 실행합니다."""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        # 체크포인터와 함께 그래프 실행
        try:
            result = self.graph.invoke(state, config)
            return result
        except Exception as e:
            print(f"❌ 그래프 실행 중 오류 발생: {e}")
            # interrupt가 발생한 경우 체크포인터에서 상태 복구 시도
            if "interrupt" in str(e).lower():
                print("🔄 interrupt가 발생했습니다. 체크포인터에서 상태를 확인하세요.")
                print("💡 Command(resume)을 사용하여 워크플로우를 재개할 수 있습니다.")
            raise

    def resume_workflow(self, resume_data: str, config: Optional[Dict] = None) -> TeacherState:
        """Command(resume)을 사용하여 중단된 워크플로우를 재개합니다."""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        # 상위 그래프 재개 시, solution 노드에서 서브그래프를 재개할 수 있도록 임시로 보관
        try:
            self._pending_user_feedback = resume_data
        except Exception:
            pass
        
        # LangGraph 버전에 따른 Command import 시도
        try:
            from langgraph.checkpoint.memory import Command
        except ImportError:
            try:
                from langgraph import Command
            except ImportError:
                try:
                    from langgraph.types import Command
                except ImportError:
                    print("❌ Command를 import할 수 없습니다. LangGraph 버전을 확인해주세요.")
                    raise ImportError("Command import 실패")
        
        try:
            print(f"🔄 워크플로우 재개 중... resume_data: {resume_data}")
            print(f"🔍 체크포인터 상태 확인: {self.checkpointer}")
            
            # 숏텀 메모리에서 solution_agent 상태 복구 시도
            try:
                from common.short_term.redis_memory import RedisMemory
                redis_memory = RedisMemory()
                
                # solution_agent의 메모리 키들을 찾아서 상태 복구
                memory_keys = redis_memory.keys("solution_*")
                if memory_keys:
                    print(f"🔍 숏텀 메모리에서 solution 상태 발견: {len(memory_keys)}개")
                    for key in memory_keys:
                        state_data = redis_memory.get(key)
                        if state_data and state_data.get("interrupt_occurred"):
                            print(f"💾 복구된 상태: {key}")
                            # 상태를 체크포인터에 저장
                            if hasattr(self, 'checkpointer') and self.checkpointer:
                                self.checkpointer.put(config.get("configurable", {}).get("thread_id", "default"), state_data)
            except Exception as mem_err:
                print(f"⚠️ 숏텀 메모리 복구 실패: {mem_err}")
            
            # Command(resume)을 사용하여 중단된 지점부터 재개
            resume_command = Command(resume={"data": resume_data})
            print(f"📤 Command(resume) 전송: {resume_command}")
            
            # 체크포인터가 설정된 그래프로 재개
            result = self.graph.invoke(resume_command, config)
            print("✅ 워크플로우 재개 완료")
            return result
        except Exception as e:
            print(f"❌ 워크플로우 재개 실패: {e}")
            print(f"🔍 오류 상세: {type(e).__name__}: {str(e)}")
            raise

    # ── Intent & Routing ────────────────────────────────────────────────────

    @traceable(name="teacher.intent_classifier")
    def intent_classifier(self, state: TeacherState) -> TeacherState:
        uq = (state.get("user_query") or "").strip()

        # LLM 기반 의도 분류만 담당
        try:
            raw = user_intent(uq) if uq else ""
            intent = raw
            print(f"🤖 LLM 기반 분류: {intent} (raw={raw!r})")
        except Exception as e:
            print(f"⚠️ LLM 분류 실패, 기본값 사용: {e}")
            intent = "retrieve"
            
        return {**state, "user_query": uq, "intent": intent}

    def select_agent(self, state: TeacherState) -> str:
        intent = (state.get("intent") or "").strip().strip('"\'' ).lower()

        mapping = {
            "retrieve": "retrieve",
            "generate": "generator",
            "analyze": "route_analysis",
            "solution": "preprocess",  # solution 의도일 때 preprocess를 먼저 거침
            "score": "route_score",
        }
        chosen = mapping.get(intent, "retrieve")
        print(f"[router] intent={intent} → {chosen}")
        return chosen

    # ── Router (의존성 자동 보장) ───────────────────────────────────────────

    # ── Nodes ───────────────────────────────────────────────────────────────
    @traceable(name="teacher.preprocess")  
    def preprocess(self, state: TeacherState) -> TeacherState:
        """
        PDF 및 이미지 파일에서 문제 추출하는 전처리 노드
        - 사용자 입력에서 파일 경로 추출 및 메타데이터 파싱
        - 파일 종류에 따라 적절한 처리 방법 선택
        - 인덱스 기록을 'extend 이전' 길이로 고정해 올바른 범위를 남깁니다.
        """
        print("📄 PDF/이미지 문제 추출 전처리 노드 실행")

        uq = state.get("user_query", "")
        current_artifacts = state.get("artifacts", {}) or {}
        
        # PDF 전처리 모듈 import (편의 함수들)
        from pdf_preprocessor import extract_pdf_paths, extract_problem_range, determine_problem_source

        # PDF 경로 추출 및 artifacts 업데이트
        extracted_pdfs = extract_pdf_paths(uq)
        if extracted_pdfs:
            # 사용자가 명시적으로 파일 경로를 제공한 경우, 해당 파일만 사용
            pdf_filenames = []
            for path in extracted_pdfs:
                filename = path.split('\\')[-1].split('/')[-1]  # Windows/Unix 경로 모두 처리
                pdf_filenames.append(filename)
            
            # 사용자 지정 파일이 우선 (기존 파일은 무시)
            current_artifacts["pdf_ids"] = pdf_filenames
            print(f"📁 사용자 지정 PDF 파일: {pdf_filenames}")
            print(f"🎯 이 파일들만 처리됩니다: {pdf_filenames}")

        # 이미지 파일 경로 추출
        extracted_images = extract_image_paths(uq)
        if extracted_images:
            image_filenames = []
            for path in extracted_images:
                filename = path.split('\\')[-1].split('/')[-1]  # Windows/Unix 경로 모두 처리
                image_filenames.append(filename)
            
            current_artifacts["image_ids"] = image_filenames
            print(f"🖼️ 사용자 지정 이미지 파일: {image_filenames}")
            print(f"🎯 이 이미지들만 처리됩니다: {image_filenames}")

        # 문제 번호 범위 추출
        problem_range = extract_problem_range(uq)
        if problem_range:
            current_artifacts["problem_range"] = problem_range
            print(f"🔢 문제 번호 범위: {problem_range}")

        # 문제 소스 결정
        problem_source = determine_problem_source(uq)
        if problem_source:
            current_artifacts["problem_source"] = problem_source
            print(f"📚 문제 소스: {problem_source}")

        # artifacts 업데이트
        state["artifacts"] = current_artifacts

        # 파일 경로 매핑
        file_mapper = FilePathMapper()
        external_file_paths = file_mapper.map_artifacts_to_paths(current_artifacts)

        if not external_file_paths:
            print("⚠️ 전처리할 파일이 없습니다.")
            print(f"🔍 user_query: {uq}")
            
            # user_query에서 문제와 보기 추출 시도
            if uq and uq.strip():
                print("🔍 user_query에서 문제와 보기 추출 시도...")
                try:
                    print(f"🔍 extract_problem_and_options 함수 호출 시작...")
                    extracted = extract_problem_and_options(uq.strip())
                    print(f"🔍 추출 결과: {extracted}")
                    
                    if extracted and isinstance(extracted, dict):
                        has_problem = extracted.get("has_problem", False)
                        problem = extracted.get("problem", "")
                        options = extracted.get("options", [])
                        
                        print(f"🔍 has_problem: {has_problem}")
                        print(f"🔍 problem: {problem}")
                        print(f"🔍 options: {options}")
                        
                        if has_problem and problem and options and len(options) > 0:
                            print(f"✅ 문제 추출 성공: {problem[:100]}...")
                            print(f"✅ 보기 추출 성공: {len(options)}개")
                            
                            # 추출된 문제를 shared 상태에 추가
                            new_state = ensure_shared({**state})
                            shared = new_state["shared"]
                            
                            # 중복 여부 확인 (동일 문제/보기 존재 시 재추가 방지)
                            shared.setdefault("question", [])
                            shared.setdefault("options", [])
                            existing_index = None
                            try:
                                for idx, (q0, o0) in enumerate(zip(shared["question"], shared["options"])):
                                    if str(q0).strip() == str(problem).strip() and [str(x).strip() for x in (o0 or [])] == [str(x).strip() for x in (options or [])]:
                                        existing_index = idx
                                        break
                            except Exception:
                                existing_index = None

                            if existing_index is not None:
                                print(f"⚠️ 중복 문제 감지 → 기존 인덱스: {existing_index}; 재처리 생략")
                                # 이미 존재하므로 이번 턴에는 solution 재호출이 일어나지 않게 count=0 처리
                                current_artifacts["extracted_problem_count"] = 0
                                # 인덱스는 변경하지 않음
                            else:
                                shared["question"].append(problem)
                                shared["options"].append(options)
                                print("✅ 문제/보기 추가 완료 (중복 아님)")
                                current_artifacts["extracted_problem_count"] = 1
                                current_artifacts["extracted_problem_start_index"] = len(shared["question"]) - 1
                                current_artifacts["extracted_problem_end_index"] = len(shared["question"]) - 1
                            
                            print(f"📝 추출된 문제를 shared state에 추가: 1개")
                            print(f"📂 shared state 총 문제 수: {len(shared['question'])}개")
                            print(f"📂 artifacts: {current_artifacts}")
                            
                            # artifacts 업데이트
                            new_state["artifacts"] = current_artifacts
                            return new_state
                        else:
                            print("⚠️ user_query에서 문제와 보기를 추출할 수 없습니다.")
                            print(f"🔍 has_problem: {has_problem}")
                            print(f"🔍 problem: {problem}")
                            print(f"🔍 options: {options}")
                    else:
                        print("⚠️ extract_problem_and_options 함수가 올바른 형식의 결과를 반환하지 않았습니다.")
                        print(f"🔍 반환된 결과: {extracted}")
                        
                except Exception as e:
                    print(f"❌ 문제 추출 중 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ user_query가 비어있거나 None입니다.")
            
            return state

        try:
            # 파일 종류별로 분류
            pdf_files = []
            image_files = []
            
            for file_path in external_file_paths:
                if file_path.lower().endswith(('.pdf')):
                    pdf_files.append(file_path)
                elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
                    image_files.append(file_path)
                else:
                    print(f"⚠️ 지원하지 않는 파일 형식: {file_path}")
            
            print(f"📁 PDF 파일: {len(pdf_files)}개, 이미지 파일: {len(image_files)}개")
            
            extracted_problems = []
            
            # PDF 파일 처리
            if pdf_files:
                print("📄 PDF 파일에서 문제 추출 중...")
                pdf_preprocessor = PDFPreprocessor()
                pdf_problems = pdf_preprocessor.extract_problems_from_pdf(pdf_files)
                extracted_problems.extend(pdf_problems or [])
                print(f"📄 PDF에서 {len(pdf_problems or [])}개 문제 추출")
            
            # 이미지 파일 처리
            if image_files:
                print("🖼️ 이미지 파일에서 문제 추출 중...")
                image_problems = extract_problems_from_images(image_files)
                extracted_problems.extend(image_problems or [])
                print(f"🖼️ 이미지에서 {len(image_problems or [])}개 문제 추출")

            new_state = ensure_shared({**state})
            shared = new_state["shared"]

            # extend 이전 길이를 고정 저장
            start_index = len(shared.get("question", []))

            questions: List[str] = []
            options: List[List[str]] = []

            for problem in extracted_problems or []:
                if not isinstance(problem, dict):
                    continue
                q = str(problem.get("question", "")).strip()
                opt = problem.get("options", [])
                if isinstance(opt, str):
                    opt = [x.strip() for x in opt.splitlines() if x.strip()]
                if not isinstance(opt, list):
                    opt = []
                opt = [str(x).strip() for x in opt if str(x).strip()]
                if q and opt:
                    questions.append(q)
                    options.append(opt)

            # 실제 반영
            if questions:
                prev_cnt = len(shared["question"])
                shared["question"].extend(questions)
                shared["options"].extend(options)
                new_cnt = len(shared["question"])

                added_count = len(questions)
                end_index = start_index + added_count - 1

                arts = new_state.setdefault("artifacts", {})
                arts["pdf_added_count"] = added_count
                arts["pdf_added_start_index"] = start_index
                arts["pdf_added_end_index"] = end_index

                print(f"📄 파일에서 문제를 shared state에 추가: {added_count}개")
                print(f"📂 shared state 총 문제 수: {prev_cnt}개 → {new_cnt}개")
                print(f"🔢 추가된 문제 인덱스: {start_index} ~ {end_index}")
            else:
                print("⚠️ 유효한 문제를 찾지 못했습니다.")

            return new_state

        except Exception as e:
            print(f"❌ 파일 문제 추출 중 오류: {e}")
            return state

    @traceable(name="teacher.solution")
    def solution(self, state: TeacherState) -> TeacherState:
        """
        문제 풀이 노드 - PDF에서 추가된 문제들만 solution_agent로 처리
        - preprocess에서 기록한 인덱스를 정확히 사용
        - agent가 요구하는 입력 키들의 변종과 호환(user_problems / pdf_extracted / problems)
        - 자동 답안집 PDF 생성(구식 호출) 제거
        """
        print("🔧 문제 풀이 노드 실행")
        new_state: TeacherState = ensure_shared({**state})
        new_state = self._ensure_work_selection(new_state)
        new_state.setdefault("solution", {})

        artifacts = new_state.get("artifacts", {}) or {}
        shared = new_state["shared"]

        pdf_added_count = int(artifacts.get("pdf_added_count", 0) or 0)
        extracted_problem_count = int(artifacts.get("extracted_problem_count", 0) or 0)
        start_index = artifacts.get("pdf_added_start_index", None)
        end_index = artifacts.get("pdf_added_end_index", None)
        extracted_start_index = artifacts.get("extracted_problem_start_index", None)
        extracted_end_index = artifacts.get("extracted_problem_end_index", None)

        # PDF/추출 문제 또는 work 기반 선택 여부 확인
        total_problems = pdf_added_count + extracted_problem_count
        work_sel = (new_state.get("work") or {})
        sel_indices: List[int] = list(work_sel.get("selected_indices", []) or [])
        sel_count: int = int(work_sel.get("select_count", 0) or 0)
        if total_problems <= 0 and not sel_indices and sel_count <= 0:
            print("⚠️ 처리할 문제가 없습니다.(선택 없음)")
            return new_state

        # PDF 문제 처리
        if pdf_added_count > 0 and start_index is not None and end_index is not None and end_index >= start_index:
            all_questions = shared.get("question", [])
            all_options = shared.get("options", [])

            # 범위 보정
            start = max(0, min(int(start_index), len(all_questions)))
            end = min(int(end_index), len(all_questions) - 1)

            pdf_questions = all_questions[start:end + 1]
            pdf_options = all_options[start:end + 1]

            print(f"🎯 [Solution] PDF 문제 처리: 인덱스 {start}~{end} ({len(pdf_questions)}개)")

            agent = self.solution_runner
            if agent is None:
                raise RuntimeError("solution_runner is not initialized (init_agents=False).")

            generated_answers: List[str] = []
            generated_explanations: List[str] = []

            for i, (q, opts) in enumerate(zip(pdf_questions, pdf_options), start=1):
                # 옵션 정규화
                if isinstance(opts, str):
                    opts = [x.strip() for x in opts.splitlines() if x.strip()]
                opts = [str(x).strip() for x in (opts or []) if str(x).strip()]

                if not q or not opts:
                    generated_answers.append("")
                    generated_explanations.append("")
                    continue

                problem_payload = {"question": q, "options": opts}
                print(f"🎯 [Solution] 처리할 문제: {problem_payload}")
                print(problem_payload["question"], problem_payload["options"])
                
                try:
                    agent_result = agent.invoke(user_problem=q, user_problem_options=opts, user_input_txt=state.get("user_query", ""))
                except Exception as e:
                    print(f"❌ SolutionAgent invoke 실행 실패({i}/{len(pdf_questions)}): {e}")
                    agent_result = None

                ans, exp = "", ""
                if agent_result:
                    if isinstance(agent_result, dict) and agent_result.get("results"):
                        r0 = agent_result["results"][0]
                        ans = r0.get("generated_answer", "")
                        exp = r0.get("generated_explanation", "")
                    else:
                        ans = agent_result.get("generated_answer", "")
                        exp = agent_result.get("generated_explanation", "")

                generated_answers.append(ans or "")
                generated_explanations.append(exp or "")

            # 결과 반영
            shared.setdefault("answer", [])
            shared.setdefault("explanation", [])
            shared["answer"].extend(generated_answers)
            shared["explanation"].extend(generated_explanations)

        # 추출된 문제 처리
        if extracted_problem_count > 0 and extracted_start_index is not None and extracted_end_index is not None:
            all_questions = shared.get("question", [])
            all_options = shared.get("options", [])

            # 범위 보정
            start = max(0, min(int(extracted_start_index), len(all_questions)))
            end = min(int(extracted_end_index), len(all_questions) - 1)

            extracted_questions = all_questions[start:end + 1]
            extracted_options = all_options[start:end + 1]

            print(f"🎯 [Solution] 추출된 문제 처리: 인덱스 {start}~{end} ({len(extracted_questions)}개)")

            agent = self.solution_runner
            if agent is None:
                raise RuntimeError("solution_runner is not initialized (init_agents=False).")

            generated_answers: List[str] = []
            generated_explanations: List[str] = []

            for i, (q, opts) in enumerate(zip(extracted_questions, extracted_options), start=1):
                print(f"🎯 [Solution] 추출된 문제 처리: {q[:100]}...")
                print(f"🎯 [Solution] 추출된 보기: {opts}")
                
                try:
                    # solution_agent는 키워드 인수를 받도록 설계됨
                    # 숏텀 메모리 키를 포함하여 호출
                    memory_key = f"solution_{start}_{i}"  # 고유한 메모리 키 생성
                    # 마지막 솔루션 쓰레드 ID를 보관하여 resume 시 사용
                    try:
                        self._last_solution_thread_id = memory_key
                    except Exception:
                        pass
                    # 상위 그래프 재개로 전달된 사용자 피드백이 있다면 서브그래프 최초 상태에 주입
                    pending_feedback = getattr(self, "_pending_user_feedback", None)
                    if pending_feedback:
                        print("🧩 상위 피드백 주입 → 서브그래프 최초 상태 전달")
                    agent_result = agent.invoke(
                        user_problem=q, 
                        user_problem_options=opts, 
                        user_input_txt=state.get("user_query", ""),
                        memory_key=memory_key,  # 숏텀 메모리 키 전달
                        user_feedback=pending_feedback if pending_feedback else None
                    )
                    print(f"✅ 추출된 문제 풀이 완료")
                    
                    # 결과에서 답변과 설명 추출
                    ans, exp = "", ""
                    if agent_result:
                        if isinstance(agent_result, dict) and agent_result.get("results"):
                            r0 = agent_result["results"][0]
                            ans = r0.get("generated_answer", "")
                            exp = r0.get("generated_explanation", "")
                        else:
                            ans = agent_result.get("generated_answer", "")
                            exp = agent_result.get("generated_explanation", "")
                    
                    generated_answers.append(ans or "")
                    generated_explanations.append(exp or "")
                    
                    # 결과를 solution 상태에도 저장
                    if "extracted_problem_results" not in new_state["solution"]:
                        new_state["solution"]["extracted_problem_results"] = []
                    new_state["solution"]["extracted_problem_results"].append(agent_result)
                    # 사용한 pending 피드백은 소비
                    if pending_feedback:
                        try:
                            delattr(self, "_pending_user_feedback")
                        except Exception:
                            pass
                    
                except Exception as e:
                    print(f"❌ 추출된 문제 풀이 중 오류 발생: {e}")
                    print(f"🔍 오류 상세: {type(e).__name__}: {e}")
                    generated_answers.append("")
                    generated_explanations.append("")
                    raise

            # 추출된 문제 결과를 shared 상태에 반영
            shared.setdefault("answer", [])
            shared.setdefault("explanation", [])
            shared["answer"].extend(generated_answers)
            shared["explanation"].extend(generated_explanations)

        # work.selected_indices 기반 처리 (pdf/추출 범위가 없을 때)
        if total_problems <= 0 and (sel_indices or sel_count > 0):
            all_questions = shared.get("question", [])
            all_options = shared.get("options", [])

            # 인덱스 보정 및 개수 적용
            if not sel_indices and sel_count > 0:
                sel_indices = list(range(0, min(sel_count, len(all_questions))))
            sel_indices = [i for i in sel_indices if 0 <= i < len(all_questions)]
            if not sel_indices:
                print("⚠️ 선택된 인덱스가 유효하지 않습니다.")
                return new_state

            sel_questions = [all_questions[i] for i in sel_indices]
            sel_options = [all_options[i] if i < len(all_options) else [] for i in sel_indices]

            print(f"🎯 [Solution] 선택 문제 처리: 인덱스 {sel_indices} ({len(sel_questions)}개)")

            agent = self.solution_runner
            if agent is None:
                raise RuntimeError("solution_runner is not initialized (init_agents=False).")

            generated_answers: List[str] = []
            generated_explanations: List[str] = []

            for i, (q, opts) in enumerate(zip(sel_questions, sel_options), start=1):
                if isinstance(opts, str):
                    opts = [x.strip() for x in opts.splitlines() if x.strip()]
                opts = [str(x).strip() for x in (opts or []) if str(x).strip()]
                if not q or not opts:
                    generated_answers.append("")
                    generated_explanations.append("")
                    continue
                try:
                    agent_result = agent.invoke(
                        user_problem=q,
                        user_problem_options=opts,
                        user_input_txt=state.get("user_query", "")
                    )
                except Exception as e:
                    print(f"❌ SolutionAgent invoke 실행 실패(선택 {i}/{len(sel_questions)}): {e}")
                    agent_result = None

                ans, exp = "", ""
                if agent_result:
                    if isinstance(agent_result, dict) and agent_result.get("results"):
                        r0 = agent_result["results"][0]
                        ans = r0.get("generated_answer", "")
                        exp = r0.get("generated_explanation", "")
                    else:
                        ans = agent_result.get("generated_answer", "")
                        exp = agent_result.get("generated_explanation", "")
                generated_answers.append(ans or "")
                generated_explanations.append(exp or "")

            shared.setdefault("answer", [])
            shared.setdefault("explanation", [])
            # 선택 인덱스에 맞춰 반영(길이 보정)
            while len(shared["answer"]) < len(shared.get("question", [])):
                shared["answer"].append("")
            while len(shared["explanation"]) < len(shared.get("question", [])):
                shared["explanation"].append("")
            for idx, (ans, exp) in zip(sel_indices, zip(generated_answers, generated_explanations)):
                shared["answer"][idx] = ans
                shared["explanation"][idx] = exp

        # subject 패딩
        need = len(shared["question"]) - len(shared.get("subject", []))
        if need > 0:
            shared.setdefault("subject", []).extend(["일반"] * need)

        validate_qas(shared)

        return new_state

    @traceable(name="teacher.generator")
    def generator(self, state: TeacherState) -> TeacherState:
        """
        문제 생성 노드 - generator_agent로 문제 생성
        """
        print("🎯 문제 생성 노드 실행")
        new_state: TeacherState = {**state}
        new_state = ensure_shared(new_state)
        new_state.setdefault("generation", {})
        
        # generator_agent 실행
        agent = self.generator_runner
        if agent is None:
            raise RuntimeError("generator_runner is not initialized (init_agents=False).")
        
        try:
            # 사용자 입력에서 생성 파라미터 추출
            user_query = state.get("user_query", "")
            agent_input = parse_generator_input(user_query)
            
            print(f"🎯 [Generator] 생성 파라미터: {agent_input}")
            
            # generator_agent를 subgraph로 실행
            agent_result = agent.invoke({
                "user_query": state.get("user_query", ""),
                "mode": agent_input.get("mode", "full_exam"),
                "selected_subjects": agent_input.get("selected_subjects", []),
                "questions_per_subject": agent_input.get("questions_per_subject", 10),
                "subject_area": agent_input.get("subject_area", ""),
                "target_count": agent_input.get("target_count", 10),
                "difficulty": agent_input.get("difficulty", "중급"),
                "save_to_file": agent_input.get("save_to_file", False)
            })
            
            if agent_result:
                # 생성된 문제를 shared state에 추가
                shared = new_state["shared"]
                
                # agent_result의 구조: {"success": True, "result": {...}}
                if agent_result.get("success") and "result" in agent_result:
                    result = agent_result["result"]
                    
                    # full_exam 모드: result.all_questions에서 추출
                    if "all_questions" in result:
                        questions = []
                        options = []
                        answers = []
                        explanations = []
                        subjects = []
                        
                        for q in result["all_questions"]:
                            if isinstance(q, dict):
                                questions.append(q.get("question", ""))
                                options.append(q.get("options", []))
                                answers.append(q.get("answer", ""))
                                explanations.append(q.get("explanation", ""))
                                subjects.append(q.get("subject", ""))
                        
                        if questions:
                            shared.setdefault("question", [])
                            shared["question"].extend(questions)
                            print(f"📝 [Generator] {len(questions)}개 문제 추가")
                        
                        if options:
                            shared.setdefault("options", [])
                            shared["options"].extend(options)
                            print(f"📝 [Generator] {len(options)}개 보기 추가")
                        
                        if answers:
                            shared.setdefault("answer", [])
                            shared["answer"].extend(answers)
                            print(f"📝 [Generator] {len(answers)}개 정답 추가")
                        
                        if explanations:
                            shared.setdefault("explanation", [])
                            shared["explanation"].extend(explanations)
                            print(f"📝 [Generator] {len(explanations)}개 해설 추가")
                        
                        if subjects:
                            shared.setdefault("subject", [])
                            shared["subject"].extend(subjects)
                    
                    # subject_quiz 모드: result.questions에서 추출
                    elif "questions" in result:
                        questions = []
                        options = []
                        answers = []
                        explanations = []
                        subjects = []
                        
                        for q in result["questions"]:
                            if isinstance(q, dict):
                                questions.append(q.get("question", ""))
                                options.append(q.get("options", []))
                                answers.append(q.get("answer", ""))
                                explanations.append(q.get("explanation", ""))
                                subjects.append(q.get("subject", result.get("subject_area", "")))
                        
                        if questions:
                            shared.setdefault("question", [])
                            shared["question"].extend(questions)
                            print(f"📝 [Generator] {len(questions)}개 문제 추가")
                        
                        if options:
                            shared.setdefault("options", [])
                            shared["options"].extend(options)
                            print(f"📝 [Generator] {len(options)}개 보기 추가")
                        
                        if answers:
                            shared.setdefault("answer", [])
                            shared["answer"].extend(answers)
                            print(f"📝 [Generator] {len(answers)}개 정답 추가")
                        
                        if explanations:
                            shared.setdefault("explanation", [])
                            shared["explanation"].extend(explanations)
                            print(f"📝 [Generator] {len(explanations)}개 해설 추가")
                        
                        if subjects:
                            shared.setdefault("subject", [])
                            shared["subject"].extend(subjects)
                    
                    # partial_exam 모드: result.all_questions에서 추출
                    elif "all_questions" in result:
                        questions = []
                        options = []
                        answers = []
                        explanations = []
                        subjects = []
                        
                        for q in result["all_questions"]:
                            if isinstance(q, dict):
                                questions.append(q.get("question", ""))
                                options.append(q.get("options", []))
                                answers.append(q.get("answer", ""))
                                explanations.append(q.get("explanation", ""))
                                subjects.append(q.get("subject", ""))
                        
                        if questions:
                            shared.setdefault("question", [])
                            shared["question"].extend(questions)
                            print(f"📝 [Generator] {len(questions)}개 문제 추가")
                        
                        if options:
                            shared.setdefault("options", [])
                            shared["options"].extend(options)
                            print(f"📝 [Generator] {len(options)}개 보기 추가")
                        
                        if answers:
                            shared.setdefault("answer", [])
                            shared["answer"].extend(answers)
                            print(f"📝 [Generator] {len(answers)}개 정답 추가")
                        
                        if explanations:
                            shared.setdefault("explanation", [])
                            shared["explanation"].extend(explanations)
                            print(f"📝 [Generator] {len(explanations)}개 해설 추가")
                        
                        if subjects:
                            shared.setdefault("subject", [])
                            shared["subject"].extend(subjects)
                    
                    # generation state에 결과 저장
                    new_state["generation"].update(agent_result)
                    
                    print(f"✅ [Generator] 문제 생성 완료: 총 {len(shared.get('question', []))}개 문제")
                else:
                    print(f"⚠️ [Generator] 문제 생성 실패: {agent_result.get('error', '알 수 없는 오류')}")
            else:
                print("⚠️ [Generator] 문제 생성 실패")
                
        except Exception as e:
            print(f"❌ [Generator] 문제 생성 중 오류: {e}")
        
        return new_state

    @traceable(name="teacher.score")
    def score(self, state: TeacherState) -> TeacherState:
        """
        채점 노드 - score_agent로 사용자 답안 채점
        """
        print("📊 채점 노드 실행")
        new_state: TeacherState = {**state}
        new_state = ensure_shared(new_state)
        new_state = self._ensure_work_selection(new_state)
        new_state.setdefault("score", {})
        
        # 사용자 답안 입력 받기
        shared = new_state["shared"]
        questions = shared.get("question", [])
        user_query = new_state.get("user_query", "")
        
        # ===== 채점 시작 전 데이터 확인 =====
        print("\n🔍 [Score] 채점 시작 전 데이터 확인:")
        print(f"  - 문제 수: {len(questions)}")
        print(f"  - 사용자 질문: {user_query}")
        print(f"  - 기존 정답: {len(shared.get('answer', []))}개")
        print(f"  - 기존 해설: {len(shared.get('explanation', []))}개")
        print(f"  - 기존 과목: {len(shared.get('subject', []))}개")
        
        if questions:
            print(f"  - 첫 번째 문제: {questions[0][:100]}{'...' if len(questions[0]) > 100 else ''}")
        
        if not questions:
            print("⚠️ 채점할 문제가 없습니다.")
            return new_state
        
        # 사용자 답안 입력: work.selected_indices 있으면 선택된 문제 수만큼 입력 유도
        work_sel = (new_state.get("work") or {})
        sel_indices: List[int] = list(work_sel.get("selected_indices", []) or [])
        sel_count: int = int(work_sel.get("select_count", 0) or 0)
        # 우선 사용자 입력 전체에서 파싱
        user_answer = get_user_answer(user_query)
        # 선택 인덱스가 있고, 파싱된 답 수가 선택 수와 불일치하면 앞에서 필요한 개수만 사용
        if sel_indices or sel_count > 0:
            need_n = len(sel_indices) if sel_indices else sel_count
            if isinstance(user_answer, list) and need_n > 0:
                user_answer = user_answer[:need_n]
        if not user_answer:
            print("⚠️ 사용자 답안을 입력받지 못했습니다.")
            return new_state
        
        # ===== 사용자 답안 파싱 결과 확인 =====
        print(f"\n📝 [Score] 사용자 답안 파싱 결과:")
        print(f"  - 원본 입력: {user_query}")
        print(f"  - 파싱된 답안: {user_answer}")
        print(f"  - 답안 개수: {len(user_answer) if isinstance(user_answer, list) else 'N/A'}")
        
        # shared state에 사용자 답안 저장
        shared["user_answer"] = user_answer
        
        # solution_agent에서 생성된 정답과 해설
        solution_answers = shared.get("answer", [])
        # 선택 인덱스 기반 채점: 선택된 문제에 대한 정답만 비교
        if (sel_indices or sel_count > 0) and isinstance(solution_answers, list):
            if not sel_indices and sel_count > 0:
                sel_indices = list(range(min(sel_count, len(questions))))
            sel_indices = [i for i in sel_indices if 0 <= i < len(solution_answers)]
            if sel_indices:
                solution_answers = [solution_answers[i] for i in sel_indices]
        if not solution_answers:
            print("⚠️ 정답이 없어서 채점할 수 없습니다.")
            return new_state
        
        # ===== 채점 실행 전 최종 데이터 확인 =====
        print(f"\n🎯 [Score] 채점 실행 전 최종 데이터:")
        print(f"  - 문제 수: {len(questions)}")
        print(f"  - 사용자 답안: {len(user_answer) if isinstance(user_answer, list) else 'N/A'}")
        print(f"  - 정답 수: {len(solution_answers)}")
        print(f"  - 답안 수: {len(shared.get('explanation', []))}")
        
        # score_agent 실행
        agent = self.score_runner
        if agent is None:
            raise RuntimeError("score_runner is not initialized (init_agents=False).")
        
        try:
            user_query = state.get("user_query", "")
            sh = shared
            
            # score_agent를 subgraph로 실행
            agent_result = agent.invoke({
                "user_answer": user_answer,
                "solution_answer": solution_answers,
                "user_query": user_query,
                "shared": sh
            })
            
            # ===== 채점 결과 확인 =====
            print(f"\n✅ [Score] 채점 결과:")
            print(f"  - agent_result 타입: {type(agent_result)}")
            print(f"  - agent_result 키: {list(agent_result.keys()) if isinstance(agent_result, dict) else 'N/A'}")
            print(f"  - agent_result 전체 내용: {agent_result}")
            
            if agent_result:
                # 채점 결과를 score state에 저장
                new_state["score"].update(agent_result)
                print(f"  - new_state['score'] 업데이트 후: {new_state['score']}")
                
                # shared state에 채점 결과 추가
                if "score_result" in agent_result:
                    shared["score_result"] = agent_result["score_result"]
                    print(f"  - shared['score_result'] 설정: {shared['score_result']}")
                else:
                    # score_result가 없으면 기본 구조 생성
                    shared["score_result"] = {
                        "correct_count": shared.get("correct_count", 0),
                        "total_count": shared.get("total_count", 0),
                        "accuracy": shared.get("correct_count", 0) / max(shared.get("total_count", 1), 1)
                    }
                    print(f"  - shared['score_result'] 기본값 설정: {shared['score_result']}")
                
                if "correct_count" in agent_result:
                    shared["correct_count"] = agent_result["correct_count"]
                    print(f"  - shared['correct_count'] 설정: {shared['correct_count']}")
                
                if "total_count" in agent_result:
                    shared["total_count"] = agent_result["total_count"]
                    print(f"  - shared['total_count'] 설정: {shared['total_count']}")
                
                # score_agent의 결과 구조에 따른 추가 처리
                if "results" in agent_result:
                    # ScoreEngine의 표준 결과 형태
                    results = agent_result["results"]
                    if isinstance(results, list):
                        correct_count = sum(1 for r in results if r == 1)
                        total_count = len(results)
                        shared["correct_count"] = correct_count
                        shared["total_count"] = total_count
                        print(f"  - results에서 계산된 정답 수: {correct_count}")
                        print(f"  - results에서 계산된 총 문제 수: {total_count}")
                
                print(f"  - 최종 정답 수: {shared.get('correct_count', 0)}")
                print(f"  - 최종 총 문제 수: {shared.get('total_count', 0)}")
                print(f"  - 정답률: {shared.get('correct_count', 0)}/{shared.get('total_count', 0)}")
                
                print(f"✅ [Score] 채점 완료: {shared.get('correct_count', 0)}/{shared.get('total_count', 0)} 정답")
            else:
                print("⚠️ [Score] 채점 실패")
                
        except Exception as e:
            print(f"❌ [Score] 채점 중 오류: {e}")
        
        # ===== 채점 완료 후 최종 상태 확인 =====
        print(f"\n🔍 [Score] 채점 완료 후 최종 상태:")
        print(f"  - shared['user_answer']: {len(shared.get('user_answer', []))}개")
        print(f"  - shared['correct_count']: {shared.get('correct_count', 'N/A')}")
        print(f"  - shared['total_count']: {shared.get('total_count', 'N/A')}")
        print(f"  - shared['score_result']: {'있음' if 'score_result' in shared else '없음'}")
        
        return new_state

    @traceable(name="teacher.analysis")
    def analysis(self, state: TeacherState) -> TeacherState:
        """분석 노드 - analysis_agent로 답안 분석"""
        print("�� 분석 노드 실행")
        new_state: TeacherState = {**state}
        new_state = ensure_shared(new_state)
        new_state.setdefault("analysis", {})
        
        # 분석에 필요한 데이터 확인
        shared = new_state["shared"]
        questions = shared.get("question", [])
        problem_types = shared.get("subject", [])
        user_answer = shared.get("user_answer", [])
        solution_answers = shared.get("answer", [])
        solution = shared.get("explanation", [])
        score_result = shared.get("score_result", {})
        
        # ===== 분석 시작 전 데이터 확인 =====
        print("\n�� [Analysis] 분석 시작 전 데이터 확인:")
        print(f"  - 문제 수: {len(questions)}")
        print(f"  - 과목 수: {len(problem_types)}")
        print(f"  - 사용자 답안: {len(user_answer) if isinstance(user_answer, list) else 'N/A'}")
        print(f"  - 정답 수: {len(solution_answers)}")
        print(f"  - 채점 결과: {shared.get('correct_count', 'N/A')}/{shared.get('total_count', 'N/A')}")
        print(f"  - score state: {new_state.get('score', {})}")
        print(f"  - shared state 키들: {list(shared.keys())}")
        
        # 채점 결과 상세 확인
        score_state = new_state.get('score', {})
        if score_state:
            print(f"  - score state 키들: {list(score_state.keys())}")
            if 'results' in score_state:
                results = score_state['results']
                print(f"  - score results: {results}")
                if isinstance(results, list):
                    print(f"  - score results 길이: {len(results)}")
                    print(f"  - score results 내용: {results[:10]}...")  # 처음 10개만
        
        if questions:
            print(f"  - 첫 번째 문제: {questions[0][:100]}{'...' if len(questions[0]) > 100 else ''}")
        
        if problem_types:
            print(f"  - 첫 번째 과목: {problem_types[0] if problem_types[0] else 'N/A'}")
        
        if user_answer:
            print(f"  - 첫 번째 사용자 답안: {user_answer[0] if isinstance(user_answer, list) and user_answer else 'N/A'}")
        
        if not questions or not user_answer or not solution_answers:
            print("⚠️ 분석에 필요한 데이터가 부족합니다.")
            print(f"    - questions: {'있음' if questions else '없음'}")
            print(f"    - user_answer: {'있음' if user_answer else '없음'}")
            print(f"    - solution_answers: {'있음' if solution_answers else '없음'}")
            return new_state
        
        # ===== 분석 실행 전 최종 데이터 검증 =====
        print(f"\n�� [Analysis] 분석 실행 전 최종 데이터 검증:")
        print(f"  - 문제와 답안 개수 일치: {'✅' if len(questions) == len(user_answer) else '❌'}")
        print(f"  - 문제와 정답 개수 일치: {'✅' if len(questions) == len(solution_answers) else '❌'}")
        print(f"  - 문제와 과목 개수 일치: {'✅' if len(questions) == len(problem_types) else '❌'}")
        
        if len(questions) != len(user_answer):
            print(f"    ⚠️ 문제 수({len(questions)})와 답안 수({len(user_answer)})가 일치하지 않습니다.")
        
        if len(questions) != len(solution_answers):
            print(f"    ⚠️ 문제 수({len(questions)})와 정답 수({len(solution_answers)})가 일치하지 않습니다.")
        
        if len(questions) != len(problem_types):
            print(f"    ⚠️ 문제 수({len(questions)})와 과목 수({len(problem_types)})가 일치하지 않습니다.")
        
        # analysis_agent 실행
        agent = self.analyst_runner
        if agent is None:
            raise RuntimeError("analyst_runner is not initialized (init_agents=False).")
        
        try:
            user_query = state.get("user_query", "")
            sh = shared
            
            # ===== analysis_agent 호출 전 최종 데이터 확인 =====
            print(f"\n🚀 [Analysis] analysis_agent 호출 전 최종 데이터:")
            print(f"  - problem: {len(questions)}개")
            print(f"  - user_answer: {len(user_answer)}개")
            print(f"  - problem_types: {len(problem_types)}개")
            print(f"  - solution_answer: {len(solution_answers)}개")
            print(f"  - solution: {len(solution)}개")
            print(f"  - user_query: {user_query}")
            
            # 데이터 상세 내용 확인
            if questions:
                print(f"  - 첫 번째 문제: {questions[0][:100]}...")
            if problem_types:
                print(f"  - 첫 번째 과목: {problem_types[0]}")
            if user_answer:
                print(f"  - 첫 번째 사용자 답안: {user_answer[0]}")
            if solution_answers:
                print(f"  - 첫 번째 정답: {solution_answers[0]}")
            if solution:
                print(f"  - 첫 번째 해설: {solution[0][:100] if len(solution[0]) > 100 else solution[0]}...")
            
            # ===== score_result 타입 확인 =====
            print(f"\n🔍 [Analysis] score_result 상세 확인:")
            print(f"  - score_result 타입: {type(score_result)}")
            print(f"  - score_result 값: {score_result}")
            
            # score state에서 results 추출
            score_state = new_state.get('score', {})
            print(f"  - score state: {score_state}")
            
            # 올바른 results 데이터 추출
            if score_state and 'results' in score_state:
                results_data = score_state['results']
                print(f"  - results_data 타입: {type(results_data)}")
                print(f"  - results_data 값: {results_data}")
            else:
                results_data = []
                print(f"  - results_data를 빈 리스트로 설정")
            
            # analysis_agent를 subgraph로 실행
            agent_input = {
                "problem": sh.get("question", []) or [],
                "user_answer": user_answer,
                "problem_types": problem_types,  # ✅ 과목 정보 전달
                "solution_answer": solution_answers,
                "user_query": user_query,
                "solution": solution,  # explanation 데이터를 solution으로 전달
                "results": results_data  # 수정: score_result 대신 results_data 사용
            }
            
            print(f"\n🔍 [Analysis] analysis_agent 입력 데이터:")
            for key, value in agent_input.items():
                if isinstance(value, list):
                    print(f"  - {key}: {len(value)}개")
                    if value and len(value) > 0:
                        print(f"    첫 번째 항목: {value[0]}")
                else:
                    print(f"  - {key}: {value}")
            
            agent_result = agent.invoke(agent_input)
            
            # ===== 분석 결과 확인 =====
            print(f"\n✅ [Analysis] 분석 결과:")
            print(f"  - agent_result 타입: {type(agent_result)}")
            print(f"  - agent_result 키: {list(agent_result.keys()) if isinstance(agent_result, dict) else 'N/A'}")
            
            if agent_result:
                # 분석 결과를 analysis state에 저장
                new_state["analysis"].update(agent_result)
                
                # shared state에 분석 결과 추가
                if "weak_type" in agent_result:
                    shared["weak_type"] = agent_result["weak_type"]
                    print(f"  - 약점 유형: {len(agent_result['weak_type'])}개")
                
                if "wrong_question" in agent_result:
                    shared["wrong_question"] = agent_result["wrong_question"]
                    print(f"  - 오답 문제: {len(agent_result['wrong_question'])}개")
                
                print("✅ [Analysis] 분석 완료")
            else:
                print("⚠️ [Analysis] 분석 실패")
                
        except Exception as e:
            print(f"❌ [Analysis] 분석 중 오류: {e}")
            import traceback
            traceback.print_exc()
        
        # ===== 분석 완료 후 최종 상태 확인 =====
        print(f"\n�� [Analysis] 분석 완료 후 최종 상태:")
        print(f"  - shared['weak_type']: {len(shared.get('weak_type', []))}개")
        print(f"  - shared['wrong_question']: {len(shared.get('wrong_question', []))}개")
        print(f"  - analysis state 키: {list(new_state.get('analysis', {}).keys())}")
        
        return new_state

    @traceable(name="teacher.generate_problem_pdf")
    def generate_problem_pdf(self, state: TeacherState) -> TeacherState:
        """
        문제집 PDF 생성 노드 (방금 추가된 범위만 출력)
        - artifacts.pdf_added_start_index / end_index / count 를 우선 사용
        - 인덱스 정보가 없거나 비정상이면 전체로 폴백
        """
        print("📄 문제집 PDF 생성 노드 실행")
        new_state: TeacherState = {**state}
        try:
            new_state = ensure_shared(new_state)
            shared = new_state["shared"]
            arts = new_state.setdefault("artifacts", {})

            questions = shared.get("question", []) or []
            options_list = shared.get("options", []) or []
            total_n = min(len(questions), len(options_list))

            if total_n == 0:
                print("⚠️ 문제집 PDF 생성할 문제가 없습니다.")
                return new_state

            # 기본값(전체)
            start, end = 0, total_n - 1

            # 방금 추가 범위 시도
            s = arts.get("pdf_added_start_index")
            e = arts.get("pdf_added_end_index")
            c = arts.get("pdf_added_count")

            if isinstance(c, int) and c > 0:
                if isinstance(s, int) and isinstance(e, int) and 0 <= s <= e < total_n:
                    start, end = s, e
                else:
                    # 인덱스 기록이 비정상인 경우: "마지막 c개"로 폴백
                    start = max(0, total_n - c)
                    end = total_n - 1

            problems = []
            for i in range(start, end + 1):
                q = questions[i]
                opts = options_list[i]
                if isinstance(opts, str):
                    opts = [x.strip() for x in opts.splitlines() if x.strip()]
                if not isinstance(opts, list):
                    opts = []
                opts = [str(x).strip() for x in opts if str(x).strip()]
                problems.append({"question": q, "options": opts})

            if not problems:
                print("⚠️ 문제집 PDF 생성할 문제가 없습니다.")
                return new_state

            from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
            generator = ComprehensivePDFGenerator()

            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "agents", "solution", "pdf_outputs")
            )
            os.makedirs(base_dir, exist_ok=True)

            uq = (state.get("user_query") or "exam").strip()
            safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
            suffix = "" if (start == 0 and end == total_n - 1) else f"_{start+1}-{end+1}"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지
            output_path = os.path.join(base_dir, f"{safe_uq}_분석리포트{suffix}_{ts}.pdf")

            # 일부 구현은 반환값이 None → 변수에 안 받습니다.
            generator.generate_problem_booklet(problems, output_path, f"{safe_uq} 문제집")
            print(f"✅ 문제집 PDF 생성 완료: {output_path}")

            arts.setdefault("generated_pdfs", []).append(output_path)
        except Exception as e:
            print(f"❌ 문제집 PDF 생성 중 오류: {e}")
        return new_state


    @traceable(name="teacher.generate_answer_pdf")
    def generate_answer_pdf(self, state: TeacherState) -> TeacherState:
        """
        답안집 PDF 생성 노드 (방금 추가된 범위만 출력)
        - artifacts.pdf_added_start_index / end_index / count 를 우선 사용
        - 인덱스 정보가 없거나 비정상이면 전체로 폴백
        """
        print("📄 답안집 PDF 생성 노드 실행")
        new_state: TeacherState = {**state}
        try:
            new_state = ensure_shared(new_state)
            shared = new_state["shared"]
            arts = new_state.setdefault("artifacts", {})

            questions     = shared.get("question", []) or []
            options_list  = shared.get("options", []) or []
            answers       = shared.get("answer", []) or []
            explanations  = shared.get("explanation", []) or []
            total_n = min(len(questions), len(options_list), len(answers), len(explanations))

            if total_n == 0:
                print("⚠️ 답안집 PDF 생성에 필요한 데이터가 없습니다.")
                return new_state

            # 기본값(전체)
            start, end = 0, total_n - 1

            # 방금 추가 범위 시도
            s = arts.get("pdf_added_start_index")
            e = arts.get("pdf_added_end_index")
            c = arts.get("pdf_added_count")

            if isinstance(c, int) and c > 0:
                if isinstance(s, int) and isinstance(e, int) and 0 <= s <= e < total_n:
                    start, end = s, e
                else:
                    # 인덱스 기록이 비정상인 경우: "마지막 c개"로 폴백
                    start = max(0, total_n - c)
                    end = total_n - 1

            problems = []
            for i in range(start, end + 1):
                q = questions[i]
                opts = options_list[i]
                if isinstance(opts, str):
                    opts = [x.strip() for x in opts.splitlines() if x.strip()]
                if not isinstance(opts, list):
                    opts = []
                opts = [str(x).strip() for x in opts if str(x).strip()]
                ans = answers[i]
                exp = explanations[i]
                problems.append({
                    "question": q,
                    "options": opts,
                    "generated_answer": ans,
                    "generated_explanation": exp,
                })

            if not problems:
                print("⚠️ 답안집 PDF 생성할 문제가 없습니다.")
                return new_state

            from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
            generator = ComprehensivePDFGenerator()

            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "agents", "solution", "pdf_outputs")
            )
            os.makedirs(base_dir, exist_ok=True)

            uq = (state.get("user_query") or "exam").strip()
            safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
            suffix = "" if (start == 0 and end == total_n - 1) else f"_{start+1}-{end+1}"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지
            output_path = os.path.join(base_dir, f"{safe_uq}_답안집{suffix}_{ts}.pdf")

            # 일부 구현은 반환값이 None → 변수에 안 받습니다.
            generator.generate_answer_booklet(problems, output_path, f"{safe_uq} 답안집")
            print(f"✅ 답안집 PDF 생성 완료: {output_path}")

            arts.setdefault("generated_pdfs", []).append(output_path)
        except Exception as e:
            print(f"❌ 답안집 PDF 생성 중 오류: {e}")
        return new_state
    
    @traceable(name="teacher.generate_analysis_pdf")
    def generate_analysis_pdf(self, state: TeacherState) -> TeacherState:
        """
        분석 리포트 PDF 생성 노드 (방금 추가된 범위만 출력)
        - analysis 에이전트의 결과(payload['analysis'])를 함께 전달
        """
        print("📄 분석 리포트 PDF 생성 노드 실행")
        new_state: TeacherState = ensure_shared({**state})

        try:
            sh   = new_state["shared"]
            arts = new_state.setdefault("artifacts", {})

            questions        = sh.get("question", []) or []
            options_list     = sh.get("options", []) or []
            user_answer      = sh.get("user_answer", []) or []
            solution_answers = sh.get("answer", []) or []
            explanations     = sh.get("explanation", []) or []
            weak_type        = sh.get("weak_type", []) or []

            total_n = min(len(questions), len(solution_answers), len(user_answer))
            if total_n == 0:
                print("⚠️ 분석 리포트 PDF 생성에 필요한 데이터가 부족합니다.")
                return new_state

            # 범위 결정 (방금 추가된 범위 우선)
            start, end = 0, total_n - 1
            c = arts.get("pdf_added_count"); s = arts.get("pdf_added_start_index"); e = arts.get("pdf_added_end_index")
            if isinstance(c, int) and c > 0:
                if isinstance(s, int) and isinstance(e, int) and 0 <= s <= e < total_n:
                    start, end = s, e
                else:
                    start = max(0, total_n - c); end = total_n - 1

            # 슬라이스 + 정규화
            def _norm_opts(x):
                if isinstance(x, str):
                    return [t.strip() for t in x.splitlines() if t.strip()]
                if isinstance(x, list):
                    return [str(t).strip() for t in x if str(t).strip()]
                return []

            sub_q    = questions[start:end + 1]
            sub_opts = [ _norm_opts(o) for o in (options_list[start:end + 1] if options_list else [[]]*(end-start+1)) ]
            sub_user = user_answer[start:end + 1] if len(user_answer) >= end + 1 else user_answer[:]
            sub_sol  = solution_answers[start:end + 1]
            sub_exp  = (explanations[start:end + 1] if explanations else [""] * (end - start + 1))

            problems = []
            for q, opts, u, s_, ex in zip(sub_q, sub_opts, sub_user, sub_sol, sub_exp):
                problems.append({
                    "question": str(q),
                    "options": opts,
                    "user_answer": str(u),
                    "generated_answer": str(s_),
                    "generated_explanation": str(ex),
                })

            # score_result 없을 때 정확도 계산 폴백
            import re
            def _num(x):
                m = re.search(r'\d+', str(x))
                return m.group(0) if m else None
            auto_results = [1 if (_num(u) and _num(s_) and _num(u) == _num(s_)) else 0 for u, s_ in zip(sub_user, sub_sol)]
            score_result = sh.get("score_result")
            if not isinstance(score_result, dict) or "total_count" not in score_result:
                score_result = {
                    "correct_count": sum(auto_results),
                    "total_count": len(auto_results),
                    "accuracy": (sum(auto_results) / len(auto_results)) if auto_results else 0.0,
                }

            # analysis 에이전트 결과(상세/총평)도 함께 전달
            analysis_payload = (new_state.get("analysis") or {}).get("analysis")

            analysis_data = {
                "problems": problems,
                "weak_types": weak_type,                 # ['문자'] 또는 [{'label':..}] 모두 허용
                "score_result": score_result,
                "analysis": analysis_payload,            # ★ 상세/총평 전달
                "range": {"start_index": start, "end_index": end},
            }

            from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
            generator = ComprehensivePDFGenerator()

            base_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "agents", "solution", "pdf_outputs"
            ))
            os.makedirs(base_dir, exist_ok=True)

            uq = (state.get("user_query") or "exam").strip()
            safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
            suffix = "" if (start == 0 and end == total_n - 1) else f"_{start+1}-{end+1}"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지
            output_path = os.path.join(base_dir, f"{safe_uq}_분석리포트{suffix}_{ts}.pdf")

            generator.generate_analysis_report(analysis_data, output_path, f"{safe_uq} 분석 리포트")
            print(f"✅ 분석 리포트 PDF 생성 완료: {output_path}")

            new_state["artifacts"].setdefault("generated_pdfs", []).append(output_path)

        except Exception as e:
            print(f"❌ 분석 리포트 PDF 생성 중 오류: {e}")

        return new_state

    @traceable(name="teacher.generate_response")
    def generate_response(self, state: TeacherState) -> TeacherState:
        """
        사용자에게 실행 결과를 요약해서 답변하는 노드
        """
        print("💬 사용자 답변 생성 노드 실행")
        new_state: TeacherState = {**state}
        
        try:
            # generate_user_response 함수를 호출하여 사용자 친화적인 답변 생성
            user_response = generate_user_response(state)
            
            # 답변을 TeacherState에 직접 저장
            new_state["llm_response"] = user_response
            
            print(f"✅ 사용자 답변 생성 완료: {user_response[:100]}{'...' if len(user_response) > 100 else ''}")
            
        except Exception as e:
            print(f"❌ 사용자 답변 생성 중 오류: {e}")
            # 오류 발생 시 기본 답변 설정
            new_state["llm_response"] = "작업이 완료되었습니다. 추가로 도움이 필요한 부분이 있으시면 말씀해 주세요."
        
        return new_state




    @traceable(name="teacher.generate_pdfs")
    def generate_pdfs(self, state: TeacherState) -> TeacherState:
        """
        통합 PDF 생성 노드 (모든 PDF를 한 번에 생성)
        """
        print("📄 통합 PDF 생성 노드 실행")
        new_state: TeacherState = {**state}
        
        try:
            # 문제집 PDF 생성
            new_state = self.generate_problem_pdf(new_state)
            
            # 답안집 PDF 생성
            new_state = self.generate_answer_pdf(new_state)
            
            # 분석 리포트 PDF 생성
            new_state = self.generate_analysis_pdf(new_state)
            
            print("✅ 모든 PDF 생성 완료")
            
        except Exception as e:
            print(f"❌ 통합 PDF 생성 중 오류: {e}")
        
        return new_state

    @traceable(name="teacher.retrieve")
    def retrieve(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("정보 검색 노드 실행")
        agent = self.retriever_runner
        if agent is None:
            raise RuntimeError("retriever_runner is not initialized (init_agents=False).")
        # retriever_agent를 subgraph로 실행
        try:
            agent_result = agent.invoke({
                "retrieval_question": state.get("user_query", ""),
                "user_query": state.get("user_query", ""),
                "shared": state.get("shared", {})
            })
        except Exception as e:
            print(f"[WARN] retriever_agent 실행 실패: {e}")
            agent_result = {}
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

        # PDF Generation nodes
        builder.add_node("generate_pdfs", RunnableLambda(self.generate_pdfs))
        builder.add_node("generate_problem_pdf", RunnableLambda(self.generate_problem_pdf))
        builder.add_node("generate_answer_pdf", RunnableLambda(self.generate_answer_pdf))
        builder.add_node("generate_analysis_pdf", RunnableLambda(self.generate_analysis_pdf))

        # User Response Generation node
        builder.add_node("generate_response", RunnableLambda(self.generate_response))

        # Routing markers
        builder.add_node("mark_after_generator_solution", RunnableLambda(mark_after_generator_solution))
        builder.add_node("mark_after_solution_score", RunnableLambda(mark_after_solution_score))
        builder.add_node("mark_after_score_analysis", RunnableLambda(mark_after_score_analysis))

        # Routers
        builder.add_node("route_solution", RunnableLambda(route_solution))
        builder.add_node("route_score", RunnableLambda(route_score))
        builder.add_node("route_analysis", RunnableLambda(route_analysis))

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
                "preprocess": "preprocess",  # solution 의도일 때 preprocess로
                "route_score": "route_score",
            },
        )

        # route_solution
        builder.add_conditional_edges(
            "route_solution",
            lambda state: state.get("routing", {}).get("solution_next", "mark_after_generator_solution"),
            {
                "solution": "solution",
                "mark_after_generator_solution": "mark_after_generator_solution",
            },
        )
        builder.add_conditional_edges(
            "preprocess",
            lambda state: "solution" if state.get("artifacts", {}).get("extracted_problem_count", 0) > 0 or state.get("artifacts", {}).get("pdf_added_count", 0) > 0 else "mark_after_generator_solution",
            {
                "solution": "solution",
                "mark_after_generator_solution": "mark_after_generator_solution",
            },
        )
        builder.add_edge("mark_after_generator_solution", "generator")

        # route_score
        builder.add_conditional_edges(
            "route_score",
            lambda state: state.get("routing", {}).get("score_next", "mark_after_solution_score"),
            {
                "score": "score",
                "mark_after_solution_score": "mark_after_solution_score",
            },
        )
        builder.add_edge("mark_after_solution_score", "solution")

        # route_analysis
        builder.add_conditional_edges(
            "route_analysis",
            lambda state: state.get("routing", {}).get("analysis_next", "mark_after_score_analysis"),
            {
                "analysis": "analysis",
                "mark_after_score_analysis": "mark_after_score_analysis",
            },
        )
        builder.add_edge("mark_after_score_analysis", "score")

        # post dependencies - 자동 PDF 생성 강화
        builder.add_conditional_edges(
            "generator",
            post_generator_route,
            {
                "solution": "solution",
                "generate_problem_pdf": "generate_problem_pdf",
            },
        )
        builder.add_conditional_edges(
            "solution",
            post_solution_route,
            {
                "score": "score",
                "generate_answer_pdf": "generate_answer_pdf",
            },
        )
        builder.add_conditional_edges(
            "score",
            post_score_route,
            {
                "analysis": "analysis",
                "generate_answer_pdf": "generate_answer_pdf",  # 채점 후 답안집 PDF 생성
            },
        )

        # retrieve → persist, analysis → generate_analysis_pdf → persist → END
        builder.add_edge("retrieve", "persist_state")
        builder.add_edge("analysis", "generate_analysis_pdf")
        builder.add_edge("generate_analysis_pdf", "persist_state")
        builder.add_edge("generate_problem_pdf", "persist_state")
        builder.add_edge("generate_answer_pdf", "persist_state")
        builder.add_edge("persist_state", "generate_response")
        builder.add_edge("generate_response", END)

        print("✅ LangGraph 워크플로우 그래프 생성 완료")
        return builder.compile(checkpointer=self.checkpointer)

    def invoke(self, state: TeacherState, config: Optional[Dict] = None) -> TeacherState:
        """LangGraph 기반으로 워크플로우를 실행합니다."""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        # 체크포인터와 함께 그래프 실행
        try:
            result = self.graph.invoke(state, config)
            return result
        except Exception as e:
            print(f"❌ 그래프 실행 중 오류 발생: {e}")
            # interrupt가 발생한 경우 체크포인터에서 상태 복구 시도
            if "interrupt" in str(e).lower():
                print("🔄 interrupt가 발생했습니다. 체크포인터에서 상태를 확인하세요.")
                print("💡 Command(resume)을 사용하여 워크플로우를 재개할 수 있습니다.")
            raise

    def resume_workflow(self, resume_data: str, config: Optional[Dict] = None) -> TeacherState:
        """Command(resume)을 사용하여 중단된 워크플로우를 재개합니다."""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        # 상위 그래프에서 재개: 사용자 피드백을 임시 저장하여 solution 노드가 서브그래프 초기 상태로 전달
        try:
            self._pending_user_feedback = resume_data
        except Exception:
            pass

        # LangGraph 버전에 따른 Command import 시도
        try:
            from langgraph.checkpoint.memory import Command
        except ImportError:
            try:
                from langgraph import Command
            except ImportError:
                try:
                    from langgraph.types import Command
                except ImportError:
                    print("❌ Command를 import할 수 없습니다. LangGraph 버전을 확인해주세요.")
                    raise ImportError("Command import 실패")
        
        try:
            print(f"🔄 워크플로우 재개 중... resume_data: {resume_data}")
            print(f"🔍 체크포인터 상태 확인: {self.checkpointer}")
            
            # 숏텀 메모리에서 solution_agent 상태 복구 시도
            try:
                from common.short_term.redis_memory import RedisLangGraphMemory
                redis_memory = RedisLangGraphMemory()
                
                # solution_agent의 메모리 키들을 찾아서 상태 복구
                memory_keys = redis_memory.keys("solution_*")
                if memory_keys:
                    print(f"🔍 숏텀 메모리에서 solution 상태 발견: {len(memory_keys)}개")
                    for key in memory_keys:
                        state_data = redis_memory.get(key)
                        if state_data and state_data.get("interrupt_occurred"):
                            print(f"💾 복구된 상태: {key}")
                            # 상태를 체크포인터에 저장
                            if hasattr(self, 'checkpointer') and self.checkpointer:
                                self.checkpointer.put(config.get("configurable", {}).get("thread_id", "default"), state_data)
            except Exception as mem_err:
                print(f"⚠️ 숏텀 메모리 복구 실패: {mem_err}")
            
            # Command(resume)을 사용하여 중단된 지점부터 재개
            resume_command = Command(resume={"data": resume_data})
            print(f"📤 Command(resume) 전송: {resume_command}")
            
            # 체크포인터가 설정된 그래프로 재개
            result = self.graph.invoke(resume_command, config)
            print("✅ 워크플로우 재개 완료")
            return result
        except Exception as e:
            print(f"❌ 워크플로우 재개 실패: {e}")
            print(f"🔍 오류 상세: {type(e).__name__}: {str(e)}")
            raise

    # ── Memory IO ────────────────────────────────────────────────────────────

    
    # Streamlit 앱에서 사용할 함수
def create_app() -> Any:
    """Streamlit 앱에서 사용할 teacher graph 앱을 생성합니다."""
    orch = Orchestrator(user_id="streamlit_user", service="teacher", chat_id="web")
    return orch.build_teacher_graph()

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
                # artifacts는 intent_classifier에서 사용자 입력을 기반으로 동적으로 설정됩니다
                "artifacts": {},
            }

            try:
                # 체크포인터 필수 키(thread_id)를 기본으로 설정하여 모든 에이전트가 정상 실행되도록 함
                result: Dict[str, Any] = orch.invoke(init_state)
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

            # 문항 미리보기(있으면)
            # if q_cnt:
            #     print(f"\n=== 생성된 {q_cnt}개 문제 ===")
            #     for i in range(q_cnt):
            #         q = shared["question"][i] if i < len(shared["question"]) else ""
            #         opts = (shared.get("options") or [[]] * q_cnt)[i] if i < len(shared.get("options") or []) else []
            #         ans = (shared.get("answer") or [""] * q_cnt)[i] if i < len(shared.get("answer") or []) else ""
            #         exp = (shared.get("explanation") or [""] * q_cnt)[i] if i < len(shared.get("explanation") or []) else ""
                    
            #         print(f"\n[문제 {i+1}] {str(q)[:150]}{'...' if len(str(q))>150 else ''}")
            #         if opts:
            #             print("  Options:", "; ".join(opts[:6]) + ("..." if len(opts) > 6 else ""))
            #         if ans:
            #             print(f"  Answer: {str(ans)[:100]}{'...' if len(str(ans))>100 else ''}")
            #         if exp:
            #             print(f"  Explanation: {str(exp)[:120]}{'...' if len(str(exp))>120 else ''}")
            #     print("=" * 50)

            # 최근 모델 풀이/해설 미리보기 제거 (각 문제마다 이미 표시됨)

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

            # 사용자 답변 출력
            llm_response = result.get("llm_response")
            if llm_response:
                print(f"\n💬 [LLM 답변] {llm_response}")

            print("-----------------\n")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 종료합니다.")
