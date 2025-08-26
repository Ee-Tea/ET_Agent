# teacher_graph.py
# uv run teacher/teacher_graph.py
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict, NotRequired

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ──────────────────────────────────────────────────────────────────────────────
# 경로는 실제 프로젝트 구조에 맞게 하나만 활성화하세요.
# from ...common.short_term.redis_memory import RedisLangGraphMemory   # 상대 임포트(패키지 실행 전제)
# from ..common.short_term.redis_memory import RedisLangGraphMemory   # 절대 임포트(권장)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from common.short_term.redis_memory import RedisLangGraphMemory

from agents.analysis.analysis_agent import AnalysisAgent
from agents.score.score_engine import ScoreEngine as score_agent
from agents.retrieve.retrieve_agent import retrieve_agent
# from agents.TestGenerator.pdf_quiz_groq_class import InfoProcessingExamAgent as generate_agent
from agents.TestGenerator.generator import InfoProcessingExamAgent as generate_agent
from agents.solution.solution_agent import SolutionAgent as solution_agent
from teacher_nodes import get_user_answer, parse_generator_input
from file_path_mapper import FilePathMapper
from datetime import datetime
# ──────────────────────────────────────────────────────────────────────────────

# ========== 타입/프로토콜 ==========
class SupportsExecute:
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...

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

# 의도 정규화 헬퍼
CANON_INTENTS = {"retrieve","generate","analyze","solution","score"}

def normalize_intent(raw: str) -> str:
    s = (raw or "").strip().strip('"\'' ).lower()  # 양끝 따옴표/공백 제거
    # 흔한 별칭/오타 흡수
    alias = {
    "generator":"generate",
    "problem_generation":"generate",
    "make":"generate","create":"generate","생성":"generate","만들":"generate",
    "analysis":"analyze","분석":"analyze",
    "search":"retrieve","lookup":"retrieve","검색":"retrieve",
    "solve":"solution","풀이":"solution",
    "grade":"score","채점":"score",
    }
    s = alias.get(s, s)
    return s if s in CANON_INTENTS else "retrieve"

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
    
    # PDF 파일이 있으면 항상 전처리 수행 (새로운 파일이므로)
    pdf_ids = art.get("pdf_ids", [])
    
    # 이미지 파일도 체크 (새로 추가)
    image_ids = art.get("image_ids", [])
    
    # 디버깅 로그 추가
    print(f"🔍 [전처리 체크] PDF 파일: {pdf_ids}")
    print(f"🔍 [전처리 체크] 이미지 파일: {image_ids}")
    result = bool(pdf_ids) or bool(image_ids)
    print(f"🔍 [전처리 체크] 결과: {result} (PDF 있음: {bool(pdf_ids)}, 이미지 있음: {bool(image_ids)})")
    
    # PDF 또는 이미지 파일이 있으면 전처리 필요 (기존 문제 상관없이)
    return result

# ========== Orchestrator ==========
class Orchestrator:
    def __init__(self, user_id: str, service: str, chat_id: str, init_agents: bool = True):
        load_dotenv()
        if not os.getenv("LANGCHAIN_API_KEY=REDACTED("경고: LANGCHAIN_API_KEY=REDACTED not os.getenv("OPENAI_VISION_MODEL"):
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

        # PDF 전처리 모듈 import (편의 함수들)
        from pdf_preprocessor import extract_pdf_paths, extract_problem_range, determine_problem_source

        # PDF 경로 추출 및 artifacts 업데이트
        extracted_pdfs = extract_pdf_paths(uq)
        current_artifacts = state.get("artifacts", {})
        
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

        # 이미지 파일 경로 추출 (새로 추가)
        extracted_images = self._extract_image_paths(uq)
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

        # LLM 기반 의도 분류
        try:
            from teacher_nodes import user_intent
            raw = user_intent(uq) if uq else ""
            intent = normalize_intent(raw or "retrieve")
            print(f"🤖 LLM 기반 분류: {intent} (raw={raw!r})")
        except Exception as e:
            print(f"⚠️ LLM 분류 실패, 기본값 사용: {e}")
            raw = "fallback"
            intent = "retrieve"
            
        return {**state, "user_query": uq, "intent": intent, "artifacts": current_artifacts}

    def select_agent(self, state: TeacherState) -> str:
        try:
            intent_norm = normalize_intent(state.get("intent", ""))
        except NameError:
            intent_norm = (state.get("intent","") or "").strip().strip('"\'' ).lower()
        mapping = {
            "retrieve": "retrieve",
            "generate": "generator",
            "analyze": "route_analysis",
            "solution": "route_solution",
            "score": "route_score",
        }
        chosen = mapping.get(intent_norm, "retrieve")
        print(f"[router] intent={intent_norm} → {chosen}")
        return chosen

    # ── Router (의존성 자동 보장) ───────────────────────────────────────────
    def route_solution(self, state: TeacherState) -> TeacherState:
        # 라우팅 정보를 state에 저장
        intent = state.get("intent", "")
        artifacts = state.get("artifacts", {})
        
        # 우선순위: 전처리 필요 → 전처리 후 solution → 기존 문제로 solution
        if has_files_to_preprocess(state):
            next_node = "preprocess"
            print("📄 PDF 파일 전처리 후 solution 실행")
        elif has_questions(state):
            next_node = "solution"
            print("📄 기존 문제로 solution 실행")
        else:
            next_node = "mark_after_generator_solution"
        
        new_state = {**state}
        new_state.setdefault("routing", {})
        new_state["routing"]["solution_next"] = next_node
        return new_state

    def route_score(self, state: TeacherState) -> TeacherState:
        next_node = "score" if has_solution_answers(state) else "mark_after_solution_score"
        new_state = {**state}
        new_state.setdefault("routing", {})
        new_state["routing"]["score_next"] = next_node
        return new_state

    def route_analysis(self, state: TeacherState) -> TeacherState:
        next_node = "analysis" if has_score(state) else "mark_after_score_analysis"
        new_state = {**state}
        new_state.setdefault("routing", {})
        new_state["routing"]["analysis_next"] = next_node
        return new_state

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
        return nxt if nxt else "generate_problem_pdf"  # 기본적으로 문제집 PDF 생성

    def post_solution_route(self, state: TeacherState) -> str:
        nxt = ((state.get("routing") or {}).get("after_solution") or "").strip()
        return nxt if nxt else "generate_answer_pdf"  # 기본적으로 답안집 PDF 생성

    def post_score_route(self, state: TeacherState) -> str:
        nxt = ((state.get("routing") or {}).get("after_score") or "").strip()
        return nxt if nxt else "analysis"  # 기본적으로 분석 진행

    def post_analysis_route(self, state: TeacherState) -> str:
        nxt = ((state.get("routing") or {}).get("after_analysis") or "").strip()
        return nxt if nxt else "generate_analysis_pdf"  # 기본적으로 분석 리포트 PDF 생성

    # ── Nodes ───────────────────────────────────────────────────────────────
    @traceable(name="teacher.preprocess")  
    def preprocess(self, state: TeacherState) -> TeacherState:
        """
        PDF 및 이미지 파일에서 문제 추출하는 전처리 노드
        - 파일 종류에 따라 적절한 처리 방법 선택
        - 인덱스 기록을 'extend 이전' 길이로 고정해 올바른 범위를 남깁니다.
        - 불필요한 장황 로그를 줄였습니다.
        """
        print("📄 PDF/이미지 문제 추출 전처리 노드 실행")

        artifacts = state.get("artifacts", {}) or {}
        file_mapper = FilePathMapper()
        external_file_paths = file_mapper.map_artifacts_to_paths(artifacts)

        if not external_file_paths:
            print("⚠️ 전처리할 파일이 없습니다.")
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
                pdf_problems = self._extract_problems_from_pdf(pdf_files)
                extracted_problems.extend(pdf_problems or [])
                print(f"📄 PDF에서 {len(pdf_problems or [])}개 문제 추출")
            
            # 이미지 파일 처리
            if image_files:
                print("🖼️ 이미지 파일에서 문제 추출 중...")
                image_problems = self._extract_problems_from_images(image_files)
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
        new_state.setdefault("solution", {})

        artifacts = new_state.get("artifacts", {}) or {}
        shared = new_state["shared"]

        pdf_added_count = int(artifacts.get("pdf_added_count", 0) or 0)
        start_index = artifacts.get("pdf_added_start_index", None)
        end_index = artifacts.get("pdf_added_end_index", None)

        if pdf_added_count <= 0 or start_index is None or end_index is None or end_index < start_index:
            print("⚠️ PDF에서 추가된 문제가 없거나 인덱스가 유효하지 않습니다.")
            return new_state

        all_questions = shared.get("question", [])
        all_options = shared.get("options", [])

        # 범위 보정
        start = max(0, min(int(start_index), len(all_questions)))
        end = min(int(end_index), len(all_questions) - 1)

        pdf_questions = all_questions[start:end + 1]
        pdf_options = all_options[start:end + 1]

        print(f"🎯 [Solution] 처리할 문제: 인덱스 {start}~{end} ({len(pdf_questions)}개)")

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
            # # 여러 구현과 호환을 위해 가능한 키들을 모두 전달
            # agent_input_state = {
            #     "user_input_txt": state.get("user_query", ""),
            #     "user_problem": problem_payload["question"],
            #     "user_problem_options": problem_payload["options"],
            # }
            # print(f"🎯 [Solution] 처리할 문제: {agent_input_state}")
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

        # subject 패딩
        need = len(shared["question"]) - len(shared.get("subject", []))
        if need > 0:
            shared.setdefault("subject", []).extend(["일반"] * need)

        validate_qas(shared)

        # (중요) 여기서 예전처럼 자동으로 답안집을 바로 만들지 않습니다.
        # 라우팅에 의해 generate_answer_pdf 노드가 실행되도록 둡니다.

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
        
        # 사용자 답안 입력 받기
        user_answer = get_user_answer(user_query)
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

    def _extract_problems_from_pdf(self, file_paths: List[str]) -> List[Dict]:
        """PDF 파일에서 문제 추출 (pdf_preprocessor 사용)"""
        results: List[Dict] = []
        for p in file_paths:
            try:
                items = self.pdf_preprocessor.extract(p)  # [{question, options}]
                if isinstance(items, list):
                    results.extend(items)
            except Exception as e:
                print(f"[WARN] PDF 추출 실패({p}): {e}")
        return results

    def _extract_problems_from_images(self, image_paths: List[str]) -> List[Dict]:
        """이미지 파일에서 문제 추출 (img2json_generation 사용)"""
        try:
            # img2json_generation 모듈 import
            from agents.solution.img2json_generation import call_gpt_on_images
            
            print(f"🖼️ 이미지에서 문제 추출 시작: {len(image_paths)}개 파일")
            
            # 이미지 파일 존재 여부 재확인
            valid_paths = []
            for path in image_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    print(f"⚠️ 이미지 파일을 찾을 수 없음: {path}")
            
            if not valid_paths:
                print("❌ 처리할 수 있는 이미지 파일이 없습니다.")
                return []
            
            print(f"🖼️ 유효한 이미지 파일: {len(valid_paths)}개")
            
            # call_gpt_on_images 함수 호출
            result = call_gpt_on_images(valid_paths)
            
            if not result or "problems" not in result:
                print("⚠️ 이미지에서 문제를 추출하지 못했습니다.")
                return []
            
            problems = result["problems"]
            print(f"🖼️ 이미지에서 {len(problems)}개 문제 추출 성공")
            
            # img2json_generation의 결과를 teacher_graph 형식에 맞게 변환
            converted_problems = []
            skipped_count = 0
            
            for problem in problems:
                if isinstance(problem, dict):
                    # skipped 문제는 제외하되 카운트
                    if problem.get("skipped", False):
                        skipped_count += 1
                        print(f"⚠️ 문제 {problem.get('number', 'N/A')} 건너뜀: {problem.get('reason', '이유 없음')}")
                        continue
                    
                    # teacher_graph 형식으로 변환
                    question = str(problem.get("question", "")).strip()
                    options = problem.get("options", [])
                    
                    # options가 리스트가 아니면 변환
                    if isinstance(options, str):
                        options = [x.strip() for x in options.splitlines() if x.strip()]
                    elif not isinstance(options, list):
                        options = []
                    
                    # 빈 문제나 보기가 없는 문제는 제외
                    if not question or not options:
                        print(f"⚠️ 문제 {problem.get('number', 'N/A')} 건너뜀: 질문 또는 보기가 비어있음")
                        continue
                    
                    converted_problem = {
                        "question": question,
                        "options": options
                    }
                    converted_problems.append(converted_problem)
            
            print(f"🖼️ 변환 완료: {len(converted_problems)}개 문제 (건너뜀: {skipped_count}개)")
            return converted_problems
            
        except ImportError as e:
            print(f"❌ img2json_generation 모듈을 불러올 수 없습니다: {e}")
            print("💡 이미지 처리를 위해서는 agents.solution.img2json_generation 모듈이 필요합니다.")
            return []
        except Exception as e:
            print(f"❌ 이미지 문제 추출 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_image_paths(self, user_query: str) -> List[str]:
        """사용자 입력에서 이미지 파일 경로 추출"""
        import re
        
        # 이미지 파일 확장자 패턴
        image_extensions = r'\.(jpg|jpeg|png|gif|bmp|tiff|webp)$'
        
        # 파일 경로 패턴 (절대 경로 또는 상대 경로)
        path_pattern = r'["\']?([^"\s]+' + image_extensions + r')["\']?'
        
        matches = re.findall(path_pattern, user_query, re.IGNORECASE)
        
        # 실제 파일 존재 여부 확인
        valid_paths = []
        for match in matches:
            path = match.strip('"\'')
            if os.path.exists(path):
                valid_paths.append(path)
                print(f"🖼️ 이미지 파일 발견: {path}")
            else:
                print(f"⚠️ 이미지 파일을 찾을 수 없음: {path}")
        
        return valid_paths

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
            lambda state: state.get("routing", {}).get("solution_next", "mark_after_generator_solution"),
            {
                "solution": "solution",
                "preprocess": "preprocess",
                "mark_after_generator_solution": "mark_after_generator_solution",
            },
        )
        builder.add_edge("preprocess", "solution")
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
            self.post_generator_route,
            {
                "solution": "solution",
                "generate_problem_pdf": "generate_problem_pdf",
            },
        )
        builder.add_conditional_edges(
            "solution",
            self.post_solution_route,
            {
                "score": "score",
                "generate_answer_pdf": "generate_answer_pdf",
            },
        )
        builder.add_conditional_edges(
            "score",
            self.post_score_route,
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

# Streamlit 앱에서 사용할 함수
def create_app() -> Any:
    """Streamlit 앱에서 사용할 teacher graph 앱을 생성합니다."""
    orch = Orchestrator(user_id="streamlit_user", service="teacher", chat_id="web")
    return orch.build_teacher_graph()

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

            print("-----------------\n")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 종료합니다.")
    