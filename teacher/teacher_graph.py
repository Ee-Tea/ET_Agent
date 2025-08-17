# teacher_graph.py
# uv run teacher/teacher_graph.py
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
    return bool(art.get("pdf_ids") or art.get("image_ids"))

# ========== Orchestrator ==========
class Orchestrator:
    def __init__(self, user_id: str, service: str, chat_id: str, init_agents: bool = True):
        load_dotenv()
        if not os.getenv("LANGCHAIN_API_KEY=REDACTED("경고: LANGCHAIN_API_KEY=REDACTED/길이 제한은 redis_memory.py에서 설정
        self.memory = RedisLangGraphMemory(user_id=user_id, service=service, chat_id=chat_id)
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

        # 규칙 기반 빠른 분기: 문제 생성 의도 강하게 감지 시 바로 generate
        def _looks_like_generation(text: str) -> bool:
            import re
            if not text:
                return False
            text_no_space = re.sub(r"\s+", "", text)
            keywords = ["문제", "문항", "출제", "만들", "생성", "모의고사"]
            if any(kw in text for kw in keywords):
                # 과목명 또는 문항수 표기가 함께 있으면 강한 시그널
                subjects = [
                    "소프트웨어설계",
                    "소프트웨어개발",
                    "데이터베이스구축",
                    "프로그래밍언어활용",
                    "정보시스템구축관리",
                ]
                has_subject = any((s in text) or (re.sub(r"\s+","",s) in text_no_space) for s in subjects)
                has_count = re.search(r"\d+\s*(?:문제|문항|개)", text) is not None
                return has_subject or has_count
            return False

        raw = ""  # raw 변수를 먼저 초기화
        if _looks_like_generation(uq):
            intent = "generate"
        else:
            try:
                from teacher_nodes import user_intent
                raw = user_intent(uq) if uq else ""
            except Exception:
                pass
            intent = normalize_intent(raw or "retrieve")
        print(f"사용자 의도 분류(정규화): {intent} (raw={raw!r})")
        return {**state, "user_query": uq, "intent": intent}

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
        agent = self.generator_runner
        if agent is None:
            raise RuntimeError("generator_runner is not initialized (init_agents=False).")
        user_query = state.get("user_query", "")

        # 사용자 입력 파싱을 nodes.parse_generator_input으로 대체
        import json as _json
        subject_candidates = list(getattr(agent, "SUBJECT_AREAS", {}).keys()) if agent else []
        parsed_subject: Optional[str] = None
        parsed_count: Optional[int] = None
        parsed_difficulty: str = "중급"

        try:
            parsed_raw = parse_generator_input(user_query)  # nodes 함수 (LLM 기반)
            print(f"parsed_raw: {parsed_raw}")
            if isinstance(parsed_raw, str):
                try:
                    parsed_obj = _json.loads(parsed_raw)
                except Exception:
                    parsed_obj = {}
            elif isinstance(parsed_raw, dict):
                parsed_obj = parsed_raw
            else:
                parsed_obj = {}

            subj = (parsed_obj.get("subject") if isinstance(parsed_obj, dict) else None) or None
            cnt = parsed_obj.get("count") if isinstance(parsed_obj, dict) else None
            diff = (parsed_obj.get("difficulty") if isinstance(parsed_obj, dict) else None) or None

            if isinstance(subj, str) and subj.strip():
                parsed_subject = subj.strip()
            if isinstance(cnt, str) and cnt.isdigit():
                parsed_count = int(cnt)
            elif isinstance(cnt, (int, float)):
                parsed_count = int(cnt)
            if isinstance(diff, str) and diff.strip():
                parsed_difficulty = diff.strip()
        except Exception:
            # 파싱 실패 시 기본값 유지
            pass

        # 에이전트 입력 구성: 새로운 모드 지원
        mode = parsed_obj.get("mode", "full_exam")
        
        if mode == "partial_exam":
            # 선택된 과목들에 대해 지정된 문제 수만큼 생성
            subjects = parsed_obj.get("subjects", [])
            count_per_subject = parsed_obj.get("count_per_subject", 10)
            
            if subjects and isinstance(subjects, list):
                agent_input = {
                    "mode": "partial_exam",
                    "selected_subjects": subjects,
                    "questions_per_subject": count_per_subject,
                    "difficulty": parsed_difficulty,
                    "save_to_file": False,
                }
                print(f"[Generator] 선택과목 {subjects} 각 {count_per_subject}문제 생성 요청")
            else:
                # fallback to full_exam
                agent_input = {
                    "mode": "full_exam",
                    "difficulty": parsed_difficulty,
                    "save_to_file": False,
                }
        elif mode == "single_subject" and parsed_subject:
            # 단일 과목 문제 생성 모드
            target_count = parsed_count if parsed_count and parsed_count > 0 else 5
            agent_input = {
                "mode": "subject_quiz",
                "subject_area": parsed_subject,
                "target_count": target_count,
                "difficulty": parsed_difficulty,
                "save_to_file": False,
            }
            print(f"[Generator] {parsed_subject} 과목 {target_count}문제 생성 요청")
        else:
            # 전체 모드 (기본값)
            agent_input = {
                "mode": "full_exam",
                "difficulty": parsed_difficulty,
                "save_to_file": False,
            }
        agent_result = safe_execute(agent, agent_input)
        print(f"문제 생성 결과: {agent_result}")
        new_state: TeacherState = {**state}
        new_state.setdefault("generation", {})
        new_state["generation"].update(agent_result)

        # 공유부 누적(Append)
        def _append_items_into_shared(target_state: Dict[str, Any], items: List[Dict[str, Any]], current_mode: str = "") -> None:
            sh = target_state.setdefault("shared", {})
            sh.setdefault("question", [])
            sh.setdefault("options", [])
            sh.setdefault("answer", [])
            sh.setdefault("explanation", [])
            sh.setdefault("subject", [])
            for item in items or []:
                if isinstance(item, dict) and ("question" in item) and ("options" in item):
                    opts = item.get("options", [])
                    if isinstance(opts, str):
                        lines = [x.strip() for x in opts.splitlines() if x.strip()]
                        opts = lines if lines else [opts.strip()]
                    elif isinstance(opts, list):
                        opts = [str(x).strip() for x in opts if str(x).strip()]
                    else:
                        opts = []
                    sh["question"].append(item.get("question", ""))
                    sh["options"].append(opts)
                    sh["answer"].append(item.get("answer", ""))
                    sh["explanation"].append(item.get("explanation", ""))
                    sh["subject"].append(item.get("subject", ""))

        # 이전 대화의 누적을 방지하기 위해 Q/A 관련 shared 리스트 초기화
        sh_init = new_state.setdefault("shared", {})
        for _k in ("question", "options", "answer", "explanation", "subject"):
            sh_init[_k] = []

        items_to_append: List[Dict[str, Any]] = []
        # 1) validated_questions 직접 제공 시
        if isinstance(agent_result, dict) and isinstance(agent_result.get("validated_questions"), list):
            items_to_append = agent_result.get("validated_questions", [])
        else:
            # 2) InfoProcessingExamAgent.execute 포맷 처리
            result_payload = agent_result.get("result") if isinstance(agent_result, dict) else None
            if isinstance(result_payload, dict):
                # partial_exam 모드: all_questions 필드 우선 사용
                if mode == "partial_exam" and isinstance(result_payload.get("all_questions"), list):
                    items_to_append = result_payload.get("all_questions", [])
                    print(f"[Generator] partial_exam 모드: {len(items_to_append)}개 문제 추출")
                # subject_quiz 모드: questions 필드 (사용자 요청 수만큼만)
                elif isinstance(result_payload.get("questions"), list):
                    target_count = parsed_count if parsed_count and parsed_count > 0 else 5
                    questions = result_payload.get("questions", [])
                    items_to_append = questions[:target_count]  # 요청한 수만큼만 추출
                    print(f"[Generator] {len(questions)}개 생성됨 → {len(items_to_append)}개 사용 (요청: {target_count}개)")
                # full_exam 모드: all_questions가 있으면 우선 사용
                elif isinstance(result_payload.get("all_questions"), list):
                    items_to_append = result_payload.get("all_questions", [])
                # subjects.*.questions 합치기 (full_exam 모드에서 실패한 과목들 처리)
                elif isinstance(result_payload.get("subjects"), dict):
                    aggregated: List[Dict[str, Any]] = []
                    for subject_name, subject_data in result_payload.get("subjects", {}).items():
                        if isinstance(subject_data, dict):
                            qs = subject_data.get("questions") if isinstance(subject_data, dict) else None
                            if isinstance(qs, list) and qs:  # questions가 있고 비어있지 않은 경우만
                                aggregated.extend(qs)
                                print(f"[Generator] {subject_name}에서 {len(qs)}개 문제 추출")
                            elif subject_data.get("status") == "FAILED":
                                print(f"[Generator] {subject_name} 과목 생성 실패: {subject_data.get('error', '알 수 없는 오류')}")
                    items_to_append = aggregated
                    print(f"[Generator] 총 {len(aggregated)}개 문제를 subjects에서 추출")

        if items_to_append:
            _append_items_into_shared(new_state, items_to_append, mode)

        validate_qas(new_state["shared"])
        
        # 문제 생성 후 상태 확인
        print(f"[DEBUG] === 문제 생성 후 상태 확인 ===")
        print(f"[DEBUG] shared 전체: {new_state.get('shared', {})}")
        print(f"[DEBUG] question 키 존재: {'question' in new_state.get('shared', {})}")
        print(f"[DEBUG] options 키 존재: {'options' in new_state.get('shared', {})}")
        print(f"[DEBUG] question 값: {new_state['shared'].get('question', [])}")
        print(f"[DEBUG] options 값: {new_state['shared'].get('options', [])}")
        print(f"[DEBUG] question 길이: {len(new_state['shared'].get('question', []))}")
        print(f"[DEBUG] options 길이: {len(new_state['shared'].get('options', []))}")
        
        # 문제 생성 후 자동으로 문제집 PDF 생성
        if new_state["shared"].get("question") and new_state["shared"].get("options"):
            print("[AUTO-PDF] 문제 생성 완료 → 문제집 PDF 자동 생성 시작")
            print(f"[DEBUG] 문제 수: {len(new_state['shared']['question'])}")
            print(f"[DEBUG] 보기 수: {len(new_state['shared']['options'])}")
            
            try:
                # 문제집 PDF 자동 생성
                print("[DEBUG] ComprehensivePDFGenerator 임포트 시도...")
                from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
                print("[DEBUG] ComprehensivePDFGenerator 임포트 성공")
                
                generator = ComprehensivePDFGenerator()
                print("[DEBUG] PDF 생성기 초기화 완료")
                
                problems = []
                questions = new_state["shared"]["question"]
                options_list = new_state["shared"]["options"]
                count = min(len(questions), len(options_list))
                
                print(f"[DEBUG] 처리할 문제 수: {count}")
                
                for i in range(count):
                    q = questions[i] if i < len(questions) else ""
                    opts = options_list[i] if i < len(options_list) else []
                    if isinstance(opts, str):
                        opts = [x.strip() for x in opts.splitlines() if x.strip()] or [opts.strip()]
                    
                    print(f"[DEBUG] 문제 {i+1}: {str(q)[:50]}...")
                    print(f"[DEBUG] 보기 {i+1}: {opts}")
                    
                    problems.append({
                        "question": q,
                        "options": opts,
                        "generated_answer": "",  # 문제집에는 정답 미포함
                        "generated_explanation": "",  # 문제집에는 풀이 미포함
                    })
                
                print(f"[DEBUG] problems 배열 생성 완료: {len(problems)}개")
                
                # 출력 디렉토리
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "agents", "solution", "pdf_outputs"))
                print(f"[DEBUG] 출력 디렉토리: {base_dir}")
                
                os.makedirs(base_dir, exist_ok=True)
                print(f"[DEBUG] 디렉토리 생성/확인 완료")
                
                uq = (state.get("user_query") or "exam").strip()
                safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
                base_filename = os.path.join(base_dir, f"{safe_uq}_문제집")
                
                print(f"[DEBUG] 파일명: {base_filename}")
                print(f"[DEBUG] 전체 경로: {base_filename}.pdf")
                
                # 문제집 PDF 생성
                print("[DEBUG] generate_problem_booklet 호출 시작...")
                problem_pdf = generator.generate_problem_booklet(problems, f"{base_filename}.pdf", f"{safe_uq} 문제집")
                print(f"[AUTO-PDF] 문제집 PDF 자동 생성 완료 → {problem_pdf}")
                
                # 파일 존재 확인
                if os.path.exists(f"{base_filename}.pdf"):
                    file_size = os.path.getsize(f"{base_filename}.pdf")
                    print(f"[DEBUG] PDF 파일 생성 확인: {f'{base_filename}.pdf'} (크기: {file_size:,} bytes)")
                else:
                    print(f"[WARNING] PDF 파일이 생성되지 않음: {f'{base_filename}.pdf'}")
                
                # artifacts에 기록
                arts = new_state.setdefault("artifacts", {})
                generated_list = arts.setdefault("generated_pdfs", [])
                generated_list.append(f"{base_filename}.pdf")
                print(f"[DEBUG] artifacts에 PDF 경로 추가: {f'{base_filename}.pdf'}")
                
            except ImportError as e:
                print(f"[ERROR] 모듈 임포트 실패: {e}")
                print(f"[DEBUG] 현재 작업 디렉토리: {os.getcwd()}")
                print(f"[DEBUG] Python 경로: {sys.path}")
            except Exception as e:
                print(f"[AUTO-PDF] 문제집 PDF 자동 생성 중 오류: {e}")
                print(f"[DEBUG] 오류 타입: {type(e).__name__}")
                import traceback
                print(f"[DEBUG] 상세 오류: {traceback.format_exc()}")
        else:
            print(f"[DEBUG] PDF 생성 조건 불충족:")
            print(f"[DEBUG] - question 존재: {bool(new_state['shared'].get('question'))}")
            print(f"[DEBUG] - options 존재: {bool(new_state['shared'].get('options'))}")
            print(f"[DEBUG] - question 내용: {new_state['shared'].get('question', [])}")
            print(f"[DEBUG] - options 내용: {new_state['shared'].get('options', [])}")
            print(f"[DEBUG] - question 길이: {len(new_state['shared'].get('question', []))}")
            print(f"[DEBUG] - options 길이: {len(new_state['shared'].get('options', []))}")
            print(f"[DEBUG] - question 타입: {type(new_state['shared'].get('question'))}")
            print(f"[DEBUG] - options 타입: {type(new_state['shared'].get('options'))}")
        
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
        agent = self.solution_runner
        if agent is None:
            raise RuntimeError("solution_runner is not initialized (init_agents=False).")

        # 파일 경로 정보 추출 (artifacts에서)
        artifacts = state.get("artifacts", {})
        
        # FilePathMapper를 사용하여 파일 경로 추출
        file_mapper = FilePathMapper()
        external_file_paths = file_mapper.map_artifacts_to_paths(artifacts)
        
        print(f"🔍 발견된 파일 경로: {external_file_paths}")
        if external_file_paths:
            print(f"   - PDF: {[f for f in external_file_paths if f.endswith('.pdf')]}")
            print(f"   - 이미지: {[f for f in external_file_paths if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]}")
        else:
            print("   ⚠️ 파일을 찾을 수 없습니다. artifacts를 확인해주세요.")
            print(f"   - artifacts: {artifacts}")

        for question, options in zip(questions, options_list):
            if isinstance(options, str):
                options = [x.strip() for x in options.splitlines() if x.strip()] or [options.strip()]
            
            # 파일이 있는 경우와 없는 경우를 구분하여 처리
            if external_file_paths:
                # 외부 파일이 있는 경우: external 모드로 실행
                agent_input = {
                    "user_question": state.get("user_query", ""),
                    "user_problem": question,
                    "user_problem_options": options,
                    "source_type": "external",
                    "external_file_paths": external_file_paths,
                    "short_term_memory": [],
                    "vectorstore": None,  # solution_agent에서 필요시 생성
                    "retrieved_docs": [],
                    "similar_questions_text": "",
                    "generated_answer": "",
                    "generated_explanation": "",
                    "results": [],
                    "validated": False,
                    "retry_count": 0,
                    "exam_title": "",
                    "difficulty": "",
                    "subject": "",
                    "chat_history": []
                }
            else:
                # 외부 파일이 없는 경우: internal 모드로 실행 (기존 방식)
                agent_input = {
                    "user_question": state.get("user_query", ""),
                    "user_problem": question,
                    "user_problem_options": options,
                    "source_type": "internal",
                    "external_file_paths": [],
                    "short_term_memory": [{"question": question, "options": options}],
                    "vectorstore": None,
                    "retrieved_docs": [],
                    "similar_questions_text": "",
                    "generated_answer": "",
                    "generated_explanation": "",
                    "results": [],
                    "validated": False,
                    "retry_count": 0,
                    "exam_title": "",
                    "difficulty": "",
                    "subject": "",
                    "chat_history": []
                }
            
            # solution_agent의 execute 메서드는 특별한 시그니처를 가짐
            if hasattr(agent, 'execute') and callable(getattr(agent, 'execute')):
                try:
                    # solution_agent의 execute 메서드 호출
                    agent_result = agent.execute(
                        user_question=agent_input["user_question"],
                        source_type=agent_input["source_type"],
                        vectorstore=agent_input.get("vectorstore"),
                        short_term_memory=agent_input.get("short_term_memory"),
                        external_file_paths=agent_input.get("external_file_paths"),
                        exam_title="정보처리기사 모의고사",
                        difficulty="중급",
                        subject="기타"
                    )
                    # 결과를 기존 형식에 맞게 변환
                    if isinstance(agent_result, list) and len(agent_result) > 0:
                        # 첫 번째 결과에서 답과 해설 추출
                        first_result = agent_result[0]
                        if "generated_answer" in first_result:
                            agent_result = {"generated_answer": first_result["generated_answer"]}
                        if "generated_explanation" in first_result:
                            agent_result["generated_explanation"] = first_result["generated_explanation"]
                    elif not isinstance(agent_result, dict):
                        agent_result = {}
                except Exception as e:
                    print(f"[WARN] solution_agent execute failed: {e}")
                    agent_result = {}
            else:
                # 기존 safe_execute 방식 사용
                agent_result = safe_execute(agent, agent_input)
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
        
        # 풀이 생성 후 자동으로 답안집 PDF 생성
        if sh.get("question") and sh.get("options") and sh.get("answer") and sh.get("explanation"):
            print("[AUTO-PDF] 풀이 생성 완료 → 답안집 PDF 자동 생성 시작")
            try:
                # 답안집 PDF 자동 생성
                from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
                generator = ComprehensivePDFGenerator()
                
                problems = []
                questions = sh["question"]
                options_list = sh["options"]
                answers = sh["answer"]
                explanations = sh["explanation"]
                count = min(len(questions), len(options_list))
                
                for i in range(count):
                    q = questions[i] if i < len(questions) else ""
                    opts = options_list[i] if i < len(options_list) else []
                    if isinstance(opts, str):
                        opts = [x.strip() for x in opts.splitlines() if x.strip()] or [opts.strip()]
                    ans = answers[i] if i < len(answers) else ""
                    exp = explanations[i] if i < len(explanations) else ""
                    problems.append({
                        "question": q,
                        "options": opts,
                        "generated_answer": ans,
                        "generated_explanation": exp,
                    })
                
                # 출력 디렉토리
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "agents", "solution", "pdf_outputs"))
                os.makedirs(base_dir, exist_ok=True)
                
                uq = (state.get("user_query") or "exam").strip()
                safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
                base_filename = os.path.join(base_dir, f"{safe_uq}_답안집")
                
                # 답안집 PDF 생성
                answer_pdf = generator.generate_answer_booklet(problems, f"{base_filename}.pdf", f"{safe_uq} 답안집")
                print(f"[AUTO-PDF] 답안집 PDF 자동 생성 완료 → {answer_pdf}")
                
                # artifacts에 기록
                arts = new_state.setdefault("artifacts", {})
                generated_list = arts.setdefault("generated_pdfs", [])
                generated_list.append(f"{base_filename}.pdf")
                
            except Exception as e:
                print(f"[AUTO-PDF] 답안집 PDF 자동 생성 중 오류: {e}")
        
        return new_state

    @traceable(name="teacher.score")
    def score(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("채점 노드 실행")
        user_query = state.get("user_query", "")
        user_answer = get_user_answer(user_query)
        sh = (state.get("shared") or {})
        solution_answers = sh.get("answer", []) or []
        agent = self.score_runner
        if agent is None:
            raise RuntimeError("score_runner is not initialized (init_agents=False).")

        agent_input = {
            "user_answer": user_answer,
            "solution_answer": solution_answers,
        }
        agent_result = safe_execute(agent, agent_input)

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
        agent = self.analyst_runner
        if agent is None:
            raise RuntimeError("analyst_runner is not initialized (init_agents=False).")
        agent_input = {
            "problem": sh.get("question", []) or [],
            "user_answer": user_answer,
            "solution_answer": sh.get("answer", []) or [],
            "solution": sh.get("explanation", []) or [],
        }
        agent_result = safe_execute(agent, agent_input)

        new_state: TeacherState = {**state}
        new_state.setdefault("analysis", {})
        new_state["analysis"].update(agent_result or {})

        if agent_result and "mistake_summary" in agent_result:
            new_state.setdefault("shared", {})
            new_state["shared"]["weak_type"] = agent_result["mistake_summary"]
        
        # 분석 완료 후 자동으로 분석 리포트 PDF 생성
        if sh.get("question") and sh.get("options") and sh.get("answer") and sh.get("explanation"):
            print("[AUTO-PDF] 분석 완료 → 분석 리포트 PDF 자동 생성 시작")
            try:
                # 분석 리포트 PDF 자동 생성
                from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
                generator = ComprehensivePDFGenerator()
                
                problems = []
                questions = sh["question"]
                options_list = sh["options"]
                answers = sh["answer"]
                explanations = sh["explanation"]
                weak_types = sh.get("weak_type", [])
                count = min(len(questions), len(options_list))
                
                for i in range(count):
                    q = questions[i] if i < len(questions) else ""
                    opts = options_list[i] if i < len(options_list) else []
                    if isinstance(opts, str):
                        opts = [x.strip() for x in opts.splitlines() if x.strip()] or [opts.strip()]
                    ans = answers[i] if i < len(answers) else ""
                    exp = explanations[i] if i < len(explanations) else ""
                    problems.append({
                        "question": q,
                        "options": opts,
                        "generated_answer": ans,
                        "generated_explanation": exp,
                    })
                
                # 출력 디렉토리
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "agents", "solution", "pdf_outputs"))
                os.makedirs(base_dir, exist_ok=True)
                
                uq = (state.get("user_query") or "exam").strip()
                safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
                base_filename = os.path.join(base_dir, f"{safe_uq}_분석리포트")
                
                # 분석 리포트 PDF 생성
                analysis_pdf = generator.generate_analysis_report(problems, f"{base_filename}.pdf", f"{safe_uq} 분석 리포트")
                print(f"[AUTO-PDF] 분석 리포트 PDF 자동 생성 완료 → {analysis_pdf}")
                
                # artifacts에 기록
                arts = new_state.setdefault("artifacts", {})
                generated_list = arts.setdefault("generated_pdfs", [])
                generated_list.append(f"{base_filename}.pdf")
                
            except Exception as e:
                print(f"[AUTO-PDF] 분석 리포트 PDF 자동 생성 중 오류: {e}")
        
        return new_state

    @traceable(name="teacher.generate_problem_pdf")
    def generate_problem_pdf(self, state: TeacherState) -> TeacherState:
        """문제 생성 후 문제집 PDF만 생성 (질문 + 보기)"""
        state = ensure_shared(state)
        sh = state.get("shared") or {}

        questions: List[str] = sh.get("question", []) or []
        options_list: List[List[str]] = sh.get("options", []) or []

        if not questions or not options_list:
            print("[PDF] 생성할 문제가 없어 문제집 PDF 생성을 건너뜁니다.")
            return state

        problems: List[Dict[str, Any]] = []
        count = min(len(questions), len(options_list))
        for i in range(count):
            q = questions[i] if i < len(questions) else ""
            opts = options_list[i] if i < len(options_list) else []
            if isinstance(opts, str):
                opts = [x.strip() for x in opts.splitlines() if x.strip()] or [opts.strip()]
            problems.append({
                "question": q,
                "options": opts,
                "generated_answer": "",  # 문제집에는 정답 미포함
                "generated_explanation": "",  # 문제집에는 풀이 미포함
            })

        # 출력 디렉토리
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "agents", "solution", "pdf_outputs"))
        os.makedirs(base_dir, exist_ok=True)

        uq = (state.get("user_query") or "exam").strip()
        safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
        base_filename = os.path.join(base_dir, f"{safe_uq}_문제집")

        try:
            # 지연 임포트: 그래프 시각화 등에서 모듈 임포트 부작용 방지
            from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
            generator = ComprehensivePDFGenerator()
            # 문제집만 생성
            problem_pdf = generator.generate_problem_booklet(problems, f"{base_filename}.pdf", f"{safe_uq} 문제집")
            print(f"[PDF] 문제집 생성 완료 → {problem_pdf}")
        except Exception as e:
            print(f"[PDF] 문제집 생성 중 오류: {e}")
            return state

        ns = {**state}
        arts = ns.setdefault("artifacts", {})
        generated_list = arts.setdefault("generated_pdfs", [])
        generated_list.append(f"{base_filename}.pdf")
        return ns

    @traceable(name="teacher.generate_answer_pdf")
    def generate_answer_pdf(self, state: TeacherState) -> TeacherState:
        """풀이 생성 후 답안집 PDF 생성 (질문 + 보기 + 정답 + 풀이)"""
        state = ensure_shared(state)
        sh = state.get("shared") or {}

        questions: List[str] = sh.get("question", []) or []
        options_list: List[List[str]] = sh.get("options", []) or []
        answers: List[str] = sh.get("answer", []) or []
        explanations: List[str] = sh.get("explanation", []) or []

        if not questions or not options_list:
            print("[PDF] 생성할 문제가 없어 답안집 PDF 생성을 건너뜁니다.")
            return state

        problems: List[Dict[str, Any]] = []
        count = min(len(questions), len(options_list))
        for i in range(count):
            q = questions[i] if i < len(questions) else ""
            opts = options_list[i] if i < len(options_list) else []
            if isinstance(opts, str):
                opts = [x.strip() for x in opts.splitlines() if x.strip()] or [opts.strip()]
            ans = answers[i] if i < len(answers) else ""
            exp = explanations[i] if i < len(explanations) else ""
            problems.append({
                "question": q,
                "options": opts,
                "generated_answer": ans,
                "generated_explanation": exp,
            })

        # 출력 디렉토리
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "agents", "solution", "pdf_outputs"))
        os.makedirs(base_dir, exist_ok=True)

        uq = (state.get("user_query") or "exam").strip()
        safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
        base_filename = os.path.join(base_dir, f"{safe_uq}_답안집")

        try:
            # 지연 임포트: 그래프 시각화 등에서 모듈 임포트 부작용 방지
            from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
            generator = ComprehensivePDFGenerator()
            # 답안집만 생성
            answer_pdf = generator.generate_answer_booklet(problems, f"{base_filename}.pdf", f"{safe_uq} 답안집")
            print(f"[PDF] 답안집 생성 완료 → {answer_pdf}")
        except Exception as e:
            print(f"[PDF] 답안집 생성 중 오류: {e}")
            return state

        ns = {**state}
        arts = ns.setdefault("artifacts", {})
        generated_list = arts.setdefault("generated_pdfs", [])
        generated_list.append(f"{base_filename}.pdf")
        return ns

    @traceable(name="teacher.generate_analysis_pdf")
    def generate_analysis_pdf(self, state: TeacherState) -> TeacherState:
        """분석 후 분석 리포트 PDF 생성"""
        state = ensure_shared(state)
        sh = state.get("shared") or {}

        questions: List[str] = sh.get("question", []) or []
        options_list: List[List[str]] = sh.get("options", []) or []
        answers: List[str] = sh.get("answer", []) or []
        explanations: List[str] = sh.get("explanation", []) or []
        weak_types: List[str] = sh.get("weak_type", []) or []

        if not questions or not options_list:
            print("[PDF] 생성할 문제가 없어 분석 리포트 PDF 생성을 건너뜁니다.")
            return state

        problems: List[Dict[str, Any]] = []
        count = min(len(questions), len(options_list))
        for i in range(count):
            q = questions[i] if i < len(questions) else ""
            opts = options_list[i] if i < len(options_list) else []
            if isinstance(opts, str):
                opts = [x.strip() for x in opts.splitlines() if x.strip()] or [opts.strip()]
            ans = answers[i] if i < len(answers) else ""
            exp = explanations[i] if i < len(explanations) else ""
            problems.append({
                "question": q,
                "options": opts,
                "generated_answer": ans,
                "generated_explanation": exp,
            })

        # 출력 디렉토리
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "agents", "solution", "pdf_outputs"))
        os.makedirs(base_dir, exist_ok=True)

        uq = (state.get("user_query") or "exam").strip()
        safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
        base_filename = os.path.join(base_dir, f"{safe_uq}_분석리포트")

        try:
            # 지연 임포트: 그래프 시각화 등에서 모듈 임포트 부작용 방지
            from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
            generator = ComprehensivePDFGenerator()
            # 분석 리포트 생성
            analysis_pdf = generator.generate_analysis_report(problems, f"{base_filename}.pdf", f"{safe_uq} 분석 리포트")
            print(f"[PDF] 분석 리포트 생성 완료 → {analysis_pdf}")
        except Exception as e:
            print(f"[PDF] 분석 리포트 생성 중 오류: {e}")
            return state

        ns = {**state}
        arts = ns.setdefault("artifacts", {})
        generated_list = arts.setdefault("generated_pdfs", [])
        generated_list.append(f"{base_filename}.pdf")
        return ns

    @traceable(name="teacher.generate_pdfs")
    def generate_pdfs(self, state: TeacherState) -> TeacherState:
        """문제/풀이 결과를 PDF로 저장하고 파일 경로를 artifacts에 기록합니다."""
        state = ensure_shared(state)
        sh = state.get("shared") or {}

        questions: List[str] = sh.get("question", []) or []
        options_list: List[List[str]] = sh.get("options", []) or []
        answers: List[str] = sh.get("answer", []) or []
        explanations: List[str] = sh.get("explanation", []) or []

        if not questions or not options_list:
            print("[PDF] 생성할 문제가 없어 PDF 생성을 건너뜁니다.")
            return state

        problems: List[Dict[str, Any]] = []
        count = min(len(questions), len(options_list))
        for i in range(count):
            q = questions[i] if i < len(questions) else ""
            opts = options_list[i] if i < len(options_list) else []
            if isinstance(opts, str):
                opts = [x.strip() for x in opts.splitlines() if x.strip()] or [opts.strip()]
            ans = answers[i] if i < len(answers) else ""
            exp = explanations[i] if i < len(explanations) else ""
            problems.append({
                "question": q,
                "options": opts,
                "generated_answer": ans,
                "generated_explanation": exp,
            })

        # 출력 디렉토리
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "agents", "solution", "pdf_outputs"))
        os.makedirs(base_dir, exist_ok=True)

        uq = (state.get("user_query") or "exam").strip()
        safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
        base_filename = os.path.join(base_dir, f"{safe_uq}")

        try:
            # 지연 임포트: 그래프 시각화 등에서 모듈 임포트 부작용 방지
            from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
            generator = ComprehensivePDFGenerator()
            files = generator.generate_all_pdfs(problems, base_filename)
        except Exception as e:
            print(f"[PDF] 생성 중 오류: {e}")
            return state

        ns = {**state}
        arts = ns.setdefault("artifacts", {})
        generated_list = arts.setdefault("generated_pdfs", [])
        for k in ("problem_pdf", "answer_pdf", "analysis_pdf"):
            if files.get(k):
                generated_list.append(files[k])
        print(f"[PDF] 생성 완료 → {generated_list}")
        return ns

    @traceable(name="teacher.retrieve")
    def retrieve(self, state: TeacherState) -> TeacherState:
        state = ensure_shared(state)
        print("정보 검색 노드 실행")
        agent = self.retriever_runner
        if agent is None:
            raise RuntimeError("retriever_runner is not initialized (init_agents=False).")
        agent_input = {"retrieval_question": state.get("user_query", "")}
        
        agent_result = safe_execute(agent, agent_input)
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
        builder.add_conditional_edges(
            "analysis",
            self.post_analysis_route,
            {
                "generate_analysis_pdf": "generate_analysis_pdf",
                "persist_state": "persist_state",
            },
        )

        # PDF 생성 노드 추가 및 연결
        builder.add_node("generate_pdfs", RunnableLambda(self.generate_pdfs))
        builder.add_node("generate_problem_pdf", RunnableLambda(self.generate_problem_pdf))
        builder.add_node("generate_answer_pdf", RunnableLambda(self.generate_answer_pdf))
        builder.add_node("generate_analysis_pdf", RunnableLambda(self.generate_analysis_pdf))

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
                # PDF 파일 테스트를 위한 예시 (실제 파일명으로 수정 필요)
                "artifacts": {"pdf_ids": ["2024년3회_정보처리기사필기기출문제.pdf"]},
                # 여러 파일 타입을 동시에 테스트할 수도 있습니다:
                # "artifacts": {
                #     "pdf_ids": ["2024년3회_정보처리기사필기기출문제.pdf", "정보처리기사"],
                #     "image_ids": ["diagram", "chart"]
                # },
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
            if q_cnt:
                print(f"\n=== 생성된 {q_cnt}개 문제 ===")
                for i in range(q_cnt):
                    q = shared["question"][i] if i < len(shared["question"]) else ""
                    opts = (shared.get("options") or [[]] * q_cnt)[i] if i < len(shared.get("options") or []) else []
                    ans = (shared.get("answer") or [""] * q_cnt)[i] if i < len(shared.get("answer") or []) else ""
                    exp = (shared.get("explanation") or [""] * q_cnt)[i] if i < len(shared.get("explanation") or []) else ""
                    
                    print(f"\n[문제 {i+1}] {str(q)[:150]}{'...' if len(str(q))>150 else ''}")
                    if opts:
                        print("  Options:", "; ".join(opts[:6]) + ("..." if len(opts) > 6 else ""))
                    if ans:
                        print(f"  Answer: {str(ans)[:100]}{'...' if len(str(ans))>100 else ''}")
                    if exp:
                        print(f"  Explanation: {str(exp)[:120]}{'...' if len(str(exp))>120 else ''}")
                print("=" * 50)

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
