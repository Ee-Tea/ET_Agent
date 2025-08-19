# teacher_graph.py
# uv run teacher/teacher_graph.py
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional, Tuple
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
    
    # PDF 파일이 있으면 항상 전처리 수행 (새로운 파일이므로)
    pdf_ids = art.get("pdf_ids", [])
    
    # 디버깅 로그 추가
    print(f"🔍 [전처리 체크] PDF 파일: {pdf_ids}")
    result = bool(pdf_ids)
    print(f"🔍 [전처리 체크] 결과: {result} (PDF 있음: {bool(pdf_ids)})")
    
    # PDF 파일이 있으면 전처리 필요 (기존 문제 상관없이)
    return result

# ========== Orchestrator ==========
class Orchestrator:
    def __init__(self, user_id: str, service: str, chat_id: str, init_agents: bool = True):
        load_dotenv()
        if not os.getenv("LANGCHAIN_API_KEY"):
            print("경고: LANGCHAIN_API_KEY 환경 변수가 설정되지 않았습니다.")
        # TTL/길이 제한은 redis_memory.py에서 설정
        self.memory = RedisLangGraphMemory(user_id=user_id, service=service, chat_id=chat_id)
        
        # PDF 전처리기 초기화
        from .pdf_preprocessor import PDFPreprocessor
        self.pdf_preprocessor = PDFPreprocessor()
        
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

        # 규칙 기반 빠른 분기: 명확한 패턴은 즉시 처리
        def _get_rule_based_intent(text: str) -> Optional[str]:
            import re
            if not text:
                return None
            
            text_lower = text.lower()
            
            # 1. 매우 명확한 solution 패턴
            solution_patterns = [
                r'\.pdf.*풀',  # PDF 풀어줘
                r'풀이.*해.*줘',  # 풀이해줘
                r'해설.*해.*줘',  # 해설해줘
                r'답.*알려.*줘',  # 답 알려줘
            ]
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in solution_patterns):
                return "solution"
            
            # 2. 매우 명확한 generate 패턴
            if re.search(r'\d+\s*(?:문제|문항|개).*(?:만들|생성|출제)', text):
                return "generate"
            
            # 3. 매우 명확한 retrieve 패턴
            retrieve_patterns = [
                r'(?:뭐|무엇|설명).*(?:야|인가|해줘)',
                r'(?:검색|찾아).*줘',
                r'.*(?:란|이란|뜻).*(?:뭐|무엇)',
            ]
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in retrieve_patterns):
                return "retrieve"
            
            # 4. 명확하지 않으면 None 반환 (LLM에 위임)
            return None

        # PDF 전처리 모듈 import (편의 함수들)
        from .pdf_preprocessor import extract_pdf_paths, extract_problem_range, determine_problem_source

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

        # 의도 분류: 규칙 기반 -> LLM 폴백
        rule_intent = _get_rule_based_intent(uq)
        
        if rule_intent:
            intent = rule_intent
            raw = f"rule_based:{rule_intent}"
            print(f"🔧 규칙 기반 분류: {intent}")
        else:
            # 규칙으로 분류되지 않으면 LLM 사용
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
        PDF 파일에서 문제 추출하는 전처리 노드
        """
        print("📄 PDF 문제 추출 전처리 노드 실행")
        
        artifacts = state.get("artifacts", {})
        file_mapper = FilePathMapper()
        external_file_paths = file_mapper.map_artifacts_to_paths(artifacts)
        
        print(f"🔍 전처리할 파일: {external_file_paths}")
        
        if not external_file_paths:
            print("⚠️ 전처리할 파일이 없습니다.")
            return state
            
        # PDF에서 문제 추출 로직
        try:
            extracted_problems = self._extract_problems_from_pdf(external_file_paths)
            
            new_state = {**state}
            new_state = ensure_shared(new_state)
            shared = new_state["shared"]
            
            questions = []
            options = []
            
            for problem in extracted_problems:
                if isinstance(problem, dict):
                    questions.append(problem.get("question", ""))
                    options.append(problem.get("options", []))
            
            # 기존 shared state에 PDF 문제 추가
            existing_questions = shared.get("question", [])
            existing_options = shared.get("options", [])
            
            # PDF 문제를 shared에 추가
            shared["question"].extend(questions)
            shared["options"].extend(options)
            
            # 추가된 문제 수를 artifacts에 기록
            added_count = len(questions)
            new_state["artifacts"]["pdf_added_count"] = added_count
            new_state["artifacts"]["pdf_added_start_index"] = len(existing_questions)
            
            print(f"📄 PDF 문제를 shared state에 추가: {added_count}개 문제")
            print(f"📂 shared state 총 문제 수: {len(existing_questions)}개 → {len(shared['question'])}개")
            print(f"🔢 추가된 문제 인덱스: {len(existing_questions)} ~ {len(shared['question']) - 1}")
            
            # PDF 전처리 전용 state에도 저장 (백업용)
            new_state["pdf_extracted"] = {
                "question": questions,
                "options": options,
                "source_files": external_file_paths,
                "extracted_count": added_count
            }
            
            print(f"✅ {added_count}개 문제 추출 및 shared state 추가 완료")
            return new_state
            
        except Exception as e:
            print(f"❌ PDF 문제 추출 중 오류: {e}")
            return state

    @traceable(name="teacher.solution")
    def solution(self, state: TeacherState) -> TeacherState:
        """
        문제 풀이 노드 - PDF에서 추가된 문제들을 solution_agent로 처리
        """
        print("🔧 문제 풀이 노드 실행")
        new_state: TeacherState = {**state}
        new_state = ensure_shared(new_state)
        new_state.setdefault("solution", {})
        
        # artifacts에서 PDF 추가 정보 확인
        artifacts = state.get("artifacts", {})
        pdf_added_count = artifacts.get("pdf_added_count", 0)
        pdf_added_start_index = artifacts.get("pdf_added_start_index", 0)
        
        print(f"📊 [Solution] PDF 추가 정보: {pdf_added_count}개 문제, 시작 인덱스: {pdf_added_start_index}")
        
        if pdf_added_count == 0:
            print("⚠️ PDF에서 추가된 문제가 없습니다.")
            return new_state
        
        # shared state에서 뒤에서부터 PDF 추가된 문제들 추출
        shared = new_state["shared"]
        all_questions = shared.get("question", [])
        all_options = shared.get("options", [])
        
        if len(all_questions) < pdf_added_count:
            print(f"⚠️ shared state의 문제 수({len(all_questions)})가 PDF 추가 수({pdf_added_count})보다 적습니다.")
            return new_state
        
        # 뒤에서부터 PDF 추가된 문제들 추출
        start_idx = len(all_questions) - pdf_added_count
        end_idx = len(all_questions)
        
        pdf_questions = all_questions[start_idx:end_idx]
        pdf_options = all_options[start_idx:end_idx]
        
        print(f"🎯 [Solution] 처리할 문제: 인덱스 {start_idx}~{end_idx-1} ({len(pdf_questions)}개)")
        
        # solution_agent 실행
        agent = self.solution_runner
        if agent is None:
            raise RuntimeError("solution_runner is not initialized (init_agents=False).")
        
        generated_answers = []
        generated_explanations = []
        
        # 모든 문제를 한 번에 solution_agent에 전달
        all_user_problems = []
        for question, options in zip(pdf_questions, pdf_options):
            if isinstance(options, str):
                options = [x.strip() for x in options.splitlines() if x.strip()] or [options.strip()]
            all_user_problems.append({"question": question, "options": options})
        
        print(f"📝 [Solution] 총 {len(all_user_problems)}개 문제를 한 번에 처리 중...")
        
        try:
            # solution_agent에 필요한 state 전달 (모든 문제를 한 번에)
            agent_input_state = {
                "user_input_txt": state.get("user_query", ""),
                "user_problems": all_user_problems,
                "user_problem": all_user_problems[0]["question"] if all_user_problems else "",
                "user_problem_options": all_user_problems[0]["options"] if all_user_problems else [],
                "source_type": "external",  # PDF 데이터이므로 external
                "input_kind": "file",
                "external_file_paths": [],
                "short_term_memory": [],
                "vectorstore": None,
                "retrieved_docs": [],
                "similar_questions_text": "",
                "generated_answer": "",
                "generated_explanation": "",
                "results": [],
                "validated": False,
                "retry_count": 0,
                "exam_title": "정보처리기사 모의고사",
                "difficulty": "중급",
                "subject": "기타",
                "chat_history": []
            }
            
            # subgraph로 실행
            agent_result = agent.invoke(agent_input_state)
            
            if agent_result:
                # results 배열에서 모든 문제의 답과 해설을 추출
                if "results" in agent_result and agent_result["results"]:
                    results = agent_result["results"]
                    print(f"📊 [Solution] {len(results)}개 문제 결과 수신")
                    
                    for i, result in enumerate(results):
                        answer = result.get("generated_answer", "")
                        explanation = result.get("generated_explanation", "")
                        
                        if answer:
                            generated_answers.append(answer)
                        else:
                            generated_answers.append("")
                            
                        if explanation:
                            generated_explanations.append(explanation)
                        else:
                            generated_explanations.append("")
                        
                        print(f"   - 문제 {i+1}: 답안 {'✅' if answer else '❌'}, 해설 {'✅' if explanation else '❌'}")
                
                # results가 없거나 비어있으면 기존 방식으로 처리 (호환성)
                elif "generated_answer" in agent_result or "generated_explanation" in agent_result:
                    print("⚠️ [Solution] results 배열이 비어있어 기존 방식으로 처리")
                    if "generated_answer" in agent_result:
                        generated_answers.append(agent_result["generated_answer"])
                    else:
                        generated_answers.append("")
                        
                    if "generated_explanation" in agent_result:
                        generated_explanations.append(agent_result["generated_explanation"])
                    else:
                        generated_explanations.append("")
                
                # 아무것도 없으면 빈 값으로 채움
                else:
                    print("⚠️ [Solution] 결과가 없어 빈 값으로 처리")
                    generated_answers.append("")
                    generated_explanations.append("")
            
            print(f"✅ [Solution] 총 {len(all_user_problems)}개 문제 처리 완료")
            
        except Exception as e:
            print(f"❌ [Solution] 문제 처리 실패: {e}")
            # 오류 시 빈 값으로 채움
            generated_answers = [""] * len(all_user_problems)
            generated_explanations = [""] * len(all_user_problems)
        
        # 결과를 shared state에 추가
        if generated_answers:
            shared.setdefault("answer", [])
            shared["answer"].extend(generated_answers)
            print(f"📝 [Solution] {len(generated_answers)}개 정답 추가")
        
        if generated_explanations:
            shared.setdefault("explanation", [])
            shared["explanation"].extend(generated_explanations)
            print(f"📝 [Solution] {len(generated_explanations)}개 해설 추가")
        
        # subject 정보도 추가
        if not shared.get("subject") or len(shared.get("subject", [])) < len(pdf_questions):
            shared.setdefault("subject", [])
            needed = len(pdf_questions) - len(shared["subject"])
            shared["subject"].extend(["일반"] * needed)
        
        validate_qas(shared)
        
        # 풀이 생성 후 자동으로 답안집 PDF 생성
        if shared.get("question") and shared.get("options") and shared.get("answer") and shared.get("explanation"):
            print("[AUTO-PDF] 풀이 생성 완료 → 답안집 PDF 자동 생성 시작")
            try:
                from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
                generator = ComprehensivePDFGenerator()
                
                problems = []
                questions = shared["question"]
                options_list = shared["options"]
                answers = shared["answer"]
                explanations = shared["explanation"]
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
        
        print(f"✅ [Solution] 총 {len(pdf_questions)}개 문제 처리 완료")
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
                
                if "questions" in agent_result:
                    questions = agent_result["questions"]
                    shared.setdefault("question", [])
                    shared["question"].extend(questions)
                    print(f"📝 [Generator] {len(questions)}개 문제 추가")
                
                if "options" in agent_result:
                    options = agent_result["options"]
                    shared.setdefault("options", [])
                    shared["options"].extend(options)
                    print(f"📝 [Generator] {len(options)}개 보기 추가")
                
                if "answers" in agent_result:
                    answers = agent_result["answers"]
                    shared.setdefault("answer", [])
                    shared["answer"].extend(answers)
                    print(f"📝 [Generator] {len(answers)}개 정답 추가")
                
                if "explanations" in agent_result:
                    explanations = agent_result["explanations"]
                    shared.setdefault("explanation", [])
                    shared["explanation"].extend(explanations)
                    print(f"📝 [Generator] {len(explanations)}개 해설 추가")
                
                # 과목 정보 추가
                if "subjects" in agent_result:
                    subjects = agent_result["subjects"]
                    shared.setdefault("subject", [])
                    shared["subject"].extend(subjects)
                
                # generation state에 결과 저장
                new_state["generation"].update(agent_result)
                
                print(f"✅ [Generator] 문제 생성 완료: 총 {len(shared.get('question', []))}개 문제")
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
        
        if not questions:
            print("⚠️ 채점할 문제가 없습니다.")
            return new_state
        
        # 사용자 답안 입력 받기
        user_answer = get_user_answer(questions)
        if not user_answer:
            print("⚠️ 사용자 답안을 입력받지 못했습니다.")
            return new_state
        
        # shared state에 사용자 답안 저장
        shared["user_answer"] = user_answer
        
        # solution_agent에서 생성된 정답과 해설
        solution_answers = shared.get("answer", [])
        if not solution_answers:
            print("⚠️ 정답이 없어서 채점할 수 없습니다.")
            return new_state
        
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
            
            if agent_result:
                # 채점 결과를 score state에 저장
                new_state["score"].update(agent_result)
                
                # shared state에 채점 결과 추가
                if "score_result" in agent_result:
                    shared["score_result"] = agent_result["score_result"]
                
                if "correct_count" in agent_result:
                    shared["correct_count"] = agent_result["correct_count"]
                
                if "total_count" in agent_result:
                    shared["total_count"] = agent_result["total_count"]
                
                print(f"✅ [Score] 채점 완료: {shared.get('correct_count', 0)}/{shared.get('total_count', 0)} 정답")
            else:
                print("⚠️ [Score] 채점 실패")
                
        except Exception as e:
            print(f"❌ [Score] 채점 중 오류: {e}")
        
        return new_state

    @traceable(name="teacher.analysis")
    def analysis(self, state: TeacherState) -> TeacherState:
        """
        분석 노드 - analysis_agent로 답안 분석
        """
        print("🔍 분석 노드 실행")
        new_state: TeacherState = {**state}
        new_state = ensure_shared(new_state)
        new_state.setdefault("analysis", {})
        
        # 분석에 필요한 데이터 확인
        shared = new_state["shared"]
        questions = shared.get("question", [])
        user_answer = shared.get("user_answer", [])
        solution_answers = shared.get("answer", [])
        
        if not questions or not user_answer or not solution_answers:
            print("⚠️ 분석에 필요한 데이터가 부족합니다.")
            return new_state
        
        # analysis_agent 실행
        agent = self.analyst_runner
        if agent is None:
            raise RuntimeError("analyst_runner is not initialized (init_agents=False).")
        
        try:
            user_query = state.get("user_query", "")
            sh = shared
            
            # analysis_agent를 subgraph로 실행
            agent_result = agent.invoke({
                "problem": sh.get("question", []) or [],
                "user_answer": user_answer,
                "solution_answer": sh.get("answer", []) or [],
                "solution": sh.get("explanation", []) or [],
                "user_query": user_query,
                "shared": sh
            })
            
            if agent_result:
                # 분석 결과를 analysis state에 저장
                new_state["analysis"].update(agent_result)
                
                # shared state에 분석 결과 추가
                if "weak_type" in agent_result:
                    shared["weak_type"] = agent_result["weak_type"]
                
                if "analysis_result" in agent_result:
                    shared["analysis_result"] = agent_result["analysis_result"]
                
                print(f"✅ [Analysis] 분석 완료: 취약 유형 {len(shared.get('weak_type', []))}개")
            else:
                print("⚠️ [Analysis] 분석 실패")
                
        except Exception as e:
            print(f"❌ [Analysis] 분석 중 오류: {e}")
        
        return new_state

    @traceable(name="teacher.generate_problem_pdf")
    def generate_problem_pdf(self, state: TeacherState) -> TeacherState:
        """
        문제집 PDF 생성 노드
        """
        print("📄 문제집 PDF 생성 노드 실행")
        new_state: TeacherState = {**state}
        
        try:
            shared = new_state.get("shared", {})
            questions = shared.get("question", [])
            options_list = shared.get("options", [])
            
            if not questions or not options_list:
                print("⚠️ PDF 생성할 문제가 없습니다.")
                return new_state
            
            from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
            generator = ComprehensivePDFGenerator()
            
            problems = []
            count = min(len(questions), len(options_list))
            
            for i in range(count):
                q = questions[i] if i < len(questions) else ""
                opts = options_list[i] if i < len(options_list) else []
                if isinstance(opts, str):
                    opts = [x.strip() for x in opts.splitlines() if x.strip()] or [opts.strip()]
                problems.append({
                    "question": q,
                    "options": opts,
                })
            
            # 출력 디렉토리
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "agents", "solution", "pdf_outputs"))
            os.makedirs(base_dir, exist_ok=True)
            
            uq = (state.get("user_query") or "exam").strip()
            safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
            base_filename = os.path.join(base_dir, f"{safe_uq}_문제집")
            
            # 문제집 PDF 생성
            problem_pdf = generator.generate_problem_booklet(problems, f"{base_filename}.pdf", f"{safe_uq} 문제집")
            print(f"✅ 문제집 PDF 생성 완료: {problem_pdf}")
            
            # artifacts에 기록
            arts = new_state.setdefault("artifacts", {})
            generated_list = arts.setdefault("generated_pdfs", [])
            generated_list.append(f"{base_filename}.pdf")
            
        except Exception as e:
            print(f"❌ 문제집 PDF 생성 중 오류: {e}")
        
        return new_state

    @traceable(name="teacher.generate_answer_pdf")
    def generate_answer_pdf(self, state: TeacherState) -> TeacherState:
        """
        답안집 PDF 생성 노드
        """
        print("📄 답안집 PDF 생성 노드 실행")
        new_state: TeacherState = {**state}
        
        try:
            shared = new_state.get("shared", {})
            questions = shared.get("question", [])
            options_list = shared.get("options", [])
            answers = shared.get("answer", [])
            explanations = shared.get("explanation", [])
            
            if not questions or not options_list or not answers or not explanations:
                print("⚠️ 답안집 PDF 생성에 필요한 데이터가 부족합니다.")
                return new_state
            
            from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
            generator = ComprehensivePDFGenerator()
            
            problems = []
            count = min(len(questions), len(options_list), len(answers), len(explanations))
            
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
            print(f"✅ 답안집 PDF 생성 완료: {answer_pdf}")
            
            # artifacts에 기록
            arts = new_state.setdefault("artifacts", {})
            generated_list = arts.setdefault("generated_pdfs", [])
            generated_list.append(f"{base_filename}.pdf")
            
        except Exception as e:
            print(f"❌ 답안집 PDF 생성 중 오류: {e}")
        
        return new_state

    @traceable(name="teacher.generate_analysis_pdf")
    def generate_analysis_pdf(self, state: TeacherState) -> TeacherState:
        """
        분석 리포트 PDF 생성 노드
        """
        print("📄 분석 리포트 PDF 생성 노드 실행")
        new_state: TeacherState = {**state}
        
        try:
            shared = new_state.get("shared", {})
            questions = shared.get("question", [])
            user_answer = shared.get("user_answer", [])
            solution_answers = shared.get("answer", [])
            explanations = shared.get("explanation", [])
            score_result = shared.get("score_result", {})
            weak_type = shared.get("weak_type", [])
            
            if not questions or not user_answer or not solution_answers:
                print("⚠️ 분석 리포트 PDF 생성에 필요한 데이터가 부족합니다.")
                return new_state
            
            from agents.solution.comprehensive_pdf_generator import ComprehensivePDFGenerator
            generator = ComprehensivePDFGenerator()
            
            # 분석 데이터 구성
            analysis_data = {
                "questions": questions,
                "user_answers": user_answer,
                "correct_answers": solution_answers,
                "explanations": explanations,
                "score_result": score_result,
                "weak_types": weak_type
            }
            
            # 출력 디렉토리
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "agents", "solution", "pdf_outputs"))
            os.makedirs(base_dir, exist_ok=True)
            
            uq = (state.get("user_query") or "exam").strip()
            safe_uq = ("".join(ch for ch in uq if ch.isalnum()))[:20] or "exam"
            base_filename = os.path.join(base_dir, f"{safe_uq}_분석리포트")
            
            # 분석 리포트 PDF 생성
            analysis_pdf = generator.generate_analysis_report(analysis_data, f"{base_filename}.pdf", f"{safe_uq} 분석 리포트")
            print(f"✅ 분석 리포트 PDF 생성 완료: {analysis_pdf}")
            
            # artifacts에 기록
            arts = new_state.setdefault("artifacts", {})
            generated_list = arts.setdefault("generated_pdfs", [])
            generated_list.append(f"{base_filename}.pdf")
            
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
        return self.pdf_preprocessor.extract_problems_from_pdf(file_paths)
    
    def _split_problem_blocks(self, raw_text: str) -> List[str]:
        """텍스트를 문제 블록으로 분할 (pdf_preprocessor 사용)"""
        return self.pdf_preprocessor._split_problem_blocks(raw_text)
    
    def _merge_blocks_by_question(self, micro_blocks: List[str]) -> List[str]:
        """미세 분할된 블록들을 문제별로 재묶기 (pdf_preprocessor 사용)"""
        return self.pdf_preprocessor._merge_blocks_by_question(micro_blocks)
    
    def normalize_docling_markdown(self, md: str) -> str:
        """Docling 마크다운 정규화 (pdf_preprocessor 사용)"""
        return self.pdf_preprocessor.normalize_docling_markdown(md)

    def _find_option_clusters(self, lines: List[str], start: int, end: int) -> List[Tuple[int, int]]:
        """옵션 클러스터 찾기 (pdf_preprocessor 사용)"""
        return self.pdf_preprocessor._find_option_clusters(lines, start, end)

    def split_problem_blocks_without_keyword(self, text: str) -> List[str]:
        """문제 키워드 없는 시험지에서 번호로 문항 분할 (pdf_preprocessor 사용)"""
        return self.pdf_preprocessor.split_problem_blocks_without_keyword(text)
    
    def _parse_block_with_llm(self, block_text: str, llm) -> Optional[Dict]:
        """LLM으로 블록을 문제 형태로 파싱 (pdf_preprocessor 사용)"""
        return self.pdf_preprocessor._parse_block_with_llm(block_text, llm)

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
        builder.add_conditional_edges(
            "analysis",
            self.post_analysis_route,
            {
                "generate_analysis_pdf": "generate_analysis_pdf",
                "persist_state": "persist_state",
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
                # 기본값으로 테스트용 PDF만 포함
                "artifacts": {"pdf_ids": ["2024년3회_정보처리기사필기기출문제.pdf"]},
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
