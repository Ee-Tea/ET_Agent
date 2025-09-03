import os
import sys
import time
import uuid
import json
from typing import Any, Dict, List, Optional, TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver as LGMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
try:
    from common.short_term.redis_memory import RedisLangGraphMemory
except Exception:
    RedisLangGraphMemory = None

# project path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# load environment variables
load_dotenv()

from teacher.teacher import Teacher, TeacherState
from farmer.farmer import Farmer


class MainReState(TypedDict):
    user_query: str
    user_id: str
    chat_id: str
    service_classification: str
    final_response: NotRequired[str]
    teacher_state: NotRequired[TeacherState]
    farmer_state: NotRequired[Dict]
    artifacts: NotRequired[Dict]
    routing: NotRequired[Dict]
    session_data: NotRequired[Dict]
    # 숏텀메모리 관련
    saved_questions: NotRequired[List[Dict[str, Any]]]
    loaded_questions: NotRequired[List[Dict[str, Any]]]


class MainRe:
    def __init__(self, user_id: str = "cli_user", chat_id: Optional[str] = None):
        self.user_id = user_id
        self.chat_id = chat_id or "cli_chat"
        self.thread_id = f"main:{self.user_id}:{self.chat_id}"

        # single shared checkpointer (prefer Redis)
        if RedisLangGraphMemory is not None:
            try:
                self.checkpointer = RedisLangGraphMemory(
                    user_id=self.user_id,
                    service="teacher",
                    chat_id=self.chat_id,
                    redis_host=os.getenv("REDIS_HOST", "localhost"),
                    redis_port=int(os.getenv("REDIS_PORT", "6380")),
                )
                print("✅ RedisLangGraphMemory 활성화 (checkpointer)")
            except Exception as e:
                print(f"⚠️ Redis 메모리 초기화 실패 → MemorySaver로 폴백: {e}")
                self.checkpointer = LGMemorySaver()
        else:
            self.checkpointer = LGMemorySaver()

        # simple LLM for classification (OpenAI)
        try:
            self.llm = ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"), temperature=0.2, api_key=os.getenv("OPENAI_API_KEY=REDACTED Exception:
            self.llm = None
        print(f"🧪 LLM 초기화: provider=OpenAI, model={os.getenv('OPENAI_LLM_MODEL', 'gpt-4o-mini')}, has_key={'YES' if os.getenv('OPENAI_API_KEY=REDACTED 'NO'}")

        # service lock flags
        self.service_locked: bool = False
        self.locked_service: Optional[str] = None

        # sub-graphs
        self.teacher = Teacher(user_id=self.user_id, service="teacher", chat_id=self.chat_id, checkpointer=self.checkpointer)
        self.farmer = Farmer(user_id=self.user_id, service="farmer", chat_id=self.chat_id)

        # 숏텀메모리 초기화
        self.short_term_memory = self._init_short_term_memory()

        # compile combined graph
        self.graph = self._create_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
        print(f"✅ MainRe 초기화 완료: user_id={self.user_id}, chat_id={self.chat_id}, thread_id={self.thread_id}")

    def _rebuild_app(self):
        # Recreate subgraphs (needed when chat_id/thread changes)
        if RedisLangGraphMemory is not None:
            try:
                self.checkpointer = RedisLangGraphMemory(
                    user_id=self.user_id,
                    service="teacher",
                    chat_id=self.chat_id,
                    redis_host=os.getenv("REDIS_HOST", "localhost"),
                    redis_port=int(os.getenv("REDIS_PORT", "6380")),
                )
                print("🔁 RedisLangGraphMemory 재생성 (새 chat_id)")
            except Exception as e:
                print(f"⚠️ Redis 재생성 실패 → 기존 checkpointer 유지: {e}")
        self.teacher = Teacher(user_id=self.user_id, service="teacher", chat_id=self.chat_id, checkpointer=self.checkpointer)
        self.farmer = Farmer(user_id=self.user_id, service="farmer", chat_id=self.chat_id)
        
        # 숏텀메모리 재초기화
        self.short_term_memory = self._init_short_term_memory()
        
        self.graph = self._create_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
        print(f"🔁 앱 리빌드: thread_id={self.thread_id}")

    def _init_short_term_memory(self) -> Optional[RedisLangGraphMemory]:
        """숏텀메모리 초기화"""
        if RedisLangGraphMemory is not None:
            try:
                memory = RedisLangGraphMemory(
                    user_id=self.user_id,
                    service="teacher",
                    chat_id=self.chat_id,
                    redis_host=os.getenv("REDIS_HOST", "localhost"),
                    redis_port=int(os.getenv("REDIS_PORT", "6380")),
                )
                print("✅ 숏텀메모리 초기화 완료")
                return memory
            except Exception as e:
                print(f"⚠️ 숏텀메모리 초기화 실패: {e}")
                return None
        return None

    def save_teacher_data(self, teacher_state: TeacherState) -> bool:
        """
        Teacher 상태에서 문제/답/풀이 보기 데이터를 숏텀메모리에 저장
        
        Args:
            teacher_state: Teacher 상태 객체
            
        Returns:
            저장 성공 여부
        """
        if not self.short_term_memory:
            print("⚠️ 숏텀메모리가 초기화되지 않았습니다.")
            return False
        
        try:
            shared = teacher_state.get("shared", {})
            questions = shared.get("question", [])
            options = shared.get("options", [])
            answers = shared.get("answer", [])
            explanations = shared.get("explanation", [])
            subjects = shared.get("subject", [])
            user_answers = shared.get("user_answer", [])
            
            if not questions:
                print("📝 저장할 문제가 없습니다.")
                return False
            
            # 문제 데이터를 Redis 메모리 형식으로 변환
            question_data = []
            for i, question in enumerate(questions):
                if i < len(options) and i < len(answers) and i < len(explanations):
                    question_data.append({
                        "question": question,
                        "options": options[i] if isinstance(options[i], list) else [],
                        "answer": answers[i] if i < len(answers) else "",
                        "explanation": explanations[i] if i < len(explanations) else "",
                        "subject": subjects[i] if i < len(subjects) else "unknown",
                        "user_answer": user_answers[i] if i < len(user_answers) else "",
                        "timestamp": int(time.time())
                    })
            
            if question_data:
                # Redis 메모리에 저장 (중복 자동 필터링)
                new_qids = self.short_term_memory.add_questions(question_data)
                print(f"💾 {len(new_qids)}개 문제가 숏텀메모리에 저장되었습니다.")
                
                # 풀이 정보도 함께 저장
                for i, qid in enumerate(new_qids):
                    if i < len(question_data):
                        q_data = question_data[i]
                        self.short_term_memory.upsert_solution(
                            qid=qid,
                            user_answer=q_data.get("user_answer", ""),
                            model_answer=q_data.get("answer", ""),
                            explanation=q_data.get("explanation", ""),
                            score=1.0 if q_data.get("user_answer") == q_data.get("answer") else 0.0,
                            is_correct=q_data.get("user_answer") == q_data.get("answer")
                        )
                
                return True
            else:
                print("📝 유효한 문제 데이터가 없습니다.")
                return False
                
        except Exception as e:
            print(f"❌ 문제 데이터 저장 중 오류: {e}")
            return False

    def load_recent_questions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        최근 저장된 문제들을 불러오기
        
        Args:
            limit: 불러올 최대 문제 수
            
        Returns:
            문제 데이터 리스트
        """
        if not self.short_term_memory:
            print("⚠️ 숏텀메모리가 초기화되지 않았습니다.")
            return []
        
        try:
            # 최근 문제 ID들 가져오기
            qids = self.short_term_memory.select_qids(limit=limit, recent_first=True)
            
            if not qids:
                print("📝 저장된 문제가 없습니다.")
                return []
            
            # 각 문제의 상세 정보 가져오기
            questions = []
            for qid in qids:
                try:
                    question = self.short_term_memory.get_question(qid)
                    solution = self.short_term_memory.get_solution(qid)
                    
                    questions.append({
                        "qid": qid,
                        "question": question.get("question_text", ""),
                        "options": question.get("options", []),
                        "answer": solution.get("model_answer", ""),
                        "explanation": solution.get("explanation", ""),
                        "subject": question.get("subject", "unknown"),
                        "user_answer": solution.get("user_answer", ""),
                        "is_correct": solution.get("is_correct", False),
                        "score": solution.get("score", 0.0),
                        "created_at": question.get("created_at", 0),
                        "updated_at": question.get("updated_at", 0)
                    })
                except Exception as e:
                    print(f"⚠️ 문제 {qid} 로드 중 오류: {e}")
                    continue
            
            print(f"📖 {len(questions)}개 문제를 숏텀메모리에서 불러왔습니다.")
            return questions
            
        except Exception as e:
            print(f"❌ 문제 데이터 로드 중 오류: {e}")
            return []

    def get_weakness_analysis(self, top_k: int = 3) -> Dict[str, Any]:
        """
        취약점 분석 결과 가져오기
        
        Args:
            top_k: 상위 취약 과목 개수
            
        Returns:
            취약점 분석 결과
        """
        if not self.short_term_memory:
            return {"error": "숏텀메모리가 초기화되지 않았습니다."}
        
        try:
            return self.short_term_memory.weakness_summary(top_k=top_k)
        except Exception as e:
            print(f"❌ 취약점 분석 중 오류: {e}")
            return {"error": str(e)}

    def clear_short_term_memory(self) -> bool:
        """숏텀메모리 데이터 삭제"""
        if not self.short_term_memory:
            return False
        
        try:
            self.short_term_memory.clear(include_questions=True)
            print("🧹 숏텀메모리가 삭제되었습니다.")
            return True
        except Exception as e:
            print(f"❌ 숏텀메모리 삭제 중 오류: {e}")
            return False

    def lock_service_session(self, service: str):
        if service in ("teacher", "farmer"):
            self.service_locked = True
            self.locked_service = service
            print(f"🔒 서비스 고정: {service}")

    def reset_session(self, new_chat: bool = True):
        if new_chat:
            self.chat_id = uuid.uuid4().hex[:8]
        self.thread_id = f"main:{self.user_id}:{self.chat_id}"
        self.service_locked = False
        self.locked_service = None
        self._rebuild_app()
        print(f"🧹 세션 초기화: chat_id={self.chat_id}")

    # ===== Nodes =====
    def classify(self, state: MainReState) -> MainReState:
        uq = (state.get("user_query") or "").strip()
        session = dict(state.get("session_data") or {})
        print(f"🧭 분류 시작 | locked={self.service_locked}({self.locked_service}), uq='{uq[:120]}'")

        # enforce lock
        if self.service_locked and self.locked_service:
            session.update({"service_locked": True, "locked_service": self.locked_service})
            print(f"↪️ 잠금 유지: {self.locked_service}")
            return {**state, "service_classification": self.locked_service, "session_data": session}

        # Reuse main.py classification prompt/behavior
        service_classification = "teacher"  # 기본값
        try:
            if self.llm is not None:
                classification_prompt = f"""
            사용자의 질문을 분석하여 어떤 서비스가 필요한지 분류해주세요.

            질문: {uq}

            다음 중 하나로만 답변해주세요:
            - "farmer": 농업, 재배, 작물, 농사, 농업기술, 작물관리 관련 질문
            - "teacher": 자격증, 시험, 학습, 교육, 문제풀이, 시험준비 관련 질문

            답변:
            """
                raw = self.llm.invoke(classification_prompt)
                resp = (getattr(raw, "content", None) or str(raw)).strip().lower()
                print(f"📝 LLM 분류 원본: {resp}")
                if "farmer" in resp:
                    service_classification = "farmer"
                elif "teacher" in resp:
                    service_classification = "teacher"
                else:
                    service_classification = "teacher"  # 기본값
            else:
                uq_lower = uq.lower()
                if any(k in uq_lower for k in ["작물", "재배", "농업", "판매처", "재해", "시장"]):
                    service_classification = "farmer"
                elif any(k in uq_lower for k in ["시험", "문제", "풀이", "정답", "자격증", "기출"]):
                    service_classification = "teacher"
                print(f"📝 키워드 분류: {service_classification}")
        except Exception as e:
            print(f"❌ 서비스 분류 중 오류: {e}")
            service_classification = "teacher"

        if service_classification in ("teacher", "farmer"):
            self.lock_service_session(service_classification)
            session.update({"service_locked": True, "locked_service": service_classification})
        else:
            session.update({"service_locked": False, "locked_service": None})

        print(f"✅ 분류 결과: {service_classification}")
        return {**state, "service_classification": service_classification, "session_data": session}

    def run_teacher(self, state: MainReState) -> MainReState:
        # 최근 저장된 문제들을 불러와서 teacher 입력에 포함
        recent_questions = self.load_recent_questions(limit=10)
        
        init: TeacherState = {
            "user_query": state.get("user_query", ""),
            "intent": "",
            "shared": {}, "work": {}, "retrieval": {}, "generation": {},
            "solution": {}, "score": {}, "analysis": {},
            "history": [], "session": {}, "artifacts": {}, "routing": {},
            "llm_response": "",
        }
        
        # 최근 문제 데이터가 있으면 shared에 추가
        if recent_questions:
            questions = [q["question"] for q in recent_questions]
            options = [q["options"] for q in recent_questions]
            answers = [q["answer"] for q in recent_questions]
            explanations = [q["explanation"] for q in recent_questions]
            subjects = [q["subject"] for q in recent_questions]
            user_answers = [q["user_answer"] for q in recent_questions]
            
            init["shared"] = {
                "question": questions,
                "options": options,
                "answer": answers,
                "explanation": explanations,
                "subject": subjects,
                "user_answer": user_answers,
                "retrieve_answer": "",
                "wrong_question": [],
                "weak_type": [],
                "notes": []
            }
            print(f"📖 최근 {len(recent_questions)}개 문제를 teacher 입력에 포함했습니다.")
        
        res = self.teacher.invoke(init, config={"configurable": {"thread_id": self.thread_id}})
        merged = dict(res)
        
        # teacher 실행 후 결과를 숏텀메모리에 저장
        if merged.get("shared"):
            saved = self.save_teacher_data(merged)
            if saved:
                print("💾 Teacher 실행 결과가 숏텀메모리에 저장되었습니다.")
        
        # expose key outputs
        out: MainReState = {**state, "teacher_state": merged, "loaded_questions": recent_questions}
        for k in ("final_response", "artifacts", "routing"):
            if k in merged:
                out[k] = merged[k]
        return out

    def run_farmer(self, state: MainReState) -> MainReState:
        # farmer expects {"query": ...}
        fr = self.farmer.invoke({"query": state.get("user_query", "")}, config={"configurable": {"thread_id": self.thread_id}})
        out = {**state, "farmer_state": fr, "final_response": fr.get("output", "")}
        return out

    def finalize(self, state: MainReState) -> MainReState:
        print("🔚 finalize 호출")
        return state

    def handle_unrelated(self, state: MainReState) -> MainReState:
        msg = (
            "해당 질문은 현재 지원 서비스와 관련이 없어 보입니다.\n"
            "지원 서비스: farmer(농업/재배/판매처/재해), teacher(자격증/기출/풀이).\n"
            "원하시는 서비스를 알려주시거나 질문을 수정해주세요. 새 세션은 /new 로 시작할 수 있어요."
        )
        print("🚫 비관련 질문 처리")
        return {**state, "final_response": msg}

    # ===== Routing =====
    def route_service(self, state: MainReState) -> str:
        svc = state.get("service_classification", "teacher")
        target = "unrelated"
        if svc == "teacher":
            target = "teacher_app"
        elif svc == "farmer":
            target = "farmer_app"
        print(f"➡️ 라우팅: {svc} -> {target}")
        return target

    # ===== Graph =====
    def _create_graph(self) -> StateGraph:
        g = StateGraph(MainReState)
        g.add_node("classify", self.classify)
        g.add_node("teacher_app", self.teacher.graph)
        g.add_node("farmer_app", self.run_farmer)
        g.add_node("unrelated", self.handle_unrelated)
        g.add_node("finalize", self.finalize)

        g.add_edge(START, "classify")
        g.add_conditional_edges(
            "classify", self.route_service, {
                "teacher_app": "teacher_app",
                "farmer_app": "farmer_app",
                "unrelated": "unrelated",
            }
        )
        # after teacher subgraph, just finalize
        g.add_edge("teacher_app", "finalize")
        g.add_edge("farmer_app", "finalize")
        g.add_edge("unrelated", "finalize")
        g.add_edge("finalize", END)
        return g


# ===== CLI with HITL =====
def _make_cfg(orchestrator: MainRe):
    return {
        "configurable": {"thread_id": orchestrator.thread_id},
        "interrupt_after": [
            # support both subgraph-qualified and traceable names
            "teacher_app.await_output_mode",
            "teacher_app.await_form_answers",
            "teacher.await_output_mode",
            "teacher.await_form_answers",
            # stop right after output-mode decision so we can present problems before answers
            "teacher_app.decide_output_mode",
            "teacher.decide_output_mode",
            "teacher_app.prepare_form",
            "teacher.prepare_form",
            # solution HITL steps bubble up as teacher_app; we still list explicit names for newer LangGraph versions
            "teacher_app.collect_feedback",
            "teacher_app.user_feedback_tool",
        ],
    }

def _make_cfg_new_turn(orchestrator: MainRe):
    # New branch to avoid carrying transient routing like output_mode between turns
    return {
        "configurable": {"thread_id": orchestrator.thread_id, "checkpoint_id": uuid.uuid4().hex},
        "interrupt_after": [
            "teacher_app.await_output_mode",
            "teacher_app.await_form_answers",
            "teacher.await_output_mode",
            "teacher.await_form_answers",
            "teacher_app.decide_output_mode",
            "teacher.decide_output_mode",
            "teacher_app.prepare_form",
            "teacher.prepare_form",
            "teacher_app.collect_feedback",
            "teacher_app.user_feedback_tool",
        ],
    }

def _resume(orchestrator: MainRe, cfg, payload: Any):
    # Robust import across versions
    Command = None
    try:
        from langgraph.types import Command as _Cmd  # type: ignore
        Command = _Cmd
    except Exception:
        try:
            from langgraph import Command as _Cmd  # type: ignore
            Command = _Cmd
        except Exception:
            try:
                from langgraph.checkpoint.memory import Command as _Cmd  # type: ignore
                Command = _Cmd
            except Exception:
                Command = None
    if Command is None:
        print("❌ Command 클래스를 가져오지 못했습니다. LangGraph 버전을 확인하세요.")
        raise RuntimeError("Command import failed")
    print(f"🔁 Resume 호출: payload={payload}")
    return orchestrator.app.invoke(Command(resume=payload), cfg)

def _get_snapshot(orchestrator: MainRe, cfg):
    try:
        snap = orchestrator.app.get_state(cfg)
        try:
            pending = getattr(snap, "next", None)
            print(f"🛰️ 스냅샷 획득: pending={list(pending) if pending else []}")
        except Exception:
            pass
        return snap
    except Exception:
        return None

def _get_pending_nodes(orchestrator: MainRe, cfg) -> List[str]:
    snap = _get_snapshot(orchestrator, cfg)
    if snap is None:
        return []
    nxt = getattr(snap, "next", None)
    try:
        lst = [str(n) for n in (nxt or [])]
        if lst:
            print(f"⏸️ 대기 노드: {lst}")
        return lst
    except Exception:
        return []

def _get_values(orchestrator: MainRe, cfg) -> Dict[str, Any]:
    snap = _get_snapshot(orchestrator, cfg)
    if snap is None:
        return {}
    vals = getattr(snap, "values", None)
    if isinstance(vals, dict):
        try:
            keys = list(vals.keys())[:10]
            print(f"📦 상태 키: {keys}{' ...' if len(vals)>10 else ''}")
        except Exception:
            pass
        return vals
    if isinstance(snap, dict):
        out = snap.get("values") or snap
        try:
            keys = list(out.keys())[:10]
            print(f"📦 상태 키(dict): {keys}{' ...' if len(out)>10 else ''}")
        except Exception:
            pass
        return out
    return {}

def _print_outputs_if_any(state: Dict[str, Any]) -> bool:
    final_resp = state.get("final_response")
    if final_resp:
        print("🖨️ 최종 응답 출력")
        print(f"Bot> {final_resp}")
        return True
    arts = state.get("artifacts") or (state.get("teacher_state") or {}).get("artifacts") or {}
    gen = arts.get("generated_pdfs") or arts.get("pdfs")
    if isinstance(gen, list) and gen:
        print("Bot> PDF 생성 완료:")
        for p in gen:
            print(f"  - {p}")
        return True
    return False


def main():
    print("🚀 MainRe 오케스트레이터 (HITL) 시작")
    orch = MainRe(user_id="cli_user", chat_id="cli_chat")

    while True:
        cfg = _make_cfg(orch)
        # pending 확인 (teacher HITL)
        pending = _get_pending_nodes(orch, cfg)
        vals = _get_values(orch, cfg)
        routing = (vals.get("routing") or (vals.get("teacher_state") or {}).get("routing") or {})
        output_mode = routing.get("output_mode")
        ua = (vals.get("shared") or {}).get("user_answer") or ((vals.get("teacher_state") or {}).get("shared") or {}).get("user_answer")
        if pending:
            print(f"🔎 감지된 인터럽트: {pending}")

        if any(p.startswith("teacher_app.await_output_mode") for p in pending):
            if output_mode in ("pdf", "form"):
                state = _resume(orch, cfg, "")
            else:
                mode = input("Bot> 출력 방식을 선택하세요 (pdf|form): ").strip().lower()
                try:
                    setattr(orch.teacher, "_pending_output_mode", mode)
                except Exception:
                    pass
                state = _resume(orch, cfg, "")
            _print_outputs_if_any(dict(state))
            continue

        if any(p.startswith("teacher_app.await_form_answers") for p in pending):
            shared_vals = (vals.get("shared") or (vals.get("teacher_state") or {}).get("shared") or {})
            qs = shared_vals.get("question") or []
            opts = shared_vals.get("options") or []
            print(f"[DEBUG] await_form_answers snapshot: questions={len(qs)}, options={len(opts)}, user_answer={(shared_vals.get('user_answer') or [])}")
            try:
                if qs:
                    print("📄 문제:")
                    for i, q in enumerate(qs, 1):
                        print(f"  {i}. {q}")
                        if i-1 < len(opts) and isinstance(opts[i-1], list):
                            print("   보기:")
                            print("    - " + " | ".join(str(x) for x in opts[i-1]))
                else:
                    print("⚠️ 아직 표시할 문제가 준비되지 않았습니다. 잠시 후 다시 시도하거나 Enter를 눌러 다음 루프로 진행하세요.")
            except Exception:
                pass
            ans = input(f"Bot> 정답을 쉼표로 입력하세요 (문항 {len(qs)}개): ").strip()
            if not ans:
                print("[DEBUG] No answer typed; staying on await_form_answers without resume.")
                # 사용자 입력을 기다리기 위해 재개하지 않고 루프 계속
                continue
            parts = [p.strip() for p in ans.replace(" ", ",").split(",") if p.strip()]
            if not parts:
                print("[DEBUG] Parsed answers empty; continue waiting.")
                continue
            try:
                setattr(orch.teacher, "_pending_form_answers", parts)
                print(f"[DEBUG] Set _pending_form_answers={parts}")
            except Exception:
                print("[DEBUG] Failed to set _pending_form_answers on teacher instance")
                pass
            state = _resume(orch, cfg, {})
            _print_outputs_if_any(dict(state))
            continue

        # Fallback: pending only shows 'teacher_app' (subgraph-level interrupt)
        if any(p == "teacher_app" for p in pending):
            print("🧩 서브그래프 인터럽트 감지(teacher_app): 세부 단계 추론 시도")
            qs = ((vals.get("shared") or {}).get("question") or ((vals.get("teacher_state") or {}).get("shared") or {}).get("question") or [])
            if output_mode not in ("pdf", "form"):
                mode = input("Bot> 출력 방식을 선택하세요 (pdf|form): ").strip().lower()
                if mode == "pdf":
                    try:
                        setattr(orch.teacher, "_pending_output_mode", mode)
                        print(f"[DEBUG] Set _pending_output_mode={mode}")
                    except Exception:
                        print("[DEBUG] Failed to set _pending_output_mode on teacher instance")
                    state = _resume(orch, cfg, "")
                    _print_outputs_if_any(dict(state))
                    continue
                elif mode == "form":
                    try:
                        setattr(orch.teacher, "_pending_output_mode", mode)
                        print(f"[DEBUG] Set _pending_output_mode={mode}")
                    except Exception:
                        print("[DEBUG] Failed to set _pending_output_mode on teacher instance")
                    # Do NOT resume yet; show questions and wait for answers first
                    shared_vals = (vals.get("shared") or (vals.get("teacher_state") or {}).get("shared") or {})
                    qs2 = shared_vals.get("question") or []
                    opts2 = shared_vals.get("options") or []
                    try:
                        if qs2:
                            print("📄 문제:")
                            for i, q in enumerate(qs2, 1):
                                print(f"  {i}. {q}")
                                if i-1 < len(opts2) and isinstance(opts2[i-1], list):
                                    print("   보기:")
                                    print("    - " + " | ".join(str(x) for x in opts2[i-1]))
                        else:
                            print("⚠️ 아직 표시할 문제가 준비되지 않았습니다. Enter로 대기 유지.")
                    except Exception:
                        pass
                    ans = input(f"Bot> 정답을 쉼표로 입력하세요 (문항 {len(qs2)}개): ").strip()
                    if not ans:
                        print("[DEBUG] No answer typed after selecting form; staying pending.")
                        continue
                    parts = [p.strip() for p in ans.replace(" ", ",").split(",") if p.strip()]
                    if not parts:
                        print("[DEBUG] Parsed empty answers; staying pending.")
                        continue
                    try:
                        setattr(orch.teacher, "_pending_form_answers", parts)
                        print(f"[DEBUG] Set _pending_form_answers={parts}")
                    except Exception:
                        print("[DEBUG] Failed to set _pending_form_answers on teacher instance")
                    state = _resume(orch, cfg, {})
                    _print_outputs_if_any(dict(state))
                    continue
                else:
                    print("⚠️ 유효한 입력이 아닙니다 (pdf|form). 대기 유지.")
                    continue
            if qs and not ua:
                ans = input(f"Bot> 정답을 쉼표로 입력하세요 (문항 {len(qs)}개): ").strip()
                parts = [p.strip() for p in ans.replace(" ", ",").split(",") if p.strip()]
                try:
                    setattr(orch.teacher, "_pending_form_answers", parts)
                    print(f"[DEBUG] Set _pending_form_answers={parts}")
                except Exception:
                    print("[DEBUG] Failed to set _pending_form_answers on teacher instance")
                    pass
                state = _resume(orch, cfg, {})
                _print_outputs_if_any(dict(state))
                continue
            # solution feedback fallback
            print("🧩 풀이 피드백이 필요한지 검사")
            maybe_feedback = input("Bot> 풀이가 이해되지 않거나 더 쉽게 설명이 필요하면 피드백을 적어주세요. (그냥 Enter로 건너뛰기): ").strip()
            if maybe_feedback:
                try:
                    setattr(orch.teacher, "_pending_user_feedback", maybe_feedback)
                    print(f"[DEBUG] Set _pending_user_feedback length={len(maybe_feedback)}")
                except Exception:
                    print("[DEBUG] Failed to set _pending_user_feedback on teacher instance")
                    pass
                state = _resume(orch, cfg, {})
                _print_outputs_if_any(dict(state))
                continue

        try:
            uq = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다.")
            break
        if not uq:
            continue

        # If there is a pending teacher interrupt, interpret direct input as resume payload
        if pending:
            lower = uq.lower()
            # output mode
            if lower in ("pdf", "form"):
                try:
                    setattr(orch.teacher, "_pending_output_mode", lower)
                except Exception:
                    pass
                state = _resume(orch, cfg, "")
                _print_outputs_if_any(dict(state))
                continue
            # form answers: csv like "1,2,3" or "a,b,c" or "1 2 3"
            if any(sep in uq for sep in [",", " "]):
                try:
                    parts = [p.strip() for p in uq.replace(" ", ",").split(",") if p.strip()]
                    if parts:
                        try:
                            setattr(orch.teacher, "_pending_form_answers", parts)
                        except Exception:
                            pass
                        state = _resume(orch, cfg, {})
                        _print_outputs_if_any(dict(state))
                        continue
                except Exception:
                    pass
            # solution feedback: simple heuristic (prefix or long text)
            if lower.startswith("feedback:") or len(uq) > 10:
                fb = uq[9:].strip() if lower.startswith("feedback:") else uq
                try:
                    setattr(orch.teacher, "_pending_user_feedback", fb)
                except Exception:
                    pass
                state = _resume(orch, cfg, {})
                _print_outputs_if_any(dict(state))
                continue

        # commands
        if uq.startswith("/"):
            parts = uq[1:].split()
            cmd = parts[0].lower() if parts else ""
            arg = parts[1].lower() if len(parts) > 1 else None
            if cmd in ("exit", "quit"):
                print("👋 종료합니다.")
                break
            if cmd in ("new", "clear"):
                orch.reset_session(new_chat=True)
                print(f"🆕 새 세션 시작: thread_id={orch.thread_id}")
                continue
            if cmd == "service":
                if orch.service_locked:
                    print(f"🔒 서비스가 '{orch.locked_service}'로 고정되어 있어 변경할 수 없습니다. /new 로 새 세션을 시작하세요.")
                    continue
                if arg in ("teacher", "farmer"):
                    orch.lock_service_session(arg)
                    print(f"🔒 서비스 고정: {arg}")
                    continue
                print("사용법: /service teacher|farmer")
                continue
            if cmd == "help":
                print("명령: /new, /clear, /service <teacher|farmer>, /memory <load|save|clear|weakness>, /exit")
                continue
            if cmd == "memory":
                if arg == "load":
                    questions = orch.load_recent_questions(limit=10)
                    if questions:
                        print(f"📖 최근 {len(questions)}개 문제를 불러왔습니다:")
                        for i, q in enumerate(questions[:5], 1):  # 최대 5개만 표시
                            print(f"  {i}. {q['question'][:50]}... (과목: {q['subject']})")
                        if len(questions) > 5:
                            print(f"  ... 외 {len(questions)-5}개")
                    else:
                        print("📝 저장된 문제가 없습니다.")
                    continue
                elif arg == "save":
                    # 현재 teacher 상태에서 데이터 저장
                    if hasattr(orch, 'last_teacher_state') and orch.last_teacher_state:
                        saved = orch.save_teacher_data(orch.last_teacher_state)
                        if saved:
                            print("💾 현재 teacher 데이터가 저장되었습니다.")
                        else:
                            print("❌ 저장에 실패했습니다.")
                    else:
                        print("❌ 저장할 teacher 데이터가 없습니다.")
                    continue
                elif arg == "clear":
                    if orch.clear_short_term_memory():
                        print("🧹 숏텀메모리가 삭제되었습니다.")
                    else:
                        print("❌ 삭제에 실패했습니다.")
                    continue
                elif arg == "weakness":
                    weakness = orch.get_weakness_analysis(top_k=3)
                    if "error" not in weakness:
                        print("📊 취약점 분석 결과:")
                        for i, (subject, avg_score, attempts) in enumerate(weakness["weak_subjects"], 1):
                            print(f"  {i}. {subject}: 평균 점수 {avg_score:.2f} (시도: {attempts}회)")
                        print(f"  총 문제 수: {weakness['total_questions']}")
                    else:
                        print(f"❌ 취약점 분석 실패: {weakness['error']}")
                    continue
                else:
                    print("사용법: /memory <load|save|clear|weakness>")
                    continue
            print("알 수 없는 명령입니다. /help 를 입력하세요.")
            continue

        init: MainReState = {
            "user_query": uq,
            "user_id": orch.user_id,
            "chat_id": orch.chat_id,
            "service_classification": "",
            "session_data": {},
        }
        print(f"▶️ 실행: thread_id={orch.thread_id}, query='{uq[:120]}' (new turn)")
        cfg_turn = _make_cfg_new_turn(orch)
        res = orch.app.invoke(init, config=cfg_turn)
        s = dict(res)
        
        # teacher 상태 저장 (메모리 명령어용)
        if "teacher_state" in s:
            orch.last_teacher_state = s["teacher_state"]
        
        try:
            top_keys = list(s.keys())[:15]
            print(f"📤 실행 결과 키: {top_keys}{' ...' if len(s)>15 else ''}")
        except Exception:
            pass
        if _print_outputs_if_any(s):
            continue
        # otherwise loop and check HITL


if __name__ == "__main__":
    main()


