import os
from typing import TypedDict, List, Dict, Literal, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import connections
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
import json, re
from langchain_openai import ChatOpenAI
from ..base_agent import BaseAgent
from docling.document_converter import DocumentConverter
from datetime import datetime

load_dotenv()
GROQ_API_KEY=REDACTED("GROQ_API_KEY=REDACTED SolutionState(TypedDict):
    user_question: str
    user_problems: List[Dict]
    user_problem: str
    user_problem_options: List[str]

    source_type: Literal["internal", "external"]
    # 내부/외부 원천
    short_term_memory: List[Dict]
    external_file_paths: List[str] 

    vectorstore: Milvus
    retrieved_docs: List[Document]
    similar_questions_text : str

    generated_answer: str         # 해답
    generated_explanation: str   # 풀이
    results: List[Dict]
    validated: bool
    retry_count: int             # 검증 실패 시 재시도 횟수

    exam_title: str
    difficulty: str
    subject: str

    chat_history: List[str]
    
class SolutionAgent(BaseAgent):
    """문제 해답/풀이 생성 에이전트"""

    def __init__(self):
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """워크플로우 그래프 생성"""

        # ✅ LangGraph 구성
        print("📚 LangGraph 흐름 구성 중...")
        
        graph = StateGraph(SolutionState)

        # 분기 & 로딩
        graph.add_node("route", self._route)
        graph.add_node("load_from_short_term_memory", self._load_from_stm)
        graph.add_node("load_from_external_docs", self._load_from_external)

        # 공통 처리
        graph.add_node("search_similarity", self._search_similar_questions)
        graph.add_node("generate_solution", self._generate_solution)
        graph.add_node("validate", self._validate_solution)
        graph.add_node("store", self._store_to_vector_db)
        graph.add_node("next_problem", self._next_problem)

        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route", 
            lambda s: s["source_type"],
            {"internal": "load_from_short_term_memory", "external": "load_from_external_docs"})
        graph.add_edge("load_from_short_term_memory", "next_problem")
        graph.add_edge("load_from_external_docs", "next_problem")

        graph.add_edge("next_problem", "search_similarity")
        graph.add_edge("search_similarity", "generate_solution")
        graph.add_edge("generate_solution", "validate")

        graph.add_conditional_edges(
            "validate", 
            lambda s: "ok" if s["validated"] else ("back" if s.get("retry_count", 0) < 5 else "fail"),
            {"ok": "store", "back": "generate_solution", "fail": "next_problem"}
        )

        # 저장 후 남은 문제가 있으면 next_problem로 루프
        graph.add_conditional_edges(
            "store",
            lambda s: "more" if len(s.get("user_problems", [])) > 0 else "done",
            {"more": "next_problem", "done": END}
        )

        return graph.compile()
    
    def _llm_extract_qas(self, text: str, llm) -> List[tuple]:
        """
        LLM에게 페이지 텍스트를 주고
        [{"question":"...","options":["...","..."]}, ...] 만 받는다.
        실패 시 [] 반환.
        """
        sys_prompt = (
            "너는 시험 문제 PDF에서 텍스트를 구조화하는 도우미다. "
            "문제 질문과 보기를 구분해서 question과 options 배열로 출력한다."
            "options는 보기 항목만 포함하고, 설명/해설/정답 등은 포함하지 않는다. "
            "응답은 반드시 JSON 배열만 출력한다. 다른 문장이나 코드는 절대 포함하지 말 것."
        )
        user_prompt = (
            "다음 텍스트에서 문항을 최대한 그대로, 정확히 추출해 JSON 배열로 만들어줘.\n"
            "요구 스키마: [{\"question\":\"...\",\"options\":[\"...\",\"...\"]}]\n"
            "규칙:\n"
            "- 문제 질문에서 번호(예: '문제 1.' 등)와 불필요한 머리글은 제거.\n"
            "- 옵션은 4개가 일반적임.\n"
            f"텍스트:\n{text}"
        )

        try:
            resp = llm.invoke([{"role":"system","content":sys_prompt},
                            {"role":"user","content":user_prompt}])
            content = (resp.content or "").strip()

            # JSON만 남기기 (혹시 모델이 불필요한 텍스트를 붙였을 때 대비)
            m = re.search(r"\[.*\]", content, re.S)
            if not m:
                return []
            arr = json.loads(m.group(0))

            results = []
            for item in arr:
                q = (item.get("question") or "").strip()
                opts = [str(o).strip() for o in (item.get("options") or [])]
                if q:
                    results.append((q, opts))
            return results
        except Exception:
            return []

    @staticmethod
    def _split_problem_blocks(raw: str) -> List[str]:
        """
        빈 줄(2개 이상) 기준으로 블록 분할.
        페이지 구분(\f)은 빈 줄로 치환.
        머리글/푸터/잡음 라인은 1차 필터링.
        """
        if not raw:
            return []

        text = raw.replace("\f", "\n\n")      # 페이지 경계는 빈 줄로
        text = re.sub(r"[ \t]+\n", "\n", text)  # 행 끝 공백 제거
        # 문서 공통 잡음 헤더/푸터(필요시 추가)
        noise_patterns = [
            r"^\s*문제\s*지\s*$", r"^\s*모의\s*고사\s*$", r"^\s*페이지\s*\d+\s*$"
        ]

        blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]

        cleaned_blocks = []
        for b in blocks:
            lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
            # 잡음 제거
            kept = []
            for ln in lines:
                if any(re.search(pat, ln, re.I) for pat in noise_patterns):
                    continue
                kept.append(ln)
            if not kept:
                continue
            cleaned_blocks.append("\n".join(kept))

        return cleaned_blocks

    
    # --------- 분기 ----------
    def _route(self, state: SolutionState) -> SolutionState:
        # 오케스트레이터가 채워준 source_type을 그대로 사용
        st = state["source_type"]
        print(f"🧭 분기: {st}")
        return state

    # --------- 내부: STM에서 문제 1개 꺼내와 state에 세팅 ----------
    def _load_from_stm(self, state: SolutionState) -> SolutionState:
        stm = state.get("short_term_memory", [])
        state["user_problems"] = [{"question": x.get("question",""),
                                "options": x.get("options",[])} for x in stm]
        state["short_term_memory"] = []  # 큐로 이관
        return state
    
    # --------- 외부: Docling으로 문서 → 텍스트 → JSON(문제/옵션) → state에 세팅 ----------
    def _load_from_external(self, state: SolutionState) -> SolutionState:
        """
        PDF/문서 → 텍스트 → [문제 블록 분할] → (블록 단위) LLM 파싱 → '문항+보기4'만 저장
        """
        print("📄 [외부] 첨부 문서 로드 → 텍스트 변환 → 블록 단위 LLM 파싱 시작")
        paths = state.get("external_file_paths", [])
        if not paths:
            raise ValueError("external_file_paths 가 비어있습니다. 외부 분기에서는 파일 경로가 필요합니다.")

        converter = DocumentConverter()

        # LLM (엄격한 구조화 전용)
        llm = ChatOpenAI(
            api_key=GROQ_API_KEY=REDACTED="https://api.groq.com/openai/v1",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0
        )

        # ----- 블록 1개를 LLM으로 파싱하는 내부 함수 -----
        def parse_block_with_llm(block_text: str) -> Optional[Dict[str, object]]:
            # 노이즈 제거 (정답/해설 라인)
            cleaned = []
            for ln in block_text.splitlines():
                if re.search(r"(정답|해설|답안|풀이|answer|solution)\s*[:：]", ln, re.I):
                    continue
                cleaned.append(ln)
            cleaned_text = "\n".join(cleaned).strip()

            if len(cleaned_text) < 5:
                return None

            sys_prompt = (
                "너는 시험 블록 텍스트를 정확히 구조화하는 도우미다. "
                "입력 블록에는 '한 문제'가 들어있다. "
                "출력은 반드시 JSON 하나의 객체로만 하며, 다음 스키마를 지켜라:\n"
                '{"question": "<질문 본문(번호/머리글 제거)>", "options": ["<보기1>","<보기2>","<보기3>","<보기4>"]}\n'
                "주의사항:\n"
                "- 반드시 options는 정확히 4개여야 한다.\n"
                "- 입력 블록에 있는 보기 텍스트만 사용하고 새로 만들지 마라.\n"
                "- 불필요한 설명/정답/해설/코드블록/문자열은 출력하지 마라. JSON만 출력하라."
            )
            user_prompt = f"다음 블록을 구조화하라:\n```\n{cleaned_text}\n```"

            try:
                resp = llm.invoke([
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ])
                content = (resp.content or "").strip()
                m = re.search(r"\{.*\}", content, re.S)  # JSON 객체만 추출
                if not m:
                    return None
                obj = json.loads(m.group(0))

                q = (obj.get("question") or "").strip()
                opts = [str(o).strip() for o in (obj.get("options") or []) if str(o).strip()]
                if not q or len(opts) != 4:
                    return None

                # 번호/머리글 정리
                q = re.sub(r"^\s*(?:문제\s*)?\d{1,3}\s*[\).:]\s*", "", q).strip()
                norm_opts = []
                for o in opts:
                    o = re.sub(r"^\s*(?:\(?[①-④1-4A-Da-d가-라]\)?[\).．\.]?)\s*", "", o).strip()
                    norm_opts.append(o)

                return {"question": q, "options": norm_opts}

            except Exception as e:
                print(f"⚠️ LLM 파싱 실패: {e}")
                return None

        extracted: List[Dict[str, object]] = []

        for p in paths:
            try:
                result = converter.convert(p)
                doc = result.document
            except Exception as e:
                print(f"⚠️ 변환 실패: {p} - {e}")
                continue

            # 문서 전체 텍스트 추출
            raw = ""
            if hasattr(doc, "export_to_markdown"):
                raw = doc.export_to_markdown()
            elif hasattr(doc, "export_to_text"):
                raw = doc.export_to_text()
            raw = (raw or "").replace("\r\n", "\n")

            # ✅ 문제 블록 분할 (빈 줄 2개 이상 기준 + 일부 헤더 제거)
            blocks = self._split_problem_blocks(raw)
            print(f"📦 {p} | 추정 문제 블록 수: {len(blocks)}")

            for idx, block in enumerate(blocks, 1):
                item = parse_block_with_llm(block)
                if item:
                    extracted.append({
                        "question": item["question"],
                        "options": item["options"],
                        "source": p,
                        "block_index": idx
                    })

        if not extracted:
            raise ValueError("문서에서 '문항 + 보기4'를 추출하지 못했습니다. LLM 파싱 규칙 또는 블록 분할 기준을 조정하세요.")

        # ✅ 질문 텍스트 기준 중복 제거
        seen, deduped = set(), []
        for it in extracted:
            key = re.sub(r"\s+", " ", it["question"]).strip()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)

        state["user_problems"] = [{"question": it["question"], "options": it["options"]} for it in deduped]
        print(f"✅ 최종 추출 문항 수(보기 4개): {len(state['user_problems'])}")

        saved_file = self.save_user_problems_to_json(state["user_problems"], "user_problems_json.json")
        print(f"💾 저장된 파일: {saved_file}")
        return state


    # 간단한 문제/보기 파서 (문서 포맷에 맞게 조정 가능)
    def _split_by_questions(self, text: str) -> List[tuple]:
        blocks = re.split(r"\n\s*\n", text)  # 빈 줄 기준 거칠게 분할
        results = []
        for b in blocks:
            lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
            if not lines:
                continue
            # 옵션 라인 감지 (숫자. 또는 숫자) 패턴)
            opts = [ln for ln in lines if re.match(r"^\(?\d+\)?[).]\s*", ln)]
            if opts:
                # 문제문은 옵션 라인 제외 첫 줄 위주로 사용
                question_lines = [ln for ln in lines if ln not in opts]
                qtext = " ".join(question_lines) if question_lines else lines[0]
                # 옵션 텍스트 정제: "1) ..." → "..." 로
                clean_opts = [re.sub(r"^\(?\d+\)?[).]\s*", "", o) for o in opts]
                results.append((qtext, clean_opts))
        return results


    def _search_similar_questions(self, state: SolutionState) -> SolutionState:
        print("\n🔍 [1단계] 유사 문제 검색 시작")
        results = state["vectorstore"].similarity_search(state["user_problem"], k=3)

        similar_questions = []
        for i, doc in enumerate(results):
            metadata = doc.metadata
            options = json.loads(metadata.get("options", "[]"))
            answer = metadata.get("answer", "")
            explanation = metadata.get("explanation", "")

            formatted = f"""[유사문제 {i+1}]
                문제: {doc.page_content}
                보기:
                """ + "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)]) + f"""
                정답: {answer}
                풀이: {explanation}
                """
            similar_questions.append(formatted)
        
        state["retrieved_docs"] = results
        state["similar_questions_text"] = "\n\n".join(similar_questions) 

        print(f"유사 문제 {len(results)}개 검색 완료.")
        print("🔍 [1단계] 유사 문제 검색 함수 종료")
        return state

    def _generate_solution(self, state: SolutionState) -> SolutionState:

        print("\n✏️ [2단계] 해답 및 풀이 생성 시작")

        llm = ChatOpenAI(
            api_key=GROQ_API_KEY=REDACTED="https://api.groq.com/openai/v1",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.5
        )

        similar_problems = state.get("similar_questions_text", "")
        print("유사 문제들:\n", similar_problems[:100])

        prompt = f"""
            사용자가 입력한 질문:
            {state['user_question']}
            다음은 사용자가 입력한 문제:
            {state['user_problem']}
            {state['user_problem_options']}

            아래는 이 문제와 유사한 문제들:
            {similar_problems}

            1. 이 문제의 **정답**만 간결하게 한 문장으로 먼저 작성해 주세요.
            2. 이어서 그 정답인 근거를 담은 **풀이 과정**을 상세히 설명해 주세요.

            출력 형식:
            정답: ...
            풀이: ...
        """

        response = llm.invoke(prompt)
        result = response.content.strip()
        print("🧠 LLM 응답 완료")

        answer_match = re.search(r"정답:\s*(.+)", result)
        explanation_match = re.search(r"풀이:\s*(.+)", result, re.DOTALL)
        state["generated_answer"] = answer_match.group(1).strip() if answer_match else ""
        state["generated_explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        state["chat_history"].append(f"Q: {state['user_question']}\nP: {state['user_problem']}\nA: {state['generated_answer']}\nE: {state['generated_explanation']}")

        return state

    # ✅ 정합성 검증 (간단히 길이 기준 사용)

    def _validate_solution(self, state: SolutionState) -> SolutionState:
        print("\n🧐 [3단계] 정합성 검증 시작")
        
        llm = ChatOpenAI(
            api_key=GROQ_API_KEY=REDACTED="https://api.groq.com/openai/v1",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0
        )

        validation_prompt = f"""
        사용자 요구사항: {state['user_question']}

        문제 질문: {state['user_problem']}
        문제 보기: {state['user_problem_options']}

        생성된 정답: {state['generated_answer']}
        생성된 풀이: {state['generated_explanation']}

        생성된 해답과 풀이가 문제와 사용자 요구사항에 맞고, 논리적 오류나 잘못된 정보가 없습니까?
        적절하다면 '네', 그렇지 않다면 '아니오'로만 답변하세요.
        """

        validation_response = llm.invoke(validation_prompt)
        result_text = validation_response.content.strip().lower()

        # ✅ '네'가 포함된 응답일 경우에만 유효한 풀이로 판단
        print("📌 검증 응답:", result_text)
        state["validated"] = "네" in result_text
        
        if not state["validated"]:
            state["retry_count"] = state.get("retry_count", 0) + 1
            print(f"⚠️ 검증 실패 (재시도 {state['retry_count']}/5)")
        else:
            print("✅ 검증 결과: 통과")
            
        return state


    # ✅ 임베딩 후 벡터 DB 저장
    def _store_to_vector_db(self, state: SolutionState) -> SolutionState:  
        print("\n🧩 [4단계] 임베딩 및 벡터 DB 저장 시작")

        # 벡터 DB 저장 (외부인 경우)
        if state["source_type"] == "external":
            vectorstore = state["vectorstore"] 

            # 중복 문제 확인
            similar = vectorstore.similarity_search(state["user_problem"], k=1)
            if similar and state["user_problem"].strip() in similar[0].page_content:
                print("⚠️ 동일한 문제가 존재하여 저장 생략")
            else:
                # 문제, 해답, 풀이를 각각 metadata로 저장
                doc = Document(
                    page_content=state["user_problem"],
                    metadata={
                        "options": json.dumps(state.get("user_problem_options", [])), 
                        "answer": state["generated_answer"],
                        "explanation": state["generated_explanation"]
                    }
                )
                vectorstore.add_documents([doc])
                print("✅ 문제+해답+풀이 저장 완료")
        else:
            print("⚠️ 내부 저장소는 벡터 DB 저장을 지원하지 않습니다. 내부 문제로만 저장합니다.")
            # 내부: 요구 스키마(JSON)로 파일 누적 저장
            store_path = "./internal_store.json"
            data = {
                "exam_title": state.get("exam_title", "내부 문제 모음"),
                "total_questions": 0,
                "difficulty": state.get("difficulty", "중급"),
                "subjects": {},  # subject명: {"requested_count":0,"actual_count":n,"questions":[...]}
            }

            if os.path.exists(store_path):
                try:
                    with open(store_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    pass

            subj = state.get("subject", "기타")
            subjects = data.setdefault("subjects", {})
            bucket = subjects.setdefault(subj, {"requested_count": 0, "actual_count": 0, "questions": []})

            bucket["questions"].append({
                "question": state["user_problem"],
                "options": [f"  {i+1}. {opt}" for i, opt in enumerate(state.get("user_problem_options", []))],
                "answer": state["generated_answer"],
                "explanation": state["generated_explanation"],
                "subject": subj,
            })
            bucket["actual_count"] = len(bucket["questions"])

            # 총 문항 수 재계산
            total = 0
            for v in subjects.values():
                total += len(v.get("questions", []))
            data["total_questions"] = total

            with open(store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ 내부 문제 저장(JSON 스키마) 완료 → {store_path}")

        # 결과를 state에 저장 (항상 실행)
        print(f"\n📝 결과 저장 시작:")
        print(f"   - 현재 문제: {state['user_problem'][:50]}...")
        print(f"   - 생성된 정답: {state['generated_answer'][:30]}...")
        print(f"   - 검증 상태: {state['validated']}")
        
        item = {
            "question": state["user_problem"],
            "options": state["user_problem_options"],
            "generated_answer": state["generated_answer"],
            "generated_explanation": state["generated_explanation"],
            "validated": state["validated"],
            "chat_history": state.get("chat_history", [])
        }
        
        
        state["results"].append(item)
        print(f"✅ 결과 저장 완료: {len(state['results'])}개")
        
        return state
    
    def _next_problem(self, state: SolutionState) -> SolutionState:
        queue = state.get("user_problems", [])
        if not queue:
            raise ValueError("처리할 문제가 없습니다. user_problems가 비어있어요.")
        
        current = queue.pop(0)
        state["user_problem"] = current.get("question", "")
        state["user_problem_options"] = current.get("options", [])
        state["user_problems"] = queue
        
        print(f"📝 다음 문제 처리: {state['user_problem'][:50]}...")
        print(f"   - 남은 문제 수: {len(queue)}")
        
        return state

    def execute(
            self, 
            user_question: str, 
            source_type: Literal["internal", "external"],
            vectorstore: Optional[Milvus] = None,
            short_term_memory: Optional[List[Dict]] = None,
            external_file_paths: Optional[List[str]] = None,
            exam_title: str = "정보처리기사 모의고사 (Groq 순차 버전)",
            difficulty: str = "중급",
            subject: str = "기타",
            recursion_limit: int = 1000,
        ) -> Dict:

        # ✅ Milvus 연결 및 벡터스토어 생성
        if vectorstore is None:

            embedding_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={"device": "cpu"}
            )

            if "default" in connections.list_connections():
                connections.disconnect("default")
            connections.connect(alias="default", host="localhost", port="19530")

            vectorstore = Milvus(
                embedding_function=embedding_model,
                collection_name="problems",
                connection_args={"host": "localhost", "port": "19530"}
            )
        
        initial_state: SolutionState = {
            "user_question": user_question,
            "user_problems": [], 
            "user_problem": "",
            "user_problem_options": [],

            "source_type": source_type,
            "short_term_memory": short_term_memory or [],
            "external_file_paths": external_file_paths or [],

            "vectorstore": vectorstore,
            "retrieved_docs": [],
            "similar_questions_text": "",

            "generated_answer": "",
            "generated_explanation": "",
            "validated": False,
            "retry_count": 0,
            "results": [],
            
            "exam_title": exam_title,
            "difficulty": difficulty,
            "subject": subject,

            "chat_history": []
        }
        
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
        
        # 결과 확인 및 디버깅
        results = final_state.get("results", [])
        print(f"\n🎯 최종 실행 결과:")
        print(f"   - 총 결과 수: {len(results)}")
        print(f"   - 결과 키 존재: {'results' in final_state}")
        print(f"   - 상태 키들: {list(final_state.keys())}")
        
        if results:
            for i, result in enumerate(results):
                print(f"   - 결과 {i+1}: {result.get('question', '')[:30]}...")
        else:
            print("   ⚠️ results가 비어있습니다!")
            print(f"   - final_state 내용: {final_state}")
        
        return results

    def save_user_problems_to_json(self, user_problems: List[Dict], filename: str = None) -> str:
        """
        user_problems를 JSON 파일로 저장합니다.
        
        Args:
            user_problems: 저장할 문제 데이터 리스트
            filename: 저장할 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"user_problems_{timestamp}.json"
        
        # 파일 경로가 상대 경로인 경우 현재 디렉토리에 저장
        if not os.path.isabs(filename):
            filename = os.path.join(os.getcwd(), filename)
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # JSON 데이터 준비
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_problems": len(user_problems),
            "problems": user_problems
        }
        
        # JSON 파일로 저장
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ user_problems가 JSON 파일로 저장되었습니다: {filename}")
        return filename

if __name__ == "__main__":
    # ✅ Milvus 연결 및 벡터스토어 생성
    embedding_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"}
    )

    if "default" in connections.list_connections():
        connections.disconnect("default")
    connections.connect(alias="default", host="localhost", port="19530")

    vectorstore = Milvus(
        embedding_function=embedding_model,
        collection_name="problems",
        connection_args={"host": "localhost", "port":"19530"}
    )

    agent = SolutionAgent()

    # 그래프 시각화
    try:
        graph_image_path = "agent_workflow.png"
        with open(graph_image_path, "wb") as f:
            f.write(agent.graph.get_graph().draw_mermaid_png())
        print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")
        print("워크플로우는 정상적으로 작동합니다.")

    # ✅ 사용자 질문 입력
    user_question = input("\n❓ 사용자 질문을 입력하세요 : ").strip()
    
    results = agent.execute(user_question, "external", vectorstore, external_file_paths=["./user_problems.pdf"])

    # 결과를 JSON 파일로 저장
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "user_question": user_question,
        "total_results": len(results),
        "results": results
    }
    
    results_filename = f"solution_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 해답 결과가 JSON 파일로 저장되었습니다: {results_filename}")
