import os
from typing import TypedDict, List, Dict, Literal, Optional, Tuple, Any
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ✅ 상태 정의
class SolutionState(TypedDict):
    # 입력
    input_kind: Literal["file", "image", "text", "stm"]  # 입력 종류
    user_input_txt: str
    external_file_paths: List[str]
    external_image_paths: List[str]
    # 문제리스트, 문제, 보기
    user_problems: List[Dict]
    user_problem: str
    user_problem_options: List[str]

    source_type: Literal["internal", "external"]
    short_term_memory: List[Dict]

    vectorstore: Milvus
    retrieved_docs: List[Document]
    similar_questions_text : str

    generated_answer: str         # 해답
    generated_explanation: str   # 풀이
    subject: str

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
        
    @property
    def name(self) -> str:
        return "SolutionAgent"

    @property
    def description(self) -> str:
        return "시험문제를 인식하여 답과 풀이, 해설을 제공하는 에이전트입니다."

    def _create_graph(self) -> StateGraph:
        """워크플로우 그래프 생성"""

        # ✅ LangGraph 구성
        print("📚 LangGraph 흐름 구성 중...")
        
        graph = StateGraph(SolutionState)

        # 분기 & 로딩
        graph.add_node("route", self._route)
        graph.add_node("load_from_short_term_memory", self._load_from_stm)
        graph.add_node("load_from_external_docs", self._load_from_external)
        graph.add_node("load_from_images", self._load_from_images)   # NEW
        graph.add_node("load_from_text", self._load_from_text)       # NEW
        # 공통 처리
        graph.add_node("search_similarity", self._search_similar_questions)
        graph.add_node("generate_solution", self._generate_solution)
        graph.add_node("validate", self._validate_solution)
        graph.add_node("store", self._store_to_vector_db)
        graph.add_node("next_problem", self._next_problem)

        graph.set_entry_point("route")

        def _decide_route(s: SolutionState) -> str:
            # 1) 내부는 무조건 STM 경로
            if s.get("source_type") == "internal":
                return "stm"

            # 2) 외부: 명시 input_kind 우선
            kind = s.get("input_kind")
            if kind:
                return kind

            # 3) 외부: 첨부 상태로 추정
            if s.get("external_file_paths"):
                return "file"
            if s.get("external_image_paths"):
                return "image"
            if s.get("short_term_memory"):
                # 외부라도 상위가 STM을 넘겨줄 수 있으므로 보조 경로
                return "stm"

            # 4) 그 외엔 텍스트
            return "text"
        
        graph.add_conditional_edges(
        "route",
            _decide_route,
            {
                "stm": "load_from_short_term_memory",
                "file": "load_from_external_docs",
                "image": "load_from_images",
                "text": "load_from_text",
            },
        )
        
        graph.add_edge("load_from_short_term_memory", "next_problem")
        graph.add_edge("load_from_external_docs", "next_problem")
        graph.add_edge("load_from_images", "next_problem")   # NEW
        graph.add_edge("load_from_text", "next_problem")     # NEW

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
    
    
    def _llm(self, temperature: float = 0):
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            model=OPENAI_LLM_MODEL,  # ✅ 환경변수에서 가져온 모델
            temperature=temperature,
        )

    
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
            resp = llm.invoke(
                "SYSTEM:\n" + sys_prompt + "\n\nUSER:\n" + user_prompt
            )
            content = (getattr(resp, "content", "") or "").strip()

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
    def normalize_docling_markdown_static(md: str) -> str:
        """Docling 마크다운 정규화 (정적 메서드)"""
        import re
        s = md
        s = re.sub(r'(?m)^\s*(\d+)\.\s*\1\.\s*', r'\1. ', s)  # '1. 1.' -> '1.'
        s = re.sub(r'(?m)^\s*(\d+)\s*\.\s*', r'\1. ', s)      # '1 . ' -> '1. '
        s = re.sub(r'[ \t]+', ' ', s).replace('\r', '')
        return s.strip()

    @staticmethod
    def _find_option_clusters_static(lines: List[str], start: int, end: int) -> List[Tuple[int, int]]:
        """옵션 클러스터 찾기 (정적 메서드)"""
        import re
        _OPT_LINE = re.compile(
            r'(?m)^\s*(?:\(?([1-5])\)?\.?|[①-⑤]|[가-하]\)|[A-Z]\))\s+\S'
        )
        
        clusters = []
        i = start
        while i < end:
            if _OPT_LINE.match(lines[i] or ''):
                j = i
                cnt = 0
                while j < end and _OPT_LINE.match(lines[j] or ''):
                    cnt += 1
                    j += 1
                if cnt >= 3:
                    clusters.append((i, j))  # [i, j) 옵션 블록
                i = j
            else:
                i += 1
        return clusters

    @staticmethod
    def split_problem_blocks_without_keyword_static(text: str) -> List[str]:
        """개선된 문제 블록 분할 (정적 메서드 버전)"""
        import re
        from typing import List, Tuple
        
        if not text:
            return []
            
        text = SolutionAgent.normalize_docling_markdown_static(text)
        lines = text.split('\n')
        n = len(lines)

        # 미리 옵션 클러스터를 계산해놓고, 그 내부 번호는 문항 헤더로 안 봄
        clusters = SolutionAgent._find_option_clusters_static(lines, 0, n)

        def in_option_cluster(idx: int) -> bool:
            for a, b in clusters:
                if a <= idx < b:
                    return True
            return False

        # 문항 헤더 후보 인덱스 수집
        _QHEAD_CAND = re.compile(r'(?m)^\s*(\d{1,3})[.)]\s+\S')
        candidates = []
        for i, ln in enumerate(lines):
            m = _QHEAD_CAND.match(ln or '')
            if not m:
                continue
            if in_option_cluster(i):
                # 보기 블록 안의 번호는 문항 헤더가 아님
                continue
            num = int(m.group(1))
            candidates.append((i, num))

        # 전역 증가 시퀀스 + 섹션 리셋 허용으로 실제 헤더 선별
        headers = []
        prev_num = 0
        last_header_idx = -9999
        for i, num in candidates:
            if num == prev_num + 1:
                headers.append(i)
                prev_num = num
                last_header_idx = i
                continue
            # 섹션 리셋: num==1이고, 최근 헤더에서 충분히 떨어져 있거나 섹션 느낌의 라인 존재 시 허용
            if num == 1:
                window = '\n'.join(lines[max(0, i-3): i+1])
                if (i - last_header_idx) >= 8 or re.search(r'(Ⅰ|Ⅱ|III|과목|파트|SECTION)', window):
                    headers.append(i)
                    prev_num = 1
                    last_header_idx = i
                    continue
            # 그 외는 옵션/노이즈로 무시

        # 헤더가 하나도 안 잡히면 폴백 전략 사용
        if not headers:
            print(f"❌ [디버그] 헤더가 하나도 선택되지 않음 - 폴백 전략 사용")
            # 폴백 1: 더 느슨한 조건으로 재시도
            if candidates:
                print(f"🔄 [폴백] 순차 조건 없이 모든 후보를 헤더로 사용")
                headers = [i for i, num in candidates]
            else:
                # 폴백 2: 기본 번호 패턴으로 분할
                print(f"🔄 [폴백] 기본 번호 패턴으로 분할")
                simple_pattern = re.compile(r'(?m)^\s*(\d{1,2})\.\s+')
                for i, ln in enumerate(lines):
                    if simple_pattern.match(ln or ''):
                        headers.append(i)
                        print(f"📌 [폴백] 라인 {i}: '{ln[:30]}...' → 헤더 추가")
            
            if not headers:
                print(f"❌ [폴백 실패] 전체를 1개 블록으로 처리")
                return [text] if text.strip() else []

        # 헤더 범위로 블록 만들기
        headers.append(n)  # sentinel
        blocks = []
        for a, b in zip(headers[:-1], headers[1:]):
            blk = '\n'.join(lines[a:b]).strip()
            if blk:
                blocks.append(blk)
        return blocks

    def _parse_block_with_llm(self, block_text: str, llm) -> Optional[Dict[str, object]]:
        # (기존 _load_from_external 내부 함수였던 내용을 그대로 이동)
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
            '출력은 반드시 JSON 하나의 객체로만 하며, 다음 스키마를 지켜라:\n'
            '{"question": "<질문 본문(번호/머리글 제거)>", "options": ["<보기1>","<보기2>","<보기3>","<보기4>"]}\n'
            "주의사항:\n"
            "- 반드시 options는 정확히 4개여야 한다.\n"
            "- 입력 블록에 있는 보기 텍스트만 사용하고 새로 만들지 마라.\n"
            "- 불필요한 설명/정답/해설/코드블록/문자열은 출력하지 마라. JSON만 출력하라."
        )
        user_prompt = f"다음 블록을 구조화하라:\n```\n{cleaned_text}\n```"

        try:
            resp = llm.invoke(
                "SYSTEM:\n" + sys_prompt + "\n\nUSER:\n" + user_prompt
            )
            content = (getattr(resp, "content", "") or "").strip()
            m = re.search(r"\{.*\}", content, re.S)
            if not m:
                return None
            obj = json.loads(m.group(0))
            q = (obj.get("question") or "").strip()
            opts = [str(o).strip() for o in (obj.get("options") or []) if str(o).strip()]
            if not q or len(opts) != 4:
                return None
            q = re.sub(r"^\s*(?:문제\s*)?\d{1,3}\s*[\).:]\s*", "", q).strip()
            norm_opts = []
            for o in opts:
                o = re.sub(r"^\s*(?:\(?[①-④1-4A-Da-d가-라]\)?[\).．\.]?)\s*", "", o).strip()
                norm_opts.append(o)
            return {"question": q, "options": norm_opts}
        except Exception as e:
            print(f"⚠️ LLM 파싱 실패: {e}")
            return None

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
    #----------------------------------------nodes------------------------------------------------------

    # --------- 분기 ----------
    def _route(self, state: SolutionState) -> SolutionState:
        print(f"🧭 분기: input_kind={state.get('input_kind')} | source_type={state.get('source_type')}")
        return state

    # --------- 내부: STM에서 문제 1개 꺼내와 state에 세팅 ----------
    def _load_from_stm(self, state: SolutionState) -> SolutionState:
        """
        내부 모드에서 문제 로드 (pdf_extracted 우선, short_term_memory 차선)
        """
        print("📊 [내부] 문제 로드 시작")
        
        # 1. pdf_extracted 우선 확인 (PDF 전처리 데이터)
        pdf_data = state.get("pdf_extracted", {})
        pdf_questions = pdf_data.get("question", []) or []
        
        if pdf_questions:
            print("📄 PDF 전처리 데이터에서 문제 로드")
            questions = pdf_questions
            options_list = pdf_data.get("options", []) or []
        else:
            # 2. short_term_memory에서 로드
            print("📊 short_term_memory에서 문제 로드")
            stm = state.get("short_term_memory", [])
            questions = [x.get("question", "") for x in stm]
            options_list = [x.get("options", []) for x in stm]
        
        # user_problems 설정
        user_problems = []
        for i, question in enumerate(questions):
            options = options_list[i] if i < len(options_list) else []
            if question and options:
                user_problems.append({
                    "question": question,
                    "options": options
                })
        
        state["user_problems"] = user_problems
        state["short_term_memory"] = []  # 큐로 이관
        
        print(f"✅ [내부] 최종 로드된 문제: {len(user_problems)}개")
        return state
    
    # --------- 외부: shared state에서 전처리된 문제 로드 ----------
    def _load_from_external(self, state: SolutionState) -> SolutionState:
        """전처리 노드에서 추출된 문제들을 로드"""
        print("📄 [외부] 문제 로드 시작")
        
        # pdf_extracted에서 문제 로드 (teacher_graph.py에서 이미 처리됨)
        pdf_data = state.get("pdf_extracted", {})
        questions = pdf_data.get("question", []) or []
        options_list = pdf_data.get("options", []) or []
        
        if not questions:
            print("⚠️ pdf_extracted에 문제가 없습니다.")
            state["user_problems"] = []
            return state
        
        # user_problems 형태로 변환
        user_problems = []
        for i, question in enumerate(questions):
            options = options_list[i] if i < len(options_list) else []
            if question and options:
                user_problems.append({
                    "question": question,
                    "options": options
                })
        
        state["user_problems"] = user_problems
        print(f"✅ 최종 로드된 문제: {len(user_problems)}개")
        
        # JSON 파일로 저장 (디버깅용)
        saved_file = self.save_user_problems_to_json(user_problems, "user_problems_json.json")
        print(f"💾 저장된 파일: {saved_file}")
        
        return state
    
    def _load_from_text(self, state: SolutionState) -> SolutionState:
        print("📝 [외부] 텍스트 입력 → 문항 파싱 시작")
        raw = (state.get("user_input_txt") or "").strip()

        if not raw:
            raise ValueError("텍스트 입력이 비었습니다. user_input_txt을 확인하세요.")

        # LLM으로 먼저 시도 (여러 문항 포함 가능)
        llm = self._llm(0)

        qas = self._llm_extract_qas(raw, llm)  # -> List[(q, options)]
        problems = []
        if qas:
            for q, opts in qas:
                # 옵션 0~4개 가능. 4개 아니어도 통과(텍스트 입력은 그대로 프롬프트에 들어갈 수 있게)
                problems.append({"question": q, "options": opts[:4]})
        else:
            # LLM 실패하면 단일 문항으로 처리
            problems = [{"question": raw, "options": []}]

        state["user_problems"] = problems
        saved_file = self.save_user_problems_to_json(state["user_problems"], "user_problems_from_text.json")
        print(f"✅ 텍스트 문항 구성 완료: {len(state['user_problems'])}개 | 저장: {saved_file}")
        return state


    def _load_from_images(self, state: SolutionState) -> SolutionState:
        
        return state


    def _search_similar_questions(self, state: SolutionState) -> SolutionState:
        print("\n🔍 [1단계] 유사 문제 검색 시작")
        
        vectorstore = state.get("vectorstore")
        if vectorstore is None:
            print("⚠️ 벡터스토어가 없어 유사 문제 검색을 건너뜁니다.")
            state["retrieved_docs"] = []
            state["similar_questions_text"] = ""
            print("🔍 [1단계] 유사 문제 검색 함수 종료 (건너뜀)")
            return state
        
        try:
            results = vectorstore.similarity_search(state["user_problem"], k=3)
        except Exception as e:
            print(f"⚠️ 유사 문제 검색 실패: {e}")
            results = []
        
        similar_questions = []
        for i, doc in enumerate(results):
            metadata = doc.metadata
            options = json.loads(metadata.get("options", "[]"))
            answer = metadata.get("answer", "")
            explanation = metadata.get("explanation", "")
            subject = metadata.get("subject", "기타")

            formatted = f"""[유사문제 {i+1}]
                문제: {doc.page_content}
                보기:
                """ + "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)]) + f"""
                정답: {answer}
                풀이: {explanation}
                과목: {subject}
                """
            similar_questions.append(formatted)
        
        state["retrieved_docs"] = results
        state["similar_questions_text"] = "\n\n".join(similar_questions) 

        print(f"유사 문제 {len(results)}개 검색 완료.")
        print("🔍 [1단계] 유사 문제 검색 함수 종료")
        return state

    def _generate_solution(self, state: SolutionState) -> SolutionState:

        print("\n✏️ [2단계] 해답 및 풀이 생성 시작")

        llm_gen = self._llm(0.5)  

        similar_problems = state.get("similar_questions_text", "")
        print("유사 문제들:\n", similar_problems[:100])

        prompt = f"""
            사용자가 입력한 질문:
            {state['user_input_txt']}
            다음은 사용자가 입력한 문제:
            {state['user_problem']}
            {state['user_problem_options']}

            아래는 이 문제와 유사한 문제들:
            {similar_problems}

            1. 이 문제의 **정답**만 간결하게 한 문장으로 먼저 작성해 주세요.
            2. 이어서 그 정답인 근거를 담은 **풀이 과정**을 상세히 설명해 주세요.
            3. 이 문제의 과목을 정보처리기사 과목 5개 중에서 가장 적합한 것으로 지정해 주세요. [소프트웨어 설계, 소프트웨어 개발, 데이터베이스 구축, 프로그래밍 언어 활용, 정보시스템 구축 관리]

            출력 형식:
            정답: ...
            풀이: ...
            과목: ...
        """

        response = llm_gen.invoke(prompt)
        result = response.content.strip()
        print("🧠 LLM 응답 완료")

        answer_match = re.search(r"정답:\s*(.+)", result)
        explanation_match = re.search(r"풀이:\s*(.+)", result, re.DOTALL)
        subject_match = re.search(r"과목:\s*(.+)", result)
        state["generated_answer"] = answer_match.group(1).strip() if answer_match else ""
        state["generated_explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        state["subject"] = subject_match.group(1).strip() if subject_match else "기타"
        state["chat_history"].append(f"Q: {state['user_input_txt']}\nP: {state['user_problem']}\nA: {state['generated_answer']}\nE: {state['generated_explanation']}")

        return state

    # ✅ 정합성 검증 (간단히 길이 기준 사용)

    def _validate_solution(self, state: SolutionState) -> SolutionState:
        print("\n🧐 [3단계] 정합성 검증 시작")
        
        llm = self._llm(0)

        validation_prompt = f"""
        사용자 요구사항: {state['user_input_txt']}

        문제 질문: {state['user_problem']}
        문제 보기: {state['user_problem_options']}

        생성된 정답: {state['generated_answer']}
        생성된 풀이: {state['generated_explanation']}
        생성된 과목: {state['subject']}

        생성된 해답과 풀이, 과목이 문제와 사용자 요구사항에 맞고, 논리적 오류나 잘못된 정보가 없습니까?
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
                        "explanation": state["generated_explanation"],
                        "subject": state["subject"],
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
            "subject": state["subject"],
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

    def invoke(
            self, 
            user_input_txt: str, 
            source_type: Literal["internal", "external"],

            input_kind: Optional[Literal["file", "image", "text", "stm"]] = None,
            external_image_paths: Optional[List[str]] = None,
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
            try:
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
                print("✅ Milvus 벡터스토어 연결 성공")
            except Exception as e:
                print(f"⚠️ Milvus 연결 실패: {e}")
                print("   - 벡터스토어 없이 실행을 계속합니다.")
                vectorstore = None
        
        if source_type == "internal":
            inferred_input = "stm"
        else:   
            inferred_input = input_kind
            if external_file_paths:
                inferred_input = "file"
            elif external_image_paths:
                inferred_input = "image"
            elif short_term_memory:
                inferred_input = "stm"
            else:
                inferred_input = "text"  # 파일/이미지/STM 없으면 텍스트로 간주

        initial_state: SolutionState = {
            "user_input_txt": user_input_txt,
            "source_type": source_type,
            "input_kind": inferred_input,
            "external_image_paths": external_image_paths or [],

            "user_problems": [], 
            "user_problem": "",
            "user_problem_options": [],

            "short_term_memory": short_term_memory or [],
            "external_file_paths": external_file_paths or [],

            "vectorstore": vectorstore,
            "retrieved_docs": [],
            "similar_questions_text": "",

            "generated_answer": "",
            "generated_explanation": "",
            "subject": "",
            "validated": False,
            "retry_count": 0,
            "results": [],
            
            "exam_title": exam_title,
            "difficulty": difficulty,
            "subject": subject,

            "chat_history": []
        }
        
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
        
        # 그래프 시각화
        try:
            graph_image_path = "./teacher/agents/solution/agent_workflow.png"
            with open(graph_image_path, "wb") as f:
                f.write(self.graph.get_graph().draw_mermaid_png())
            print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
        except Exception as e:
            print(f"그래프 시각화 중 오류 발생: {e}")
            print("워크플로우는 정상적으로 작동합니다.")

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
        
        return final_state

    

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph 그래프를 subgraph로 실행하는 메서드입니다.
        
        Args:
            input_data (Dict[str, Any]): 에이전트 실행에 필요한 입력 데이터입니다.
            
        Returns:
            Dict[str, Any]: 에이전트 실행 결과 데이터입니다.
        """
        try:
            # LangGraph 그래프 실행
            final_state = self.graph.invoke(input_data)
            
            # 결과 추출 및 반환
            results = final_state.get("results", [])
            generated_answer = final_state.get("generated_answer", "")
            generated_explanation = final_state.get("generated_explanation", "")
            
            return {
                "results": results,
                "generated_answer": generated_answer,
                "generated_explanation": generated_explanation,
                "final_state": final_state
            }
            
        except Exception as e:
            print(f"❌ SolutionAgent invoke 실행 실패: {e}")
            return {
                "results": [],
                "generated_answer": "",
                "generated_explanation": "",
                "error": str(e)
            }

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

    # 그래프 시각화 (선택)
    try:
        graph_image_path = "./teacher/agents/solution/agent_workflow.png"
        with open(graph_image_path, "wb") as f:
            f.write(agent.graph.get_graph().draw_mermaid_png())
        print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")
        print("워크플로우는 정상적으로 작동합니다.")

    # ✅ 사용자 질문 입력
    user_input_txt = input("\n❓ 사용자 질문을 입력하세요 : ").strip()

    # ✅ (수정) 키워드 인자 + input_kind 명시
    final_state = agent.execute(
        user_input_txt=user_input_txt,
        source_type="external",
        input_kind="text",
        vectorstore=vectorstore,
        # external_image_paths=["./teacher/agents/solution/user_problems.png"],
    )

    # 결과를 JSON 파일로 저장
    results = final_state.get("results", [])
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "user_input_txt": user_input_txt,
        "total_results": len(results),
        "results": results
    }

    results_filename = os.path.join("./teacher/agents/solution", f"solution_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    with open(results_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 해답 결과가 JSON 파일로 저장되었습니다: {results_filename}")
