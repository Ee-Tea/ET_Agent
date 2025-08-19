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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ 상태 정의
class SolutionState(TypedDict):
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
        개선된 문제 블록 분할 알고리즘 (정적 메서드 버전)
        """
        return SolutionAgent.split_problem_blocks_without_keyword_static(raw)
    
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

    
    # --------- 분기 ----------
    def _route(self, state: SolutionState) -> SolutionState:
        # 오케스트레이터가 채워준 source_type을 그대로 사용
        st = state["source_type"]
        print(f"🧭 분기: {st}")
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
    
    def _filter_problems_by_range(self, problems: List[Dict], problem_range: Dict) -> List[Dict]:
        """문제 범위에 따라 문제들을 필터링"""
        if not problem_range:
            return problems
        
        range_type = problem_range.get("type")
        
        if range_type == "single":
            # 단일 번호
            target_num = problem_range.get("number")
            return [p for p in problems if p.get("index") == target_num]
        
        elif range_type == "range":
            # 범위
            start = problem_range.get("start")
            end = problem_range.get("end")
            return [p for p in problems if start <= p.get("index", 0) <= end]
        
        elif range_type == "specific":
            # 특정 번호들
            target_numbers = problem_range.get("numbers", [])
            return [p for p in problems if p.get("index") in target_numbers]
        
        else:
            print(f"⚠️ 알 수 없는 범위 타입: {range_type}")
            return problems
    
    # --------- 외부: shared state에서 전처리된 문제 로드 ----------
    def _load_from_external(self, state: SolutionState) -> SolutionState:
        """
        전처리 노드에서 추출된 문제들을 적절한 소스에서 가져와서 user_problems에 설정
        """
        print("📄 [외부] 문제 로드 시작")
        
        # artifacts에서 문제 소스와 범위 정보 가져오기
        artifacts = state.get("artifacts", {})
        problem_source = artifacts.get("problem_source")
        problem_range = artifacts.get("problem_range")
        
        print(f"📚 문제 소스: {problem_source}")
        print(f"🔢 문제 범위: {problem_range}")
        
        # 문제 소스 결정 (우선순위: PDF 존재 여부 > 명시적 지정 > shared)
        pdf_data = state.get("pdf_extracted", {})
        pdf_questions = pdf_data.get("question", []) or []
        
        if pdf_questions:
            # PDF 데이터가 있으면 무조건 PDF 우선
            questions = pdf_questions
            options_list = pdf_data.get("options", []) or []
            print("📄 PDF 전처리 state에서 문제 로드 (PDF 데이터 존재)")
        elif problem_source == "shared":
            questions = state.get("question", []) or []
            options_list = state.get("options", []) or []
            print("📊 shared state에서 문제 로드")
        elif problem_source == "pdf_extracted" or artifacts.get("pdf_ids"):
            questions = pdf_questions  # 빈 리스트
            options_list = pdf_data.get("options", []) or []
            print("📄 PDF 전처리 state에서 문제 로드 (명시적 지정)")
        else:
            # 기본: shared state 사용
            questions = state.get("question", []) or []
            options_list = state.get("options", []) or []
            print("📊 shared state에서 문제 로드 (기본값)")
            
        # 디버그 정보
        print(f"🔍 [디버그] 최종 선택된 소스의 문제 수: {len(questions)}")
        print(f"🔍 [디버그] 전체 state 키들: {list(state.keys())}")
        print(f"🔍 [디버그] pdf_extracted 존재: {'pdf_extracted' in state}")
        if 'pdf_extracted' in state:
            pdf_debug = state['pdf_extracted']
            print(f"🔍 [디버그] pdf_extracted 문제 수: {len(pdf_debug.get('question', []))}")
        
        if not questions:
            print("⚠️ 선택된 소스에 문제가 없습니다.")
            state["user_problems"] = []
            return state
        
        # user_problems 형태로 변환
        all_problems = []
        for i, question in enumerate(questions):
            options = options_list[i] if i < len(options_list) else []
            if question and options:
                all_problems.append({
                    "question": question,
                    "options": options,
                    "index": i + 1  # 1-based 번호
                })
        
        # 문제 범위 필터링
        if problem_range:
            filtered_problems = self._filter_problems_by_range(all_problems, problem_range)
            print(f"🎯 범위 필터링: {len(all_problems)}개 → {len(filtered_problems)}개")
        else:
            filtered_problems = all_problems
            print(f"📝 전체 문제 로드: {len(filtered_problems)}개")
        
        # index 제거 (solution_agent 내부에서는 필요 없음)
        user_problems = []
        for problem in filtered_problems:
            user_problems.append({
                "question": problem["question"],
                "options": problem["options"]
            })
        
        state["user_problems"] = user_problems
        print(f"✅ 최종 로드된 문제: {len(user_problems)}개")
        
        # JSON 파일로 저장 (디버깅용)
        saved_file = self.save_user_problems_to_json(user_problems, "user_problems_json.json")
        print(f"💾 저장된 파일: {saved_file}")
        
        return state
    
    # --------- 기존 PDF 추출 로직 (백업용) ----------
    def _load_from_external_OLD_BACKUP(self, state: SolutionState) -> SolutionState:
        """
        PDF/문서 → 텍스트 → [문제 블록 분할] → (블록 단위) LLM 파싱 → '문항+보기4'만 저장
        """
        print("📄 [외부] 첨부 문서 로드 → 텍스트 변환 → 블록 단위 LLM 파싱 시작")
        paths = state.get("external_file_paths", [])
        if not paths:
            raise ValueError("external_file_paths 가 비어있습니다. 외부 분기에서는 파일 경로가 필요합니다.")

        # 환경변수 설정으로 권한 문제 해결
        import os
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        os.environ['HF_HOME'] = 'C:\\temp\\huggingface_cache'
        
        # cv2 setNumThreads 문제 해결
        try:
            import cv2
            if not hasattr(cv2, 'setNumThreads'):
                # setNumThreads가 없으면 더미 함수 추가
                cv2.setNumThreads = lambda x: None
        except ImportError:
            pass
        
        converter = DocumentConverter()

        # LLM (엄격한 구조화 전용)
        llm = ChatOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="moonshotai/kimi-k2-instruct",
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
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="moonshotai/kimi-k2-instruct",
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
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="moonshotai/kimi-k2-instruct",
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
