# teacher/pdf_preprocessor_ai.py
# -*- coding: utf-8 -*-
"""
PDF 전처리 & 문제 추출 (4단계 흐름)
1. 좌우 텍스트 분리
2. 세그먼트 구분 (숫자. 패턴)
3. 문항/보기 추출
4. 저장 및 출력
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
# 파일 상단 import 아래에 추가
_OPTION_HEAD = r"(?:[①-⑩]|\(\d{1,2}\)|\d{1,2}[.)]|[A-Ea-e]\)|[ㄱ-ㅎ]\)|[가-하]\))"

# pdfplumber import
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️ pdfplumber를 사용할 수 없습니다.")

# LLM 설정
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))


class PDFPreprocessor:
    """PDF 파일 전처리 및 문제 추출 (4단계 흐름)"""

    def __init__(self):
        self.use_pdfplumber = PDFPLUMBER_AVAILABLE

    def _pre_normalize_text(self, text: str) -> str:
        """
        PDF에서 뽑힌 컬럼 텍스트를 세그먼트 분리 전에 정리:
        - 개행 정규화
        - 하이픈 줄바꿈 제거
        - 머리말(--- 페이지 n ---) 제거
        - 줄바꿈으로 찢어진 번호(세자리/두자리, dot/paren) 복원
        - 한 줄 내 숫자 사이 공백 복원
        - '1 .' → '1.' / '1 )' → '1)'
        - 원문자 보기 공백 정리
        - '정답/해설/출처' 라인 제거
        """
        if not text:
            return ""

        # 0) 개행 정규화
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 1) 하이픈 줄바꿈: "개발-\n자" → "개발자"
        text = re.sub(r"-\s*\n", "", text)

        # 2) 다중 공백 축소 (개행은 살림)
        text = re.sub(r"[ \t]+", " ", text)

        # 3) 디버그 머리말 제거 (저장용 마커)
        text = re.sub(r"(?m)^---.*?---\s*$", "", text)

        # 4) 줄바꿈으로 찢어진 문제번호 복원 (우선순위: 세자리 → 두자리)
        #    예: "1\\n0\\n0." / "1\\n\\n0\\n .", "( 1 )\\n( 0 )\\n( 0 )" 등
        # 4-1) 세자리 (dot)
        text = re.sub(
            r"(?m)^\s*([1-9])\s*(?:\n\s*)+\s*([0-9])\s*(?:\n\s*)+\s*([0-9])\s*([.)])",
            r"\1\2\3\4 ",
            text,
        )
        # 4-2) 세자리 (paren)
        text = re.sub(
            r"(?m)^\s*\(\s*([1-9])\s*\)\s*(?:\n\s*)+\(?\s*([0-9])\s*\)?\s*(?:\n\s*)+\(?\s*([0-9])\s*\)?\s*\)",
            r"(\1\2\3) ",
            text,
        )
        # 4-3) 두자리 (dot)
        text = re.sub(
            r"(?m)^\s*([1-9])\s*(?:\n\s*)+\s*([0-9])\s*([.)])",
            r"\1\2\3 ",
            text,
        )
        # 4-4) 두자리 (paren)
        text = re.sub(
            r"(?m)^\s*\(\s*([1-9])\s*\)\s*(?:\n\s*)+\(?\s*([0-9])\s*\)?\s*\)",
            r"(\1\2) ",
            text,
        )

        # 5) 한 줄 안에서 숫자 사이 공백 복원 (예: "1 0 ." → "10.", "( 1 0 )" → "(10)")
        text = re.sub(r"(?m)^\s*\(\s*(\d)\s+(\d)\s*\)\s*", r"(\1\2) ", text)
        text = re.sub(r"(?m)^\s*(\d)\s+(\d)\s*([.)])", r"\1\2\3 ", text)
        text = re.sub(r"(?m)^\s*(\d)\s+(\d)\s+(\d)\s*([.)])", r"\1\2\3\4 ", text)

        # 6) "1 ." → "1." , "1 )" → "1)"
        text = re.sub(r"(?m)^\s*(\d{1,3})\s*\.\s+", r"\1. ", text)
        text = re.sub(r"(?m)^\s*(\d{1,3})\s*\)\s+", r"\1) ", text)

        # 7) 원문자 보기(①~⑩) 뒤 과다 공백 정리
        text = re.sub(r"(?m)^([①-⑩])\s+", r"\1 ", text)

        # 8) '정답/해설/출처' 단독 라인 제거 (본문 혼입 방지)
        text = re.sub(r"(?mi)^\s*(정답|해설|정답\s*및\s*해설|출처)\s*[:：]?.*$", "", text)

        # 9) 앞뒤 공백 정리
        return text.strip()


    def _normalize_option_head(self, s: str) -> str:
        s = re.sub(r"^\s*"+_OPTION_HEAD+r"\s*", "", s)
        return s.strip()

    def _ensure_line_breaks_before_questions(self, text: str) -> str:
        """
        컬럼 텍스트 안에서 '문제 시작'만 줄머리로 강제 → 세그먼트는 문제 기준으로만 분할
        """
        if not text:
            return ""

        # 1) 1. / 12. / 3)
        text = re.sub(r"(?m)(?<!^)\s*(?=\d{1,3}\s*[.]\s)", r"\n", text)
        # 2) (1) / (12)
        text = re.sub(r"(?m)(?<!^)\s*(?=\(\d{1,3}\)\s)", r"\n", text)

        # ⚠️ ①~⑳ 에 대한 줄바꿈 주입은 삭제! (보기로 과분할되기 때문)
        # text = re.sub(r"(?m)(?<!^)\s*(?=[①-⑳]\s)", r"\n", text)
        text = re.sub(
            r"(?m)^\s*([1-9])\s*(?:\n\s*)+\s*([0-9])\s*(?:\n\s*)+\s*([0-9])\s*([.)])",
            r"\1\2\3\4 ",
            text,
        )
        # 4-3) 두자리 (dot)
        text = re.sub(
            r"(?m)^\s*([1-9])\s*(?:\n\s*)+\s*([0-9])\s*([.)])",
            r"\1\2\3 ",
            text,
        )

        text = re.sub(r"\n{3,}", "\n\n", text)
        return text



    def extract_problems_with_pdfplumber(self, file_paths: List[str]) -> List[Dict]:
        """pdfplumber를 사용하여 PDF에서 문제 추출"""
        if not self.use_pdfplumber:
            print("⚠️ pdfplumber를 사용할 수 없습니다.")
            return []
        
        all_problems: List[Dict] = []
        
        for path in file_paths:
            try:
                print(f"📖 파일 처리 중: {path}")
                problems = self._extract_problems_from_single_pdf(path)
                if problems:
                    all_problems.extend(problems)
                    print(f"✅ {path}에서 {len(problems)}개 문제 추출")
                else:
                    print(f"⚠️ {path}에서 문제 추출 실패")
            except Exception as e:
                print(f"❌ {path} 처리 실패: {e}")
                continue
        
        # 문제 번호 순으로 정렬
        sorted_problems = self._sort_problems_by_number(all_problems)
        print(f"🎯 총 {len(sorted_problems)}개 문제 추출 및 번호 순 정렬 완료")
        
        return sorted_problems

    def _extract_problems_from_single_pdf(self, pdf_path: str) -> List[Dict]:
        """단일 PDF에서 문제 추출 (4단계 흐름)"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                print(f"📄 총 {len(pdf.pages)}페이지 처리 중...")
                
                # 1단계: 좌우 텍스트 분리
                left_column_text = ""
                right_column_text = ""
                
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"🔄 페이지 {page_num} 처리 중...")
                    left_col, right_col = self._split_page_into_columns(page)
                    left_column_text += f"\n\n--- 페이지 {page_num} 왼쪽 ---\n{left_col}"
                    right_column_text += f"\n\n--- 페이지 {page_num} 오른쪽 ---\n{right_col}"
                left_column_text  = self._pre_normalize_text(left_column_text)
                right_column_text = self._pre_normalize_text(right_column_text)
                # 컬럼별 텍스트를 txt 파일로 저장 (디버깅용)
                left_column_text  = self._ensure_line_breaks_before_questions(left_column_text)
                right_column_text = self._ensure_line_breaks_before_questions(right_column_text)
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                
                with open(f"{base_name}_left_column.txt", "w", encoding="utf-8") as f:
                    f.write("=== 왼쪽 컬럼 전체 내용 ===\n\n")
                    f.write(left_column_text)
                print(f"💾 왼쪽 컬럼 저장: {base_name}_left_column.txt")
                
                with open(f"{base_name}_right_column.txt", "w", encoding="utf-8") as f:
                    f.write("=== 오른쪽 컬럼 전체 내용 ===\n\n")
                    f.write(right_column_text)
                print(f"💾 오른쪽 컬럼 저장: {base_name}_right_column.txt")
                
                # 2단계: 세그먼트 구분 (숫자. 패턴)
                print("\n🔍 세그먼트 구분 시작...")
                left_segments = self._split_text_by_problems(left_column_text)
                right_segments = self._split_text_by_problems(right_column_text)
                
                print(f"  - 왼쪽 컬럼: {len(left_segments)}개 세그먼트")
                print(f"  - 오른쪽 컬럼: {len(right_segments)}개 세그먼트")
                
                # 3단계: 문항/보기 추출
                llm = ChatOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY=REDACTED=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
                    model=OPENAI_LLM_MODEL,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )
                
                problems = []
                
                # 왼쪽 컬럼에서 문제 추출
                if left_segments:
                    print(f"\n📝 왼쪽 컬럼에서 문제 추출 중...")
                    left_problems = self._extract_problems_from_segments(left_segments, llm)
                    if left_problems:
                        problems.extend(left_problems)
                        print(f"✅ 왼쪽 컬럼에서 {len(left_problems)}개 문제 추출")
                
                # 오른쪽 컬럼에서 문제 추출
                if right_segments:
                    print(f"\n📝 오른쪽 컬럼에서 문제 추출 중...")
                    right_problems = self._extract_problems_from_segments(right_segments, llm)
                    if right_problems:
                        problems.extend(right_problems)
                        print(f"✅ 오른쪽 컬럼에서 {len(right_problems)}개 문제 추출")
                
                print(f"\n📊 추출 결과: 총 {len(problems)}개 문제")
                
                # 4단계: 문제 번호 순 정렬
                if problems:
                    problems = self._sort_problems_by_number(problems)
                    print(f"📋 정렬 완료: {len(problems)}개 문제")
                
                return problems
                
        except Exception as e:
            print(f"❌ PDF 처리 실패: {e}")
            return []

    def _split_page_into_columns(self, page) -> Tuple[str, str]:
        """1단계: 페이지를 왼쪽/오른쪽 컬럼으로 분리"""
        try:
            # 페이지의 텍스트 객체들을 가져옴
            text_objects = page.extract_words()
            if not text_objects:
                return "", ""
            
            # x 좌표 기준으로 정렬
            text_objects.sort(key=lambda x: x['x0'])
            
            # 페이지 중간점 계산
            x_coords = [obj['x0'] for obj in text_objects]
            if not x_coords:
                return "", ""
            
            mid_x = sum(x_coords) / len(x_coords)
            
            left_text = ""
            right_text = ""
            
            # y 좌표 기준으로 정렬 (위에서 아래로)
            text_objects.sort(key=lambda x: x['top'])
            
            for obj in text_objects:
                text = obj['text']
                x0 = obj['x0']
                
                if x0 < mid_x:
                    left_text += text + " "
                else:
                    right_text += text + " "
            
            return left_text.strip(), right_text.strip()
            
        except Exception as e:
            print(f"⚠️ 컬럼 분리 실패: {e}")
            return "", ""

    def _split_text_by_problems(self, text: str) -> List[str]:
        if not text:
            return []
        # 문제 시작(1., 1), (1)) 기준으로만 분할
        pattern = r"(?m)^(?=\s*(?:\d{1,3}\s*[.)]|\(\d{1,3}\)))"
        chunks = re.split(pattern, text)

        cleaned = []
        for c in chunks:
            c = c.strip()
            if len(c) < 20:
                continue
            # ✅ 유효성 검사: 줄머리 번호 + '보기' 최소 2개(줄머리 X, 본문 어디든)
            has_qnum = re.match(r"^\s*(?:\d{1,3}\s*[.)]|\(\d{1,3}\))", c)
            opt_count = len(re.findall(_OPTION_HEAD, c))  # ← ^ 제거!
            if has_qnum and opt_count >= 2:
                cleaned.append(c)
        return cleaned



    def _extract_problems_from_segments(self, segments: List[str], llm) -> List[Dict]:
        """3단계: 세그먼트에서 문항/보기 추출"""
        all_problems = []
        
        for i, segment in enumerate(segments, 1):
            if len(segment.strip()) < 50:
                continue
            
            print(f"🔄 세그먼트 {i}/{len(segments)} 처리 중...")
            
            # LLM으로 문제 추출 시도
            problem = self._extract_single_problem_with_llm(segment, llm)
            if problem:
                all_problems.append(problem)
                print(f"✅ 세그먼트 {i}에서 문제 추출 성공")
            else:
                # LLM 실패 시 정규표현식 폴백
                regex_problem = self._extract_single_problem_with_regex(segment)
                if regex_problem:
                    all_problems.append(regex_problem)
                    print(f"✅ 세그먼트 {i}에서 정규표현식으로 문제 추출")
                else:
                    print(f"⚠️ 세그먼트 {i} 추출 실패")
        
        return all_problems

    def _extract_single_problem_with_llm(self, segment: str, llm) -> Optional[Dict]:
        try:
            sys_prompt = (
                "당신은 한국어 객관식 시험 문제를 분석하는 전문가입니다. "
                "문제(번호 포함)와 보기만 추출하고 정답·해설·풀이·출처는 절대 포함하지 마세요."
            )
            user_prompt = (
                "아래 텍스트에서 객관식 문제 1개를 찾아 JSON으로 반환하세요.\n"
                '형식: {"question":"질문(번호 포함)","options":["보기1","보기2","보기3","보기4"]}\n'
                "규칙:\n"
                "1) 문제는 '1.', '1)', '(1)', '①' 등으로 시작할 수 있습니다.\n"
                "2) 보기 머리표시는 제거(①, 1), (1), 1., A), ㄱ) 등)하고 내용만 남기세요.\n"
                "3) 정답·해설·풀이·출처 등은 제거하세요.\n\n"
                f"텍스트:\n{segment[:3500]}"
            )
            resp = llm.invoke([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ])
            content = (resp.content or "").strip()
            if not content:
                return None

            json_part = self._extract_json_from_response(content)
            if not json_part:
                return None

            data = json.loads(json_part)
            if not (isinstance(data, dict) and "question" in data and "options" in data):
                return None

            q = re.sub(r"\s+", " ", str(data["question"]).strip())
            opts = [self._normalize_option_head(str(x)) for x in data.get("options", [])]
            # 보기에서 정답/해설 꼬리 제거
            opts = [re.sub(r"(?i)(정답|해설)\s*[:：].*$", "", o).strip() for o in opts]
            opts = [o for o in opts if o]
            if len(q) >= 10 and len(opts) >= 2:
                return {"question": q, "options": opts[:4]}
        except Exception as e:
            print(f"⚠️ LLM 문제 추출 실패: {e}")
        return None

    def _extract_single_problem_with_regex(self, segment: str) -> Optional[Dict]:
        """
        줄바꿈/공백으로 찢어진 번호(예: '1\\n0.' / '1 1 .')까지 인식해
        2자리/3자리 문제번호를 안정적으로 복원하고 보기(①, 1), (1), 1., A), ㄱ))를 추출한다.
        반환: {"question": "<번호>. <본문>", "options": [..최대4개..]}
        """
        try:
            s = segment.strip()
            if len(s) < 10:
                return None

            # 1) 문제 번호 + 본문 매칭
            #    - 숫자 사이에 공백/줄바꿈 허용: (?:\d\s*){1,3}
            #    - 두 가지 머리표기: "10." / "(10)"
            head_num = r"(?P<num>(?:\d\s*){1,3})"
            num_dot_pat   = rf"(?m)^\s*{head_num}\s*[.)]\s*(?P<body>.+)"
            num_paren_pat = rf"(?m)^\s*\(\s*{head_num}\s*\)\s*(?P<body>.+)"

            m = re.search(num_dot_pat, s, flags=re.DOTALL) or re.search(num_paren_pat, s, flags=re.DOTALL)
            if not m:
                return None

            raw_num = re.sub(r"\s+", "", m.group("num"))   # '1 0' / '1\n1' → '10' / '11'
            # '0', '00' 같은 비정상 번호 방지
            if not raw_num.isdigit():
                return None
            num_int = int(raw_num)
            if num_int == 0:
                return None

            body = (m.group("body") or "").strip()
            if not body:
                return None

            # 2) 보기 추출
            options: List[str] = []

            # 2-1) 줄머리 기반(가장 깔끔)
            #      다음 머리표시 전까지 비탐욕
            line_opt_pat = rf"(?m)^\s*(?:{_OPTION_HEAD})\s*(.+?)(?=\n\s*(?:{_OPTION_HEAD})|\Z)"
            for mm in re.finditer(line_opt_pat, s, flags=re.DOTALL):
                opt = self._normalize_option_head(mm.group(1))
                opt = re.sub(r"(?i)(정답|해설)\s*[:：].*$", "", opt).strip()
                if len(opt) >= 2:
                    options.append(opt)
                if len(options) >= 4:
                    break

            # 2-2) 인라인 보조(질문과 한 줄에 ①②③④가 붙는 경우)
            if len(options) < 2:
                inline_opt_pat = rf"(?:{_OPTION_HEAD})\s*(.+?)(?=(?:{_OPTION_HEAD})|\s*$)"
                for mm in re.finditer(inline_opt_pat, s, flags=re.DOTALL):
                    opt = self._normalize_option_head(mm.group(1))
                    opt = re.sub(r"(?i)(정답|해설)\s*[:：].*$", "", opt).strip()
                    if len(opt) >= 2:
                        options.append(opt)
                    if len(options) >= 4:
                        break

            # 3) 본문/보기 최소 요건 확인
            body = re.sub(r"(?m)^\s*0\s*[.)]\s*", "", body).strip()  # 앞에 잘못 들어온 '0.' 방지
            body = re.sub(r"\s+", " ", body)
            options = [re.sub(r"\s+", " ", o) for o in options]

            if len(body) < 10 or len(options) < 2:
                return None

            question = f"{num_int}. {body}"
            return {"question": question, "options": options[:4]}

        except Exception as e:
            print(f"⚠️ 정규표현식 문제 추출 실패: {e}")
            return None




    def _extract_json_from_response(self, content: str) -> Optional[str]:
        """LLM 응답에서 JSON 부분 추출"""
        try:
            # 코드블록 우선
            m = re.search(r"```(?:json)?\s*(\{[^`]*\})\s*```", content, re.DOTALL)
            if m:
                return m.group(1).strip()
            
            # 중괄호로 둘러싸인 JSON 객체 찾기
            m = re.search(r"\{[^{}]*\"question\"[^{}]*\"options\"[^{}]*\}", content, re.DOTALL)
            if m:
                return m.group(0)
            
            return None
            
        except Exception as e:
            print(f"⚠️ JSON 추출 오류: {e}")
            return None

    def _sort_problems_by_number(self, problems: List[Dict]) -> List[Dict]:
        """문제를 번호 순으로 정렬"""
        def extract_number(problem):
            question = problem.get('question', '')
            
            # 1. 질문 시작 부분에서 문제 번호 찾기 (1. 2. 3. 등)
            number_match = re.search(r'^(\d+)\s*\.', question)
            if number_match:
                return int(number_match.group(1))
            
            # 2. 질문 내용에서 문제 번호 찾기 (더 유연한 패턴)
            patterns = [
                r'(\d+)번\s*문제',
                r'(\d+)번',
                r'(\d+)\.\s*문제',
                r'문제\s*(\d+)',
                r'(\d+)\s*\.',
                r'(\d+)\s*[\.\s]'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, question)
                if match:
                    return int(match.group(1))
            
            # 3. 보기에서 문제 번호 찾기 (마지막 수단)
            options = problem.get('options', [])
            for option in options:
                # 보기에서 "15번" 같은 패턴 찾기
                option_match = re.search(r'(\d+)번', option)
                if option_match:
                    return int(option_match.group(1))
            
            print(f"⚠️ 문제 번호를 찾을 수 없음: {question[:50]}...")
            return 999999  # 번호가 없는 경우 맨 뒤로
        
        # 번호 순으로 정렬
        sorted_problems = sorted(problems, key=extract_number)
        
        print(f"📊 문제 번호 순 정렬 완료: {len(sorted_problems)}개")
        
        # 번호별 요약 출력 및 누락 확인
        number_counts = {}
        found_numbers = set()
        
        for problem in sorted_problems:
            question = problem.get('question', '')
            # 문제 번호 추출 (정렬에 사용한 것과 동일한 로직)
            number = extract_number(problem)
            if number != 999999:  # 유효한 번호인 경우만
                number_counts[number] = number_counts.get(number, 0) + 1
                found_numbers.add(number)
        
        print(f"  - 발견된 문제 번호: {sorted(found_numbers)}")
        
        # 누락된 번호 확인
        expected_numbers = set(range(1, 30))  # 1~29번
        missing_numbers = expected_numbers - found_numbers
        
        if missing_numbers:
            print(f"  ⚠️ 누락된 문제 번호: {sorted(missing_numbers)}")
        else:
            print(f"  ✅ 모든 문제 번호가 발견됨!")
        
        # 처음 5개와 마지막 5개 출력
        if len(sorted_problems) > 0:
            print(f"  📝 처음 5개:")
            for i, problem in enumerate(sorted_problems[:5], 1):
                question = problem.get('question', '')
                number_match = re.search(r'^(\d+)\s*\.', question)
                number = number_match.group(1) if number_match else "??"
                print(f"    {i}. [{number:>2}] {question[:80]}...")
            
            if len(sorted_problems) > 5:
                print(f"  📝 마지막 5개:")
                for i, problem in enumerate(sorted_problems[-5:], len(sorted_problems)-4):
                    question = problem.get('question', '')
                    number_match = re.search(r'^(\d+)\s*\.', question)
                    number = number_match.group(1) if number_match else "??"
                    print(f"    {i}. [{number:>2}] {question[:80]}...")
        
        return sorted_problems


# 편의 함수
def extract_pdf_paths(text: str) -> List[str]:
    pre = PDFPreprocessor()
    return pre.extract_pdf_paths(text)

def extract_problem_range(text: str) -> Optional[Dict]:
    pre = PDFPreprocessor()
    return pre.extract_problem_range(text)

def determine_problem_source(text: str) -> Optional[str]:
    pre = PDFPreprocessor()
    return pre.determine_problem_source(text)

# 간단한 구현
PDF_PATH_PATTERNS = [
    r"([^\s]+\.pdf)",
    r"([C-Z]:[\\\/][^\\\/\s]*\.pdf)",
    r"([\.\/][^\\\/\s]*\.pdf)",
]

def _findall(pattern, text):
    return re.findall(pattern, text, re.IGNORECASE)

def _extract_pdf_paths_impl(self, text: str) -> List[str]:
    paths = []
    for p in PDF_PATH_PATTERNS:
        paths.extend(_findall(p, text))
    return list(set(paths))

def _extract_problem_range_impl(self, text: str) -> Optional[Dict]:
    m = re.search(r'(\d+)번만', text)
    if m:
        return {"type": "single", "number": int(m.group(1))}
    m = re.search(r'(\d+)번\s*풀', text)
    if m:
        return {"type": "single", "number": int(m.group(1))}
    
    m = re.search(r'(\d+)\s*[-~]\s*(\d+)번', text)
    if m:
        return {"type": "range", "start": int(m.group(1)), "end": int(m.group(2))}
    
    return None

def _determine_problem_source_impl(self, text: str) -> Optional[str]:
    tl = text.lower()
    if any(k in tl for k in ['pdf', '파일', '문서']):
        return "pdf_extracted"
    if any(k in tl for k in ['기존', 'shared', '저장된', '이전']):
        return "shared"
    if _extract_pdf_paths_impl(self, text):
        return "pdf_extracted"
    return None

# 클래스 메소드로 바인딩
PDFPreprocessor.extract_pdf_paths = _extract_pdf_paths_impl
PDFPreprocessor.extract_problem_range = _extract_problem_range_impl
PDFPreprocessor.determine_problem_source = _determine_problem_source_impl
