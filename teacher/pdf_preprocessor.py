# -*- coding: utf-8 -*-
"""
PDF 전처리 관련 함수들을 모아놓은 모듈 (개선판)
teacher_graph.py에서 PDF 관련 로직을 분리하여 가독성을 높임
- 번호 상한 제거(또는 상향)로 100문항 이상 대응
- 문제 경계(## 31. 등) 보강
- 과도한 블록 길이 컷으로 LLM JSON 잘림 완화
- 폴백 트리거 강화(기대치 대비 부족 시 2단 전체→일괄 LLM, 필요 시 페이지 배치 폴백)
- Docling .text 분기 제거
"""

import os
import re
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from docling.document_converter import DocumentConverter
from langchain_openai import ChatOpenAI

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ------------ 유틸: 기대 문제 수 추정 ------------
_HEADER_NUM_PAT = re.compile(r'^\s*(?:##\s*)?(?:문제\s*)?(\d+)\s*\.', re.UNICODE)


def _estimate_expected_count_from_text(text: str, upper: int = 100) -> int:
    nums = []
    for ln in text.split("\n"):
        m = _HEADER_NUM_PAT.match(ln.strip())
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= upper:
                    nums.append(n)
            except Exception:
                pass
    return max(nums) if nums else 0


class PDFPreprocessor:
    """PDF 파일 전처리 및 문제 추출 클래스"""

    def __init__(self):
        # 환경변수 설정으로 권한 문제 해결
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HOME"] = os.getenv("HF_HOME", "C:\\temp\\huggingface_cache")

        # cv2 setNumThreads 문제 해결
        try:
            import cv2  # noqa: F401

            if not hasattr(cv2, "setNumThreads"):
                # setNumThreads가 없으면 더미 함수 추가
                cv2.setNumThreads = lambda x: None  # type: ignore[attr-defined]
        except ImportError:
            pass

    # -------------------- 편의 추출 --------------------

    def extract_pdf_paths(self, text: str) -> List[str]:
        """PDF 파일 경로 추출"""
        pdf_patterns = [
            r"([^\s]+\.pdf)",  # 기본 .pdf 파일 경로
            r"([C-Z]:[\\\/][^\\\/\s]*\.pdf)",  # Windows 절대 경로
            r"([\.\/][^\\\/\s]*\.pdf)",  # 상대 경로
        ]
        pdf_paths = []
        for pattern in pdf_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            pdf_paths.extend(matches)
        return list(set(pdf_paths))  # 중복 제거

    def extract_problem_range(self, text: str) -> Optional[Dict]:
        """문제 번호 범위 추출"""
        patterns = [
            r"(\d+)번만",
            r"(\d+)번\s*풀",
            r"(\d+)\s*[-~]\s*(\d+)번",
            r"(\d+)번부터\s*(\d+)번",
            r"(\d+(?:\s*,\s*\d+)*)번",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    if "," in groups[0]:
                        numbers = [int(x.strip()) for x in groups[0].split(",")]
                        return {"type": "specific", "numbers": numbers}
                    else:
                        return {"type": "single", "number": int(groups[0])}
                elif len(groups) == 2:
                    start, end = int(groups[0]), int(groups[1])
                    return {"type": "range", "start": start, "end": end}
        return None

    def determine_problem_source(self, text: str) -> Optional[str]:
        """문제 소스 결정"""
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ["pdf", "파일", "문서"]):
            return "pdf_extracted"
        elif any(keyword in text_lower for keyword in ["기존", "shared", "저장된", "이전"]):
            return "shared"
        if self.extract_pdf_paths(text):
            return "pdf_extracted"
        return None

    # -------------------- 핵심 파이프라인 --------------------

    def extract_problems_from_pdf(self, file_paths: List[str]) -> List[Dict]:
        """PDF 파일에서 문제 추출 (Docling 사용) + 폴백 강화"""
        try:
            print("🔧 DocumentConverter 초기화 중...")
            converter = DocumentConverter()
            try:
                converter.config.image_processing = False
                print("✅ 이미지 처리 비활성화 설정")
            except Exception:
                print("⚠️ 이미지 처리 설정 변경 불가")
            try:
                converter.config.text_extraction_priority = "text"
                print("✅ 텍스트 추출 우선순위 설정")
            except Exception:
                print("⚠️ 텍스트 추출 우선순위 설정 불가")
            print("✅ DocumentConverter 초기화 완료")
        except Exception as e:
            print(f"❌ DocumentConverter 초기화 실패: {e}")
            import traceback

            traceback.print_exc()
            return []

        # LLM 설정
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY=REDACTED=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            model=OPENAI_LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        all_problems: List[Dict] = []

        for path in file_paths:
            try:
                print(f"📖 파일 처리 중: {path}")
                doc_result = converter.convert(path)

                # 방법 1: 마크다운 추출
                raw_text = doc_result.document.export_to_markdown()
                print(f"📝 [방법1] 마크다운 길이: {len(raw_text)}")

                # 방법 2/3 제거: DoclingDocument에 .text 없음 / pages 요소가 int되는 경우 있음
                # -> 안정성 위해 제거(로그 소음 방지)

                # 방법 4: 마크다운 HTML 주석/태그 제거
                if raw_text.startswith("<!--"):
                    try:
                        cleaned = re.sub(r"<!--.*?-->", "", raw_text, flags=re.DOTALL)
                        cleaned = re.sub(r"<[^>]+>", "", cleaned)
                        cleaned = re.sub(r"^\s*-\s*", "", cleaned, flags=re.MULTILINE)
                        cleaned = "\n".join(
                            line for line in cleaned.split("\n") if line.strip()
                        )
                        if cleaned and len(cleaned) > 50:
                            raw_text = cleaned
                            print("✅ HTML 태그/주석 제거 채택")
                    except Exception as e:
                        print(f"⚠️ HTML 정리 실패: {e}")

                if not raw_text.strip() or raw_text.startswith("<!--"):
                    print("❌ Docling으로 텍스트 추출 실패")
                    continue

                raw_text = self.normalize_docling_markdown(raw_text)
                print(f"📊 최종 텍스트 길이: {len(raw_text)}")

                # 기대 문제 수(헤더 기반) 추정
                expected = _estimate_expected_count_from_text(raw_text)
                if expected:
                    print(f"📈 헤더 기반 기대 문제 수(추정): 약 {expected}문항")

                # 1) 기본 분할 → LLM 블록 파싱
                blocks = self._process_pdf_text(raw_text, path)
                print(f"📝 1차 분할 블록 수: {len(blocks)}")

                successful_parses = 0
                local_problems: List[Dict] = []

                for i, block in enumerate(blocks):
                    blk = block.strip()
                    if len(blk) < 20:
                        continue
                    try:
                        # 과도한 블록 길이 컷(LLM 안전 가드)
                        if len(blk) > 4000:
                            blk = self._smart_truncate_block(blk)
                        problem = self._parse_block_with_llm(blk, llm)
                        if problem:
                            local_problems.append(problem)
                            successful_parses += 1
                    except Exception as e:
                        print(f"⚠️ 블록 {i+1} 파싱 실패: {e}")

                # --- 폴백 조건 판단 강화 ---
                need_fallback = False
                if successful_parses == 0:
                    need_fallback = True
                elif len(blocks) > 0 and successful_parses / max(1, len(blocks)) < 0.2:
                    need_fallback = True
                # 기대치 70% 미만이면 폴백
                if expected and len(local_problems) < int(0.7 * expected):
                    need_fallback = True

                if need_fallback:
                    print("🔁 폴백 발동: 2단 재정렬 후 전체 텍스트를 LLM으로 일괄 추출 (실패 시 페이지 배치)")

                    reordered_text = self._reorder_two_columns_with_pdfminer(path)
                    if not reordered_text.strip():
                        print("⚠️ 2단 재정렬 텍스트 없음 → 원본으로 일괄 추출 시도")
                        reordered_text = raw_text
                    reordered_text = self.normalize_docling_markdown(reordered_text)

                    # (1) 전체 일괄
                    batch = self._parse_whole_text_with_llm(reordered_text, llm)

                    # (2) 실패 시 페이지 배치
                    if not batch:
                        print("🔁 폴백 2단계: 페이지 배치 LLM 추출 시도")
                        try:
                            batch = self._parse_by_pages_with_llm(doc_result, llm)
                        except Exception as e:
                            print(f"⚠️ 페이지 배치 폴백 실패: {e}")

                    if batch:
                        print(f"✅ 폴백 추출 성공: {len(batch)}개")
                        local_problems = batch
                    else:
                        print("❌ 폴백 추출 실패 → 기존 부분 성과 유지")
                else:
                    print("✅ 1차 블록 파싱 결과가 충분하여 폴백 생략")

                all_problems.extend(local_problems)
                print(f"📊 누적 문제 수: {len(all_problems)}")

            except Exception as e:
                print(f"❌ 파일 {path} 처리 실패: {e}")
                continue

        print(f"🎯 총 {len(all_problems)}개 문제 추출 완료")
        return all_problems

    # -------------------- 텍스트 전처리/분할 --------------------

    def _process_pdf_text(self, raw_text: str, pdf_path: str) -> List[str]:
        """PDF 텍스트를 1단/2단 구분하여 처리"""
        print("🔍 [레이아웃 분석] 1단/2단 구조 파악 중...")

        blocks = self._split_problem_blocks(raw_text)

        # 1단 파싱 결과가 부족하면 2단 구조로 재시도
        if len(blocks) <= 2:
            print("⚠️ 1단 파싱 결과 부족 - 2단 구조로 재시도")
            try:
                reordered_text = self._reorder_two_columns_with_pdfminer(pdf_path)
                reordered_text = self.normalize_docling_markdown(reordered_text)
                blocks = self._split_problem_blocks(reordered_text)
                print(f"🔄 2단 재정렬 후: {len(blocks)}개 블록")

                if len(blocks) <= 2:
                    print("⚠️ 2단 파싱도 부족 - 숫자 헤더 폴백 사용")
                    blocks = self._split_problem_blocks_without_keyword(reordered_text)
                    print(f"🔄 폴백 후: {len(blocks)}개 블록")
            except Exception as e:
                print(f"⚠️ 2단 처리 실패: {e}")
                blocks = self._split_problem_blocks_without_keyword(raw_text)

        return blocks

    def _reorder_two_columns_with_pdfminer(self, pdf_path: str) -> str:
        """PDFMiner를 사용하여 2단 PDF를 1단으로 재정렬"""
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer

            print("🔄 [2단 재정렬] PDFMiner로 좌우 컬럼 재정렬 중...")

            pages_text = []
            for page_layout in extract_pages(pdf_path):
                left, right = [], []
                xs = []
                for el in page_layout:
                    if isinstance(el, LTTextContainer):
                        xs.append(el.bbox[0])

                if not xs:
                    continue

                sorted_xs = sorted(xs)
                mid = sorted_xs[len(sorted_xs) // 2]

                for el in page_layout:
                    if isinstance(el, LTTextContainer):
                        (x0, y0, x1, y1) = el.bbox
                        text = el.get_text().strip()
                        if text:
                            (left if x0 < mid else right).append((y1, text))

                left.sort(key=lambda t: -t[0])
                right.sort(key=lambda t: -t[0])

                page_text = "".join(t for _, t in left) + "\n" + "".join(t for _, t in right)
                pages_text.append(page_text)

            result = "\n\n".join(pages_text)
            print(f"✅ [2단 재정렬 완료] 총 {len(pages_text)}페이지 처리")
            return result

        except ImportError:
            print("⚠️ PDFMiner가 설치되지 않음 - 2단 재정렬 불가")
            return ""
        except Exception as e:
            print(f"⚠️ 2단 재정렬 실패: {e}")
            return ""

    def _split_problem_blocks_without_keyword(self, text: str) -> List[str]:
        """문제 키워드가 없는 시험지에서 번호(1., 2., …)만으로 문항 단위를 분할"""
        print("🔄 [폴백 파싱] 문제 키워드 없이 번호만으로 분할 시도")

        text = self.normalize_docling_markdown(text)
        lines = text.split("\n")
        n = len(lines)

        _QHEAD_CAND = re.compile(r"(?m)^\s*(\d{1,3})[.)]\s+\S")
        candidates = []

        for i, ln in enumerate(lines):
            m = _QHEAD_CAND.match(ln or "")
            if m:
                num = int(m.group(1))
                if not re.match(r"^\s*\d+\)\s*", ln):
                    if len(ln.strip()) > 10:
                        candidates.append((i, num))
                        print(f"🔍 [폴백] 라인 {i}: '{ln[:50]}...' → 후보 번호 {num}")
                else:
                    print(f"🔍 [폴백] 라인 {i}: '{ln[:50]}...' → 보기 번호로 판단하여 제외")

        print(f"🔍 [폴백] 총 후보 수: {len(candidates)}")

        headers = []
        prev_num = 0
        last_header_idx = -9999

        for i, num in candidates:
            if num == prev_num + 1:
                headers.append(i)
                prev_num = num
                last_header_idx = i
                print(f"✅ [폴백] 라인 {i}: 번호 {num} - 순차 증가로 헤더 선택")
                continue

            if num == 1:
                window = "\n".join(lines[max(0, i - 3) : i + 1])
                if (i - last_header_idx) >= 8 or re.search(r"(Ⅰ|Ⅱ|III|과목|파트|SECTION)", window):
                    headers.append(i)
                    prev_num = 1
                    last_header_idx = i
                    print(f"✅ [폴백] 라인 {i}: 번호 {num} - 섹션 리셋으로 헤더 선택")
                else:
                    print(f"❌ [폴백] 라인 {i}: 번호 {num} - 섹션 리셋 조건 불충족 (거리: {i - last_header_idx})")

        if not headers:
            print("❌ [폴백] 헤더가 하나도 선택되지 않음 - 단순 패턴으로 재시도")
            simple_pattern = re.compile(r"(?m)^\s*(\d{1,3})\.\s+")
            for i, ln in enumerate(lines):
                if simple_pattern.match(ln or "") and not re.match(r"^\s*\d+\)\s*", ln):
                    headers.append(i)
                    print(f"📌 [폴백] 라인 {i}: '{ln[:30]}...' → 헤더 추가")

            if not headers:
                print("❌ [폴백 실패] 전체를 1개 블록으로 처리")
                return [text] if text.strip() else []

        headers.append(n)
        blocks = []
        for a, b in zip(headers[:-1], headers[1:]):
            blk = "\n".join(lines[a:b]).strip()
            if blk:
                blocks.append(blk)
                print(f"📦 [폴백] 블록 {len(blocks)}: 라인 {a}-{b-1} ({len(blk)}자)")

        print(f"🎯 [폴백] 최종 블록 수: {len(blocks)}")
        return blocks

    def _split_problem_blocks(self, raw_text: str) -> List[str]:
        """텍스트를 문제 블록으로 분할 (정교한 개선된 로직)"""
        print("🔍 [구조 분석] 정교한 문제 헤더 기반으로 파싱 방식 결정")

        lines = raw_text.split("\n")

        # 정교한 문제 헤더 패턴들 (우선순위 상단에 범용형 추가)
        problem_header_patterns = [
            r"^\s*(?:#+\s*)?문제?\s*(\d+)\s*\.\s*",  # "## 31.", "문제 31.", "31."
            r"^\s*#+\s*문제\s*(\d+)\s*\.\s*",
            r"^\s*문제\s*(\d+)\s*\.\s*",
            r"^\s*(\d+)\s*\.\s*[^가-힣]*[가-힣]",
            r"^\s*(\d+)\s*\.\s*\S",
            r"^\s*Q\s*(\d+)\s*\.\s*",
            r"^\s*\[(\d+)\]\s*",
            r"^\s*#+\s*.*?(\d+)\s*\.\s*[가-힣]",
            r"^\s*#+\s*[^가-힣]*(\d+)\.\s*[가-힣]",
            r"^\s*-\s*[^가-힣]*(\d+)\.\s*[가-힣]",
            r"^\s*[^가-힣]*(\d+)\.\s*[가-힣]",
        ]

        option_patterns = [
            r"^\s*(\d+)\.\s*\1\.\s*",
            r"^\s*(\d+)\s*[)]\s*",
            r"^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*",
            r"^\s*[가-하]\s*[)]\s*",
            r"^\s*[A-E]\s*[)]\s*",
            r"^\s*-\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*",
            r"^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*[가-힣]",
        ]

        problem_headers = []
        seen_numbers = set()

        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue

            # 보기 번호 먼저 걸러내기
            if any(re.match(p, s) for p in option_patterns):
                continue

            for pattern in problem_header_patterns:
                match = re.match(pattern, s)
                if match:
                    try:
                        problem_num = int(match.group(1))
                    except Exception:
                        continue

                    # 번호 상한 완화 (또는 제거). 안전하게 300으로 둠.
                    if 1 <= problem_num <= 100 and problem_num not in seen_numbers:
                        problem_headers.append((i, problem_num, s))
                        seen_numbers.add(problem_num)
                        print(
                            f"✅ [문제 헤더 발견] 라인 {i+1}: '{s[:80]}...' (문제 {problem_num}번)"
                        )
                    break

        if not problem_headers:
            print("⚠️ 문제 헤더를 찾을 수 없음 - 전체를 1개 블록으로 처리")
            return [raw_text] if raw_text.strip() else []

        problem_headers.sort(key=lambda x: x[1])

        # 누락 번호 (상한은 현재 문서에서 최대 번호 기준)
        found_numbers = {h[1] for h in problem_headers}
        max_num = max(found_numbers) if found_numbers else 0
        expected_range = set(range(1, max_num + 1))
        missing_numbers = expected_range - found_numbers
        if missing_numbers:
            print(f"⚠️ 누락된 문제 번호: {sorted(missing_numbers)}")

        # 문제 블록 생성
        blocks = []
        for idx, (header_idx, problem_num, header_text) in enumerate(problem_headers):
            start_line = header_idx
            if idx + 1 < len(problem_headers):
                next_header_idx = problem_headers[idx + 1][0]
                end_line = next_header_idx
            else:
                end_line = len(lines)

            problem_text = "\n".join(lines[start_line:end_line]).strip()

            # 복합 문제 분리 로직 유지
            sub_blocks = self._split_composite_problem(problem_text, problem_num)
            if len(sub_blocks) > 1:
                print(f"🔧 문제 {problem_num}번 복합 문제 분리: {len(sub_blocks)}개 블록")
                blocks.extend(sub_blocks)
            else:
                blocks.append(problem_text)

            print(
                f"📦 문제 {problem_num}번: 라인 {start_line+1}-{end_line} ({len(problem_text)}자)"
            )
            print(f"   헤더: '{header_text[:50]}...'")

        print(f"✅ 총 {len(blocks)}개 문제 블록 생성 완료")

        # 누락 문제 추가 탐색
        if missing_numbers and len(blocks) < max_num:
            print(f"🔄 누락된 문제 {len(missing_numbers)}개 추가 시도 중...")
            add = self._find_missing_problems(lines, missing_numbers)
            if add:
                blocks.extend(add)
                print(f"✅ 추가 문제 {len(add)}개 발견 - 총 {len(blocks)}개")

        return blocks

    # -------------------- LLM 파서 --------------------

    def _parse_block_with_llm(self, block_text: str, llm) -> Optional[Dict]:
        """LLM으로 블록을 문제 형태로 파싱"""
        cleaned_text = self._clean_problem_block(block_text)

        sys_prompt = (
            "너는 시험 문제 PDF에서 텍스트를 구조화하는 도우미다. "
            "문제 질문과 보기를 구분해서 question과 options 배열로 출력한다. "
            "options는 보기 항목만 포함하고, 설명/해설/정답 등은 포함하지 않는다. "
            "응답은 반드시 JSON 형태로만 출력한다. 다른 문장이나 코드는 절대 포함하지 말 것."
        )

        user_prompt = (
            "다음 텍스트에서 문항을 최대한 그대로, 정확히 추출해 JSON으로 만들어줘.\n"
            '요구 스키마: {"question":"...","options":["...","..."]}\n'
            "규칙:\n"
            "- 문제 질문에서 번호(예: '문제 1.' 등)와 불필요한 머리글은 제거.\n"
            "- 옵션은 실제 보기 개수에 맞춤.\n"
            "- 보기 번호(①, ②, ③, ④ 등)는 제거하고 내용만 추출.\n"
            "- 문제가 명확하지 않으면 null을 반환.\n"
            "- 응답은 반드시 JSON만 출력하고 다른 텍스트는 포함하지 말 것.\n"
            f"텍스트:\n{cleaned_text[:1500]}"
        )

        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            content = response.content.strip()
            print(f"🔍 LLM 응답 원본: {content[:200]}...")

            json_content = self._extract_json_from_response(content)
            if not json_content:
                print("❌ JSON을 추출할 수 없음")
                return None

            try:
                data = json.loads(json_content)
            except json.JSONDecodeError:
                fixed = self._fix_json_format(json_content)
                try:
                    data = json.loads(fixed)
                    print("✅ JSON 수정 후 파싱 성공")
                except json.JSONDecodeError as e2:
                    print(f"❌ JSON 수정 후에도 파싱 실패: {e2}")
                    return None

            if isinstance(data, dict) and "question" in data and "options" in data:
                if data["question"].strip() and isinstance(data["options"], list) and len(data["options"]) > 0:
                    print("✅ 문제 파싱 성공")
                    return data
                else:
                    print("❌ 유효하지 않은 데이터 구조")
            else:
                print("❌ 필수 필드 누락")

        except Exception as e:
            print(f"⚠️ LLM 파싱 실패: {e}")

        return None

    def _parse_whole_text_with_llm(self, full_text: str, llm) -> Optional[List[Dict]]:
        """2단 재정렬(또는 원문) 전체 텍스트를 한 번에 LLM에 넣어 일괄 추출"""
        cleaned = self.normalize_docling_markdown(full_text)

        sys_prompt = (
            "너는 시험지에서 문항을 구조화해 추출하는 도우미다. "
            "문항의 질문과 보기만을 뽑아내고, 해설/정답/출처 등은 제외한다. "
            "반드시 JSON 배열로만 응답한다."
        )
        user_prompt = (
            "다음 전체 텍스트에서 각 문항을 최대한 누락 없이 추출해줘.\n"
            '요구 스키마(배열): [{"question":"...","options":["...","..."]}, ...]\n'
            "규칙:\n"
            "- 질문에 붙은 번호(예: '1.', '문제 1.') 등 머리글은 제거.\n"
            "- 보기 번호 표기(①, ②, 1), 가) 등)는 제거하고 내용만.\n"
            "- 보기 개수는 원문에 맞춤.\n"
            "- 해설/정답/풀이/설명 등은 절대 포함하지 말 것.\n"
            "- JSON 외 다른 텍스트 금지.\n"
            f"텍스트:\n{cleaned[:40000]}"
        )

        try:
            resp = llm.invoke(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            content = resp.content.strip()
            print(f"🔍 [일괄] LLM 응답 미리보기: {content[:200]}...")

            json_part = self._extract_json_from_response(content)
            if not json_part:
                print("❌ [일괄] JSON 추출 실패")
                return None

            try:
                data = json.loads(json_part)
            except json.JSONDecodeError:
                fixed = self._fix_json_format(json_part)
                try:
                    data = json.loads(fixed)
                except json.JSONDecodeError as e2:
                    print(f"❌ [일괄] JSON 파싱 실패: {e2}")
                    return None

            if isinstance(data, list):
                cleaned_list: List[Dict] = []
                for idx, item in enumerate(data, 1):
                    if (
                        isinstance(item, dict)
                        and isinstance(item.get("question", ""), str)
                        and item.get("question", "").strip()
                        and isinstance(item.get("options", []), list)
                        and len(item.get("options", [])) > 0
                    ):
                        cleaned_list.append(
                            {
                                "question": item["question"].strip(),
                                "options": [str(opt).strip() for opt in item["options"] if str(opt).strip()],
                            }
                        )
                    else:
                        print(f"⚠️ [일괄] 항목 {idx} 무시(스키마 불일치)")
                return cleaned_list if cleaned_list else None

            print("❌ [일괄] 최종 구조가 배열이 아님")
            return None

        except Exception as e:
            print(f"⚠️ [일괄] LLM 호출 실패: {e}")
            return None

    def _parse_by_pages_with_llm(self, doc_result, llm, max_pages: int = 9999) -> Optional[List[Dict]]:
        """페이지 단위로 나눠 일괄 추출 (폴백 2단계)"""
        items: List[Dict] = []
        pages = getattr(doc_result.document, "pages", []) or []
        for pidx, page in enumerate(pages[:max_pages], start=1):
            # 페이지 오브젝트에 따라 text/export_to_markdown 접근이 다를 수 있음
            page_text = ""
            if hasattr(page, "text") and isinstance(page.text, str):
                page_text = page.text
            elif hasattr(page, "export_to_markdown"):
                try:
                    page_text = page.export_to_markdown()
                except Exception:
                    page_text = ""
            if not page_text:
                continue

            chunk = self.normalize_docling_markdown(str(page_text))
            # 길면 분할 호출
            chunks = [chunk[i : i + 8000] for i in range(0, len(chunk), 8000)]
            for c in chunks:
                batch = self._parse_whole_text_with_llm(c, llm)
                if batch:
                    items.extend(batch)

        return items if items else None

    # -------------------- JSON 후처리/정리 --------------------

    def _extract_json_from_response(self, content: str) -> Optional[str]:
        """LLM 응답에서 JSON 부분만 정확히 추출"""
        code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        json_match = re.search(r"\{.*?\}", content, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            if self._is_valid_json_structure(json_text):
                return json_text

        array_match = re.search(r"\[.*?\]", content, re.DOTALL)
        if array_match:
            array_text = array_match.group(0)
            if self._is_valid_json_structure(array_text):
                return array_text

        if '"question"' in content and '"options"' in content:
            return self._construct_json_from_parts(content)

        return None

    def _is_valid_json_structure(self, text: str) -> bool:
        """JSON 구조가 유효한지 기본 검사"""
        brace_count = text.count("{") - text.count("}")
        bracket_count = text.count("[") - text.count("]")
        if brace_count != 0 or bracket_count != 0:
            return False
        if not (text.startswith("{") and text.endswith("}")) and not (
            text.startswith("[") and text.endswith("]")
        ):
            return False
        return True

    def _construct_json_from_parts(self, content: str) -> Optional[str]:
        """LLM 응답에서 question과 options 부분을 찾아 JSON 구성"""
        try:
            question_match = re.search(r'"question"\s*:\s*"([^"]*)"', content)
            if not question_match:
                return None
            question = question_match.group(1)

            options_match = re.search(r'"options"\s*:\s*\[(.*?)\]', content, re.DOTALL)
            if not options_match:
                return None
            options_text = options_match.group(1)

            options = []
            option_matches = re.findall(r'"([^"]*)"', options_text)
            for opt in option_matches:
                if opt.strip():
                    options.append(opt.strip())

            if not options:
                return None

            json_data = {"question": question, "options": options}
            return json.dumps(json_data, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ JSON 구성 실패: {e}")
            return None

    # -------------------- 블록 전처리/보조 --------------------

    def _clean_problem_block(self, block_text: str) -> str:
        """문제 블록 텍스트를 정리하여 파싱에 적합하게 만듦"""
        lines = block_text.split("\n")
        cleaned_lines = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            s = re.sub(r"<!--.*?-->", "", s)
            s = re.sub(r"<[^>]+>", "", s)
            s = re.sub(r"^\d+\.\s*", "", s)  # 번호 제거
            s = re.sub(r"^#+\s*", "", s)  # MD 헤더 제거
            if s:
                cleaned_lines.append(s)
        return "\n".join(cleaned_lines)

    def _fix_json_format(self, content: str) -> str:
        """JSON 형식을 수정하여 파싱 가능하게 만듦 (강화)"""
        print(f"🔧 JSON 수정 전: {content[:100]}...")
        content = re.sub(r",\s*}", "}", content)
        content = re.sub(r",\s*]", "]", content)
        content = re.sub(r"\n|\r", " ", content)
        content = re.sub(r"\s+", " ", content)

        # 중괄호 및 대괄호 범위 정리
        if content.count("{") > 0 and content.count("}") > 0:
            start = content.find("{")
            end = content.rfind("}") + 1
            content = content[start:end]
        if content.count("[") > 0 and content.count("]") > 0:
            start = content.find("[")
            end = content.rfind("]") + 1
            content = content[start:end]

        print(f"🔧 JSON 수정 후: {content[:100]}...")
        return content

    def _split_composite_problem(self, block_text: str, problem_num: int) -> List[str]:
        """복합 문제를 개별 문제로 분리"""
        lines = block_text.split("\n")
        sub_blocks = []
        current_block = []
        current_problem_num = problem_num

        for line in lines:
            s = line.strip()
            m = re.match(r"^(\d+)\.\s*", s)
            if m:
                new_num = int(m.group(1))
                if current_block:
                    sub_text = "\n".join(current_block).strip()
                    if sub_text and len(sub_text) > 20:
                        sub_blocks.append(sub_text)
                        print(
                            f"   📝 하위 블록 {len(sub_blocks)}: 문제 {current_problem_num}번 관련 ({len(sub_text)}자)"
                        )
                current_block = [line]
                current_problem_num = new_num
            else:
                current_block.append(line)

        if current_block:
            sub_text = "\n".join(current_block).strip()
            if sub_text and len(sub_text) > 20:
                sub_blocks.append(sub_text)
                print(
                    f"   📝 하위 블록 {len(sub_blocks)}: 문제 {current_problem_num}번 관련 ({len(sub_text)}자)"
                )

        if len(sub_blocks) <= 1:
            return [block_text]
        print(f"🔧 문제 {problem_num}번 복합 문제 분리: {len(sub_blocks)}개 블록")
        return sub_blocks

    def _find_missing_problems(self, lines: List[str], missing_numbers: set) -> List[str]:
        """누락된 문제들을 찾아서 추가 블록 생성 (경계 보강)"""
        additional_blocks = []
        for missing_num in sorted(missing_numbers):
            print(f"🔍 누락된 문제 {missing_num}번 검색 중...")
            for i, line in enumerate(lines):
                if str(missing_num) in line and any(
                    keyword in line for keyword in ["문제", "설명", "것은", "?", "다음"]
                ):
                    print(f"   ✅ 문제 {missing_num}번 후보 발견 - 라인 {i+1}: '{line[:50]}...'")
                    start_line = max(0, i - 1)
                    end_line = min(len(lines), i + 10)
                    for j in range(i + 1, min(len(lines), i + 40)):
                        # 경계: "## 31." 형태 포함
                        if re.search(r"^(?:\s*##\s*)?\s*\d+\.", lines[j]):
                            end_line = j
                            break
                    block_text = "\n".join(lines[start_line:end_line]).strip()
                    if block_text and len(block_text) > 20:
                        additional_blocks.append(block_text)
                        print(
                            f"   📦 추가 블록 생성: 라인 {start_line+1}-{end_line} ({len(block_text)}자)"
                        )
                        break
        return additional_blocks

    def _smart_truncate_block(self, text: str, body_limit: int = 4000) -> str:
        """너무 긴 블록을 보기 이후 적당한 빈줄에서 컷하여 LLM 안정성 확보"""
        if len(text) <= body_limit:
            return text
        lines = text.split("\n")
        # 보기 패턴 이후 첫 빈 줄에서 컷
        option_pat = re.compile(
            r"^\s*(?:-|\*|•)?\s*(?:①|②|③|④|⑤|\d+\)|[가-하]\)|[A-E]\))"
        )
        cut_idx = None
        for k in range(len(lines)):
            if option_pat.match(lines[k].strip()):
                for m in range(k + 1, min(k + 80, len(lines))):
                    if not lines[m].strip():
                        cut_idx = m
                        break
                break
        if cut_idx:
            return "\n".join(lines[:cut_idx]).strip()
        return "\n".join(lines[: max(1, body_limit // 80)]).strip()

    # -------------------- 정규화 --------------------

    def normalize_docling_markdown(self, md: str) -> str:
        """Docling 마크다운 정규화"""
        s = md
        s = re.sub(r"(?m)^\s*(\d+)\.\s*\1\.\s*", r"\1. ", s)  # '1. 1.' -> '1.'
        s = re.sub(r"(?m)^\s*(\d+)\s*\.\s*", r"\1. ", s)  # '1 . ' -> '1. '
        s = re.sub(r"[ \t]+", " ", s).replace("\r", "")
        return s.strip()


# -------------------- 모듈 레벨 편의 함수 --------------------

def extract_pdf_paths(text: str) -> List[str]:
    preprocessor = PDFPreprocessor()
    return preprocessor.extract_pdf_paths(text)


def extract_problem_range(text: str) -> Optional[Dict]:
    preprocessor = PDFPreprocessor()
    return preprocessor.extract_problem_range(text)


def determine_problem_source(text: str) -> Optional[str]:
    preprocessor = PDFPreprocessor()
    return preprocessor.determine_problem_source(text)
