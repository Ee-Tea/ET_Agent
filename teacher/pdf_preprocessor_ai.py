# teacher/pdf_preprocessor_ai.py
# -*- coding: utf-8 -*-
"""
PDF 전처리 & 문제 추출 (열 단위 LLM 파이프라인, 정리판)

핵심 아이디어
- PDFMiner로 페이지마다 좌/우 열 텍스트 분리
- 각 열을 통으로(안전 청크) LLM에 넘겨 문항 배열(JSON) 추출
- 헤더/번호 기반 분할, 복잡한 헤더 추정 로직 제거
"""

import os
import re, traceback
import json
import hashlib
import html
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from docling.document_converter import DocumentConverter
from langchain_openai import ChatOpenAI

# ── LLM 설정 ──────────────────────────────────────────────────────────────────
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))  # 안정성↑
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))


class PDFPreprocessor:
    """PDF 파일 전처리 및 문제 추출 (열 단위)"""

    def __init__(self):
        # 권한/캐시 관련 (Windows 환경 대응)
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HOME"] = os.getenv("HF_HOME", "C:\\temp\\huggingface_cache")

        # 일부 OpenCV 빌드에서 setNumThreads 미존재 이슈 회피
        try:
            import cv2  # noqa: F401
            if not hasattr(cv2, "setNumThreads"):
                cv2.setNumThreads = lambda x: None  # type: ignore[attr-defined]
        except ImportError:
            pass
        

        
    # ====== JSON 파싱 유틸: PDFPreprocessor 내부 메서드로 추가 ======

    def _strip_code_fences(self, text: str) -> str:
        # ```json ... ``` 같은 코드펜스 제거
        return re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()

    def _repair_brackets(self, text: str) -> str:
        # 가장 바깥의 대괄호 구간만 보존
        if "[" in text and "]" in text:
            s = text.find("["); e = text.rfind("]")
            return text[s:e+1]
        return text

    def _find_largest_json_array(self, text: str) -> str:
        """응답 본문에서 가장 큰 JSON 배열 구간을 찾아 반환(문항/보기 키 포함 우선)."""
        text = self._strip_code_fences(text)
        # 괄호 스팬 수집
        spans, stack = [], []
        for i, ch in enumerate(text):
            if ch == "[":
                stack.append(i)
            elif ch == "]" and stack:
                start = stack.pop()
                spans.append((start, i))
        spans.sort(key=lambda t: t[1]-t[0], reverse=True)
        for s, e in spans:
            sub = text[s:e+1]
            if '"question"' in sub and '"options"' in sub:
                return sub
        return self._repair_brackets(text)

    def _parse_mcq_json_safely(self, text: str):
        """LLM 응답 → JSON 배열(문항 최소 정합성 필터 포함). 실패 시 []."""
        try:
            cleaned = html.unescape(text or "")
            candidate = self._find_largest_json_array(cleaned)
            if not candidate:
                return []
            
            # JSON 파싱 시도
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 수정 시도
                fixed = self._fix_json_format(candidate)
                try:
                    data = json.loads(fixed)
                except json.JSONDecodeError:
                    return []
            
            # 결과 검증 및 정리
            out = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = [data]
            else:
                return []
                
            for item in items:
                if not isinstance(item, dict):
                    continue
                    
                q = str(item.get("question", "")).strip()
                opts = [str(o).strip() for o in (item.get("options") or []) if str(o).strip()]
                
                # 최소 검증: 질문 3자 이상, 보기 2개 이상
                if len(q) >= 3 and len(opts) >= 2:
                    # 보기 정리 (번호/원문자 제거)
                    clean_opts = []
                    for opt in opts[:4]:  # 최대 4개까지만
                        clean_opt = re.sub(r"^(?:[①②③④⑤⑥⑦⑧⑨⑩]|\d+\)|[A-E]\)|[가-하]\))\s*", "", opt)
                        if clean_opt.strip():
                            clean_opts.append(clean_opt.strip())
                    
                    if len(clean_opts) >= 2:  # 최소 2개 보기 필요
                        out.append({
                            "question": re.sub(r"\s+", " ", q), 
                            "options": clean_opts
                        })
            
            return out
            
        except Exception as e:
            print(f"⚠️ JSON 파싱 오류: {e}")
            return []


    # ========== 공개 API =======================================================
    

    def extract_problems_from_pdf(self, file_paths: List[str]) -> List[Dict]:
        """PDF 파일들에서 문제(question, options[]) 추출"""
        # Docling 초기화
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
            return []

        # LLM 클라이언트
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

                # Docling에서 마크다운 추출(원문 보관: 폴백용)
                raw_text = doc_result.document.export_to_markdown()
                raw_text = self.normalize_docling_markdown(raw_text)
                raw_text = self._strip_headers_for_llm(raw_text)
                raw_text = self._fix_korean_spacing_noise(raw_text)
                print(f"📊 Docling 텍스트 길이: {len(raw_text)}")

                # 1) 전체 텍스트 우선 파싱 (가장 정확)
                print("🧭 전체 텍스트 우선 파싱 시작")
                full_text_problems = self._parse_full_text_with_fallback(raw_text, llm)
                
                local: List[Dict] = []
                if full_text_problems:
                    print(f"✅ 전체 텍스트에서 {len(full_text_problems)}개 문제 추출")
                    local = full_text_problems
                else:
                    # 2) 폴백: 컬럼별 파싱
                    print("🔁 전체 텍스트 파싱 실패 → 컬럼별 파싱 시도")
                    col_batch = self._parse_by_columns_with_llm(path, llm)
                    if col_batch:
                        local = col_batch
                        print(f"✅ 컬럼별 파싱에서 {len(col_batch)}개 문제 추출")
                    else:
                        # 3) 최종 폴백: 원문 전체를 안전 청크로 나눠 일괄 추출
                        print("🔁 최종 폴백: 원문 전체 일괄 추출")
                        for chunk in self._chunk_by_paragraph(raw_text, max_chars=16000):
                            batch = self._parse_whole_text_with_llm(chunk, llm)
                            if batch:
                                local.extend(batch)

                # 중복 제거
                before = len(local)
                local = self._dedupe_problems(local)
                print(f"🧹 dedupe: {before} → {len(local)}")

                all_problems.extend(local)
                print(f"📊 누적 문제 수: {len(all_problems)}")

            except Exception as e:
                print(f"❌ 파일 처리 실패: {e}")
                continue

        print(f"🎯 총 {len(all_problems)}개 문제 추출 완료")
        return all_problems

    # ========== 열(컬럼) 추출 & 일괄 LLM =======================================

    def _extract_columns_with_pdfminer(self, pdf_path: str):
        """각 페이지를 (left_text, right_text)로 분리"""
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer
        except Exception as e:
            print(f"⚠️ pdfminer import 실패: {e}")
            return []

        pages_cols = []
        try:
            for page_layout in extract_pages(pdf_path):
                left, right = [], []
                xs = []
                for el in page_layout:
                    if isinstance(el, LTTextContainer):
                        xs.append(el.bbox[0])
                if not xs:
                    pages_cols.append({"left": "", "right": ""})
                    continue

                mid = sorted(xs)[len(xs) // 2]

                for el in page_layout:
                    if isinstance(el, LTTextContainer):
                        x0, y0, x1, y1 = el.bbox
                        txt = el.get_text().strip()
                        if not txt:
                            continue
                        (left if x0 < mid else right).append((y1, txt))

                left.sort(key=lambda t: -t[0])
                right.sort(key=lambda t: -t[0])

                left_text = "".join(t for _, t in left)
                right_text = "".join(t for _, t in right)
                pages_cols.append({"left": left_text, "right": right_text})
        except Exception as e:
            print(f"⚠️ 컬럼 추출 실패: {e}")
            return []

        print(f"✅ [컬럼 추출] 총 {len(pages_cols)}페이지 처리 완료")
        return pages_cols

    def _parse_by_columns_with_llm(self, pdf_path: str, llm) -> Optional[List[Dict]]:
        """좌/우 열을 통으로 LLM에 넘겨 문항 배열 추출"""
        cols = self._extract_columns_with_pdfminer(pdf_path)
        if not cols:
            return None

        left_stream = "\n\n".join(p["left"] for p in cols if p.get("left"))
        right_stream = "\n\n".join(p["right"] for p in cols if p.get("right"))

        results: List[Dict] = []
        for label, stream in (("LEFT", left_stream), ("RIGHT", right_stream)):
            text = self.normalize_docling_markdown(stream)
            text = self._strip_headers_for_llm(text)
            text = self._fix_korean_spacing_noise(text)

            chunks = self._chunk_by_paragraph(text, max_chars=16000)
            print(f"🧱 [{label}] 청크 {len(chunks)}개")
            for idx, chunk in enumerate(chunks, 1):
                batch = self._parse_whole_text_with_llm(chunk, llm)
                if batch:
                    print(f"✅ [{label}] 청크 {idx} → {len(batch)}개 추출")
                    results.extend(batch)
                else:
                    print(f"⚠️ [{label}] 청크 {idx} 추출 0개")

        return results if results else None

    # ========== 전체 텍스트 우선 파싱 (폴백 메커니즘 포함) =====================

    def _parse_full_text_with_fallback(self, full_text: str, llm) -> Optional[List[Dict]]:
        """전체 텍스트에서 문제를 추출하고, 짝이 없는 항목들을 폴백으로 처리"""
        print("🔍 전체 텍스트 파싱 시작")
        
        # 1단계: 청크 내부를 순차적으로 탐색하면서 문제를 하나씩 추출
        problems = self._parse_text_incrementally(full_text, llm)
        if problems:
            print(f"✅ 순차 파싱에서 {len(problems)}개 문제 추출 성공")
            return problems
        
        # 2단계: 폴백 - 정규표현식으로 문제-보기 패턴 찾기
        print("🔄 LLM 파싱 실패 → 정규표현식 폴백 시작")
        regex_problems = self._extract_problems_with_regex(full_text)
        if regex_problems:
            print(f"✅ 정규표현식으로 {len(regex_problems)}개 문제 추출")
            return regex_problems
        
        print("❌ 모든 파싱 방법 실패")
        return None

    def _parse_text_incrementally(self, full_text: str, llm) -> List[Dict]:
        """청크 내부를 순차적으로 탐색하면서 문제를 하나씩 추출"""
        all_problems = []
        
        # 텍스트를 문제 단위로 분할 (정규표현식으로 문제 번호 기준)
        problem_segments = self._split_text_by_problems(full_text)
        print(f"🔍 총 {len(problem_segments)}개 문제 세그먼트 발견")
        
        for i, segment in enumerate(problem_segments, 1):
            if len(segment.strip()) < 50:  # 너무 짧은 세그먼트는 건너뜀
                continue
                
            print(f"🔄 세그먼트 {i}/{len(problem_segments)} 처리 중...")
            
            # 각 세그먼트에서 문제 추출 시도
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

    def _split_text_by_problems(self, text: str) -> List[str]:
        """텍스트를 문제 번호 기준으로 분할"""
        # 문제 번호 패턴으로 분할 (1. 2. 3. 등)
        pattern = r'(?=\d+\s*\.\s*)'
        segments = re.split(pattern, text)
        
        # 빈 세그먼트 제거하고 정리
        cleaned_segments = []
        for segment in segments:
            segment = segment.strip()
            if segment and len(segment) > 20:  # 최소 길이 필터
                cleaned_segments.append(segment)
        
        return cleaned_segments

    def _extract_single_problem_with_llm(self, segment: str, llm) -> Optional[Dict]:
        """단일 세그먼트에서 LLM을 사용해 하나의 문제 추출"""
        try:
            # 매우 간단한 프롬프트로 단일 문제만 추출
            sys_prompt = "한국어 객관식 문제를 JSON으로 반환해라."
            user_prompt = (
                f"다음 텍스트에서 객관식 문제 1개를 찾아 JSON으로 반환해라.\n"
                f"형식: {{\"question\": \"질문내용\", \"options\": [\"보기1\", \"보기2\", \"보기3\", \"보기4\"]}}\n"
                f"텍스트:\n{segment[:2000]}"  # 길이 제한
            )
            
            resp = llm.invoke([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            content = (resp.content or "").strip()
            if not content:
                return None
            
            # JSON 추출 및 파싱
            json_part = self._extract_json_from_response(content)
            if not json_part:
                return None
            
            try:
                data = json.loads(json_part)
                if isinstance(data, dict) and "question" in data and "options" in data:
                    question = str(data["question"]).strip()
                    options = [str(opt).strip() for opt in data.get("options", []) if str(opt).strip()]
                    
                    if len(question) > 5 and len(options) >= 2:
                        # 보기에서 번호 제거
                        clean_options = []
                        for opt in options[:4]:
                            clean_opt = re.sub(r'^[①②③④⑤⑥⑦⑧⑨⑩]?\s*', '', opt)
                            clean_opt = re.sub(r'^\d+\)\s*', '', clean_opt)
                            if clean_opt.strip():
                                clean_options.append(clean_opt.strip())
                        
                        if len(clean_options) >= 2:
                            return {
                                "question": re.sub(r'\s+', ' ', question),
                                "options": clean_options
                            }
                
            except json.JSONDecodeError:
                pass
            
        except Exception as e:
            print(f"⚠️ LLM 단일 문제 추출 실패: {e}")
        
        return None

    def _extract_single_problem_with_regex(self, segment: str) -> Optional[Dict]:
        """단일 세그먼트에서 정규표현식으로 하나의 문제 추출"""
        try:
            # 문제 텍스트 찾기
            question_match = re.search(r'\d+\s*\.\s*(.+?)(?=[①②③④⑤⑥⑦⑧⑨⑩]|\d+\))', segment, re.DOTALL)
            if not question_match:
                return None
            
            question = question_match.group(1).strip()
            if len(question) < 10:
                return None
            
            # 보기들 찾기
            options = []
            
            # 원문자 보기 찾기
            for opt_match in re.finditer(r'[①②③④⑤⑥⑦⑧⑨⑩]\s*([^①②③④⑤⑥⑦⑧⑨⑩]+?)(?=[①②③④⑤⑥⑦⑧⑨⑩]|\d+\s*\.|$)', segment, re.DOTALL):
                option_text = opt_match.group(1).strip()
                if len(option_text) > 3:
                    options.append(option_text)
                    if len(options) >= 4:
                        break
            
            # 숫자 보기 찾기 (원문자가 부족한 경우)
            if len(options) < 2:
                for opt_match in re.finditer(r'\d+\)\s*([^1-4\)]+?)(?=\d+\)|\d+\s*\.|$)', segment, re.DOTALL):
                    option_text = opt_match.group(1).strip()
                    if len(option_text) > 3 and option_text not in options:
                        options.append(option_text)
                        if len(options) >= 4:
                            break
            
            if len(options) >= 2:
                return {
                    "question": re.sub(r'\s+', ' ', question),
                    "options": options[:4]
                }
            
        except Exception as e:
            print(f"⚠️ 정규표현식 단일 문제 추출 실패: {e}")
        
        return None

    def _extract_problems_with_regex(self, text: str) -> List[Dict]:
        """정규표현식으로 문제-보기 패턴을 찾아 추출"""
        problems = []
        
        # 문제 번호 패턴 (1. 2. 3. 등)
        question_pattern = r'(\d+)\s*\.\s*([^①-⑩1-4\)]+?)(?=\d+\s*\.|$)'
        
        # 보기 패턴 (① ② ③ ④ 또는 1) 2) 3) 4))
        option_patterns = [
            r'[①-⑩]\s*([^①-⑩]+?)(?=[①-⑩]|\d+\s*\.|$)',
            r'(\d+\))\s*([^1-4\)]+?)(?=\d+\)|\d+\s*\.|$)'
        ]
        
        # 문제 찾기
        for match in re.finditer(question_pattern, text, re.DOTALL):
            question_num = match.group(1)
            question_text = match.group(2).strip()
            
            # 해당 문제 다음에 오는 보기들 찾기
            start_pos = match.end()
            options = []
            
            # 원문자 보기 찾기
            for opt_match in re.finditer(r'[①-⑩]\s*([^①-⑩]+?)(?=[①-⑩]|\d+\s*\.|$)', text[start_pos:], re.DOTALL):
                option_text = opt_match.group(1).strip()
                if len(option_text) > 5:  # 너무 짧은 보기는 제외
                    options.append(option_text)
                    if len(options) >= 4:  # 최대 4개까지만
                        break
            
            # 숫자 보기 찾기 (원문자가 부족한 경우)
            if len(options) < 4:
                for opt_match in re.finditer(r'(\d+\))\s*([^1-4\)]+?)(?=\d+\)|\d+\s*\.|$)', text[start_pos:], re.DOTALL):
                    option_text = opt_match.group(2).strip()
                    if len(option_text) > 5 and option_text not in options:
                        options.append(option_text)
                        if len(options) >= 4:
                            break
            
            # 최소 2개 보기가 있는 경우만 추가
            if len(question_text) > 10 and len(options) >= 2:
                problems.append({
                    "question": re.sub(r'\s+', ' ', question_text),
                    "options": options[:4]  # 최대 4개까지만
                })
        
        return problems

    # ========== LLM 일괄 파서 ===================================================

    def _parse_whole_text_with_llm(self, full_text: str, llm) -> Optional[List[Dict]]:
        """주어진 텍스트에서 문항 배열을 추출(JSON 배열만 허용) - 프롬프트 강화판"""
        cleaned = self.normalize_docling_markdown(full_text)
        cleaned = self._strip_headers_for_llm(cleaned)
        cleaned = self._fix_korean_spacing_noise(cleaned)

        # ── 프롬프트(형식 규칙을 매우 구체적으로) ─────────────────────────────
        sys_prompt = "한국어 객관식 시험지에서 문제를 찾아 JSON 배열로 반환해라."

        # 형식 규칙: 질문/보기의 시작과 끝을 명시
        user_prompt = (
            "다음 텍스트에서 객관식 문제들을 찾아 JSON 배열로 반환해라.\n"
            "형식: [{\"question\": \"질문내용\", \"options\": [\"보기1\", \"보기2\", \"보기3\", \"보기4\"]}]\n"
            "문제 번호로 시작하는 줄을 찾고, 그 다음에 ① ② ③ ④로 시작하는 보기들을 찾아라.\n"
            "정답이나 해설은 제외하고 문제와 보기만 추출해라.\n\n"
            f"텍스트:\n{cleaned[:40000]}"
        )

        try:
            # LLM 호출 시도 (최대 3번)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"🔄 LLM 호출 시도 {attempt + 1}/{max_retries}")
                    resp = llm.invoke(
                        [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    )
                    content = (resp.content or "").strip()
                    if content:
                        print(f"✅ LLM 호출 성공 (시도 {attempt + 1})")
                        break
                    else:
                        print(f"⚠️ LLM 응답이 비어있음 (시도 {attempt + 1})")
                        if attempt == max_retries - 1:
                            return None
                        continue
                        
                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠️ LLM 호출 실패 (시도 {attempt + 1}): {error_msg}")
                    
                    # 괄호 오류인 경우 프롬프트를 더 단순하게 만들어 재시도
                    if "unbalanced parenthesis" in error_msg.lower() or "position" in error_msg.lower():
                        if attempt < max_retries - 1:
                            print("🔄 괄호 오류 감지, 프롬프트 단순화하여 재시도...")
                            # 더 단순한 프롬프트로 재시도
                            simple_sys = "문제를 찾아 JSON으로 반환해라."
                            simple_user = f"텍스트에서 객관식 문제를 찾아 [{{\"question\": \"질문\", \"options\": [\"보기1\", \"보기2\", \"보기3\", \"보기4\"]}}] 형태로 반환해라.\n\n{cleaned[:20000]}"
                            
                            try:
                                resp = llm.invoke([
                                    {"role": "system", "content": simple_sys},
                                    {"role": "user", "content": simple_user}
                                ])
                                content = (resp.content or "").strip()
                                if content:
                                    print(f"✅ 단순화된 프롬프트로 성공")
                                    break
                            except Exception as retry_e:
                                print(f"⚠️ 단순화된 프롬프트도 실패: {retry_e}")
                                continue
                        else:
                            print("❌ 최대 재시도 횟수 초과")
                            return None
                    else:
                        # 다른 오류인 경우 잠시 대기 후 재시도
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(1)
                            continue
                        else:
                            print("❌ 최대 재시도 횟수 초과")
                            return None
            
            if not content:
                return None

            # 단 하나의 JSON 코드블록/배열만 추출
            print(f"🔍 [DEBUG] LLM 응답 내용:")
            print(f"--- 응답 시작 ---")
            print(content[:500] + "..." if len(content) > 500 else content)
            print(f"--- 응답 끝 ---")
            
            json_part = self._extract_json_from_response(content)
            if not json_part:
                print("❌ [일괄] JSON 추출 실패")
                return None
            
            print(f"🔍 [DEBUG] 추출된 JSON 부분:")
            print(f"--- JSON 시작 ---")
            print(json_part[:300] + "..." if len(json_part) > 300 else json_part)
            print(f"--- JSON 끝 ---")

            # 1차 파싱
            try:
                data = json.loads(json_part)
            except json.JSONDecodeError:
                # 남아있는 잔여 텍스트로 인한 'Extra data' 대응: 가장 큰 유효 배열만 파싱
                try:
                    from json import JSONDecoder
                    dec = JSONDecoder()
                    obj, _ = dec.raw_decode(json_part.lstrip())
                    data = obj
                except Exception:
                    fixed = self._fix_json_format(json_part)
                    try:
                        data = json.loads(fixed)
                    except json.JSONDecodeError as e2:
                        print(f"❌ [일괄] JSON 파싱 실패: {e2}")
                        return None

            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                print("❌ [일괄] JSON 최상위가 배열이 아님")
                return None

            cleaned_list: List[Dict] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = (item.get("question") or "").strip()
                opts = item.get("options") if isinstance(item.get("options"), list) else []
                if not q or len(opts) < 2:
                    continue
                q = re.sub(r"\\s+", " ", q)[:800]
                norm_opts = []
                seen = set()
                for o in opts:
                    s = re.sub(r"\\s+", " ", str(o)).strip()
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    # 보기 앞 번호/원문자 제거(혹시 남아있다면)
                    s = re.sub(r"^(?:[①②③④⑤⑥⑦⑧⑨⑩]|\\d+\\)|[A-E]\\)|[가-하]\\))\\s*", "", s)
                    norm_opts.append(s)
                if 2 <= len(norm_opts) <= 6:
                    cleaned_list.append({"question": q, "options": norm_opts})
            return cleaned_list or None

        except Exception as e:
            print(f"⚠️ [일괄] LLM 호출 실패: {e}")
            return None


    # ========== 텍스트/청크/정규화 =============================================

    def _chunk_by_paragraph(self, text: str, max_chars: int = 16000) -> List[str]:
        """빈 줄 기준으로 적절히 합쳐 LLM 입력 길이 제어"""
        paras = [p for p in text.split("\n\n") if p.strip()]
        if not paras:  # 빈 줄 없는 문서 대비
            paras = [text]

        chunks, cur, size = [], [], 0
        for p in paras:
            if size + len(p) + 2 > max_chars and cur:
                chunks.append("\n\n".join(cur))
                cur, size = [p], len(p)
            else:
                cur.append(p)
                size += len(p) + 2
        if cur:
            chunks.append("\n\n".join(cur))
        return chunks

    def normalize_docling_markdown(self, md: str) -> str:
        """Docling 마크다운 정규화(사소한 번호/공백 보정)"""
        s = md
        s = re.sub(r"(?m)^\s*(\d+)\.\s*\1\.\s*", r"\1. ", s)  # '1. 1.' → '1.'
        s = re.sub(r"(?m)^\s*(\d+)\s*\.\s*", r"\1. ", s)      # '1 . ' → '1. '
        s = re.sub(r"[ \t]+", " ", s).replace("\r", "")
        return s.strip()

    def _strip_headers_for_llm(self, s: str) -> str:
        """헤더/메타 라인 제거 및 잉여 공백 정리(LLM 전처리)"""
        s = re.sub(r"<!--.*?-->", "", s, flags=re.DOTALL)
        s = re.sub(r"(?m)^\s*<!--\s*image\s*-->\s*$", "", s)
        # 마크다운 헤더 프리픽스 제거(내용 보존)
        s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)
        # 과도한 빈 줄 축소
        s = re.sub(r"\n{3,}", "\n\n", s)
        s = re.sub(r"[ \t]+", " ", s)
        return s.strip()

    def _fix_korean_spacing_noise(self, s: str) -> str:
        """한글-한글 사이에 생긴 불필요 공백 제거(예: '익스트 림')"""
        for _ in range(2):  # 두 번 정도 반복
            s = re.sub(r"([\uAC00-\uD7A3])\s+(?=[\uAC00-\uD7A3])", r"\1", s)
        return s

    # ========== JSON 추출 유틸 ================================================

    def _extract_json_from_response(self, content: str) -> Optional[str]:
        """
        응답에서 '단 하나의' JSON 배열을 추출합니다.
        1) ```json 코드블록``` 안의 배열 우선
        2) 본문 전체에서 가장 큰 유효 JSON 배열(대괄호 균형 스캔)
        3) 마지막 폴백: 단일 객체가 있으면 반환(상위에서 dict→list로 감쌈)
        """
        try:
            # 1) 코드블록 우선
            m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", content, re.DOTALL)
            if m:
                return m.group(1).strip()

            # 2) 본문에서 가장 큰 유효 배열
            arr = self._find_largest_json_array(content)
            if arr:
                return arr

            # 3) 객체 폴백(질문/보기 키를 가진 객체만 허용)
            obj = self._find_first_question_object(content)
            if obj:
                return obj

            return None
            
        except Exception as e:
            print(f"⚠️ JSON 추출 오류: {e}")
            return None

    def _find_first_question_object(self, content: str) -> Optional[str]:
        """질문/보기 키를 가진 첫 번째 JSON 객체를 찾아 반환"""
        try:
            # question과 options 키를 모두 포함하는 객체 찾기
            pattern = r'\{[^{}]*"question"\s*:\s*"[^"]*"[^{}]*"options"\s*:\s*\[[^{}]*\][^{}]*\}'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(0)
            
            # 더 유연한 패턴으로 시도
            pattern2 = r'\{[^{}]*"question"[^{}]*"options"[^{}]*\}'
            match2 = re.search(pattern2, content, re.DOTALL)
            if match2:
                return match2.group(0)
                
            return None
            
        except Exception as e:
            print(f"⚠️ 객체 검색 오류: {e}")
            return None

    def _is_valid_json_structure(self, text: str) -> bool:
        """중괄호/대괄호 짝 간단 검사 + 시작/끝 문자 검사"""
        if text.count("{") != text.count("}"):
            return False
        if text.count("[") != text.count("]"):
            return False
        if not (text.startswith("{") and text.endswith("}")) and not (
            text.startswith("[") and text.endswith("]")
        ):
            return False
        return True

    def _construct_json_from_parts(self, content: str) -> Optional[str]:
        """흩어진 question/options를 찾아 최소 JSON 구성"""
        try:
            q = re.search(r'"question"\s*:\s*"([^"]*)"', content)
            o = re.search(r'"options"\s*:\s*\[(.*?)\]', content, re.DOTALL)
            if not (q and o):
                return None
            question = q.group(1)
            options = [x for x in re.findall(r'"([^"]*)"', o.group(1)) if x.strip()]
            if not options:
                return None
            return json.dumps({"question": question, "options": options}, ensure_ascii=False)
        except Exception:
            return None

    def _fix_json_format(self, content: str) -> str:
        """가벼운 JSON 포맷 보정"""
        try:
            # 기본 정리
            content = re.sub(r",\s*}", "}", content)
            content = re.sub(r",\s*]", "]", content)
            content = re.sub(r"\n|\r", " ", content)
            content = re.sub(r"\s+", " ", content)
            
            # 중괄호/대괄호 범위 찾기
            if content.count("{") > 0 and content.count("}") > 0:
                start = content.find("{")
                end = content.rfind("}")
                if start < end:
                    content = content[start:end + 1]
                    
            if content.count("[") > 0 and content.count("]") > 0:
                start = content.find("[")
                end = content.rfind("]")
                if start < end:
                    content = content[start:end + 1]
            
            # 따옴표 문제 수정
            content = re.sub(r'([^\\])"([^"]*?)([^\\])"', r'\1"\2\3"', content)
            
            return content.strip()
            
        except Exception as e:
            print(f"⚠️ JSON 포맷 수정 오류: {e}")
            return content

    # ========== dedupe =========================================================

    def _fingerprint(self, q: str) -> str:
        s = q.lower()
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[^a-z0-9\uAC00-\uD7A3]", "", s)
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _dedupe_problems(self, items: List[Dict]) -> List[Dict]:
        seen = set()
        out: List[Dict] = []
        for it in items:
            q = (it.get("question") or "").strip()
            opts = it.get("options") or []
            if not q or not isinstance(opts, list) or len(opts) == 0:
                continue
            fp = self._fingerprint(q)
            if fp in seen:
                continue
            seen.add(fp)

            clean_opts = []
            oseen = set()
            for o in opts:
                o2 = re.sub(r"\s+", " ", str(o)).strip()
                if o2 and o2 not in oseen:
                    oseen.add(o2)
                    clean_opts.append(o2)
            it["options"] = clean_opts[:6]
            out.append(it)
        return out


# ── 모듈 레벨 편의 함수들 ─────────────────────────────────────────────────────

def extract_pdf_paths(text: str) -> List[str]:
    pre = PDFPreprocessor()
    return pre.extract_pdf_paths(text)

def extract_problem_range(text: str) -> Optional[Dict]:
    pre = PDFPreprocessor()
    return pre.extract_problem_range(text)

def determine_problem_source(text: str) -> Optional[str]:
    pre = PDFPreprocessor()
    return pre.determine_problem_source(text)

# ↑ 위 3개 편의 함수는 아래 간단 구현으로 대체합니다.
# 필요 없으면 삭제해도 되지만 기존 코드 호환을 위해 유지.

PDF_PATH_PATTERNS = [
    r"([^\s]+\.pdf)",  # 기본 .pdf
    r"([C-Z]:[\\\/][^\\\/\s]*\.pdf)",  # Windows 절대 경로
    r"([\.\/][^\\\/\s]*\.pdf)",  # 상대 경로
]

def _findall(pattern, text):
    return re.findall(pattern, text, re.IGNORECASE)

def _extract_pdf_paths_impl(self, text: str) -> List[str]:
    paths = []
    for p in PDF_PATH_PATTERNS:
        paths.extend(_findall(p, text))
    return list(set(paths))

def _extract_problem_range_impl(self, text: str) -> Optional[Dict]:
    # 단일 번호: "5번만", "5번 풀어줘"
    m = re.search(r'(\d+)번만', text)
    if m:
        return {"type": "single", "number": int(m.group(1))}
    m = re.search(r'(\d+)번\s*풀', text)
    if m:
        return {"type": "single", "number": int(m.group(1))}

    # 범위: "1-10번", "3번부터 7번"
    m = re.search(r'(\d+)\s*[-~]\s*(\d+)번', text)
    if m:
        return {"type": "range", "start": int(m.group(1)), "end": int(m.group(2))}
    m = re.search(r'(\d+)번부터\s*(\d+)번', text)
    if m:
        return {"type": "range", "start": int(m.group(1)), "end": int(m.group(2))}

    # 묶음: "1,3,5번"
    m = re.search(r'(\d+(?:\s*,\s*\d+)*)번', text)
    if m:
        numbers = [int(x.strip()) for x in m.group(1).split(',')]
        return {"type": "specific", "numbers": numbers}

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
