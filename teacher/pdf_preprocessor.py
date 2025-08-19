"""
PDF 전처리 관련 함수들을 모아놓은 모듈
teacher_graph.py에서 PDF 관련 로직을 분리하여 가독성을 높임
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple
from docling.document_converter import DocumentConverter
from langchain_openai import ChatOpenAI


class PDFPreprocessor:
    """PDF 파일 전처리 및 문제 추출 클래스"""
    
    def __init__(self):
        # 환경변수 설정으로 권한 문제 해결
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
    
    def extract_pdf_paths(self, text: str) -> List[str]:
        """PDF 파일 경로 추출"""
        # PDF 파일 경로 패턴 매칭
        pdf_patterns = [
            r'([^\s]+\.pdf)',  # 기본 .pdf 파일 경로
            r'([C-Z]:[\\\/][^\\\/\s]*\.pdf)',  # Windows 절대 경로
            r'([\.\/][^\\\/\s]*\.pdf)',  # 상대 경로
        ]
        
        pdf_paths = []
        for pattern in pdf_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            pdf_paths.extend(matches)
        
        return list(set(pdf_paths))  # 중복 제거
    
    def extract_problem_range(self, text: str) -> Optional[Dict]:
        """문제 번호 범위 추출"""
        # 패턴들: "5번", "1-10번", "3번부터 7번까지", "1,3,5번"
        patterns = [
            r'(\d+)번만',  # "5번만"
            r'(\d+)번\s*풀',  # "5번 풀어줘"
            r'(\d+)\s*[-~]\s*(\d+)번',  # "1-10번", "1~10번"
            r'(\d+)번부터\s*(\d+)번',  # "3번부터 7번까지"
            r'(\d+(?:\s*,\s*\d+)*)번',  # "1,3,5번"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    if ',' in groups[0]:
                        # 콤마로 구분된 번호들
                        numbers = [int(x.strip()) for x in groups[0].split(',')]
                        return {"type": "specific", "numbers": numbers}
                    else:
                        # 단일 번호
                        return {"type": "single", "number": int(groups[0])}
                elif len(groups) == 2:
                    # 범위
                    start, end = int(groups[0]), int(groups[1])
                    return {"type": "range", "start": start, "end": end}
        return None
    
    def determine_problem_source(self, text: str) -> Optional[str]:
        """문제 소스 결정"""
        text_lower = text.lower()
        
        # 명시적 소스 지정
        if any(keyword in text_lower for keyword in ['pdf', '파일', '문서']):
            return "pdf_extracted"
        elif any(keyword in text_lower for keyword in ['기존', 'shared', '저장된', '이전']):
            return "shared"
        
        # PDF 파일이 명시되었으면 pdf_extracted 우선
        if self.extract_pdf_paths(text):
            return "pdf_extracted"
        
        # 아무것도 명시되지 않으면 None (자동 결정)
        return None
    
    def extract_problems_from_pdf(self, file_paths: List[str]) -> List[Dict]:
        """PDF 파일에서 문제 추출 (Docling 사용)"""
        # Docling 변환기 초기화
        converter = DocumentConverter()
        
        # LLM 설정
        llm = ChatOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1", 
            model="moonshotai/kimi-k2-instruct",
            temperature=0
        )
        
        all_problems = []
        
        for path in file_paths:
            try:
                print(f"📖 파일 처리 중: {path}")
                
                # Docling으로 PDF 변환
                doc_result = converter.convert(path)
                raw_text = doc_result.document.export_to_markdown()
                
                if not raw_text.strip():
                    print(f"⚠️ PDF에서 텍스트를 추출할 수 없음: {path}")
                    continue
                
                # 디버깅: 추출된 텍스트 일부 출력
                print(f"📝 추출된 텍스트 미리보기 (처음 500자):")
                print(f"'{raw_text[:500]}...'")
                print(f"📊 총 텍스트 길이: {len(raw_text)} 문자")
                
                # 텍스트를 블록으로 분할
                blocks = self._split_problem_blocks(raw_text)
                print(f"📝 {len(blocks)}개 블록으로 분할")
                
                # 디버깅: 첫 번째 블록 미리보기
                if blocks:
                    print(f"🔍 첫 번째 블록 미리보기:")
                    print(f"'{blocks[0][:300]}...'")
                    if len(blocks) > 1:
                        print(f"🔍 두 번째 블록 미리보기:")
                        print(f"'{blocks[1][:300]}...'")
                        print(f"🔍 마지막 블록 미리보기:")
                        print(f"'{blocks[-1][:300]}...')")
                
                # 각 블록을 LLM으로 파싱
                successful_parses = 0
                for i, block in enumerate(blocks):
                    block_len = len(block.strip())
                    if block_len < 20:  # 필터링 조건을 완화 (50 → 20)
                        print(f"⚠️ 블록 {i+1} 스킵 (너무 짧음: {block_len}자): '{block[:50]}...'")
                        continue
                    
                    print(f"🔄 블록 {i+1}/{len(blocks)} 파싱 중 ({block_len}자)...")
                    print(f"   미리보기: '{block[:100]}...'")
                        
                    try:
                        problem = self._parse_block_with_llm(block, llm)
                        if problem:
                            all_problems.append(problem)
                            successful_parses += 1
                            print(f"✅ 블록 {i+1} 파싱 성공! (총 {successful_parses}개)")
                        else:
                            print(f"❌ 블록 {i+1} 파싱 실패: LLM이 유효한 문제로 인식하지 못함")
                    except Exception as e:
                        print(f"⚠️ 블록 {i+1} 파싱 실패: {e}")
                        continue
                        
                print(f"📊 파싱 결과: {successful_parses}/{len(blocks)} 블록 성공")
                        
            except Exception as e:
                print(f"❌ 파일 {path} 처리 실패: {e}")
                continue
        
        print(f"🎯 총 {len(all_problems)}개 문제 추출 완료")
        return all_problems
    
    def _split_problem_blocks(self, raw_text: str) -> List[str]:
        """텍스트를 문제 블록으로 분할 (실제 문제 헤더 기반)"""
        print("🔍 [구조 분석] 실제 문제 헤더 기반으로 파싱 방식 결정")
        
        lines = raw_text.split('\n')
        
        # 실제 문제 헤더 패턴들 (우선순위 순)
        problem_header_patterns = [
            r'^\s*##\s*문제\s*(\d+)\s*[.)]\s*',  # "## 문제 1." (마크다운 헤더)
            r'^\s*#+\s*문제\s*(\d+)\s*[.)]\s*',  # "# 문제 1.", "### 문제 1." 등
            r'^\s*문제\s*(\d+)\s*[.)]\s*',       # "문제 1." 또는 "문제 1)"
            r'^\s*Q\s*(\d+)\s*[.)]\s*',          # "Q1." 또는 "Q1)"
            r'^\s*\[(\d+)\]\s*',                 # "[1]"
        ]
        
        # 보기 번호 패턴들 (문제 헤더가 아님)
        option_patterns = [
            r'^\s*(\d+)\.\s*\1\.\s*',           # "4. 4." (중복 번호)
            r'^\s*(\d+)\s*[.)]\s*',              # "1)", "2." (보기 번호)
            r'^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*',      # 원문자 보기
            r'^\s*[가-하]\s*[)]\s*',            # "가)", "나)" (보기)
            r'^\s*[A-E]\s*[)]\s*',              # "A)", "B)" (보기)
        ]
        
        # 문제 헤더 위치 찾기
        problem_headers = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # 보기 번호인지 먼저 확인
            is_option = False
            for pattern in option_patterns:
                if re.match(pattern, line_stripped):
                    is_option = True
                    break
            
            if is_option:
                continue  # 보기 번호는 스킵
            
            # 문제 헤더인지 확인
            for pattern in problem_header_patterns:
                match = re.match(pattern, line_stripped)
                if match:
                    problem_num = int(match.group(1))
                    problem_headers.append((i, problem_num, line_stripped))
                    print(f"✅ [문제 헤더 발견] 라인 {i+1}: '{line_stripped}' (문제 {problem_num}번)")
                    break
        
        if not problem_headers:
            print("⚠️ 문제 헤더를 찾을 수 없음 - 전체를 1개 블록으로 처리")
            return [raw_text] if raw_text.strip() else []
        
        print(f"🔍 총 {len(problem_headers)}개 문제 헤더 발견")
        
        # 문제 헤더를 번호 순으로 정렬
        problem_headers.sort(key=lambda x: x[1])
        
        # 문제 블록 생성
        problem_blocks = []
        
        for i, (header_idx, problem_num, header_text) in enumerate(problem_headers):
            # 현재 문제의 시작
            start_line = header_idx
            
            # 다음 문제의 시작 (또는 마지막)
            if i + 1 < len(problem_headers):
                end_line = problem_headers[i + 1][0]
            else:
                end_line = len(lines)
            
            # 문제 블록 텍스트 생성
            problem_text = '\n'.join(lines[start_line:end_line]).strip()
            
            if problem_text:
                problem_blocks.append(problem_text)
                print(f"📦 문제 {problem_num}번: 라인 {start_line+1}-{end_line} ({len(problem_text)}자)")
                print(f"   헤더: '{header_text}'")
        
        print(f"✅ 총 {len(problem_blocks)}개 문제 블록 생성 완료")
        return problem_blocks
    
    def _merge_blocks_by_question(self, micro_blocks: List[str]) -> List[str]:
        """미세 분할된 블록들을 문제별로 재묶기"""
        if not micro_blocks:
            return []
        
        print(f"🔄 [재묶기] {len(micro_blocks)}개 미세 블록을 문제별로 묶는 중...")
        
        # 문제 헤더 패턴들 (마크다운 헤더 우선, 다양한 형식 지원)
        question_patterns = [
            r'^\s*##\s*문제\s*(\d+)\s*[.)]\s*',  # "## 문제 1." (마크다운 헤더 우선)
            r'^\s*#+\s*문제\s*(\d+)\s*[.)]\s*',  # "# 문제 1.", "### 문제 1." 등
            r'^\s*문제\s*(\d+)\s*[.)]\s*',       # "문제 1." 또는 "문제 1)"
            r'^\s*(\d+)\s*[.)]\s*(?![①②③④⑤])', # "1." (보기가 아닌 경우)
            r'^\s*Q\s*(\d+)\s*[.)]\s*',          # "Q1." 또는 "Q1)"
            r'^\s*\[(\d+)\]\s*',                 # "[1]"
        ]
        
        # 보기 패턴들 (문제와 구분하기 위해)
        option_patterns = [
            r'^\s*[①②③④⑤⑥⑦⑧⑨⑩]',      # 원문자 보기
            r'^\s*[1-5]\s*[)]\s*\S',        # "1) 내용" (짧은 숫자 + 내용)
            r'^\s*[가-하]\s*[)]\s*',        # "가) 내용"
            r'^\s*[A-E]\s*[)]\s*',          # "A) 내용"
        ]
        
        merged_blocks = []
        current_block = ""
        current_question_num = 0
        
        for i, block in enumerate(micro_blocks):
            block = block.strip()
            if not block:
                continue
            
            # 문제 헤더인지 확인
            is_question_header = False
            question_num = 0
            
            for pattern in question_patterns:
                match = re.match(pattern, block, re.IGNORECASE)
                if match:
                    # 보기가 아닌지 추가 확인
                    is_option = any(re.match(opt_pattern, block) for opt_pattern in option_patterns)
                    if not is_option:
                        is_question_header = True
                        question_num = int(match.group(1))
                        print(f"✅ [문제 헤더 발견] 블록 {i+1}: '{block[:50]}...' (문제 {question_num}번)")
                        break
            
            if is_question_header and current_block:
                # 새로운 문제 시작 - 이전 블록 저장
                merged_blocks.append(current_block.strip())
                current_block = block
                current_question_num = question_num
                print(f"📦 [블록 완성] {len(merged_blocks)}번째 문제 블록 생성 ({len(current_block)}자)")
            else:
                # 현재 문제에 추가
                if current_block:
                    current_block += "\n\n" + block
                else:
                    current_block = block
                    if is_question_header:
                        current_question_num = question_num
        
        # 마지막 블록 추가
        if current_block:
            merged_blocks.append(current_block.strip())
            print(f"📦 [블록 완성] {len(merged_blocks)}번째 문제 블록 생성 ({len(current_block)}자)")
        
        print(f"🎯 [재묶기 완료] {len(micro_blocks)}개 → {len(merged_blocks)}개 문제 블록")
        
        # 디버깅: 첫 번째 블록 미리보기
        if merged_blocks:
            print(f"🔍 [재묶기 결과] 첫 번째 문제 블록:")
            print(f"'{merged_blocks[0][:200]}...'")
        
        return merged_blocks
    
    def normalize_docling_markdown(self, md: str) -> str:
        """Docling 마크다운 정규화"""
        s = md
        s = re.sub(r'(?m)^\s*(\d+)\.\s*\1\.\s*', r'\1. ', s)  # '1. 1.' -> '1.'
        s = re.sub(r'(?m)^\s*(\d+)\s*\.\s*', r'\1. ', s)      # '1 . ' -> '1. '
        s = re.sub(r'[ \t]+', ' ', s).replace('\r', '')
        return s.strip()
    
    def _find_option_clusters(self, lines: List[str], start: int, end: int) -> List[Tuple[int, int]]:
        """
        [start, end) 라인 구간에서 옵션 라인이 3개 이상 연속되는 구간들을 반환.
        (보기 영역 식별용)
        """
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
    
    def split_problem_blocks_without_keyword(self, text: str) -> List[str]:
        """
        '문제' 키워드가 없는 시험지에서 번호(1., 2., …)만으로 문항 단위를 분할.
        - 전역 증가 시퀀스(prev+1) 휴리스틱
        - 섹션 리셋(번호=1) 제한적 허용
        - 옵션 클러스터(연속 3+)는 문항 헤더로 취급하지 않음
        """
        text = self.normalize_docling_markdown(text)
        lines = text.split('\n')
        n = len(lines)

        # 미리 옵션 클러스터를 계산해놓고, 그 내부 번호는 문항 헤더로 안 봄
        clusters = self._find_option_clusters(lines, 0, n)

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
                print(f"🔍 [디버그] 라인 {i}: '{ln[:50]}...' (옵션 클러스터 내부 - 스킵)")
                continue
            num = int(m.group(1))
            candidates.append((i, num))
            print(f"🔍 [디버그] 라인 {i}: '{ln[:50]}...' → 후보 번호 {num}")
        
        print(f"🔍 [디버그] 총 후보 수: {len(candidates)}")
        print(f"🔍 [디버그] 옵션 클러스터 수: {len(clusters)}")

        # 전역 증가 시퀀스 + 섹션 리셋 허용으로 실제 헤더 선별
        headers = []
        prev_num = 0
        last_header_idx = -9999
        for i, num in candidates:
            if num == prev_num + 1:
                headers.append(i)
                prev_num = num
                last_header_idx = i
                print(f"✅ [디버그] 라인 {i}: 번호 {num} - 순차 증가로 헤더 선택")
                continue
            # 섹션 리셋: num==1이고, 최근 헤더에서 충분히 떨어져 있거나 섹션 느낌의 라인 존재 시 허용
            if num == 1:
                window = '\n'.join(lines[max(0, i-3): i+1])
                if (i - last_header_idx) >= 8 or re.search(r'(Ⅰ|Ⅱ|III|과목|파트|SECTION)', window):
                    headers.append(i)
                    prev_num = 1
                    last_header_idx = i
                    print(f"✅ [디버그] 라인 {i}: 번호 {num} - 섹션 리셋으로 헤더 선택")
                    continue
                else:
                    print(f"❌ [디버그] 라인 {i}: 번호 {num} - 섹션 리셋 조건 불충족 (거리: {i - last_header_idx})")
            else:
                print(f"❌ [디버그] 라인 {i}: 번호 {num} - 순차 증가 아님 (예상: {prev_num + 1})")
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

        print(f"✅ [디버그] 최종 선택된 헤더 수: {len(headers)}")
        
        # 헤더 범위로 블록 만들기
        headers.append(n)  # sentinel
        blocks = []
        for a, b in zip(headers[:-1], headers[1:]):
            blk = '\n'.join(lines[a:b]).strip()
            if blk:
                blocks.append(blk)
                print(f"📦 [디버그] 블록 {len(blocks)}: 라인 {a}-{b-1} ({len(blk)}자)")
        
        print(f"🎯 [디버그] 최종 블록 수: {len(blocks)}")
        return blocks
    
    def _parse_block_with_llm(self, block_text: str, llm) -> Optional[Dict]:
        """LLM으로 블록을 문제 형태로 파싱"""
        sys_prompt = (
            "너는 시험 문제 PDF에서 텍스트를 구조화하는 도우미다. "
            "문제 질문과 보기를 구분해서 question과 options 배열로 출력한다. "
            "options는 보기 항목만 포함하고, 설명/해설/정답 등은 포함하지 않는다. "
            "응답은 반드시 JSON 형태로만 출력한다. 다른 문장이나 코드는 절대 포함하지 말 것."
        )
        
        user_prompt = (
            "다음 텍스트에서 문항을 최대한 그대로, 정확히 추출해 JSON으로 만들어줘.\n"
            "요구 스키마: {\"question\":\"...\",\"options\":[\"...\",\"...\"]}\n"
            "규칙:\n"
            "- 문제 질문에서 번호(예: '문제 1.' 등)와 불필요한 머리글은 제거.\n"
            "- 옵션은 4개가 일반적임.\n"
            f"텍스트:\n{block_text[:1000]}"  # 너무 긴 텍스트는 잘라서
        )
        
        try:
            response = llm.invoke([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            content = response.content.strip()
            
            # JSON 추출
            if content.startswith('```'):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
            
            data = json.loads(content)
            
            # 유효성 검사
            if isinstance(data, dict) and "question" in data and "options" in data:
                if data["question"].strip() and isinstance(data["options"], list) and len(data["options"]) > 0:
                    return data
                    
        except Exception as e:
            print(f"⚠️ LLM 파싱 실패: {e}")
            
        return None


# 편의를 위한 함수들 (기존 코드와의 호환성)
def extract_pdf_paths(text: str) -> List[str]:
    """PDF 파일 경로 추출 (편의 함수)"""
    preprocessor = PDFPreprocessor()
    return preprocessor.extract_pdf_paths(text)


def extract_problem_range(text: str) -> Optional[Dict]:
    """문제 번호 범위 추출 (편의 함수)"""
    preprocessor = PDFPreprocessor()
    return preprocessor.extract_problem_range(text)


def determine_problem_source(text: str) -> Optional[str]:
    """문제 소스 결정 (편의 함수)"""
    preprocessor = PDFPreprocessor()
    return preprocessor.determine_problem_source(text)
