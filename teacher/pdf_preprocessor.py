"""
PDF 전처리 관련 함수들을 모아놓은 모듈
teacher_graph.py에서 PDF 관련 로직을 분리하여 가독성을 높임
"""

import os
import re
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

from docling.document_converter import DocumentConverter
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))


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
        try:
            # Docling 변환기 초기화 - 설정 개선
            print("🔧 DocumentConverter 초기화 중...")
            converter = DocumentConverter()
            
            # Docling 설정 조정
            try:
                # 이미지 처리 비활성화 시도
                converter.config.image_processing = False
                print("✅ 이미지 처리 비활성화 설정")
            except:
                print("⚠️ 이미지 처리 설정 변경 불가")
            
            try:
                # 텍스트 추출 우선순위 설정
                converter.config.text_extraction_priority = "text"
                print("✅ 텍스트 추출 우선순위 설정")
            except:
                print("⚠️ 텍스트 추출 우선순위 설정 불가")
                
            print("✅ DocumentConverter 초기화 완료")
        except Exception as e:
            print(f"❌ DocumentConverter 초기화 실패: {e}")
            print(f"❌ 에러 타입: {type(e)}")
            import traceback
            traceback.print_exc()
            return []
        
        # LLM 설정
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"), 
            model=OPENAI_LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        all_problems = []
        
        for path in file_paths:
            try:
                print(f"📖 파일 처리 중: {path}")
                
                # Docling으로 PDF 변환 - 여러 방법 시도
                doc_result = converter.convert(path)
                
                # 방법 1: 마크다운 추출
                raw_text = doc_result.document.export_to_markdown()
                print(f"📝 [방법1] 마크다운 추출 결과 (길이: {len(raw_text)}자)")
                print(f"   미리보기: '{raw_text[:200]}...'")
                
                # 방법 2: 텍스트 직접 추출
                try:
                    raw_text2 = doc_result.document.text
                    print(f"📝 [방법2] 텍스트 직접 추출 결과 (길이: {len(raw_text2)}자)")
                    print(f"   미리보기: '{raw_text2[:200]}...'")
                    
                    # 텍스트 직접 추출이 더 나으면 사용
                    if len(raw_text2) > len(raw_text) and not raw_text2.startswith('<!--'):
                        raw_text = raw_text2
                        print("✅ 텍스트 직접 추출 방식 사용")
                except Exception as e:
                    print(f"⚠️ 텍스트 직접 추출 실패: {e}")
                
                # 방법 3: 페이지별 텍스트 추출
                try:
                    pages_text = []
                    for page in doc_result.document.pages:
                        page_text = page.text
                        if page_text and not page_text.startswith('<!--'):
                            pages_text.append(page_text)
                    
                    if pages_text:
                        raw_text3 = '\n\n'.join(pages_text)
                        print(f"📝 [방법3] 페이지별 텍스트 추출 결과 (길이: {len(raw_text3)}자)")
                        print(f"   미리보기: '{raw_text3[:200]}...'")
                        
                        # 페이지별 추출이 더 나으면 사용
                        if len(raw_text3) > len(raw_text) and not raw_text3.startswith('<!--'):
                            raw_text = raw_text3
                            print("✅ 페이지별 텍스트 추출 방식 사용")
                except Exception as e:
                    print(f"⚠️ 페이지별 텍스트 추출 실패: {e}")
                
                # 방법 4: 마크다운에서 HTML 태그 제거
                if raw_text.startswith('<!--'):
                    print("🔄 마크다운에서 HTML 태그 제거 시도...")
                    try:
                        # HTML 주석과 태그 제거
                        import re
                        cleaned_text = re.sub(r'<!--.*?-->', '', raw_text, flags=re.DOTALL)
                        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
                        cleaned_text = re.sub(r'^\s*-\s*', '', cleaned_text, flags=re.MULTILINE)
                        cleaned_text = re.sub(r'^\s*$', '', cleaned_text, flags=re.MULTILINE)
                        cleaned_text = '\n'.join(line for line in cleaned_text.split('\n') if line.strip())
                        
                        if cleaned_text and len(cleaned_text) > 50:
                            raw_text = cleaned_text
                            print(f"✅ HTML 태그 제거 성공 (길이: {len(raw_text)}자)")
                            print(f"   미리보기: '{raw_text[:200]}...'")
                        else:
                            print("⚠️ HTML 태그 제거 후 텍스트가 너무 짧음")
                    except Exception as e:
                        print(f"⚠️ HTML 태그 제거 실패: {e}")
                
                if not raw_text.strip() or raw_text.startswith('<!--'):
                    print(f"❌ 모든 Docling 방법으로도 텍스트 추출 실패")
                    print(f"⚠️ PDF 파일 자체에 문제가 있을 수 있음")
                    continue
                
                # 디버깅: 추출된 텍스트 일부 출력
                print(f"📝 최종 추출된 텍스트 미리보기 (처음 500자):")
                print(f"'{raw_text[:500]}...'")
                print(f"📊 총 텍스트 길이: {len(raw_text)} 문자")
                
                # 1단/2단 구분 및 처리
                blocks = self._process_pdf_text(raw_text, path)
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
    
    def _process_pdf_text(self, raw_text: str, pdf_path: str) -> List[str]:
        """PDF 텍스트를 1단/2단 구분하여 처리"""
        print("🔍 [레이아웃 분석] 1단/2단 구조 파악 중...")
        
        # 1단 구조로 먼저 시도
        blocks = self._split_problem_blocks(raw_text)
        
        # 1단 파싱 결과가 부족하면 2단 구조로 재시도
        if len(blocks) <= 2:
            print("⚠️ 1단 파싱 결과 부족 - 2단 구조로 재시도")
            try:
                # 2단 재정렬
                reordered_text = self._reorder_two_columns_with_pdfminer(pdf_path)
                reordered_text = self.normalize_docling_markdown(reordered_text)
                
                # 2단 재정렬 후 파싱 시도
                blocks = self._split_problem_blocks(reordered_text)
                print(f"🔄 2단 재정렬 후: {len(blocks)}개 블록")
                
                # 여전히 부족하면 숫자 헤더 폴백 사용
                if len(blocks) <= 2:
                    print("⚠️ 2단 파싱도 부족 - 숫자 헤더 폴백 사용")
                    blocks = self._split_problem_blocks_without_keyword(reordered_text)
                    print(f"🔄 폴백 후: {len(blocks)}개 블록")
                    
            except Exception as e:
                print(f"⚠️ 2단 처리 실패: {e}")
                # 2단 처리 실패 시 원본 텍스트로 폴백
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
                
                # x 분할 기준값을 페이지 폭의 중간쯤으로 설정 (휴리스틱)
                # LTTextContainer의 bbox=(x0,y0,x1,y1)
                # 먼저 평균 x0를 보고 중앙값을 추정하는 보정 로직
                xs = []
                for el in page_layout:
                    if isinstance(el, LTTextContainer):
                        xs.append(el.bbox[0])
                
                if not xs:
                    continue
                    
                # 중앙값 계산 (더 안정적인 방법)
                sorted_xs = sorted(xs)
                mid = sorted_xs[len(sorted_xs)//2]
                
                # 좌우 컬럼 분리
                for el in page_layout:
                    if isinstance(el, LTTextContainer):
                        (x0, y0, x1, y1) = el.bbox
                        text = el.get_text().strip()
                        if text:  # 빈 텍스트 제외
                            (left if x0 < mid else right).append((y1, text))
                
                # y1 기준으로 위→아래 정렬 (y1이 클수록 위쪽)
                left.sort(key=lambda t: -t[0])
                right.sort(key=lambda t: -t[0])
                
                # 왼쪽 전체 → 오른쪽 전체 순으로 합치기
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
        lines = text.split('\n')
        n = len(lines)
        
        # 문항 헤더 후보 인덱스 수집
        _QHEAD_CAND = re.compile(r'(?m)^\s*(\d{1,3})[.)]\s+\S')
        candidates = []
        
        for i, ln in enumerate(lines):
            m = _QHEAD_CAND.match(ln or '')
            if m:
                num = int(m.group(1))
                # 보기 번호가 아닌지 확인 (1), 2), 3), 4)는 보기)
                if not re.match(r'^\s*\d+\)\s*', ln):
                    # 추가 검증: 실제 문제 내용이 있는지 확인
                    if len(ln.strip()) > 10:  # 최소 10자 이상
                        candidates.append((i, num))
                        print(f"🔍 [폴백] 라인 {i}: '{ln[:50]}...' → 후보 번호 {num}")
                    else:
                        print(f"🔍 [폴백] 라인 {i}: '{ln[:50]}...' → 너무 짧아서 제외")
                else:
                    print(f"🔍 [폴백] 라인 {i}: '{ln[:50]}...' → 보기 번호로 판단하여 제외")
        
        print(f"🔍 [폴백] 총 후보 수: {len(candidates)}")
        
        # 전역 증가 시퀀스 + 섹션 리셋 허용으로 실제 헤더 선별
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
            
            # 섹션 리셋: num==1이고, 최근 헤더에서 충분히 떨어져 있거나 섹션 느낌의 라인 존재 시 허용
            if num == 1:
                window = '\n'.join(lines[max(0, i-3): i+1])
                if (i - last_header_idx) >= 8 or re.search(r'(Ⅰ|Ⅱ|III|과목|파트|SECTION)', window):
                    headers.append(i)
                    prev_num = 1
                    last_header_idx = i
                    print(f"✅ [폴백] 라인 {i}: 번호 {num} - 섹션 리셋으로 헤더 선택")
                    continue
                else:
                    print(f"❌ [폴백] 라인 {i}: 번호 {num} - 섹션 리셋 조건 불충족 (거리: {i - last_header_idx})")
            else:
                print(f"❌ [폴백] 라인 {i}: 번호 {num} - 순차 증가 아님 (예상: {prev_num + 1})")
        
        # 헤더가 하나도 안 잡히면 폴백 전략 사용
        if not headers:
            print(f"❌ [폴백] 헤더가 하나도 선택되지 않음 - 폴백 전략 사용")
            if candidates:
                print(f"🔄 [폴백] 순차 조건 없이 모든 후보를 헤더로 사용")
                headers = [i for i, num in candidates]
            else:
                # 기본 번호 패턴으로 분할
                print(f"🔄 [폴백] 기본 번호 패턴으로 분할")
                simple_pattern = re.compile(r'(?m)^\s*(\d{1,2})\.\s+')
                for i, ln in enumerate(lines):
                    if simple_pattern.match(ln or ''):
                        # 보기 번호가 아닌지 확인
                        if not re.match(r'^\s*\d+\)\s*', ln):
                            headers.append(i)
                            print(f"📌 [폴백] 라인 {i}: '{ln[:30]}...' → 헤더 추가")
                        else:
                            print(f"📌 [폴백] 라인 {i}: '{ln[:30]}...' → 보기 번호로 판단하여 제외")
            
            if not headers:
                print(f"❌ [폴백 실패] 전체를 1개 블록으로 처리")
                return [text] if text.strip() else []
        
        print(f"✅ [폴백] 최종 선택된 헤더 수: {len(headers)}")
        
        # 헤더 범위로 블록 만들기
        headers.append(n)  # sentinel
        blocks = []
        for a, b in zip(headers[:-1], headers[1:]):
            blk = '\n'.join(lines[a:b]).strip()
            if blk:
                blocks.append(blk)
                print(f"📦 [폴백] 블록 {len(blocks)}: 라인 {a}-{b-1} ({len(blk)}자)")
        
        print(f"🎯 [폴백] 최종 블록 수: {len(blocks)}")
        return blocks
    
    def _split_problem_blocks(self, raw_text: str) -> List[str]:
        """텍스트를 문제 블록으로 분할 (실제 문제 헤더 기반)"""
        print("🔍 [구조 분석] 실제 문제 헤더 기반으로 파싱 방식 결정")
        
        lines = raw_text.split('\n')
        
        # 디버깅: 전체 라인 구조 분석
        print(f"📊 전체 라인 수: {len(lines)}")
        print("🔍 라인별 내용 분석 (처음 50줄):")
        for i, line in enumerate(lines[:50]):
            if line.strip():
                print(f"   라인 {i+1:2d}: '{line.strip()}'")
        
        # 숫자로 시작하는 라인들 찾기
        print("\n🔍 숫자로 시작하는 라인들:")
        number_lines = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped and re.match(r'^\d+\.', line_stripped):
                number_lines.append((i, line_stripped))
                print(f"   라인 {i+1:2d}: '{line_stripped[:100]}...'")
        
        print(f"📊 숫자로 시작하는 라인 수: {len(number_lines)}")
        
        # 실제 문제 헤더 패턴들 (우선순위 순)
        problem_header_patterns = [
            r'^\s*##\s*문제\s*(\d+)\s*\.\s*',   # "## 문제 1." (마크다운 헤더)
            r'^\s*#+\s*문제\s*(\d+)\s*\.\s*',   # "# 문제 1.", "### 문제 1." 등
            r'^\s*문제\s*(\d+)\s*\.\s*',        # "문제 1." (점만)
            r'^\s*(\d+)\s*\.\s*',                # "1." (숫자. + 공백)
            r'^\s*(\d+)\s*\.\s*\S',             # "1. 텍스트" (숫자. + 공백 + 텍스트)
            r'^\s*Q\s*(\d+)\s*\.\s*',           # "Q1." (점만)
            r'^\s*\[(\d+)\]\s*',                 # "[1]"
            # 마크다운 헤더 안의 숫자 패턴 추가
            r'^\s*#+\s*.*?(\d+)\s*\.\s*',       # "## ... 1. ..." (마크다운 헤더 안의 숫자)
        ]
        
        # 보기 번호 패턴들 (문제 헤더가 아님)
        option_patterns = [
            r'^\s*(\d+)\.\s*\1\.\s*',           # "4. 4." (중복 번호)
            r'^\s*(\d+)\s*[)]\s*',              # "1)", "2)" (보기 번호 - 괄호만)
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
    
    def normalize_docling_markdown(self, md: str) -> str:
        """Docling 마크다운 정규화"""
        s = md
        s = re.sub(r'(?m)^\s*(\d+)\.\s*\1\.\s*', r'\1. ', s)  # '1. 1.' -> '1.'
        s = re.sub(r'(?m)^\s*(\d+)\s*\.\s*', r'\1. ', s)      # '1 . ' -> '1. '
        s = re.sub(r'[ \t]+', ' ', s).replace('\r', '')
        return s.strip()
    
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
