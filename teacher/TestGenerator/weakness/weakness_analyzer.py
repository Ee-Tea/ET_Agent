import json
import re
import sys
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from pathlib import Path

# Add the TestGenerator directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import WEAKNESS_ANALYSIS_PROMPT

class WeaknessAnalyzer:
    """학습자 취약점 분석을 담당하는 클래스"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_weakness_with_llm(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM을 사용하여 분석 데이터에서 취약점을 분석하고 학습 개념을 추출합니다.
        
        Args:
            analysis_data: 분석 결과 데이터
            
        Returns:
            취약점 분석 결과
        """
        try:
            # 분석 데이터를 텍스트로 변환
            analysis_text = self._convert_analysis_to_text(analysis_data)
            
            analysis_prompt = PromptTemplate(
                input_variables=["analysis_text"],
                template=WEAKNESS_ANALYSIS_PROMPT
            )
            
            prompt = analysis_prompt.format(analysis_text=analysis_text)
            
            response = self.llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # JSON 추출
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                weakness_analysis = json.loads(match.group(0))
                return weakness_analysis
            else:
                return {"error": "LLM 분석 결과를 파싱할 수 없습니다."}
                
        except Exception as e:
            return {"error": f"취약점 LLM 분석 중 오류: {str(e)}"}

    def _convert_analysis_to_text(self, analysis_data: Dict[str, Any]) -> str:
        """분석 데이터를 텍스트로 변환"""
        try:
            text_parts = []
            
            # overall_assessment에서 정보 추출
            overall = analysis_data.get("analysis", {}).get("overall_assessment", {})
            if overall.get("weaknesses"):
                text_parts.append(f"취약점: {overall['weaknesses']}")
            if overall.get("strengths"):
                text_parts.append(f"강점: {overall['strengths']}")
            
            # detailed_analysis에서 정보 추출
            detailed = analysis_data.get("analysis", {}).get("detailed_analysis", [])
            for item in detailed:
                concept_path = item.get("concept_path", "")
                mistake_type = item.get("mistake_type", "")
                analysis = item.get("analysis", "")
                
                detail_text = f"개념경로: {concept_path}, 실수유형: {mistake_type}, 분석: {analysis}"
                text_parts.append(detail_text)
            
            return "\n\n".join(text_parts)
            
        except Exception:
            # 원본 데이터를 문자열로 변환
            return str(analysis_data)

    def extract_weakness_concepts_from_analysis(self, analysis_data: Dict[str, Any]) -> List[str]:
        """
        분석 결과 JSON에서 취약점 개념들을 추출합니다.
        
        Args:
            analysis_data: 분석 결과 딕셔너리
            
        Returns:
            취약점 개념 리스트
        """
        try:
            weakness_concepts = []
            
            # detailed_analysis에서 개념 추출
            detailed_analysis = analysis_data.get("analysis", {}).get("detailed_analysis", [])
            for item in detailed_analysis:
                analysis_text = item.get("analysis", "")
                if analysis_text:
                    # 분석 텍스트에서 핵심 개념 추출
                    concepts = self._extract_concepts_from_text(analysis_text)
                    weakness_concepts.extend(concepts)
            
            # 중복 제거 및 정리
            unique_concepts = list(set(weakness_concepts))
            
            # 빈 문자열이나 너무 짧은 개념 제거
            filtered_concepts = [concept for concept in unique_concepts if concept and len(concept.strip()) > 1]
            
            # 의미 있는 기술 개념만 필터링
            meaningful_concepts = []
            for concept in filtered_concepts:
                # 기술 용어 사전에 있는 개념들 우선 선택
                if any(tech_term.lower() in concept.lower() for tech_term in [
                    "자료 흐름도", "dfd", "미들웨어", "middleware", "프로세스", "process",
                    "자료 저장소", "data store", "종단점", "terminator", "sql", "정규화",
                    "uml", "다이어그램", "트랜잭션", "보안", "네트워크", "알고리즘",
                    "요구사항", "설계", "개발", "테스트", "구현", "모듈", "인터페이스",
                    "객체지향", "패턴", "데이터베이스", "관계형", "정규화", "무결성"
                ]):
                    meaningful_concepts.append(concept)
                # 3글자 이상의 한글 개념 중 의미 있는 것들
                elif len(concept) >= 3 and all(ord(c) > 127 for c in concept):
                    # 일반적인 조사나 부사 제외
                    if concept not in ["대한", "설명을", "책임자에", "유형에", "특징을", "학생은", "이해하지", "오해했습니다", "원칙과", "충분히", "통한", "위한", "있는", "하는", "되는", "있는", "하는", "되는"]:
                        meaningful_concepts.append(concept)
            
            # 의미 있는 개념이 없으면 기본 기술 개념 반환
            if not meaningful_concepts:
                meaningful_concepts = [
                    "자료 흐름도", "미들웨어", "프로세스", "자료 저장소", "SQL", "정규화",
                    "UML", "다이어그램", "트랜잭션", "보안", "네트워크", "알고리즘"
                ]
            
            return meaningful_concepts[:10]  # 최대 10개까지 반환
            
        except Exception as e:
            print(f"취약점 개념 추출 중 오류: {e}")
            return []

    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """
        텍스트에서 핵심 개념을 추출합니다.
        
        Args:
            text: 분석 텍스트
            
        Returns:
            추출된 개념 리스트
        """
        try:
            concepts = []
            
            # 괄호 안의 영문 용어 추출 (예: "자료 흐름도(DFD)", "미들웨어(Middleware)")
            bracket_pattern = r'([가-힣\s]+)\(([A-Za-z\s]+)\)'
            bracket_matches = re.findall(bracket_pattern, text)
            for korean, english in bracket_matches:
                concepts.append(korean.strip())
                concepts.append(english.strip())
            
            # 일반적인 기술 용어들 추출
            tech_terms = [
                "자료 흐름도", "DFD", "미들웨어", "Middleware", "프로세스", "Process",
                "자료 저장소", "Data Store", "종단점", "Terminator", "SQL", "정규화",
                "UML", "다이어그램", "트랜잭션", "보안", "네트워크", "알고리즘",
                "요구사항", "설계", "개발", "테스트", "구현", "모듈", "인터페이스",
                "객체지향", "패턴", "데이터베이스", "관계형", "무결성", "인덱스",
                "뷰", "트리거", "저장프로시저", "트랜잭션", "동시성", "데드락"
            ]
            
            for term in tech_terms:
                if term.lower() in text.lower():
                    concepts.append(term)
            
            # 의미 있는 한글 개념 추출 (3글자 이상, 조사/부사 제외)
            meaningful_korean_pattern = r'[가-힣]{3,}'
            korean_matches = re.findall(meaningful_korean_pattern, text)
            
            # 의미 없는 조사/부사 목록
            meaningless_words = {
                "대한", "설명을", "책임자에", "유형에", "특징을", "학생은", "이해하지", 
                "오해했습니다", "원칙과", "충분히", "통한", "위한", "있는", "하는", 
                "되는", "있습니다", "합니다", "됩니다", "입니다", "것입니다", "것을",
                "것이", "것은", "것에", "것으로", "것을", "것이", "것은", "것에"
            }
            
            for match in korean_matches:
                if match not in concepts and match not in meaningless_words:
                    # 기술적 맥락에서 의미 있는 단어인지 확인
                    if any(tech_word in match for tech_word in ["설계", "개발", "테스트", "구현", "분석", "관리", "보안", "네트워크", "데이터", "시스템", "프로그램", "소프트웨어", "하드웨어"]):
                        concepts.append(match)
            
            return concepts
            
        except Exception as e:
            print(f"개념 추출 중 오류: {e}")
            return []

    def load_analysis_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        분석 결과 JSON 파일을 로드합니다.
        
        Args:
            file_path: 분석 결과 JSON 파일 경로
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"분석 파일 로드 실패: {e}")
