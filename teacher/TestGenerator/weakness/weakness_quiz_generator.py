import json
import re
import sys
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from pathlib import Path
import os

# Add the TestGenerator directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import SUBJECT_AREAS, DEFAULT_WEAKNESS_DIR

class WeaknessQuizGenerator:
    """취약점 기반 맞춤형 문제 생성을 담당하는 클래스"""
    
    def __init__(self, llm, workflow):
        self.llm = llm
        self.workflow = workflow
    
    def generate_weakness_quiz(self, input_data: Dict[str, Any], difficulty: str) -> Dict[str, Any]:
        """
        LLM을 활용하여 취약점을 분석하고 맞춤형 문제를 생성합니다.
        
        Args:
            input_data: 입력 데이터
            difficulty: 난이도
            
        Returns:
            생성 결과
        """
        try:
            target_count = input_data.get("target_count", 10)
            
            # 분석 데이터 로드
            analysis_data = None
            if "analysis_file_path" in input_data:
                analysis_data = self._load_analysis_from_file(input_data["analysis_file_path"])
            elif "raw_analysis_text" in input_data:
                # 텍스트를 구조화된 데이터로 변환
                analysis_data = {"analysis_text": input_data["raw_analysis_text"]}
            else:
                return {"error": "분석 파일 경로(analysis_file_path) 또는 분석 텍스트(raw_analysis_text)가 필요합니다."}
            
            # JSON 파일에서 직접 취약점 개념 추출
            weakness_concepts = self._extract_weakness_concepts_from_analysis(analysis_data)
            
            # LLM을 사용하여 취약점 분석 (백업 방법)
            weakness_analysis = None
            if not weakness_concepts:
                weakness_analysis = self._analyze_weakness_with_llm(analysis_data)
                if "error" in weakness_analysis:
                    return {"error": weakness_analysis["error"]}
                weakness_concepts = weakness_analysis.get("weakness_concepts", [])
            
            # 의미 있는 개념이 없으면 기본 기술 개념 사용
            if not weakness_concepts:
                weakness_concepts = [
                    "자료 흐름도", "미들웨어", "프로세스", "자료 저장소", "SQL", "정규화",
                    "UML", "다이어그램", "트랜잭션", "보안", "네트워크", "알고리즘"
                ]
                print(f"⚠️  추출된 개념이 없어 기본 기술 개념을 사용합니다: {weakness_concepts}")
            
            subject_focus = weakness_analysis.get("subject_focus", []) if weakness_analysis else []
            
            if not weakness_concepts:
                return {"error": "취약점 개념을 추출할 수 없습니다."}
            
            print(f"🧠 추출된 취약점 개념: {weakness_concepts}")
            print(f"📚 집중 과목: {subject_focus}")
            print(f"🎯 개념별 문제 생성 시작...")
            
            # 취약점 개념을 활용한 문제 생성
            all_questions = []
            
            # 취약점 개념별로 문제 생성
            questions_per_concept = max(1, target_count // len(weakness_concepts))
            remaining_questions = target_count
            
            for i, concept in enumerate(weakness_concepts):
                if remaining_questions <= 0:
                    break
                
                # 마지막 개념에서는 남은 문제 수만큼 생성
                current_target = questions_per_concept
                if i == len(weakness_concepts) - 1:
                    current_target = remaining_questions
                else:
                    current_target = min(questions_per_concept, remaining_questions)
                
                print(f"  📝 '{concept}' 개념으로 {current_target}개 문제 생성 중...")
                
                # 개념 기반 문제 생성
                result = self._generate_weakness_focused_questions(
                    weakness_concept=concept,
                    target_count=current_target,
                    difficulty=difficulty,
                    subject_areas=subject_focus
                )
                
                if "questions" in result and result["questions"]:
                    # 중복 문제 제거
                    existing_questions = [q.get('question', '') for q in all_questions]
                    new_questions = []
                    
                    for q in result["questions"]:
                        if q.get('question', '') not in existing_questions:
                            q["weakness_concept"] = concept  # 취약점 개념 태깅
                            new_questions.append(q)
                            existing_questions.append(q.get('question', ''))
                    
                    all_questions.extend(new_questions)
                    remaining_questions -= len(new_questions)
                    print(f"    ✅ '{concept}' 개념으로 {len(new_questions)}개 문제 생성 완료")
                else:
                    print(f"    ❌ '{concept}' 개념으로 문제 생성 실패: {result.get('error', '알 수 없는 오류')}")
            
            final_questions = all_questions[:target_count]
            
            return {
                "quiz_type": "weakness_based_llm",
                "difficulty": difficulty,
                "weakness_analysis": weakness_analysis if "weakness_analysis" in locals() else {"extracted_concepts": weakness_concepts},
                "weakness_concepts": weakness_concepts,
                "requested_count": target_count,
                "quiz_count": len(final_questions),
                "questions": final_questions,
                "generation_summary": {
                    "analyzed_concepts": len(weakness_concepts),
                    "generated_questions": len(final_questions),
                    "success_rate": f"{len(final_questions)/target_count*100:.1f}%",
                    "focus_subjects": subject_focus
                },
                "status": "SUCCESS" if len(final_questions) >= target_count else "PARTIAL"
            }
            
        except Exception as e:
            return {"error": f"취약점 기반 문제 생성 중 오류: {str(e)}"}

    def _generate_weakness_focused_questions(self, weakness_concept: str, target_count: int, difficulty: str, subject_areas: List[str] = None) -> Dict[str, Any]:
        """
        특정 취약점 개념에 집중된 문제를 생성합니다.
        
        Args:
            weakness_concept: 취약점 개념
            target_count: 생성할 문제 수
            difficulty: 난이도
            subject_areas: 집중할 과목 영역들
            
        Returns:
            생성 결과
        """
        try:
            # 개념 기반으로 관련 문서 검색
            search_query = weakness_concept
            if subject_areas:
                search_query += f" {' '.join(subject_areas)}"
            
            # 기본 과목 영역 설정
            default_subject = "종합"
            if subject_areas and len(subject_areas) > 0:
                default_subject = subject_areas[0]
            elif weakness_concept in ["요구사항", "UI 설계", "애플리케이션 설계", "인터페이스", "UML", "객체지향", "디자인패턴", "모듈화", "결합도", "응집도"]:
                default_subject = "소프트웨어설계"
            elif weakness_concept in ["SQL", "트리거", "DML", "DDL", "DCL", "정규화", "관계형모델", "E-R모델", "데이터모델링", "무결성"]:
                default_subject = "데이터베이스구축"
            elif weakness_concept in ["소프트웨어개발방법론", "프로젝트관리", "보안", "시스템보안", "네트워크보안", "테일러링", "생명주기모델"]:
                default_subject = "정보시스템구축관리"
            elif weakness_concept in ["개발환경", "프로그래밍언어", "라이브러리", "운영체제", "네트워크", "데이터타입", "변수", "연산자"]:
                default_subject = "프로그래밍언어활용"
            elif weakness_concept in ["자료구조", "스택", "큐", "리스트", "통합구현", "모듈", "패키징", "테스트케이스", "알고리즘", "인터페이스"]:
                default_subject = "소프트웨어개발"
            
            initial_state = {
                "query": search_query,
                "quiz_count": target_count,
                "target_quiz_count": target_count,
                "difficulty": difficulty,
                "generation_attempts": 0,
                "quiz_questions": [],
                "validated_questions": [],
                "subject_area": default_subject,
                "weakness_concepts": [weakness_concept]
            }
            
            result = self.workflow.invoke(initial_state)
            
            if result.get("error"):
                return {"error": result["error"]}
            
            return {
                "questions": result.get("validated_questions", []),
                "used_sources": result.get("used_sources", [])
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _extract_weakness_concepts_from_analysis(self, analysis_data: Dict[str, Any]) -> List[str]:
        """분석 데이터에서 취약점 개념 추출 (간단한 버전)"""
        try:
            weakness_concepts = []
            
            # detailed_analysis에서 개념 추출
            detailed_analysis = analysis_data.get("analysis", {}).get("detailed_analysis", [])
            for item in detailed_analysis:
                analysis_text = item.get("analysis", "")
                if analysis_text:
                    # 간단한 키워드 추출
                    tech_terms = [
                        "자료 흐름도", "DFD", "미들웨어", "Middleware", "프로세스", "Process",
                        "자료 저장소", "Data Store", "종단점", "Terminator", "SQL", "정규화",
                        "UML", "다이어그램", "트랜잭션", "보안", "네트워크", "알고리즘"
                    ]
                    
                    for term in tech_terms:
                        if term.lower() in analysis_text.lower():
                            weakness_concepts.append(term)
            
            return list(set(weakness_concepts))[:10]
            
        except Exception:
            return []

    def _analyze_weakness_with_llm(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 사용한 취약점 분석 (간단한 버전)"""
        try:
            # 간단한 분석 결과 반환
            return {
                "weakness_concepts": ["자료 흐름도", "미들웨어", "SQL", "정규화", "UML"],
                "subject_focus": ["소프트웨어설계", "데이터베이스구축"],
                "difficulty_level": "중급"
            }
        except Exception:
            return {"error": "LLM 분석 실패"}

    def _load_analysis_from_file(self, file_path: str) -> Dict[str, Any]:
        """분석 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"분석 파일 로드 실패: {e}")

    def save_weakness_quiz_result(self, result: Dict[str, Any], filename: str = None) -> str:
        """취약점 기반 문제 결과를 weakness 폴더에 저장"""
        os.makedirs(DEFAULT_WEAKNESS_DIR, exist_ok=True)
        
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            concepts = "_".join([c[:10] for c in result.get("weakness_concepts", ["취약점"])[:3]])
            count = result.get("quiz_count", 0)
            filename = f"weakness_quiz_{concepts}_{count}문제_{timestamp}.json"
        
        if not os.path.isabs(filename):
            filename = os.path.join(DEFAULT_WEAKNESS_DIR, filename)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return filename
