import json
import re
import sys
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from pathlib import Path

# Add the TestGenerator directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import MAX_GENERATION_ATTEMPTS

from typing_extensions import TypedDict
from typing import List, Dict, Any

class GraphState(TypedDict):
    """그래프 상태 정의"""
    query: str
    documents: List
    context: str
    quiz_questions: List[Dict[str, Any]]
    quiz_count: int
    difficulty: str
    error: str
    used_sources: List[str]
    generation_attempts: int
    target_quiz_count: int
    subject_area: str
    validated_questions: List[Dict[str, Any]]
    weakness_analysis: Dict[str, Any]
    weakness_concepts: List[str]

class QuizWorkflow:
    """문제 생성 워크플로우를 관리하는 클래스"""
    
    def __init__(self, llm, rag_engine):
        self.llm = llm
        self.rag_engine = rag_engine
        self.workflow = None
        self._build_graph()
    
    def _build_graph(self):
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("generate_quiz", self._generate_quiz_incremental)
        workflow.add_node("validate_quiz", self._validate_quiz_incremental)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "prepare_context")
        workflow.add_edge("prepare_context", "generate_quiz")
        workflow.add_edge("generate_quiz", "validate_quiz")
        
        workflow.add_conditional_edges(
            "validate_quiz",
            self._check_completion,
            {
                "generate_more": "generate_quiz",
                "complete": END
            }
        )
        
        self.workflow = workflow.compile()
    
    def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """워크플로우 실행"""
        return self.workflow.invoke(initial_state)
    
    def _retrieve_documents(self, state: GraphState) -> GraphState:
        """문서 검색 및 출처 분석 노드"""
        try:
            query = state["query"]
            subject_area = state.get("subject_area", "")
            weakness_concepts = state.get("weakness_concepts", [])
            
            print(f"🔍 문서 검색 시작: query='{query}', subject_area='{subject_area}', weakness_concepts={weakness_concepts}")
            
            # RAG 엔진을 사용하여 문서 검색
            result = self.rag_engine.retrieve_documents(
                query=query,
                subject_area=subject_area,
                weakness_concepts=weakness_concepts
            )
            
            print(f"📋 RAG 엔진 결과: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            if isinstance(result, dict):
                print(f"   - documents 키 존재: {'documents' in result}")
                print(f"   - documents 타입: {type(result.get('documents'))}")
                if 'documents' in result:
                    print(f"   - documents 길이: {len(result['documents']) if result['documents'] else 0}")
            
            if "error" in result:
                print(f"❌ RAG 엔진 오류: {result['error']}")
                return {**state, "error": result["error"]}
            
            if "documents" not in result:
                print(f"❌ 'documents' 키가 결과에 없습니다!")
                return {**state, "error": "RAG 엔진에서 'documents' 키를 찾을 수 없습니다."}
            
            print(f"✅ 문서 검색 성공: {len(result['documents'])}개 문서, {len(result.get('used_sources', []))}개 소스")
            
            return {
                **state, 
                "documents": result["documents"], 
                "used_sources": result["used_sources"]
            }
        except Exception as e:
            print(f"❌ 문서 검색 중 예외 발생: {e}")
            return {**state, "error": f"문서 검색 오류: {e}"}

    def _prepare_context(self, state: GraphState) -> GraphState:
        """컨텍스트 준비 노드"""
        documents = state["documents"]
        weakness_concepts = state.get("weakness_concepts", [])
        
        # RAG 엔진을 사용하여 컨텍스트 준비
        context = self.rag_engine.prepare_context(
            documents=documents,
            weakness_concepts=weakness_concepts
        )
        
        return {**state, "context": context}

    def _generate_quiz_incremental(self, state: GraphState) -> GraphState:
        """취약점 집중 문제 생성"""
        try:
            context = state["context"]
            target_quiz_count = state.get("target_quiz_count", 5)
            validated_questions = state.get("validated_questions", [])
            subject_area = state.get("subject_area", "")
            weakness_concepts = state.get("weakness_concepts", [])
            difficulty = state.get("difficulty", "중급")
            needed_count = target_quiz_count - len(validated_questions)
            
            if needed_count <= 0:
                return {**state, "quiz_questions": validated_questions}
            
            if not context.strip():
                return {**state, "error": "검색된 문서 내용이 없습니다."}
            
            generate_count = max(needed_count, 3)
            
            # 취약점 개념 집중 문제 생성 프롬프트
            if weakness_concepts:
                weakness_focus = f"""학습자가 특히 어려워하는 취약점 개념들: {', '.join(weakness_concepts)}
이 개념들에 대해 학습자의 이해도를 높일 수 있는 문제를 집중적으로 출제하세요.

취약점 개념별 문제 생성 지침:
- 개념의 정의나 특징을 묻는 기본 문제
- 개념을 실제 상황에 적용하는 응용 문제  
- 비슷한 개념들과의 차이점을 구분하는 문제
- 개념의 구성요소나 절차를 묻는 문제"""
                
                template_text = """당신은 정보처리기사 출제 전문가입니다. 아래 문서 내용을 바탕으로 학습자의 취약점을 보강할 수 있는 {subject_area} 관련 객관식 문제 {quiz_count}개를 생성하세요.

{weakness_focus}

난이도: {difficulty}

[문서 내용]
{context}

[출제 기준]
1. 정보처리기사 시험 출제 기준에 맞는 4지선다 객관식 문제
2. 취약점 개념에 대한 정확한 이해를 확인할 수 있는 문제
3. 실무에서 활용 가능한 실용적인 문제
4. 명확한 정답과 해설 포함

반드시 JSON 형식으로만 출력하세요:

{{
  "questions": [
    {{
      "question": "구체적이고 명확한 문제 내용",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "answer": "정답 번호(1, 2, 3, 4 중 하나)",
      "explanation": "정답에 대한 상세한 해설과 취약점 개념 설명",
      "weakness_focus": "집중한 취약점 개념명"
    }}
  ]
}}"""
            else:
                weakness_focus = ""
                template_text = """아래 문서 내용을 바탕으로 {subject_area} 과목의 객관식 문제 {quiz_count}개를 반드시 생성하세요.
각 문제는 4지선다, 정답 번호와 간단한 해설을 포함해야 하며, 반드시 JSON만 출력하세요.

[문서]
{context}

[출력 예시]
{{
  "questions": [
    {{
      "question": "문제 내용",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "answer": "정답 번호(예: 1)",
      "explanation": "간단한 해설"
    }}
  ]
}}"""
            
            prompt_template = PromptTemplate(
                input_variables=["context", "quiz_count", "subject_area", "weakness_focus", "difficulty"],
                template=template_text
            )
            
            prompt = prompt_template.format(
                context=context,
                quiz_count=generate_count,
                subject_area=subject_area,
                weakness_focus=weakness_focus,
                difficulty=difficulty
            )
            
            self.llm.temperature = 0.2
            self.llm.max_tokens = 1500
            response = self.llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            new_questions = self._parse_quiz_response(response_content, subject_area)
            if not new_questions:
                return {**state, "error": "유효한 문제를 생성하지 못했습니다."}
            
            return {
                **state,
                "quiz_questions": new_questions,
                "validated_questions": validated_questions,
                "generation_attempts": state.get("generation_attempts", 0) + 1
            }
        except Exception as e:
            return {**state, "error": f"문제 생성 중 오류 발생: {e}"}

    def _validate_quiz_incremental(self, state: GraphState) -> GraphState:
        """취약점 집중 문제 검증"""
        subject_area = state.get("subject_area", "")
        weakness_concepts = state.get("weakness_concepts", [])
        
        previously_validated = state.get("validated_questions", [])
        new_questions = state.get("quiz_questions", [])
        context = state.get("context", "")
        target_quiz_count = state.get("target_quiz_count", 5)
        generation_attempts = state.get("generation_attempts", 0)

        if not new_questions:
            return {**state, "validated_questions": previously_validated}

        newly_validated = []
        
        # 취약점 집중 검증 프롬프트
        validation_template = """당신은 정보처리기사 출제 전문가입니다. 아래 제공된 '문서 내용'에 근거하여 '퀴즈 문제'가 취약점 보강에 적합한지 평가해주세요.

[문서 내용]
{context}

[취약점 개념들]
{weakness_concepts}

[평가할 퀴즈 문제]
{question_data}

[평가 기준]
1. 문제가 문서 내용에서 직접적으로 확인 가능한가?
2. 취약점 개념에 대한 이해를 확인할 수 있는가?
3. 정보처리기사 시험 수준에 적합한 난이도인가?
4. 4개 선택지가 명확하고 정답이 유일한가?
5. 해설이 취약점 개념을 잘 설명하고 있는가?

평가 결과를 JSON 형식으로 출력하세요:
{{
  "is_valid": true/false,
  "reason": "평가 이유 설명",
  "weakness_relevance": true/false
}}"""
        
        validation_prompt_template = PromptTemplate(
            input_variables=["context", "question_data", "weakness_concepts"],
            template=validation_template
        )
        
        needed = target_quiz_count - len(previously_validated)
        weakness_concepts_str = ', '.join(weakness_concepts) if weakness_concepts else "일반"
        
        for i, q in enumerate(new_questions):
            if len(newly_validated) >= needed:
                break
                
            try:
                question_str = json.dumps(q, ensure_ascii=False, indent=2)
                prompt = validation_prompt_template.format(
                    context=context[:4000], 
                    question_data=question_str,
                    weakness_concepts=weakness_concepts_str
                )
                
                response = self.llm.invoke(prompt)
                response_str = response.content if hasattr(response, 'content') else str(response)
                
                match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if not match:
                    continue

                validation_result = json.loads(match.group(0))

                # 유효성과 취약점 관련성 모두 확인
                if (validation_result.get("is_valid") is True and 
                    validation_result.get("weakness_relevance", True) is True):
                    newly_validated.append(q)

            except Exception:
                continue
        
        all_validated = previously_validated + newly_validated
        
        need_more_questions = (len(all_validated) < target_quiz_count and 
                              generation_attempts < MAX_GENERATION_ATTEMPTS)
        
        return {
            **state,
            "validated_questions": all_validated,
            "quiz_questions": all_validated,
            "need_more_questions": need_more_questions
        }

    def _check_completion(self, state: GraphState) -> str:
        """문제 생성 완료 여부를 체크하는 조건부 노드"""
        validated_count = len(state.get("validated_questions", []))
        target_count = state.get("target_quiz_count", 5)
        generation_attempts = state.get("generation_attempts", 0)
        
        if validated_count >= target_count:
            return "complete"
        elif generation_attempts < MAX_GENERATION_ATTEMPTS:
            return "generate_more"
        else:
            return "complete"

    def _parse_quiz_response(self, response: str, subject_area: str = "") -> List[Dict[str, Any]]:
        """응답에서 JSON 형식의 문제를 파싱"""
        try:
            json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1)
            else:
                json_str_match = re.search(r'\{.*\}', response.strip(), re.DOTALL)
                if not json_str_match:
                    return []
                json_str = json_str_match.group(0)

            json_str = json_str.replace('\\u312f', '').replace('\\n', ' ')
            data = json.loads(json_str)
            
            if "questions" not in data or not isinstance(data["questions"], list):
                return []
            
            for question in data["questions"]:
                if "options" in question and isinstance(question["options"], list):
                    numbered_options = []
                    for i, option_text in enumerate(question["options"], 1):
                        cleaned_text = re.sub(r'^\s*\d+\.\s*', '', option_text).strip()
                        numbered_options.append(f"  {i}. {cleaned_text}")
                    question["options"] = numbered_options
                
                if "subject" not in question:
                    question["subject"] = subject_area
            
            return data.get("questions", [])
        except:
            return []
