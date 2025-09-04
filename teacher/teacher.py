import json
from pathlib import Path
import sys
from typing import Dict, Any, cast
import os
from dotenv import load_dotenv
from langsmith import traceable

from .agents.analysis.analysis_agent import AnalysisAgent, print_analysis_result
from .agents.base_agent import BaseAgent
from .agents.score.score_engine import ScoreEngine, print_score_result, ScoreResult

class Orchestrator:
    """
    전체 워크플로우를 관리하고, 사용자 요청에 따라 
    적절한 에이전트를 선택하고 실행하는 오케스트레이터 클래스입니다.
    """
    
    def __init__(self):
        # .env 파일에서 환경 변수 로드
        load_dotenv()
        # LangSmith 추적 환경 변수 확인
        if not os.getenv("LANGCHAIN_API_KEY=REDACTED("경고: LANGCHAIN_API_KEY=REDACTED(".env 파일에 키를 추가하거나 직접 환경 변수를 설정해주세요.")

        self.agents: Dict[str, BaseAgent] = {
            "analysis": AnalysisAgent(),
            "score": ScoreEngine(),  # ScoreEngine 등록
        }

    def get_available_agents(self) -> Dict[str, str]:
        """등록된 에이전트들의 이름과 설명을 반환합니다."""
        return {agent_key: agent.description for agent_key, agent in self.agents.items()}

    @traceable(name="Orchestrator Run")
    def run(self, agent_name: str, input_file_path: str):
        """
        지정된 에이전트를 실행하고, 파일 입출력을 처리합니다.
        
        Args:
            agent_name (str): 실행할 에이전트의 이름입니다.
            input_file_path (str): 에이전트에 전달할 입력 데이터 파일 경로입니다.
        """
        # 1. 에이전트 선택
        agent = self.agents.get(agent_name)
        if not agent:
            print(f"오류: '{agent_name}'이라는 이름의 에이전트를 찾을 수 없습니다.")
            sys.exit(1)
            
        # 2. 입력 파일 로드
        input_file = Path(input_file_path)
        if not input_file.exists():
            print(f"오류: 입력 파일을 찾을 수 없습니다. ({input_file})")
            sys.exit(1)
            
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            print(f"파일 '{input_file}' 로드 성공")
        except json.JSONDecodeError:
            print(f"오류: '{input_file}'이 올바른 JSON 형식이 아닙니다.")
            sys.exit(1)
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
            sys.exit(1)

        # 3. 에이전트 실행
        try:
            print(f"🚀 '{agent_name}' 에이전트 실행 시작...")
            result = agent.execute(input_data)
            
            # 결과 검증
            if not isinstance(result, dict):
                raise ValueError("에이전트가 올바른 형식의 결과를 반환하지 않았습니다.")
                
            if result.get("status") == "error":
                print(f"⚠️  에이전트 실행 중 내부 오류 발생: {result.get('error_message', '알 수 없는 오류')}")
            else:
                print(f"✅ '{agent_name}' 에이전트 실행 완료.")
                
        except Exception as e:
            print(f"❌ 에이전트 실행 중 오류 발생: {e}")
            # 오류 발생 시에도 기본 구조로 결과 생성
            result = {
                "status": "error",
                "error_message": str(e),
                "metadata": {
                    "total_problems": 0,
                    "correct_count": 0,
                    "score": 0
                },
                "grading": {"results": [], "details": []},
                "analysis": {},
                "raw_data": {}
            }
            
        # 4. 결과 처리 및 저장
        self.handle_result(result, agent_name, input_file)

    def handle_result(self, result: Dict[str, Any], agent_name: str, input_file: Path):
        """
        에이전트 실행 결과를 처리하고 저장합니다.
        
        Args:
            result (Dict[str, Any]): 에이전트 실행 결과.
            agent_name (str): 실행된 에이전트의 이름.
            input_file (Path): 원본 입력 파일의 경로.
        """
        # 결과 출력
        if agent_name == "analysis":
            print_analysis_result(result)
        elif agent_name == "score":
            print_score_result(cast(ScoreResult, result))
        else:
            # 다른 에이전트들의 결과 출력 로직
            print("\n--- 실행 결과 ---")
            if result.get("status") == "success":
                print("✅ 성공적으로 완료되었습니다.")
                if "metadata" in result:
                    print("\n📊 메타데이터:")
                    for key, value in result["metadata"].items():
                        print(f"  - {key}: {value}")
            else:
                print("❌ 오류가 발생했습니다.")
                if "error_message" in result:
                    print(f"오류 메시지: {result['error_message']}")
            
            print("\n전체 결과:")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        # 결과 파일 저장
        output_file = input_file.with_name(f"{input_file.stem}_{agent_name}_result.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 결과가 '{output_file}'에 저장되었습니다.")

        except Exception as e:
            print(f"❌ 결과 저장 중 오류 발생: {e}")
            print("결과가 저장되지 않았지만, 분석은 완료되었습니다.")


if __name__ == "__main__":
    # 명령줄 인자 파싱
    # 예: python teacher.py analysis "path/to/your/input.json"
    if len(sys.argv) < 3:
        print("🔧 사용법: python teacher.py [agent_name] [input_file_path]")
        
        # 오케스트레이터 인스턴스 생성하여 등록된 에이전트 정보 가져오기
        orchestrator = Orchestrator()
        available_agents = orchestrator.get_available_agents()
        
        print("\n📋 사용 가능한 에이전트:")
        for agent_name, description in available_agents.items():
            print(f"  - {agent_name}: {description}")
        
        print("\n💡 예시:")
        if "analysis" in available_agents:
            print("  python teacher.py analysis ./test_sample/analysis_sample.json")
            print("  python teacher.py analysis C:/path/to/student_answers.json")
        else:
            # 첫 번째 등록된 에이전트를 예시로 사용
            first_agent = next(iter(available_agents.keys())) if available_agents else "agent_name"
            print(f"  python teacher.py {first_agent} ./path/to/input.json")
        sys.exit(1)
        
    agent_to_run = sys.argv[1]
    file_path = sys.argv[2]
    
    print("🎓 LLM-T Teacher System")
    print("=" * 40)
    
    orchestrator = Orchestrator()
    orchestrator.run(agent_name=agent_to_run, input_file_path=file_path)
