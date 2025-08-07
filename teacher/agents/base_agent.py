from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """
    모든 에이전트가 상속받아야 하는 기본 추상 클래스입니다.
    모든 에이전트는 'execute' 메서드를 구현해야 합니다.
    """
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트의 주된 로직을 실행하는 메서드입니다.
        
        Args:
            input_data (Dict[str, Any]): 에이전트 실행에 필요한 입력 데이터입니다.
            
        Returns:
            Dict[str, Any]: 에이전트 실행 결과 데이터입니다.
        """
        pass
