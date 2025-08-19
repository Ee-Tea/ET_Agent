from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """
    모든 에이전트가 상속받아야 하는 기본 추상 클래스입니다.
    모든 에이전트는 'execute' 메서드를 구현해야 합니다.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """에이전트의 고유한 이름을 반환합니다."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """에이전트의 역할에 대한 설명을 반환합니다."""
        pass
    
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
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph 그래프를 가진 에이전트를 subgraph로 실행하는 메서드입니다.
        기본적으로 execute를 호출하지만, 하위 클래스에서 오버라이드할 수 있습니다.
        
        Args:
            input_data (Dict[str, Any]): 에이전트 실행에 필요한 입력 데이터입니다.
            
        Returns:
            Dict[str, Any]: 에이전트 실행 결과 데이터입니다.
        """
        return self.execute(input_data)
