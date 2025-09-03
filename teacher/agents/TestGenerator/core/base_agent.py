from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """모든 에이전트가 상속받아야 하는 기본 추상 클래스입니다."""
    
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
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트의 주된 로직을 실행하는 메서드입니다."""
        pass
