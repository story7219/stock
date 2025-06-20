"""
투자 전략의 기본 인터페이스를 정의하는 모듈입니다.

이 모듈은 모든 투자 전략이 구현해야 하는 공통 인터페이스를 제공하며,
전략 패턴(Strategy Pattern)을 통해 일관성 있는 분석 구조를 보장합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Protocol


class AnalysisStrategy(Protocol):
    """투자 전략의 프로토콜 인터페이스입니다."""
    
    def create_prompt(self, comprehensive_data: Dict[str, Any], news_data: str) -> str:
        """분석 프롬프트를 생성합니다."""
        ...
    
    @property
    def style_name(self) -> str:
        """전략의 고유 식별자를 반환합니다."""
        ...
    
    @property
    def required_keys(self) -> List[str]:
        """분석 결과에 필수로 포함되어야 하는 키들을 반환합니다."""
        ...
    
    @property
    def component_keys(self) -> List[str]:
        """component_scores에 포함되어야 하는 키들을 반환합니다."""
        ...


class BaseStrategy(ABC):
    """
    모든 투자 전략의 기본 클래스입니다.
    
    이 클래스는 투자 전략의 공통 인터페이스를 정의하며,
    각 구체적인 전략 클래스는 이를 상속받아 구현해야 합니다.
    
    Attributes:
        style_name: 전략의 고유 식별자
        required_keys: 분석 결과에 필수로 포함되어야 하는 키들
        component_keys: component_scores에 포함되어야 하는 키들
    """

    @property
    @abstractmethod
    def style_name(self) -> str:
        """
        전략의 고유 식별자를 반환합니다.
        
        Returns:
            str: 전략을 식별하는 고유한 문자열
        """
        pass

    @abstractmethod
    def create_prompt(self, comprehensive_data: Dict[str, Any], news_data: str) -> str:
        """
        주어진 데이터를 바탕으로 AI 분석용 프롬프트를 생성합니다.
        
        Args:
            comprehensive_data: 종목의 종합 데이터 (재무, 기술적 지표 등)
            news_data: 뉴스 및 공시 정보
            
        Returns:
            str: AI 분석을 위한 구조화된 프롬프트
        """
        pass

    @property
    @abstractmethod
    def required_keys(self) -> List[str]:
        """
        분석 결과에 필수로 포함되어야 하는 키들을 반환합니다.
        
        Returns:
            List[str]: 필수 키 목록
        """
        pass

    @property
    @abstractmethod
    def component_keys(self) -> List[str]:
        """
        component_scores에 포함되어야 하는 키들을 반환합니다.
        
        Returns:
            List[str]: 컴포넌트 점수 키 목록
        """
        pass

    def validate_result(self, result: Dict[str, Any]) -> bool:
        """
        분석 결과가 전략의 요구사항을 만족하는지 검증합니다.
        
        Args:
            result: AI 분석 결과
            
        Returns:
            bool: 검증 통과 여부
        """
        # 기본 필수 키 확인
        base_required_keys = ['total_score', 'component_scores', 'rationale']
        all_required_keys = base_required_keys + self.required_keys
        
        for key in all_required_keys:
            if key not in result:
                return False
        
        # 컴포넌트 점수 키 확인
        component_scores = result.get('component_scores', {})
        for key in self.component_keys:
            if key not in component_scores:
                return False
        
        return True

    def __str__(self) -> str:
        """전략의 문자열 표현을 반환합니다."""
        return f"{self.__class__.__name__}(style_name='{self.style_name}')"

    def __repr__(self) -> str:
        """전략의 개발자용 문자열 표현을 반환합니다."""
        return self.__str__() 