"""
사용자 인터페이스(UI)의 메뉴와 입력 처리를 담당하는 모듈입니다.
"""

import asyncio
from typing import Callable, List, Optional, Awaitable
from dataclasses import dataclass

# 비동기 액션을 위한 타입 정의
AsyncAction = Callable[[], Awaitable[None]]

@dataclass
class MenuOption:
    """메뉴의 각 항목을 나타내는 데이터 클래스"""
    id: str
    description: str
    action: Optional[AsyncAction] = None
    is_exit: bool = False

class Menu:
    """메뉴의 표시와 실행을 관리하는 기본 클래스"""

    def __init__(self, title: str):
        self.title = title
        self._options: List[MenuOption] = []

    def add_option(self, option_id: str, description: str, action: Optional[AsyncAction] = None, is_exit: bool = False):
        """메뉴에 옵션을 추가합니다."""
        self._options.append(MenuOption(id=option_id, description=description, action=action, is_exit=is_exit))

    def display(self):
        """메뉴를 화면에 출력합니다."""
        print("\n" + "="*80)
        print(f"🚀 {self.title}")
        print("="*80)
        for option in self._options:
            print(f"  {option.id}. {option.description}")
        print("="*80)

    async def get_and_execute_choice(self) -> bool:
        """사용자로부터 선택을 받고 해당 액션을 실행합니다."""
        choice = input("\n선택하세요: ").strip()
        
        for option in self._options:
            if option.id == choice:
                if option.is_exit:
                    print(f"👋 {option.description}...")
                    return False  # 루프 종료 신호
                
                if option.action:
                    await option.action()
                    return True  # 루프 계속
        
        print("❌ 잘못된 선택입니다. 다시 선택해주세요.")
        return True

def create_main_menu(system) -> Menu:
    """메인 메뉴를 생성하고 시스템의 메서드와 연결합니다."""
    main_menu = Menu("주식 분석 시스템 - 메인 메뉴")
    
    # 투자 전략 분석 메뉴 그룹
    main_menu.add_option('s', "--- 📊 투자 전략별 TOP 5 종목 자동 추천 ---")
    main_menu.add_option('1', "윌리엄 오닐", system.analyze_william_oneil)
    main_menu.add_option('2', "제시 리버모어", system.analyze_jesse_livermore)
    main_menu.add_option('3', "워렌 버핏", system.analyze_warren_buffett)
    main_menu.add_option('4', "피터 린치", system.analyze_peter_lynch)
    main_menu.add_option('5', "일목균형표", system.analyze_ichimoku)
    main_menu.add_option('6', "블랙록", system.analyze_blackrock)
    
    # 기타 기능 메뉴 그룹
    main_menu.add_option('f', "--- 🔧 기타 기능 ---")
    main_menu.add_option('7', "개별 종목 상세 분석", system.analyze_individual_stock)
    main_menu.add_option('8', "실시간 모니터링 시작", system.start_monitoring)
    main_menu.add_option('9', "거래대금 TOP 20 실시간 전략 매칭", system.start_trading_volume_analysis)
    main_menu.add_option('10', "시스템 상태 확인", system.show_system_status)
    
    # 종료 옵션
    main_menu.add_option('e', "---")
    main_menu.add_option('0', "프로그램 종료", is_exit=True)
    
    return main_menu 