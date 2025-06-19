"""
프로젝트에서 사용되는 핵심 데이터 모델들을 정의합니다.
- dataclasses를 활용하여 명확하고 안정적인 데이터 구조를 제공합니다.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

class MarketType(Enum):
    """시장 구분"""
    KOSPI = "J"
    KOSDAQ = "Q"
    ALL = "ALL"

@dataclass
class MarketSignal:
    """시장 신호 데이터 클래스"""
    symbol: str
    signal_type: str  # 예: 'price_surge', 'volume_spike', 'orderbook_imbalance'
    strength: float   # 0-10 신호 강도
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StockInfo:
    """종목 기본 정보 데이터 클래스"""
    code: str
    name: str
    current_price: int
    market_cap: int
    volume: int
    volume_value: int
    market_type: str
    sector: str
    per: Optional[float] = None
    pbr: Optional[float] = None
    roe: Optional[float] = None
    debt_ratio: Optional[float] = None
    score: float = 0.0
    # DART 정보
    corp_code: Optional[str] = None
    ceo_name: Optional[str] = None
    establishment_date: Optional[str] = None
    main_business: Optional[str] = None
    employee_count: Optional[int] = None
    recent_disclosure_count: int = 0

@dataclass
class FilterCriteria:
    """종목 필터링 기준"""
    min_market_cap: int = 500
    min_volume: int = 100000
    min_volume_value: int = 1000
    market_types: List[str] = field(default_factory=lambda: ["KOSPI", "KOSDAQ"])
    exclude_sectors: List[str] = field(default_factory=lambda: ["금융업", "보험업"])
    max_stocks: int = 50

@dataclass
class DartCompanyInfo:
    """DART 기업 개요 정보"""
    corp_code: str
    corp_name: str
    corp_cls: str
    ceo_nm: str
    adres: str
    hm_url: str
    ir_url: str
    phn_no: str
    fax_no: str
    induty_code: str
    est_dt: str
    acc_mt: str 