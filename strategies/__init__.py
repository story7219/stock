"""
투자 대가별 전략 구현 모듈

20명의 유명 투자자 전략을 투자 스타일별로 분류하여 구현합니다.
"""

# 가치투자 전략
from .value.buffett import BuffettStrategy
from .value.graham import GrahamStrategy
from .value.munger import MungerStrategy

# 성장투자 전략
from .growth.lynch import LynchStrategy
from .growth.oneill import ONeillStrategy
from .growth.fisher import FisherStrategy
from .growth.wood import WoodStrategy

# 매크로 전략
from .macro.soros import SorosStrategy
from .macro.dalio import DalioStrategy
from .macro.druckenmiller import DruckenmillerStrategy

# 기술적분석/단타 전략
from .technical.williams import WilliamsStrategy
from .technical.raschke import RaschkeStrategy
from .technical.livermore import LivermoreStrategy
from .technical.tudor_jones import TudorJonesStrategy

# 시스템매매 전략
from .systematic.dennis import DennisStrategy
from .systematic.seykota import SeykotaStrategy
from .systematic.henry import HenryStrategy

# 퀀트/혼합 전략
from .quantitative.greenblatt import GreenblattStrategy
from .quantitative.k_fisher import KFisherStrategy

# 패시브 전략
from .passive.bogle import BogleStrategy

# 혁신성장 전략
from .innovation.minervini import MinerviniStrategy

# 공통 모듈
from .common import BaseStrategy, StrategyResult

__all__ = [
    # 가치투자
    'BuffettStrategy',
    'GrahamStrategy', 
    'MungerStrategy',
    
    # 성장투자
    'LynchStrategy',
    'ONeillStrategy',
    'FisherStrategy',
    'WoodStrategy',
    
    # 매크로
    'SorosStrategy',
    'DalioStrategy',
    'DruckenmillerStrategy',
    
    # 기술적분석
    'WilliamsStrategy',
    'RaschkeStrategy',
    'LivermoreStrategy',
    'TudorJonesStrategy',
    
    # 시스템매매
    'DennisStrategy',
    'SeykotaStrategy',
    'HenryStrategy',
    
    # 퀀트
    'GreenblattStrategy',
    'KFisherStrategy',
    
    # 패시브
    'BogleStrategy',
    
    # 혁신성장
    'MinerviniStrategy',
    
    # 공통
    'BaseStrategy',
    'StrategyResult'
] 