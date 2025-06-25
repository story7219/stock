#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚙️ 시스템 설정 관리 모듈 (System Configuration Manager)
====================================================

투자 분석 시스템의 모든 설정값을 중앙에서 관리하는 설정 모듈입니다.
전략별 가중치, API 설정, 시스템 파라미터를 체계적으로 관리합니다.

주요 구성 요소:
1. 투자 전략 설정 (Investment Strategy Settings)
   - 15개 투자 대가별 전략 가중치
   - 전략별 최소/최대 점수 임계값
   - 전략 조합 및 앙상블 설정
   - 전략별 리스크 파라미터

2. 기술적 분석 설정 (Technical Analysis Settings)
   - RSI, MACD, 볼린저밴드 등 지표 파라미터
   - 신호 생성 임계값
   - 추세 분석 기간 설정
   - 거래량 분석 가중치

3. AI 분석 설정 (AI Analysis Settings)
   - Gemini AI API 설정
   - 추론 모드 및 온도 설정
   - 컨텍스트 길이 및 토큰 제한
   - AI 신뢰도 계산 파라미터

4. 데이터 수집 설정 (Data Collection Settings)
   - 지원 시장별 데이터 소스 설정
   - API 요청 제한 및 재시도 설정
   - 캐시 정책 및 만료 시간
   - 데이터 품질 검증 기준

5. 시스템 운영 설정 (System Operation Settings)
   - 로깅 레벨 및 출력 형식
   - 성능 모니터링 임계값
   - 자동 백업 및 복구 설정
   - 보안 및 인증 설정

설정 카테고리:
- STRATEGY_WEIGHTS: 전략별 가중치 (총합 1.0)
- TECHNICAL_PARAMS: 기술적 지표 파라미터
- AI_CONFIG: AI 분석 관련 설정
- DATA_CONFIG: 데이터 수집 및 처리 설정
- SYSTEM_CONFIG: 시스템 운영 설정

특징:
- 중앙 집중식: 모든 설정을 한 곳에서 관리
- 타입 안전성: 타입 힌트로 설정값 검증
- 환경별 설정: 개발/운영 환경별 설정 분리
- 동적 조정: 런타임 중 설정 변경 가능

이 모듈을 통해 시스템의 모든 동작을 세밀하게 제어하고
최적의 성능을 위한 파라미터 튜닝이 가능합니다.
"""
from typing import Dict
from ..core.base_interfaces import StrategyType, MarketType, RiskLevel

# 전략별 가중치 설정
STRATEGY_WEIGHTS: Dict[StrategyType, float] = {
    StrategyType.BENJAMIN_GRAHAM: 0.12,
    StrategyType.WARREN_BUFFETT: 0.15,
    StrategyType.PETER_LYNCH: 0.10,
    StrategyType.GEORGE_SOROS: 0.08,
    StrategyType.JAMES_SIMONS: 0.09,
    StrategyType.RAY_DALIO: 0.07,
    StrategyType.JOEL_GREENBLATT: 0.06,
    StrategyType.WILLIAM_ONEIL: 0.08,
    StrategyType.JESSE_LIVERMORE: 0.05,
    StrategyType.PAUL_TUDOR_JONES: 0.06,
    StrategyType.RICHARD_DENNIS: 0.04,
    StrategyType.ED_SEYKOTA: 0.03,
    StrategyType.LARRY_WILLIAMS: 0.03,
    StrategyType.MARTIN_SCHWARTZ: 0.02,
    StrategyType.STANLEY_DRUCKENMILLER: 0.02
}

# 지원 시장 목록
SUPPORTED_MARKETS = {
    'KOSPI200': 'Korean KOSPI 200',
    'NASDAQ100': 'NASDAQ 100',
    'SP500': 'S&P 500'
}

# 위험도별 기본 포지션 크기
POSITION_SIZES: Dict[RiskLevel, float] = {
    RiskLevel.LOW: 8.0,
    RiskLevel.MEDIUM: 5.0,
    RiskLevel.HIGH: 2.0
}

# API 설정
API_SETTINGS = {
    'GEMINI_API_TIMEOUT': 30,
    'MAX_WORKERS': 5,
    'RETRY_ATTEMPTS': 3,
    'RATE_LIMIT_DELAY': 1.0
}

# 분석 설정
ANALYSIS_CONFIG = {
    'TOP_RECOMMENDATIONS': 20,
    'MIN_CONFIDENCE_THRESHOLD': 0.3,
    'TECHNICAL_WEIGHT': 0.3,
    'STRATEGY_WEIGHT': 0.7,
    'DEFAULT_TARGET_RETURN': 0.15,
    'MAX_STOP_LOSS': 0.10
} 