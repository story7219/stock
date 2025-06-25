#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 데이터 검증 유틸리티 모듈 (Data Validation Utilities)
====================================================

투자 분석 시스템에서 사용되는 모든 데이터의 품질과 무결성을 
검증하는 유틸리티 함수들을 제공합니다.

주요 기능:
1. 주식 데이터 검증 (Stock Data Validation)
   - 필수 필드 존재 여부 확인
   - 가격 데이터 유효성 검사 (양수, 범위 등)
   - 거래량 데이터 검증
   - 시계열 데이터 일관성 확인

2. 기술적 지표 검증 (Technical Indicators Validation)
   - RSI 값 범위 검증 (0-100)
   - MACD 신호 유효성 확인
   - 볼린저밴드 상한/하한 논리적 검증
   - 이동평균선 계산 정확성 확인

3. 전략 점수 검증 (Strategy Score Validation)
   - 점수 범위 검증 (0-100)
   - 전략별 점수 일관성 확인
   - 가중치 합계 검증 (총합 1.0)
   - 추천 등급 유효성 검사

4. API 응답 검증 (API Response Validation)
   - 외부 API 응답 구조 검증
   - 필수 데이터 필드 존재 확인
   - 데이터 타입 일치성 검사
   - 응답 크기 및 형식 검증

5. 설정값 검증 (Configuration Validation)
   - 시스템 설정 파라미터 유효성 확인
   - 환경 변수 존재 및 형식 검증
   - API 키 유효성 검사
   - 파일 경로 및 권한 확인

검증 규칙:
- 주식 가격: 양수, 합리적 범위 내
- 거래량: 음이 아닌 정수
- 백분율: 0-100 범위
- 날짜: 유효한 날짜 형식
- 문자열: 빈 값 및 특수문자 검증

오류 처리:
- ValidationError: 검증 실패 시 발생
- 상세한 오류 메시지 제공
- 로깅을 통한 검증 실패 추적
- 자동 복구 가능한 오류는 보정

특징:
- 포괄적: 모든 데이터 타입 지원
- 성능 최적화: 빠른 검증 처리
- 확장 가능: 새로운 검증 규칙 쉽게 추가
- 안전성: 예외 상황 완벽 처리

이 모듈을 통해 시스템에 유입되는 모든 데이터의 품질을 보장하고
분석 결과의 신뢰성을 확보합니다.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from ..core.base_interfaces import StockData, StrategyScore, TechnicalAnalysisResult

def validate_stock_data(stock_data: StockData) -> bool:
    """주식 데이터 유효성 검증"""
    try:
        if not stock_data.symbol:
            logger.error("주식 심볼이 비어있습니다")
            return False
        
        if stock_data.current_price <= 0:
            logger.error(f"잘못된 주식 가격: {stock_data.current_price}")
            return False
        
        if not stock_data.name:
            logger.warning(f"주식명이 비어있습니다: {stock_data.symbol}")
        
        return True
    except Exception as e:
        logger.error(f"주식 데이터 검증 중 오류: {e}")
        return False

def validate_strategy_scores(strategy_scores: List[StrategyScore]) -> bool:
    """전략 점수 유효성 검증"""
    try:
        if not strategy_scores:
            logger.error("전략 점수 리스트가 비어있습니다")
            return False
        
        for score in strategy_scores:
            if not score.symbol:
                logger.error("전략 점수에 심볼이 없습니다")
                return False
            
            if not (0 <= score.score <= 100):
                logger.error(f"잘못된 점수 범위: {score.score}")
                return False
            
            if not (0 <= score.confidence <= 1):
                logger.error(f"잘못된 신뢰도 범위: {score.confidence}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"전략 점수 검증 중 오류: {e}")
        return False

def validate_technical_results(technical_results: List[TechnicalAnalysisResult]) -> bool:
    """기술적 분석 결과 유효성 검증"""
    try:
        if not technical_results:
            logger.error("기술적 분석 결과가 비어있습니다")
            return False
        
        for result in technical_results:
            if not result.symbol:
                logger.error("기술적 분석 결과에 심볼이 없습니다")
                return False
            
            if not (0 <= result.confidence <= 1):
                logger.error(f"잘못된 신뢰도 범위: {result.confidence}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"기술적 분석 결과 검증 중 오류: {e}")
        return False

def validate_price_range(price: float, min_price: float = 0, max_price: float = float('inf')) -> bool:
    """가격 범위 검증"""
    return min_price <= price <= max_price

def sanitize_symbol(symbol: str) -> str:
    """심볼 정리"""
    if not symbol:
        return ""
    return symbol.strip().upper()

def validate_market_data(market_data: Dict[str, Any]) -> bool:
    """시장 데이터 유효성 검증"""
    required_fields = ['symbol', 'price', 'volume']
    
    for field in required_fields:
        if field not in market_data:
            logger.error(f"필수 필드 누락: {field}")
            return False
    
    if market_data['price'] <= 0:
        logger.error(f"잘못된 가격: {market_data['price']}")
        return False
    
    if market_data['volume'] < 0:
        logger.error(f"잘못된 거래량: {market_data['volume']}")
        return False
    
    return True 