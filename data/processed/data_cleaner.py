#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧹 데이터 클리너
수집된 원시 주식 데이터를 정제하고 표준화
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

@dataclass
class CleanedStockData:
    """정제된 주식 데이터 구조"""
    symbol: str
    name: str
    price: float
    market_cap: float
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_ratio: Optional[float] = None
    current_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_growth: Optional[float] = None
    eps_growth: Optional[float] = None
    price_momentum_3m: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market: str = "KR"
    data_quality: float = 0.0
    last_updated: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

class DataCleaner:
    """데이터 정제 클래스"""
    
    def __init__(self):
        self.required_fields = ['symbol', 'name', 'price', 'market_cap']
        self.numeric_fields = [
            'price', 'market_cap', 'pe_ratio', 'pb_ratio', 'roe', 'roa',
            'debt_ratio', 'current_ratio', 'dividend_yield', 'revenue_growth',
            'profit_growth', 'eps_growth', 'price_momentum_3m'
        ]
        self.percentage_fields = [
            'roe', 'roa', 'debt_ratio', 'dividend_yield', 'revenue_growth',
            'profit_growth', 'eps_growth', 'price_momentum_3m'
        ]
        
        # 데이터 검증 규칙
        self.validation_rules = {
            'price': (0, 1000000),  # 0원 ~ 100만원
            'market_cap': (100, 100000000),  # 100백만원 ~ 100조원
            'pe_ratio': (0, 1000),  # 0 ~ 1000배
            'pb_ratio': (0, 50),  # 0 ~ 50배
            'roe': (-100, 200),  # -100% ~ 200%
            'roa': (-50, 100),  # -50% ~ 100%
            'debt_ratio': (0, 1000),  # 0% ~ 1000%
            'current_ratio': (0, 20),  # 0 ~ 20배
            'dividend_yield': (0, 50),  # 0% ~ 50%
            'revenue_growth': (-100, 1000),  # -100% ~ 1000%
            'profit_growth': (-1000, 10000),  # -1000% ~ 10000%
            'price_momentum_3m': (-100, 1000)  # -100% ~ 1000%
        }
    
    def clean_stock_data(self, raw_data: Dict[str, Any]) -> Optional[CleanedStockData]:
        """개별 주식 데이터 정제"""
        try:
            # 1단계: 필수 필드 검증
            if not self._validate_required_fields(raw_data):
                return None
            
            # 2단계: 데이터 정제
            cleaned_data = self._clean_raw_data(raw_data)
            
            # 3단계: 데이터 검증
            validated_data = self._validate_data(cleaned_data)
            
            # 4단계: 품질 점수 계산
            quality_score = self._calculate_data_quality(validated_data)
            
            # 5단계: CleanedStockData 객체 생성
            cleaned_stock = CleanedStockData(
                symbol=validated_data['symbol'],
                name=validated_data['name'],
                price=validated_data['price'],
                market_cap=validated_data['market_cap'],
                pe_ratio=validated_data.get('pe_ratio'),
                pb_ratio=validated_data.get('pb_ratio'),
                roe=validated_data.get('roe'),
                roa=validated_data.get('roa'),
                debt_ratio=validated_data.get('debt_ratio'),
                current_ratio=validated_data.get('current_ratio'),
                dividend_yield=validated_data.get('dividend_yield'),
                revenue_growth=validated_data.get('revenue_growth'),
                profit_growth=validated_data.get('profit_growth'),
                eps_growth=validated_data.get('eps_growth'),
                price_momentum_3m=validated_data.get('price_momentum_3m'),
                sector=validated_data.get('sector'),
                industry=validated_data.get('industry'),
                market=validated_data.get('market', 'KR'),
                data_quality=quality_score,
                last_updated=datetime.now().isoformat()
            )
            
            return cleaned_stock
            
        except Exception as e:
            logger.error(f"데이터 정제 오류 ({raw_data.get('symbol', 'Unknown')}): {e}")
            return None
    
    def clean_batch_data(self, raw_data_list: List[Dict[str, Any]]) -> List[CleanedStockData]:
        """여러 주식 데이터 일괄 정제"""
        cleaned_stocks = []
        
        for raw_data in raw_data_list:
            cleaned_stock = self.clean_stock_data(raw_data)
            if cleaned_stock:
                cleaned_stocks.append(cleaned_stock)
        
        logger.info(f"데이터 정제 완료: {len(cleaned_stocks)}/{len(raw_data_list)} 종목")
        return cleaned_stocks
    
    def _validate_required_fields(self, data: Dict[str, Any]) -> bool:
        """필수 필드 검증"""
        for field in self.required_fields:
            if field not in data or data[field] is None:
                logger.warning(f"필수 필드 누락: {field} in {data.get('symbol', 'Unknown')}")
                return False
        return True
    
    def _clean_raw_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """원시 데이터 정제"""
        cleaned = {}
        
        for key, value in raw_data.items():
            if key in self.numeric_fields:
                cleaned[key] = self._clean_numeric_value(value)
            elif key in ['symbol', 'name', 'sector', 'industry']:
                cleaned[key] = self._clean_string_value(value)
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _clean_numeric_value(self, value: Any) -> Optional[float]:
        """숫자 데이터 정제"""
        if value is None or value == '' or value == 'N/A':
            return None
        
        try:
            # 문자열인 경우 정제
            if isinstance(value, str):
                # 특수문자 제거
                cleaned = re.sub(r'[,\s%원달러$]', '', value)
                cleaned = cleaned.replace('억', 'e8').replace('조', 'e12')
                
                if cleaned == '' or cleaned.lower() in ['n/a', 'na', '-', '없음']:
                    return None
                
                # 과학적 표기법 처리
                if 'e' in cleaned.lower():
                    value = float(cleaned)
                else:
                    value = float(cleaned)
            
            # 무한대나 NaN 처리
            if not np.isfinite(value):
                return None
            
            return float(value)
            
        except (ValueError, TypeError, OverflowError):
            logger.debug(f"숫자 변환 실패: {value}")
            return None
    
    def _clean_string_value(self, value: Any) -> Optional[str]:
        """문자열 데이터 정제"""
        if value is None:
            return None
        
        try:
            cleaned = str(value).strip()
            
            # 빈 문자열이나 무의미한 값 처리
            if not cleaned or cleaned.lower() in ['n/a', 'na', '-', '없음', 'null']:
                return None
            
            # 특수문자 정제 (종목명의 경우)
            if len(cleaned) > 50:  # 너무 긴 문자열 제한
                cleaned = cleaned[:50]
            
            return cleaned
            
        except Exception:
            return None
    
    def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 검증 및 이상치 처리"""
        validated = data.copy()
        
        for field, value in data.items():
            if field in self.validation_rules and value is not None:
                min_val, max_val = self.validation_rules[field]
                
                if not (min_val <= value <= max_val):
                    logger.warning(f"이상치 감지 ({data.get('symbol', 'Unknown')}): {field}={value}")
                    
                    # 이상치 처리 방법
                    if field in ['pe_ratio', 'pb_ratio']:
                        # 음수나 과도한 밸류에이션 지표는 None으로
                        if value < 0 or value > max_val:
                            validated[field] = None
                    elif field in self.percentage_fields:
                        # 퍼센트 필드는 범위 제한
                        validated[field] = max(min_val, min(value, max_val))
                    else:
                        # 기타 필드는 None으로
                        validated[field] = None
        
        return validated
    
    def _calculate_data_quality(self, data: Dict[str, Any]) -> float:
        """데이터 품질 점수 계산 (0-100)"""
        total_fields = len(self.numeric_fields) + len(['sector', 'industry'])
        filled_fields = 0
        
        # 필수 필드는 가중치 2배
        for field in self.required_fields:
            if data.get(field) is not None:
                filled_fields += 2
        
        # 선택 필드
        optional_fields = [f for f in self.numeric_fields if f not in self.required_fields]
        optional_fields.extend(['sector', 'industry'])
        
        for field in optional_fields:
            if data.get(field) is not None:
                filled_fields += 1
        
        # 품질 점수 계산
        max_score = len(self.required_fields) * 2 + len(optional_fields)
        quality_score = (filled_fields / max_score) * 100
        
        return round(quality_score, 2)
    
    def filter_by_quality(self, stocks: List[CleanedStockData], min_quality: float = 50.0) -> List[CleanedStockData]:
        """품질 기준으로 필터링"""
        filtered_stocks = [stock for stock in stocks if stock.data_quality >= min_quality]
        logger.info(f"품질 필터링: {len(filtered_stocks)}/{len(stocks)} 종목 (최소 품질: {min_quality})")
        return filtered_stocks
    
    def get_quality_statistics(self, stocks: List[CleanedStockData]) -> Dict[str, Any]:
        """데이터 품질 통계"""
        if not stocks:
            return {}
        
        qualities = [stock.data_quality for stock in stocks]
        
        return {
            "총_종목수": len(stocks),
            "평균_품질": round(np.mean(qualities), 2),
            "최고_품질": round(max(qualities), 2),
            "최저_품질": round(min(qualities), 2),
            "품질_표준편차": round(np.std(qualities), 2),
            "고품질_종목": len([q for q in qualities if q >= 80]),
            "중품질_종목": len([q for q in qualities if 60 <= q < 80]),
            "저품질_종목": len([q for q in qualities if q < 60])
        }
    
    def save_cleaned_data(self, stocks: List[CleanedStockData], filename: str = None) -> str:
        """정제된 데이터 저장"""
        if filename is None:
            filename = f"cleaned_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            data_to_save = {
                "metadata": {
                    "total_stocks": len(stocks),
                    "created_at": datetime.now().isoformat(),
                    "quality_stats": self.get_quality_statistics(stocks)
                },
                "stocks": [stock.to_dict() for stock in stocks]
            }
            
            with open(f"data/processed/{filename}", 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
            logger.info(f"정제된 데이터 저장 완료: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"데이터 저장 오류: {e}")
            raise
    
    def load_cleaned_data(self, filename: str) -> List[CleanedStockData]:
        """정제된 데이터 로드"""
        try:
            with open(f"data/processed/{filename}", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stocks = []
            for stock_data in data.get('stocks', []):
                # None 값들을 적절히 처리하여 CleanedStockData 객체 생성
                stock = CleanedStockData(**stock_data)
                stocks.append(stock)
            
            logger.info(f"정제된 데이터 로드 완료: {len(stocks)} 종목")
            return stocks
            
        except Exception as e:
            logger.error(f"데이터 로드 오류: {e}")
            raise

# 편의 함수들
def clean_stock_data_batch(raw_data_list: List[Dict[str, Any]], min_quality: float = 50.0) -> List[CleanedStockData]:
    """주식 데이터 일괄 정제 편의 함수"""
    cleaner = DataCleaner()
    cleaned_stocks = cleaner.clean_batch_data(raw_data_list)
    return cleaner.filter_by_quality(cleaned_stocks, min_quality)

def get_data_quality_report(stocks: List[CleanedStockData]) -> Dict[str, Any]:
    """데이터 품질 보고서 생성"""
    cleaner = DataCleaner()
    return cleaner.get_quality_statistics(stocks) 