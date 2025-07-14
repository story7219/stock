#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: stock_screener.py
모듈: 종목 스크리닝 및 포트폴리오 관리
목적: 기술적/기본적 분석으로 종목 선별 및 포트폴리오 구성

Author: World-Class Python Engineer
Created: 2025-07-13
Version: 1.0.0
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import logging
import os
import glob

logger = logging.getLogger(__name__)

class StockScreener:
    """종목 스크리닝 및 포트폴리오 관리"""
    
    def __init__(self, data_dir: str = "backup/krx_k200_kosdaq50/krx_backup_20250712_054858/"):
        self.data_dir = data_dir
        self.available_stocks = self._get_available_stocks()
    
    def _get_available_stocks(self) -> List[str]:
        """사용 가능한 종목 목록 조회"""
        try:
            parquet_files = glob.glob(os.path.join(self.data_dir, "*.parquet"))
            stocks = []
            for file in parquet_files:
                if "KOSPI_" in file or "KOSDAQ_" in file:
                    stock_code = file.split("_")[-2]  # 파일명에서 종목코드 추출
                    stocks.append(stock_code)
            logger.info(f"사용 가능한 종목: {len(stocks)}개")
            return stocks
        except Exception as e:
            logger.error(f"종목 목록 조회 오류: {e}")
            return []
    
    def screen_stocks(self, market_condition: str, strategy: str, max_stocks: int = 10) -> List[Dict[str, Any]]:
        """종목 스크리닝: 시장 상황과 전략에 맞는 종목 선별"""
        try:
            screened_stocks = []
            
            for stock_code in self.available_stocks[:20]:  # 상위 20개 종목만 분석 (성능 고려)
                try:
                    # 종목 데이터 로드
                    stock_data = self._load_stock_data(stock_code)
                    if stock_data is None or len(stock_data) < 100:
                        continue
                    
                    # 기술적 지표 계산
                    technical_score = self._calculate_technical_score(stock_data, strategy)
                    
                    # 기본적 지표 계산 (거래량 기반)
                    fundamental_score = self._calculate_fundamental_score(stock_data)
                    
                    # 종합 점수 계산
                    total_score = technical_score * 0.7 + fundamental_score * 0.3
                    
                    if total_score > 0.5:  # 임계값 이상만 선별
                        screened_stocks.append({
                            'stock_code': stock_code,
                            'technical_score': technical_score,
                            'fundamental_score': fundamental_score,
                            'total_score': total_score,
                            'strategy': strategy,
                            'market_condition': market_condition
                        })
                        
                except Exception as e:
                    logger.warning(f"종목 {stock_code} 분석 오류: {e}")
                    continue
            
            # 점수순 정렬 후 상위 종목 선택
            screened_stocks.sort(key=lambda x: x['total_score'], reverse=True)
            selected_stocks = screened_stocks[:max_stocks]
            
            logger.info(f"종목 스크리닝 완료: {len(selected_stocks)}개 선별")
            return selected_stocks
            
        except Exception as e:
            logger.error(f"종목 스크리닝 오류: {e}")
            raise
    
    def _load_stock_data(self, stock_code: str) -> pd.DataFrame:
        """종목 데이터 로드"""
        try:
            file_pattern = f"*_{stock_code}_backup_backup.parquet"
            files = glob.glob(os.path.join(self.data_dir, file_pattern))
            
            if not files:
                return None
            
            df = pd.read_parquet(files[0])
            df = df.sort_values('날짜')
            df['return'] = df['종가'].pct_change()
            df = df.dropna()
            
            return df
        except Exception as e:
            logger.warning(f"종목 {stock_code} 데이터 로드 오류: {e}")
            return None
    
    def _calculate_technical_score(self, df: pd.DataFrame, strategy: str) -> float:
        """기술적 점수 계산"""
        try:
            # 이동평균 분석
            ma_20 = df['종가'].rolling(20).mean()
            ma_50 = df['종가'].rolling(50).mean()
            
            # 현재가 대비 이동평균 비율
            current_price = df['종가'].iloc[-1]
            ma_ratio = current_price / ma_20.iloc[-1]
            
            # RSI 계산 (14일)
            delta = df['종가'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 전략별 점수 계산
            if strategy == "DAY_TRADING":
                # 데이트레이딩: 변동성과 단기 모멘텀 중시
                volatility = df['return'].rolling(5).std().iloc[-1] * np.sqrt(252)
                score = min(1.0, volatility * 2) * (current_rsi / 100)
                
            elif strategy == "SWING_TRADING":
                # 스윙매매: 중기 추세와 RSI 중시
                trend_score = 1.0 if ma_ratio > 1.02 else 0.5 if ma_ratio > 0.98 else 0.0
                rsi_score = 1.0 if 30 < current_rsi < 70 else 0.5
                score = (trend_score + rsi_score) / 2
                
            else:  # MEDIUM_TERM
                # 중기투자: 장기 추세와 안정성 중시
                long_trend = current_price / ma_50.iloc[-1]
                stability = 1.0 - df['return'].rolling(20).std().iloc[-1] * np.sqrt(252)
                score = (long_trend + stability) / 2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"기술적 점수 계산 오류: {e}")
            return 0.0
    
    def _calculate_fundamental_score(self, df: pd.DataFrame) -> float:
        """기본적 점수 계산 (거래량 기반)"""
        try:
            # 거래량 분석
            avg_volume = df['거래량'].rolling(20).mean().iloc[-1]
            current_volume = df['거래량'].iloc[-5:].mean()
            volume_ratio = current_volume / avg_volume
            
            # 거래대금 분석
            avg_amount = df['거래대금'].rolling(20).mean().iloc[-1]
            current_amount = df['거래대금'].iloc[-5:].mean()
            amount_ratio = current_amount / avg_amount
            
            # 유동성 점수
            liquidity_score = min(1.0, (volume_ratio + amount_ratio) / 2)
            
            return liquidity_score
            
        except Exception as e:
            logger.warning(f"기본적 점수 계산 오류: {e}")
            return 0.0
    
    def create_portfolio(self, selected_stocks: List[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
        """포트폴리오 구성"""
        try:
            portfolio = {
                'strategy': strategy,
                'stocks': selected_stocks,
                'weights': self._calculate_weights(selected_stocks, strategy),
                'total_stocks': len(selected_stocks)
            }
            
            logger.info(f"포트폴리오 구성 완료: {len(selected_stocks)}개 종목")
            return portfolio
            
        except Exception as e:
            logger.error(f"포트폴리오 구성 오류: {e}")
            raise
    
    def _calculate_weights(self, stocks: List[Dict[str, Any]], strategy: str) -> List[float]:
        """포트폴리오 가중치 계산"""
        try:
            scores = [stock['total_score'] for stock in stocks]
            total_score = sum(scores)
            
            if total_score == 0:
                # 균등 가중치
                return [1.0 / len(stocks)] * len(stocks)
            
            # 점수 기반 가중치
            weights = [score / total_score for score in scores]
            
            # 전략별 조정
            if strategy == "DAY_TRADING":
                # 데이트레이딩: 상위 종목에 집중
                weights = [w * 1.5 if i < len(weights)//2 else w * 0.5 for i, w in enumerate(weights)]
            elif strategy == "MEDIUM_TERM":
                # 중기투자: 분산 투자
                weights = [1.0 / len(stocks)] * len(stocks)
            
            # 정규화
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            return weights
            
        except Exception as e:
            logger.error(f"가중치 계산 오류: {e}")
            return [1.0 / len(stocks)] * len(stocks) 