#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 AI 기반 투자 분석 시스템 - 중앙 설정 관리
모든 시스템 설정을 통합 관리하는 핵심 설정 모듈
"""

import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class Config:
    """중앙 설정 클래스"""
    
    # 🔑 API 키 설정
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
    
    # 📊 시장 설정
    MARKETS = {
        "KOSPI200": {
            "name": "코스피200",
            "suffix": ".KS",
            "timezone": "Asia/Seoul",
            "trading_hours": {"open": "09:00", "close": "15:30"}
        },
        "NASDAQ100": {
            "name": "나스닥100",
            "suffix": "",
            "timezone": "America/New_York", 
            "trading_hours": {"open": "09:30", "close": "16:00"}
        },
        "SP500": {
            "name": "S&P500",
            "suffix": "",
            "timezone": "America/New_York",
            "trading_hours": {"open": "09:30", "close": "16:00"}
        }
    }
    
    # 🎯 투자 대가 전략 설정 (15명)
    INVESTMENT_MASTERS = {
        "benjamin_graham": {"weight": 0.12, "name": "벤저민 그레이엄"},
        "warren_buffett": {"weight": 0.15, "name": "워런 버핏"},
        "peter_lynch": {"weight": 0.10, "name": "피터 린치"},
        "george_soros": {"weight": 0.08, "name": "조지 소로스"},
        "james_simons": {"weight": 0.09, "name": "제임스 사이먼스"},
        "ray_dalio": {"weight": 0.07, "name": "레이 달리오"},
        "joel_greenblatt": {"weight": 0.06, "name": "조엘 그린블랫"},
        "william_oneil": {"weight": 0.08, "name": "윌리엄 오닐"},
        "jesse_livermore": {"weight": 0.05, "name": "제시 리버모어"},
        "paul_tudor_jones": {"weight": 0.06, "name": "폴 튜더 존스"},
        "richard_dennis": {"weight": 0.04, "name": "리처드 데니스"},
        "ed_seykota": {"weight": 0.03, "name": "에드 세이코타"},
        "larry_williams": {"weight": 0.03, "name": "래리 윌리엄스"},
        "martin_schwartz": {"weight": 0.02, "name": "마틴 슈바르츠"},
        "stanley_druckenmiller": {"weight": 0.02, "name": "스탠리 드러켄밀러"}
    }
    
    # 📈 기술적 분석 설정
    TECHNICAL_INDICATORS = {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bollinger_period": 20,
        "bollinger_std": 2,
        "sma_periods": [5, 10, 20, 50, 200],
        "ema_periods": [12, 26],
        "stochastic_k": 14,
        "stochastic_d": 3,
        "williams_r_period": 14,
        "atr_period": 14
    }
    
    # 🤖 AI 분석 설정
    AI_SETTINGS = {
        "gemini_model": "gemini-pro",
        "max_retries": 3,
        "retry_delay": 1,
        "temperature": 0.1,
        "max_tokens": 2048,
        "top_p": 0.8,
        "top_k": 40
    }
    
    # 📁 파일 경로 설정
    PATHS = {
        "data": "data",
        "reports": "reports",
        "logs": "logs",
        "cache": "cache",
        "models": "models",
        "backups": "backups"
    }
    
    # 🔄 데이터 수집 설정
    DATA_COLLECTION = {
        "default_period": "1y",
        "batch_size": 10,
        "request_delay": 0.1,
        "max_retries": 3,
        "timeout": 30,
        "cache_duration_hours": 6
    }
    
    # 🧠 ML 모델 설정
    ML_SETTINGS = {
        "models": {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6
            },
            "svm": {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale"
            }
        },
        "feature_selection": {
            "n_features": 20,
            "selection_method": "mutual_info"
        },
        "validation": {
            "test_size": 0.2,
            "cv_folds": 5,
            "scoring": "accuracy"
        }
    }
    
    # 📊 리포트 설정
    REPORT_SETTINGS = {
        "formats": ["html", "pdf", "json"],
        "include_charts": True,
        "chart_style": "seaborn",
        "max_recommendations": 5,
        "confidence_threshold": 0.7
    }
    
    # 🚨 알림 설정
    NOTIFICATION_SETTINGS = {
        "enabled": True,
        "telegram_enabled": False,
        "console_output": True,
        "log_level": "INFO"
    }
    
    # 🔒 보안 설정
    SECURITY_SETTINGS = {
        "encrypt_cache": False,
        "api_rate_limit": 100,
        "max_concurrent_requests": 10,
        "request_timeout": 30
    }
    
    @classmethod
    def get_market_config(cls, market: str) -> Dict[str, Any]:
        """특정 시장 설정 반환"""
        return cls.MARKETS.get(market, {})
    
    @classmethod
    def get_strategy_weight(cls, strategy: str) -> float:
        """전략별 가중치 반환"""
        return cls.INVESTMENT_MASTERS.get(strategy, {}).get("weight", 0.0)
    
    @classmethod
    def get_all_strategies(cls) -> List[str]:
        """모든 전략 이름 리스트 반환"""
        return list(cls.INVESTMENT_MASTERS.keys())
    
    @classmethod
    def validate_config(cls) -> bool:
        """설정 유효성 검증"""
        # API 키 확인
        if not cls.GEMINI_API_KEY:
            print("⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")
            return False
        
        # 가중치 합계 확인
        total_weight = sum(strategy["weight"] for strategy in cls.INVESTMENT_MASTERS.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"⚠️ 전략 가중치 합계가 1.0이 아닙니다: {total_weight}")
            return False
        
        return True

# 전역 설정 인스턴스
config = Config()

# 설정 검증
if __name__ == "__main__":
    if config.validate_config():
        print("✅ 설정 검증 완료")
    else:
        print("❌ 설정 검증 실패") 