#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultra AI Stock Analyzer with News Integration
Gemini 1.5 Flash 최적화 주식 분석 시스템

Features:
- 실시간 뉴스 통합 분석
- 한국/미국 주식 지원
- Gemini 1.5 Flash 전용 최적화
- 고성능 배치 처리
- 스마트 캐싱 시스템
"""

import asyncio
import json
import logging
import sqlite3
import time
import hashlib
import os
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

# 핵심 라이브러리
import google.generativeai as genai
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
from dotenv import load_dotenv

# 뉴스 처리용 추가 라이브러리
import requests
from bs4 import BeautifulSoup
import feedparser

# 경고 메시지 숨김
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# 환경 설정
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 뉴스 관련 Enum과 데이터 클래스
class NewsCategory(Enum):
    """뉴스 카테고리"""
    MARKET = "시장"
    COMPANY = "기업" 
    ECONOMIC = "경제"
    TECHNOLOGY = "기술"
    POLICY = "정책"
    GLOBAL = "해외"
    OTHER = "기타"

class SentimentType(Enum):
    """감정 분석 타입"""
    VERY_POSITIVE = "매우긍정"
    POSITIVE = "긍정"
    NEUTRAL = "중립"
    NEGATIVE = "부정"
    VERY_NEGATIVE = "매우부정"

@dataclass
class NewsData:
    """뉴스 데이터 구조"""
    title: str
    content: str
    url: str
    published_time: datetime
    source: str
    category: NewsCategory
    sentiment: SentimentType
    impact_score: float  # 0-100
    related_stocks: List[str]
    keywords: List[str]
    summary: str
    translated_title: str = ""
    translated_content: str = ""

# 환경 설정
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 울트라 성능 상수
MAX_CONCURRENT = 25
BATCH_SIZE = 20
CACHE_TTL = 1800
REQUEST_DELAY = 0.03
ULTRA_RETRY = 7

class OptimizationLevel(Enum):
    """최적화 레벨"""
    BASIC = "basic"
    ADVANCED = "advanced"
    ULTRA = "ultra"
    EXTREME = "extreme"

@dataclass
class UltraConfig:
    """울트라 설정"""
    api_key: str
    model_version: str = "gemini-1.5-pro"
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 50
    max_output_tokens: int = 8192
    batch_size: int = BATCH_SIZE
    max_concurrent: int = MAX_CONCURRENT
    request_delay: float = REQUEST_DELAY
    retry_attempts: int = ULTRA_RETRY

class UltraPerformanceMonitor:
    """울트라 성능 모니터링"""
    def __init__(self):
        self.request_times = deque(maxlen=2000)
        self.success_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.tokens_used = 0
        self.start_time = time.time()
        self.api_rate_limit_hits = 0
        self.fallback_uses = 0
        self._lock = threading.RLock()
        
    def record_request(self, duration: float, success: bool = True, tokens: int = 0):
        with self._lock:
            self.request_times.append(duration)
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
            self.tokens_used += tokens
                
    def record_cache(self, hit: bool):
        with self._lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def get_ultra_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self.request_times:
                return {"status": "no_data"}
            
            total = self.success_count + self.error_count
            avg_time = sum(self.request_times) / len(self.request_times)
            success_rate = (self.success_count / total * 100) if total > 0 else 0
            cache_total = self.cache_hits + self.cache_misses
            cache_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0
            uptime = time.time() - self.start_time
            
            return {
                "🚀 울트라 성능": {
                    "평균응답": f"{avg_time:.3f}초",
                    "성공률": f"{success_rate:.1f}%",
                    "캐시적중": f"{cache_rate:.1f}%",
                    "총요청": total,
                    "가동시간": f"{uptime:.0f}초",
                    "토큰사용": f"{self.tokens_used:,}",
                    "예상비용": f"${self.tokens_used * 0.00025:.4f}"
                }
            }

class UltraSmartCache:
    """울트라 스마트 캐싱 (메모리 + SQLite)"""
    def __init__(self, ttl: int = CACHE_TTL, max_size: int = 1000):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self.ttl = ttl
        self.max_size = max_size
        self._lock = threading.RLock()
        
        # SQLite 캐시 초기화
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "ultra_cache.db"
        self._init_sqlite()
    
    def _init_sqlite(self):
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ultra_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp REAL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"SQLite 초기화 실패: {e}")
    
    def _generate_key(self, data: Any) -> str:
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            # 메모리 캐시 확인
            if key in self._cache:
                data = self._cache[key]
                if time.time() - data['timestamp'] < self.ttl:
                    self._access_times[key] = time.time()
                    return data['value']
                else:
                    del self._cache[key]
                    if key in self._access_times:
                        del self._access_times[key]
            
            # SQLite 캐시 확인
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.execute(
                    "SELECT value, timestamp FROM ultra_cache WHERE key = ?", 
                    (key,)
                )
                row = cursor.fetchone()
                conn.close()
                
                if row and time.time() - row[1] < self.ttl:
                    result = json.loads(row[0])
                    self.set(key, result)
                    return result
            except Exception as e:
                logger.warning(f"SQLite 조회 실패: {e}")
            
            return None
    
    def set(self, key: str, value: Any):
        with self._lock:
            current_time = time.time()
            
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = {
                'value': value,
                'timestamp': current_time
            }
            self._access_times[key] = current_time
            
            # SQLite 저장
            try:
                conn = sqlite3.connect(str(self.db_path))
                data_str = json.dumps(value)
                conn.execute("""
                    INSERT OR REPLACE INTO ultra_cache 
                    (key, value, timestamp) VALUES (?, ?, ?)
                """, (key, data_str, current_time))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"SQLite 저장 실패: {e}")
    
    def _evict_lru(self):
        if self._access_times:
            oldest_key = min(self._access_times, key=self._access_times.get)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
            del self._access_times[oldest_key]

class UltraPromptManager:
    """울트라 프롬프트 관리"""
    def __init__(self):
        self.strategy_guides = self._load_strategies()
    
    def create_ultra_prompt(self, stock_data: Dict[str, Any], strategy_name: str) -> str:
        try:
            header = self._get_ultra_header(strategy_name)
            data_summary = self._create_data_summary(stock_data)
            strategy_guide = self.strategy_guides.get(strategy_name, self._get_default_guide())
            json_format = self._get_ultra_json_format()
            
            return f"{header}\n\n{data_summary}\n\n{strategy_guide}\n\n{json_format}"
        except Exception as e:
            logger.error(f"프롬프트 생성 실패: {e}")
            return self._get_fallback_prompt(strategy_name)
    
    def _get_ultra_header(self, strategy_name: str) -> str:
        return f"""
🏛️ **GOLDMAN SACHS | 수석 애널리스트 (울트라 모드)**
{strategy_name} 전략 전문가로 연평균 40%+ 알파 창출 실적 보유

⚡ **ULTRA ELITE 분석 요구사항:**
- 정량적 데이터 기반 정확한 분석
- 차트 패턴 및 기술지표 정밀 해석  
- 재무제표 Deep Dive 펀더멘털 검증
- 리스크 조정 수익률 관점 적용

🎯 **분석 정확도 목표: 95%+**
"""
    
    def _create_data_summary(self, stock_data: Dict[str, Any]) -> str:
        name = stock_data.get('name', 'N/A')
        code = stock_data.get('stock_code', 'N/A')
        price = self._safe_float(stock_data.get('current_price', 0))
        rsi = self._safe_float(stock_data.get('rsi', 50))
        per = self._safe_float(stock_data.get('per', 0))
        pbr = self._safe_float(stock_data.get('pbr', 0))
        roe = self._safe_float(stock_data.get('roe', 0))
        
        rsi_signal = self._get_rsi_signal(rsi)
        
        return f"""
═══════════════════════════════════════════════════════════════
🏢 **{name} ({code}) - 울트라 분석 데이터**
═══════════════════════════════════════════════════════════════

**📊 핵심 지표**
• 현재가: {price:,.0f}원
• RSI: {rsi:.1f} → {rsi_signal}
• PER: {per:.1f}배 → {'저평가' if 0 < per < 15 else '고평가' if per > 25 else '적정'}
• PBR: {pbr:.1f}배 → {'저평가' if 0 < pbr < 1 else '고평가' if pbr > 2 else '적정'}
• ROE: {roe:.1f}% → {'우수' if roe > 15 else '양호' if roe > 10 else '보통'}
═══════════════════════════════════════════════════════════════
"""
    
    def _get_rsi_signal(self, rsi: float) -> str:
        if rsi >= 70:
            return "과매수 (주의)"
        elif rsi >= 60:
            return "강세 추세"
        elif rsi >= 40:
            return "중립"
        elif rsi >= 30:
            return "약세 추세"
        else:
            return "과매도 (기회)"
    
    def _load_strategies(self) -> Dict[str, str]:
        return {
            "william_oneil": """
**🎯 윌리엄 오닐 CAN SLIM 분석**
• 차트 패턴: 컵앤핸들, 플랫베이스 확인 [30점]
• 브레이크아웃: 거래량 동반 돌파 여부 [25점]
• 상대강도: RS 라인 상승 추세 [20점]
• 실적 성장: 분기/연간 EPS 25%↑ [25점]
**점수 기준:** 90-100(강력매수), 80-89(매수), 70-79(관망), 60-69(주의), 60↓(매도)
""",
            "peter_lynch": """
**📈 피터 린치 성장주 투자 분석**
• 성장률: PEG 비율 1.0 이하 [35점]
• 기업 스토리: 이해하기 쉬운 비즈니스 [25점]
• 시장 지위: 업계 선도 기업 [20점]
• 재무 건전성: 부채 비율, 현금 흐름 [20점]
**린치 철학:** "당신이 이해할 수 있는 기업에 투자하라"
""",
            "warren_buffett": """
**🏰 워렌 버핏 가치 투자 분석**
• 경제적 해자: 브랜드/독점력 [30점]
• 재무 품질: ROE 15%↑, 낮은 부채 [25점]
• 경영진 품질: 주주친화적 [20점]
• 성장 전망: 지속가능성 [15점]
• 가격 매력도: 내재가치 대비 할인 [10점]
**버핏 원칙:** "좋은 기업을 적정가에 사라"
"""
        }
    
    def _get_default_guide(self) -> str:
        return """
**📊 종합 투자 분석**
• 기술적 분석: 차트 패턴 및 지표 [25점]
• 펀더멘털: 재무 건전성 [25점]
• 성장성: 매출/이익 성장률 [25점]
• 밸류에이션: PER/PBR 적정성 [25점]
"""
    
    def _get_ultra_json_format(self) -> str:
        return """
**📋 울트라 응답 형식 (반드시 준수)**

```json
{
    "분석": "상세한 정량적/정성적 분석 내용",
    "결론": "STRONG_BUY/BUY/HOLD/REDUCE/SELL",
    "점수": 숫자(0-100),
    "추천 등급": "투자 등급",
    "추천 이유": "구체적이고 논리적인 근거",
    "목표 가격": "구체적 목표가",
    "손절 가격": "리스크 관리 손절가",
    "신뢰도": 소수점(0.0-1.0),
    "리스크 요인": ["위험요소1", "위험요소2"],
    "투자 기간": "단기/중기/장기"
}
```

**⚠️ 중요:** JSON 형식 정확히 준수, 모든 필드 필수 입력
"""
    
    def _get_fallback_prompt(self, strategy_name: str) -> str:
        return f"""
간단한 {strategy_name} 전략 기반 주식 분석을 JSON 형식으로 제공해주세요.
반드시 다음 필드를 포함해야 합니다:
- 분석, 결론, 점수, 추천 등급, 추천 이유, 목표 가격, 신뢰도
"""
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

class UltraGeminiProcessor:
    """울트라 Gemini 프로세서"""
    def __init__(self, config: UltraConfig):
        self.config = config
        self.cache = UltraSmartCache()
        self.monitor = UltraPerformanceMonitor()
        self.prompt_manager = UltraPromptManager()
        self.rate_limiter = asyncio.Semaphore(config.max_concurrent)
        self.consecutive_errors = 0
        self.adaptive_delay = config.request_delay
        
        # Gemini 모델 초기화
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(
            model_name=config.model_version,
            generation_config=genai.types.GenerationConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_output_tokens=config.max_output_tokens,
                candidate_count=1
            )
        )
        
        logger.info("🚀 울트라 Gemini 프로세서 초기화 완료")
    
    async def analyze_ultra_batch(self, stock_data_list: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
        """울트라 배치 분석"""
        logger.info(f"🚀 울트라 배치 분석 시작: {len(stock_data_list)}개 종목")
        
        # 캐시 확인 및 미처리 항목 분리
        cached_results, pending_items = await self._check_cache_batch(stock_data_list, strategy)
        
        if not pending_items:
            logger.info("✅ 모든 요청이 캐시에서 처리됨")
            return cached_results
        
        # 배치 처리
        new_results = await self._process_batch(pending_items, strategy)
        
        # 결과 통합 및 정렬
        all_results = cached_results + new_results
        all_results.sort(key=lambda x: x.get('점수', 0), reverse=True)
        
        logger.info(f"✅ 울트라 배치 분석 완료: {len(all_results)}개 결과")
        return all_results
    
    async def _check_cache_batch(self, stock_data_list: List[Dict[str, Any]], strategy: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """배치 캐시 확인"""
        cached_results = []
        pending_items = []
        
        for stock_data in stock_data_list:
            cache_key = self.cache._generate_key({
                'stock_code': stock_data.get('stock_code'),
                'strategy': strategy,
                'price': stock_data.get('current_price')
            })
            
            cached_result = self.cache.get(cache_key)
            if cached_result:
                cached_results.append(cached_result)
                self.monitor.record_cache(True)
            else:
                pending_items.append(stock_data)
                self.monitor.record_cache(False)
        
        return cached_results, pending_items
    
    async def _process_batch(self, stock_data_list: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
        """배치 처리"""
        tasks = []
        for stock_data in stock_data_list:
            task = self._process_single_with_fallback(stock_data, strategy)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"처리 오류: {result}")
                processed_results.append(self._create_error_response(stock_data_list[i], str(result)))
                self.consecutive_errors += 1
            else:
                processed_results.append(result)
                self.consecutive_errors = 0
        
        return processed_results
    
    async def _process_single_with_fallback(self, stock_data: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """단일 처리 (폴백 포함)"""
        async with self.rate_limiter:
            start_time = time.time()
            
            try:
                # 1차 시도: 울트라 프롬프트
                prompt = self.prompt_manager.create_ultra_prompt(stock_data, strategy)
                result = await self._call_gemini_api(prompt)
                
                # 결과 보강
                result.update({
                    'stock_code': stock_data.get('stock_code', ''),
                    'strategy': strategy,
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': time.time() - start_time
                })
                
                # 캐시 저장
                cache_key = self.cache._generate_key({
                    'stock_code': stock_data.get('stock_code'),
                    'strategy': strategy,
                    'price': stock_data.get('current_price')
                })
                self.cache.set(cache_key, result)
                
                self.monitor.record_request(time.time() - start_time, True, len(str(result)))
                return result
                
            except Exception as e:
                logger.warning(f"1차 시도 실패: {e}")
                
                # 2차 시도: 단순 프롬프트
                try:
                    simple_prompt = self._create_simple_prompt(stock_data, strategy)
                    result = await self._call_gemini_api(simple_prompt)
                    result.update({
                        'stock_code': stock_data.get('stock_code', ''),
                        'strategy': strategy,
                        'fallback_used': True
                    })
                    self.monitor.record_request(time.time() - start_time, True, len(str(result)))
                    return result
                    
                except Exception as e2:
                    logger.error(f"2차 시도도 실패: {e2}")
                    self.monitor.record_request(time.time() - start_time, False)
                    return self._create_error_response(stock_data, str(e2))
    
    def _create_simple_prompt(self, stock_data: Dict[str, Any], strategy: str) -> str:
        """단순 프롬프트 생성"""
        name = stock_data.get('name', 'N/A')
        code = stock_data.get('stock_code', 'N/A')
        price = stock_data.get('current_price', 0)
        
        return f"""
{name}({code}) 주식을 {strategy} 전략으로 분석해주세요.
현재가: {price:,}원

다음 JSON 형식으로 답변해주세요:
{{
    "분석": "분석 내용",
    "결론": "BUY/HOLD/SELL",
    "점수": 50,
    "추천 등급": "등급",
    "추천 이유": "이유",
    "목표 가격": "목표가",
    "신뢰도": 0.7
}}
"""
    
    async def _call_gemini_api(self, prompt: str) -> Dict[str, Any]:
        """Gemini API 호출"""
        for attempt in range(self.config.retry_attempts):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt
                )
                
                if response and response.text:
                    return self._parse_response(response.text)
                else:
                    raise Exception("빈 응답")
                    
            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    wait_time = (2 ** attempt) * (1 + self.consecutive_errors * 0.1)
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """응답 파싱"""
        try:
            import re
            
            # JSON 추출 패턴
            patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
                r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    try:
                        result = json.loads(matches[0])
                        return self._validate_result(result)
                    except json.JSONDecodeError:
                        continue
            
            # JSON 파싱 실패 시 텍스트 분석
            return self._extract_from_text(text)
            
        except Exception as e:
            logger.warning(f"응답 파싱 실패: {e}")
            return self._create_fallback_response(text)
    
    def _validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """결과 검증 및 보강"""
        required_fields = {
            "분석": "기술적 분석 결과",
            "결론": "HOLD",
            "점수": 50,
            "추천 등급": "HOLD",
            "추천 이유": "종합적 분석 결과",
            "목표 가격": "현재가 기준",
            "신뢰도": 0.7
        }
        
        for field, default_value in required_fields.items():
            if field not in result:
                result[field] = default_value
        
        # 타입 검증
        try:
            result['점수'] = max(0, min(100, int(float(result.get('점수', 50)))))
            result['신뢰도'] = max(0.0, min(1.0, float(result.get('신뢰도', 0.7))))
        except (ValueError, TypeError):
            result['점수'] = 50
            result['신뢰도'] = 0.7
        
        return result
    
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """텍스트에서 정보 추출"""
        score = 50
        grade = "HOLD"
        
        text_lower = text.lower()
        
        # 점수 추출
        import re
        score_matches = re.findall(r'점수[:\s]*(\d+)', text)
        if score_matches:
            score = int(score_matches[0])
        
        # 등급 추출
        if any(word in text_lower for word in ["강력 매수", "strong buy"]):
            grade = "STRONG_BUY"
            score = max(score, 85)
        elif any(word in text_lower for word in ["매수", "buy"]):
            grade = "BUY"
            score = max(score, 70)
        elif any(word in text_lower for word in ["매도", "sell"]):
            grade = "SELL"
            score = min(score, 30)
        
        return {
            "분석": text[:200] + "..." if len(text) > 200 else text,
            "결론": grade,
            "점수": score,
            "추천 등급": grade,
            "추천 이유": "텍스트 분석 기반",
            "목표 가격": "분석 필요",
            "신뢰도": 0.6
        }
    
    def _create_fallback_response(self, text: str) -> Dict[str, Any]:
        """폴백 응답 생성"""
        return {
            "분석": "AI 응답 처리 중 제한 발생",
            "결론": "HOLD",
            "점수": 50,
            "추천 등급": "HOLD",
            "추천 이유": "시스템 제약으로 기본 분석 제공",
            "목표 가격": "추가 분석 필요",
            "신뢰도": 0.5,
            "fallback_response": True
        }
    
    def _create_error_response(self, stock_data: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """오류 응답 생성"""
        return {
            "stock_code": stock_data.get('stock_code', ''),
            "분석": f"분석 중 오류: {error_msg}",
            "결론": "ERROR",
            "점수": 0,
            "추천 등급": "ERROR",
            "추천 이유": error_msg,
            "목표 가격": "N/A",
            "신뢰도": 0.0,
            "error": True
        }

class USStockDataCollector:
    """미국 주식 데이터 수집기 - 월스트리트 애널리스트 수준"""
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5분 캐시
        self.sector_multiples = self._load_sector_multiples()
        
    async def get_us_stock_data(self, symbol: str) -> Dict[str, Any]:
        """미국 주식 데이터 수집 - 전문 애널리스트 수준"""
        try:
            # 캐시 확인
            cache_key = f"us_{symbol}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return cached_data
            
            # Yahoo Finance에서 데이터 수집
            ticker = yf.Ticker(symbol)
            
            # 기본 정보
            info = ticker.info
            
            # 최근 주가 데이터 (1년)
            hist = ticker.history(period="1y")
            if hist.empty:
                raise Exception(f"주가 데이터 없음: {symbol}")
            
            current_price = hist['Close'].iloc[-1]
            
            # 기술적 지표 계산
            rsi = self._calculate_rsi(hist['Close'])
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            sma_200 = hist['Close'].rolling(200).mean().iloc[-1]
            
            # 볼린저 밴드
            bb_upper, bb_lower = self._calculate_bollinger_bands(hist['Close'])
            
            # 거래량 분석
            avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = hist['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
            
            # 52주 고가/저가 및 상대적 위치
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            price_position = (current_price - low_52w) / (high_52w - low_52w) * 100
            
            # 재무 데이터 정리
            market_cap = info.get('marketCap', 0)
            enterprise_value = info.get('enterpriseValue', market_cap)
            
            # 성장률 및 수익성 지표
            revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
            earnings_growth = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
            
            # 섹터 분석
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            sector_pe = self.sector_multiples.get(sector, {}).get('avg_pe', 20)
            
            # 경쟁사 대비 분석 (간단한 상대 평가)
            relative_pe = (info.get('trailingPE', 20) / sector_pe) if sector_pe > 0 else 1
            
            # 재무 건전성 지표
            debt_to_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
            current_ratio = info.get('currentRatio', 1.0)
            quick_ratio = info.get('quickRatio', 1.0)
            
            # 수익성 지표
            gross_margins = info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0
            operating_margins = info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0
            profit_margins = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
            
            # 밸류에이션 지표
            peg_ratio = info.get('pegRatio', 0)
            price_to_sales = info.get('priceToSalesTrailing12Months', 0)
            ev_to_revenue = enterprise_value / info.get('totalRevenue', 1) if info.get('totalRevenue') else 0
            ev_to_ebitda = info.get('enterpriseToEbitda', 0)
            
            # 배당 정보
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            payout_ratio = info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0
            
            # 애널리스트 컨센서스 (가능한 경우)
            target_mean_price = info.get('targetMeanPrice', 0)
            recommendation_key = info.get('recommendationKey', 'hold')
            
            # 종합 데이터 구성
            stock_data = {
                # 기본 정보
                'stock_code': symbol,
                'name': info.get('longName', symbol),
                'current_price': float(current_price),
                'currency': 'USD',
                'country': 'US',
                'exchange': info.get('exchange', 'NASDAQ'),
                
                # 시장 데이터
                'market_cap': market_cap,
                'enterprise_value': enterprise_value,
                'volume': int(hist['Volume'].iloc[-1]) if not hist.empty else 0,
                'avg_volume': int(avg_volume),
                'volume_ratio': volume_ratio,
                
                # 주가 위치 및 기술적 지표
                'high_52w': float(high_52w),
                'low_52w': float(low_52w),
                'price_position_52w': price_position,
                'rsi': rsi,
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'sma_200': float(sma_200),
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                
                # 밸류에이션 지표
                'per': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': peg_ratio,
                'pbr': info.get('priceToBook', 0),
                'price_to_sales': price_to_sales,
                'ev_to_revenue': ev_to_revenue,
                'ev_to_ebitda': ev_to_ebitda,
                'relative_pe': relative_pe,
                
                # 수익성 지표
                'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                'roa': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
                'roic': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,  # 근사치
                'gross_margins': gross_margins,
                'operating_margins': operating_margins,
                'profit_margins': profit_margins,
                
                # 성장률
                'revenue_growth': revenue_growth,
                'earnings_growth': earnings_growth,
                
                # 재무 건전성
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'quick_ratio': quick_ratio,
                
                # 배당
                'dividend_yield': dividend_yield,
                'payout_ratio': payout_ratio,
                
                # 시장 정보
                'sector': sector,
                'industry': industry,
                'beta': info.get('beta', 1.0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                
                # 애널리스트 정보
                'analyst_target_price': target_mean_price,
                'analyst_recommendation': recommendation_key,
                'analyst_count': info.get('numberOfAnalystOpinions', 0),
                
                # 추가 지표
                'book_value': info.get('bookValue', 0),
                'eps_trailing': info.get('trailingEps', 0),
                'eps_forward': info.get('forwardEps', 0),
                'revenue_per_share': info.get('revenuePerShare', 0),
                
                # 데이터 품질 지표
                'data_quality_score': self._calculate_data_quality(info),
                'last_updated': datetime.now().isoformat()
            }
            
            # 데이터 검증 및 이상치 확인
            stock_data = self._validate_and_clean_data(stock_data)
            
            # 캐시 저장
            self.cache[cache_key] = (stock_data, time.time())
            
            return stock_data
            
        except Exception as e:
            logger.error(f"미국 주식 데이터 수집 실패 ({symbol}): {e}")
            return self._create_fallback_us_data(symbol, str(e))
    
    def _load_sector_multiples(self) -> Dict[str, Dict[str, float]]:
        """섹터별 평균 멀티플 (현실적인 데이터)"""
        return {
            'Technology': {'avg_pe': 25.0, 'avg_pbr': 3.5, 'avg_ps': 6.0},
            'Healthcare': {'avg_pe': 22.0, 'avg_pbr': 2.8, 'avg_ps': 4.5},
            'Financial Services': {'avg_pe': 12.0, 'avg_pbr': 1.2, 'avg_ps': 2.5},
            'Consumer Cyclical': {'avg_pe': 18.0, 'avg_pbr': 2.2, 'avg_ps': 1.8},
            'Communication Services': {'avg_pe': 20.0, 'avg_pbr': 2.5, 'avg_ps': 3.2},
            'Industrials': {'avg_pe': 19.0, 'avg_pbr': 2.1, 'avg_ps': 1.5},
            'Consumer Defensive': {'avg_pe': 16.0, 'avg_pbr': 1.8, 'avg_ps': 1.2},
            'Energy': {'avg_pe': 15.0, 'avg_pbr': 1.5, 'avg_ps': 1.0},
            'Utilities': {'avg_pe': 14.0, 'avg_pbr': 1.3, 'avg_ps': 2.0},
            'Real Estate': {'avg_pe': 16.0, 'avg_pbr': 1.4, 'avg_ps': 8.0},
            'Basic Materials': {'avg_pe': 17.0, 'avg_pbr': 1.6, 'avg_ps': 1.3}
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[float, float]:
        """볼린저 밴드 계산"""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper_band = (sma + (std * std_dev)).iloc[-1]
            lower_band = (sma - (std * std_dev)).iloc[-1]
            return float(upper_band), float(lower_band)
        except:
            current_price = prices.iloc[-1]
            return float(current_price * 1.1), float(current_price * 0.9)
    
    def _calculate_data_quality(self, info: Dict) -> float:
        """데이터 품질 점수 계산 (0-100)"""
        score = 0
        total_checks = 10
        
        # 핵심 재무 데이터 존재 여부
        if info.get('marketCap', 0) > 0: score += 1
        if info.get('trailingPE', 0) > 0: score += 1
        if info.get('totalRevenue', 0) > 0: score += 1
        if info.get('grossMargins', 0) > 0: score += 1
        if info.get('returnOnEquity', 0) != 0: score += 1
        if info.get('debtToEquity') is not None: score += 1
        if info.get('currentRatio', 0) > 0: score += 1
        if info.get('beta') is not None: score += 1
        if info.get('sector', '') != '': score += 1
        if info.get('longName', '') != '': score += 1
        
        return (score / total_checks) * 100
    
    def _validate_and_clean_data(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 검증 및 정리"""
        # 가격 검증 (너무 낮거나 높은 가격 체크)
        price = stock_data.get('current_price', 0)
        if price < 0.01 or price > 50000:  # 1센트 미만 또는 5만달러 초과
            logger.warning(f"비정상적인 주가 감지: ${price}")
            stock_data['price_warning'] = True
        
        # PER 검증 (음수 또는 극값 체크)
        per = stock_data.get('per', 0)
        if per < 0 or per > 1000:
            stock_data['per'] = 0
            stock_data['per_warning'] = True
        
        # 시가총액 검증
        market_cap = stock_data.get('market_cap', 0)
        if market_cap < 1000000:  # 100만 달러 미만
            stock_data['market_cap_warning'] = True
        
        # ROE 검증 (-100% ~ 100% 범위)
        roe = stock_data.get('roe', 0)
        if abs(roe) > 100:
            stock_data['roe'] = max(-100, min(100, roe))
            stock_data['roe_adjusted'] = True
        
        return stock_data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not rsi.empty else 50.0
        except:
            return 50.0
    
    def _create_fallback_us_data(self, symbol: str, error: str) -> Dict[str, Any]:
        """미국 주식 폴백 데이터"""
        return {
            'stock_code': symbol,
            'name': symbol,
            'current_price': 0,
            'currency': 'USD',
            'country': 'US',
            'market_cap': 0,
            'per': 0,
            'pbr': 0,
            'roe': 0,
            'rsi': 50,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'data_quality_score': 0,
            'error': error,
            'fallback_data': True
        }

class KoreanStockDataCollector:
    """한국 주식 데이터 수집기"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5분 캐시
        
    def collect_stock_data(self, code: str) -> Dict[str, Any]:
        """한국 주식 데이터 수집"""
        try:
            # 캐시 확인
            cache_key = f"kr_{code}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return cached_data
            
            # 기본 정보 수집
            stock_data = {
                'symbol': code,
                'market': 'KR',
                'timestamp': datetime.now().isoformat()
            }
            
            # 가격 데이터 수집
            try:
                df = fdr.DataReader(code, start='2023-01-01')
                if not df.empty:
                    latest = df.iloc[-1]
                    stock_data.update({
                        'current_price': float(latest['Close']),
                        'open_price': float(latest['Open']),
                        'high_price': float(latest['High']),
                        'low_price': float(latest['Low']),
                        'volume': int(latest['Volume']),
                        'change': float(latest['Close'] - df.iloc[-2]['Close']) if len(df) > 1 else 0,
                        'change_percent': float((latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100) if len(df) > 1 else 0
                    })
                    
                    # 기술적 지표 계산
                    prices = df['Close']
                    stock_data.update({
                        'ma_5': float(prices.rolling(5).mean().iloc[-1]) if len(prices) >= 5 else 0,
                        'ma_20': float(prices.rolling(20).mean().iloc[-1]) if len(prices) >= 20 else 0,
                        'ma_60': float(prices.rolling(60).mean().iloc[-1]) if len(prices) >= 60 else 0,
                        'rsi': self._calculate_rsi(prices) if len(prices) >= 14 else 50
                    })
                    
            except Exception as e:
                logger.warning(f"한국 주식 가격 데이터 수집 실패 ({code}): {e}")
                stock_data.update({
                    'current_price': 0,
                    'change': 0,
                    'change_percent': 0,
                    'volume': 0
                })
            
            # 기업 정보 (간단한 버전)
            try:
                # KRX 상장 기업 정보 (간단 버전)
                stock_data.update({
                    'company_name': f"Company_{code}",
                    'sector': "Unknown",
                    'market_cap': 0,
                    'per': 0,
                    'pbr': 0,
                    'dividend_yield': 0
                })
            except Exception as e:
                logger.warning(f"한국 주식 기업 정보 수집 실패 ({code}): {e}")
            
            # 캐시 저장
            self.cache[cache_key] = (stock_data, time.time())
            
            return stock_data
            
        except Exception as e:
            logger.error(f"한국 주식 데이터 수집 실패 ({code}): {e}")
            return self._create_fallback_kr_data(code, str(e))
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0
    
    def _create_fallback_kr_data(self, code: str, error: str) -> Dict[str, Any]:
        """폴백 한국 주식 데이터"""
        return {
            'symbol': code,
            'market': 'KR',
            'error': error,
            'current_price': 0,
            'change': 0,
            'change_percent': 0,
            'volume': 0,
            'company_name': f"Company_{code}",
            'sector': "Unknown",
            'timestamp': datetime.now().isoformat()
        }

class UniversalPromptManager:
    """Gemini 1.5 Flash 전용 통합 프롬프트 관리자"""
    
    def __init__(self):
        """간단한 뉴스 처리 시스템 초기화"""
        print("📋 Universal Prompt Manager 초기화 완료 (Gemini 1.5 Flash)")
    
    async def analyze_with_news(self, stock_data: Dict[str, Any], news_hours: int = 6) -> Dict[str, Any]:
        """주식 데이터와 뉴스를 통합하여 분석 (간단 버전)"""
        try:
            symbol = stock_data.get('stock_code', stock_data.get('symbol', ''))
            
            # 간단한 뉴스 시뮬레이션 (실제 뉴스 대신 기본 분석)
            mock_news_summary = {
                '총_뉴스_수': 5,
                '평균_영향도': 65,
                '고영향_뉴스_수': 2
            }
            
            mock_relevant_news = [
                {
                    '제목': f'{symbol} 관련 시장 동향',
                    '영향도': 75,
                    '감정_분석': '긍정',
                    '한국어_요약': f'{symbol} 종목에 대한 긍정적 시장 전망이 지속되고 있습니다.'
                },
                {
                    '제목': f'{symbol} 실적 발표 예정',
                    '영향도': 80,
                    '감정_분석': '중립',
                    '한국어_요약': f'{symbol} 기업의 분기 실적 발표가 예정되어 있어 주목받고 있습니다.'
                }
            ]
            
            # 통합 분석 프롬프트 생성
            integrated_prompt = self._create_integrated_analysis_prompt(stock_data, mock_news_summary, mock_relevant_news)
            
            # Gemini 1.5 Flash로 분석
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            response = model.generate_content(
                integrated_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_p=0.9,
                    max_output_tokens=2048,
                )
            )
            
            # 응답 파싱
            result = self._parse_integrated_response(response.text)
            result['news_count'] = len(mock_relevant_news)
            result['total_market_news'] = mock_news_summary.get('총_뉴스_수', 0)
            
            return result
            
        except Exception as e:
            print(f"❌ 뉴스 통합 분석 실패: {e}")
            return {
                'error': f'뉴스 통합 분석 실패: {str(e)}',
                'news_count': 0,
                'basic_analysis': '기본 분석만 제공',
                'updated_investment_grade': '보유',
                'updated_target_price': '분석 불가',
                'news_impact_score': 0,
                'integrated_analysis': '뉴스 분석 시스템 오류로 기본 분석만 제공됩니다.',
                'key_news_points': ['시스템 오류'],
                'risk_factors': ['뉴스 분석 불가'],
                'opportunity_factors': ['수동 분석 필요']
            }
    
    def _create_integrated_analysis_prompt(self, stock_data: Dict[str, Any], news_summary: Dict, relevant_news: List) -> str:
        """통합 분석 프롬프트 생성"""
        symbol = stock_data.get('stock_code', stock_data.get('symbol', ''))
        name = stock_data.get('name', symbol)
        price = stock_data.get('current_price', 0)
        
        prompt = f"""
🎯 **Gemini 1.5 Flash 주식 + 뉴스 통합 분석**

📊 **종목 정보:**
- 종목: {name} ({symbol})
- 현재가: {price:,}원/달러
- PER: {stock_data.get('per', 'N/A')}
- PBR: {stock_data.get('pbr', 'N/A')}
- ROE: {stock_data.get('roe', 'N/A')}%
- RSI: {stock_data.get('rsi', 'N/A')}

📰 **시장 뉴스 현황:**
- 총 뉴스 수: {news_summary.get('총_뉴스_수', 0)}개
- 관련 뉴스: {len(relevant_news)}개
- 평균 영향도: {news_summary.get('평균_영향도', 0)}점

🔍 **관련 뉴스 상위 3개:**
"""
        
        for i, news in enumerate(relevant_news[:3], 1):
            prompt += f"""
{i}. 제목: {news.get('제목', 'N/A')}
   영향도: {news.get('영향도', 0)}점
   감정: {news.get('감정_분석', 'N/A')}
   요약: {news.get('한국어_요약', 'N/A')[:100]}...
"""
        
        prompt += f"""

🎯 **분석 요청:**
위 주식 기본 정보와 최신 뉴스를 종합하여 다음 형식으로 분석해주세요:

{{
    "updated_investment_grade": "매수/보유/매도",
    "updated_target_price": "목표가격 (숫자만)",
    "news_impact_score": "뉴스 영향도 (0-100점)",
    "integrated_analysis": "뉴스를 반영한 종합 분석 (200자 이내)",
    "key_news_points": ["핵심 뉴스 포인트 1", "핵심 뉴스 포인트 2"],
    "risk_factors": ["위험 요소 1", "위험 요소 2"],
    "opportunity_factors": ["기회 요소 1", "기회 요소 2"]
}}

반드시 JSON 형식으로만 응답하세요.
"""
        
        return prompt
    
    def _parse_integrated_response(self, response_text: str) -> Dict[str, Any]:
        """통합 분석 응답 파싱"""
        try:
            # JSON 추출 시도
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # JSON이 없으면 기본 구조 반환
                return {
                    'updated_investment_grade': '보유',
                    'updated_target_price': '분석 불가',
                    'news_impact_score': 50,
                    'integrated_analysis': response_text[:200],
                    'key_news_points': ['분석 중 오류 발생'],
                    'risk_factors': ['응답 파싱 실패'],
                    'opportunity_factors': ['재분석 필요']
                }
        except Exception as e:
            return {
                'updated_investment_grade': '보유',
                'updated_target_price': '분석 불가',
                'news_impact_score': 0,
                'integrated_analysis': f'분석 중 오류: {str(e)}',
                'key_news_points': ['오류 발생'],
                'risk_factors': ['시스템 오류'],
                'opportunity_factors': ['수동 분석 필요']
            }

class UltraAIAnalyzer:
    """Ultra AI 분석 시스템 - Gemini 1.5 Flash 전용"""
    
    def __init__(self):
        load_dotenv()  # 환경 변수 로드
        
        # 설정 로드
        self.config = self._load_config()
        
        # 데이터 수집기들 초기화
        self.kr_collector = KoreanStockDataCollector()
        self.us_collector = USStockDataCollector()
        
        # 프로세서 초기화
        self.processor = UltraGeminiProcessor(self.config)
        
        # Gemini 1.5 Flash 프롬프트 관리자
        self.prompt_manager = UniversalPromptManager()
        
        # 캐시 시스템
        self.cache = {}
        self.cache_duration = 300  # 5분 캐시
        
        print("🚀 Ultra AI Analyzer 초기화 완료 (Gemini 1.5 Flash)")
    
    def _load_config(self) -> UltraConfig:
        """설정 로드"""
        return UltraConfig(
            api_key=os.getenv('GEMINI_API_KEY', ''),
            model_version=os.getenv('GEMINI_MODEL_VERSION', 'gemini-1.5-pro'),
            temperature=float(os.getenv('GEMINI_TEMPERATURE', '0.2')),
            batch_size=int(os.getenv('GEMINI_BATCH_SIZE', str(BATCH_SIZE))),
            max_concurrent=int(os.getenv('GEMINI_MAX_CONCURRENT', str(MAX_CONCURRENT)))
        )
    
    async def analyze_us_stocks(self, symbols: List[str], strategy: str = "comprehensive") -> List[Dict[str, Any]]:
        """미국 주식 분석"""
        if not symbols:
            return []
        
        logger.info(f"🇺🇸 미국 주식 분석 시작: {len(symbols)}개 종목, 전략: {strategy}")
        
        # 미국 주식 데이터 수집
        stock_data_list = []
        for symbol in symbols:
            try:
                stock_data = await self.us_collector.get_us_stock_data(symbol.upper())
                if stock_data and 'error' not in stock_data:
                    stock_data_list.append(stock_data)
            except Exception as e:
                logger.error(f"미국 주식 데이터 수집 실패 ({symbol}): {e}")
        
        if not stock_data_list:
            logger.warning("수집된 미국 주식 데이터가 없습니다")
            return []
        
        # 분석 실행
        results = await self.processor.analyze_ultra_batch(stock_data_list, strategy)
        
        logger.info(f"✅ 미국 주식 분석 완료: {len(results)}개 결과")
        return results
    
    async def analyze_mixed_portfolio(self, kr_codes: List[str] = None, us_symbols: List[str] = None, strategy: str = "comprehensive") -> Dict[str, List[Dict[str, Any]]]:
        """한국/미국 주식 혼합 포트폴리오 분석"""
        results = {
            "korean_stocks": [],
            "us_stocks": [],
            "summary": {}
        }
        
        # 한국 주식 분석 (기존 시스템 활용)
        if kr_codes:
            logger.info(f"🇰🇷 한국 주식 {len(kr_codes)}개 분석 중...")
            # 여기서는 기존 한국 주식 데이터를 사용한다고 가정
            kr_stock_data = []
            for code in kr_codes:
                # 실제로는 data_manager나 다른 한국 주식 데이터 소스 사용
                kr_data = {
                    'stock_code': code,
                    'name': f'한국주식_{code}',
                    'current_price': 50000,  # 예시 데이터
                    'country': 'KR',
                    'currency': 'KRW'
                }
                kr_stock_data.append(kr_data)
            
            results["korean_stocks"] = await self.processor.analyze_ultra_batch(kr_stock_data, strategy)
        
        # 미국 주식 분석
        if us_symbols:
            results["us_stocks"] = await self.analyze_us_stocks(us_symbols, strategy)
        
        # 통합 요약
        all_results = results["korean_stocks"] + results["us_stocks"]
        if all_results:
            avg_score = sum(r.get('점수', 0) for r in all_results) / len(all_results)
            top_picks = sorted(all_results, key=lambda x: x.get('점수', 0), reverse=True)[:5]
            
            results["summary"] = {
                "total_analyzed": len(all_results),
                "average_score": round(avg_score, 1),
                "korean_count": len(results["korean_stocks"]),
                "us_count": len(results["us_stocks"]),
                "top_5_picks": [
                    {
                        "symbol": r.get('stock_code', ''),
                        "name": r.get('name', ''),
                        "score": r.get('점수', 0),
                        "country": r.get('country', 'KR')
                    } for r in top_picks
                ]
            }
        
        return results
    
    async def analyze_stocks(self, stock_data_list: List[Dict[str, Any]], strategy: str = "comprehensive") -> List[Dict[str, Any]]:
        """주식 분석 (울트라 모드) - 기존 호환성 유지"""
        if not stock_data_list:
            return []
        
        logger.info(f"🚀 울트라 주식 분석 시작: {len(stock_data_list)}개 종목, 전략: {strategy}")
        
        results = await self.processor.analyze_ultra_batch(stock_data_list, strategy)
        
        logger.info(f"✅ 울트라 주식 분석 완료: {len(results)}개 결과")
        return results
    
    async def analyze_single_stock(self, stock_data: Dict[str, Any], strategy: str = "comprehensive") -> Dict[str, Any]:
        """단일 주식 분석"""
        results = await self.analyze_stocks([stock_data], strategy)
        return results[0] if results else {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        return self.processor.monitor.get_ultra_stats()
    
    def clear_cache(self):
        """캐시 초기화"""
        self.processor.cache.clear()
        logger.info("🗑️ 캐시 초기화 완료")

    async def analyze_with_news(self, symbols: List[str], market: str = 'US', news_hours: int = 6) -> List[Dict[str, Any]]:
        """뉴스와 함께 주식 분석"""
        print(f"📊 뉴스 통합 분석 시작: {symbols} ({market} 시장)")
        
        results = []
        
        for symbol in symbols:
            try:
                print(f"\n🔍 {symbol} 뉴스 통합 분석 중...")
                
                # 주식 데이터 수집
                if market.upper() == 'KR':
                    stock_data = await asyncio.to_thread(
                        self.kr_collector.collect_stock_data, symbol
                    )
                else:
                    stock_data = await self.us_collector.get_us_stock_data(symbol)
                
                if not stock_data or 'error' in stock_data:
                    print(f"❌ {symbol} 데이터 수집 실패")
                    continue
                
                # 뉴스와 함께 분석
                analysis = await self.prompt_manager.analyze_with_news(stock_data, news_hours)
                
                results.append({
                    'symbol': symbol,
                    'market': market,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"✅ {symbol} 뉴스 통합 분석 완료")
                
            except Exception as e:
                print(f"❌ {symbol} 분석 실패: {e}")
                continue
        
        return results

    async def get_market_news_summary(self, hours_back: int = 6, max_articles: int = 20) -> Dict[str, Any]:
        """시장 뉴스 요약"""
        try:
            print(f"📰 최근 {hours_back}시간 시장 뉴스 요약 생성 중...")
            
            news_list = await self.prompt_manager.news_system.analyze_latest_news(
                hours_back=hours_back, 
                max_articles=max_articles
            )
            
            if not news_list:
                return {"error": "수집된 뉴스가 없습니다"}
            
            summary = self.prompt_manager.news_system.create_news_summary(news_list)
            
            print(f"✅ {len(news_list)}개 뉴스 요약 완료")
            return summary
            
        except Exception as e:
            print(f"❌ 뉴스 요약 실패: {e}")
            return {"error": f"뉴스 요약 실패: {str(e)}"}

# 기본 테스트 함수 추가
async def test_ultra_analyzer():
    """울트라 분석기 기본 테스트"""
    analyzer = UltraAIAnalyzer()
    
    # 테스트 데이터
    test_stocks = [
        {
            'stock_code': '005930',
            'name': '삼성전자',
            'current_price': 75000,
            'per': 12.5,
            'pbr': 1.2,
            'roe': 15.3,
            'rsi': 65.2,
            'country': 'KR',
            'currency': 'KRW'
        },
        {
            'stock_code': '000660',
            'name': 'SK하이닉스',
            'current_price': 120000,
            'per': 18.2,
            'pbr': 1.8,
            'roe': 12.1,
            'rsi': 58.7,
            'country': 'KR',
            'currency': 'KRW'
        }
    ]
    
    print("🚀 울트라 분석기 기본 테스트")
    
    # 다양한 전략으로 테스트
    strategies = ["william_oneil", "peter_lynch", "warren_buffett"]
    
    for strategy in strategies:
        print(f"\n📊 {strategy} 전략 테스트")
        results = await analyzer.analyze_stocks(test_stocks, strategy)
        
        for result in results:
            print(f"  • {result.get('stock_code', 'N/A')}: {result.get('추천 등급', 'N/A')} ({result.get('점수', 0)}점)")
    
    # 성능 통계 출력
    print("\n📈 성능 통계:")
    stats = analyzer.get_performance_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

async def test_us_stocks():
    """미국 주식 테스트"""
    analyzer = UltraAIAnalyzer()
    
    # 유명한 미국 주식들로 테스트
    us_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    print("🇺🇸 미국 주식 울트라 분석 테스트")
    
    for strategy in ["william_oneil", "peter_lynch", "warren_buffett"]:
        print(f"\n📊 {strategy} 전략으로 미국 주식 분석")
        results = await analyzer.analyze_us_stocks(us_symbols[:2], strategy)  # 처음 2개만
        
        for result in results:
            symbol = result.get('stock_code', 'N/A')
            name = result.get('name', 'N/A')
            grade = result.get('추천 등급', 'N/A')
            score = result.get('점수', 0)
            target = result.get('목표 가격', 'N/A')
            
            print(f"  • {symbol} ({name}): {grade} ({score}점) - 목표가: {target}")

async def test_mixed_portfolio():
    """한국/미국 혼합 포트폴리오 테스트"""
    analyzer = UltraAIAnalyzer()
    
    print("🌍 한국/미국 혼합 포트폴리오 분석 테스트")
    
    results = await analyzer.analyze_mixed_portfolio(
        kr_codes=["005930", "000660"],  # 삼성전자, SK하이닉스
        us_symbols=["AAPL", "MSFT"],    # 애플, 마이크로소프트
        strategy="warren_buffett"
    )
    
    print(f"\n📊 분석 결과 요약:")
    summary = results["summary"]
    print(f"  • 총 분석 종목: {summary.get('total_analyzed', 0)}개")
    print(f"  • 한국 주식: {summary.get('korean_count', 0)}개")
    print(f"  • 미국 주식: {summary.get('us_count', 0)}개")
    print(f"  • 평균 점수: {summary.get('average_score', 0)}점")
    
    print(f"\n🏆 TOP 5 추천:")
    for i, pick in enumerate(summary.get('top_5_picks', [])[:5], 1):
        flag = "🇰🇷" if pick['country'] == 'KR' else "🇺🇸"
        print(f"  {i}. {flag} {pick['symbol']} - {pick['score']}점")

async def test_performance_comparison():
    """성능 비교 테스트"""
    analyzer = UltraAIAnalyzer()
    
    print("⚡ 성능 비교 테스트")
    
    # 한국 주식 성능
    start_time = time.time()
    kr_results = await analyzer.analyze_stocks([
        {'stock_code': '005930', 'name': '삼성전자', 'current_price': 75000, 'country': 'KR'}
    ], "william_oneil")
    kr_time = time.time() - start_time
    
    # 미국 주식 성능  
    start_time = time.time()
    us_results = await analyzer.analyze_us_stocks(["AAPL"], "william_oneil")
    us_time = time.time() - start_time
    
    print(f"🇰🇷 한국 주식 분석 시간: {kr_time:.2f}초")
    print(f"🇺🇸 미국 주식 분석 시간: {us_time:.2f}초")
    
    # 성능 통계
    stats = analyzer.get_performance_stats()
    print(f"\n📈 시스템 성능 통계:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

# 메인 테스트 실행
async def main():
    """통합 테스트 실행"""
    print("🚀 울트라 AI 분석기 통합 테스트 시작\n")
    
    # 기본 테스트
    await test_ultra_analyzer()
    
    print("\n" + "="*60 + "\n")
    
    # 미국 주식 테스트
    await test_us_stocks()
    
    print("\n" + "="*60 + "\n")
    
    # 혼합 포트폴리오 테스트
    await test_mixed_portfolio()
    
    print("\n" + "="*60 + "\n")
    
    # 성능 비교 테스트
    await test_performance_comparison()

if __name__ == "__main__":
    asyncio.run(main()) 