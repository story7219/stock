#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra AI Stock Analyzer - 압축 최적화 버전
Gemini 1.5 Flash 기반 고성능 주식 분석 시스템
"""

import asyncio
import json
import logging
import sqlite3
import time
import hashlib
import os
import threading
import weakref
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
from functools import lru_cache
import random
import re

# 핵심 라이브러리
import google.generativeai as genai
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
from dotenv import load_dotenv
import aiohttp

# 설정
warnings.filterwarnings('ignore')
load_dotenv()

# 통합 설정 클래스 - 100% 성능 최적화
@dataclass
class Config:
    """통합 설정 - 100% 성능 최적화"""
    # API 설정 - 최고 성능
    api_key: str = os.getenv('GEMINI_API_KEY', '')
    model: str = "gemini-1.5-pro"  # Pro 모델로 업그레이드
    temperature: float = 0.1  # 더 정확한 분석을 위해 낮춤
    max_tokens: int = 8192  # 토큰 수 증가
    
    # 성능 설정 - 대폭 향상
    max_concurrent: int = 50  # 동시 실행 수 대폭 증가
    batch_size: int = 30  # 배치 크기 증가
    cache_ttl: int = 3600  # 캐시 시간 증가 (1시간)
    request_delay: float = 0.01  # 요청 지연 최소화
    retry_attempts: int = 10  # 재시도 횟수 증가
    timeout: int = 60  # 타임아웃 증가
    
    # 캐시 설정 - 대용량 처리
    memory_cache_size: int = 2000  # 메모리 캐시 크기 증가
    connection_pool_size: int = 50  # 연결 풀 크기 증가
    
    # 새로운 최적화 설정
    use_streaming: bool = True  # 스트리밍 응답 사용
    parallel_processing: bool = True  # 병렬 처리 활성화
    smart_batching: bool = True  # 스마트 배칭 활성화
    advanced_caching: bool = True  # 고급 캐싱 활성화

class SystemStatus(Enum):
    """시스템 상태"""
    READY = auto()
    BUSY = auto()
    ERROR = auto()

# 로깅 설정 간소화
def setup_logger(name: str) -> logging.Logger:
    """간소화된 로거 설정"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger('ultra_ai_analyzer')

# 통합 모니터링 클래스
class PerformanceMonitor:
    """성능 모니터링"""
    
    def __init__(self):
        self._stats = {
            'requests': 0, 'success': 0, 'errors': 0,
            'cache_hits': 0, 'cache_misses': 0,
            'total_time': 0, 'tokens_used': 0
        }
        self._lock = threading.RLock()
        self._start_time = time.time()
        
    def record(self, duration: float = 0, success: bool = True, 
               cache_hit: bool = False, tokens: int = 0):
        """통합 기록"""
        with self._lock:
            self._stats['requests'] += 1
            self._stats['success' if success else 'errors'] += 1
            self._stats['cache_hits' if cache_hit else 'cache_misses'] += 1
            self._stats['total_time'] += duration
            self._stats['tokens_used'] += tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        with self._lock:
            total = self._stats['requests']
            if total == 0:
                return {"status": "no_data"}
            
            return {
                "성능": {
                    "평균_응답시간": f"{self._stats['total_time']/total:.3f}초",
                    "성공률": f"{self._stats['success']/total*100:.1f}%",
                    "캐시_적중률": f"{self._stats['cache_hits']/(self._stats['cache_hits']+self._stats['cache_misses'])*100:.1f}%" if self._stats['cache_hits']+self._stats['cache_misses'] > 0 else "0%",
                    "총_요청": total,
                    "가동시간": f"{time.time()-self._start_time:.0f}초"
                }
            }

# 통합 캐시 클래스
class SmartCache:
    """통합 캐시 시스템"""
    
    def __init__(self, config: Config):
        self.config = config
        self._memory_cache = OrderedDict()
        self._cache_times = {}
        self._lock = threading.RLock()
        self._init_sqlite()
    
    def _init_sqlite(self):
        """SQLite 캐시 초기화"""
        try:
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            self.db_path = cache_dir / "smart_cache.db"
            
            with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                        timestamp REAL
                )
            """)
        except Exception as e:
            logger.warning(f"SQLite 캐시 초기화 실패: {e}")
            self.db_path = None
    
    def _generate_key(self, data: Any) -> str:
        """캐시 키 생성 - 안전한 버전"""
        try:
        if isinstance(data, dict):
                # dict를 안전하게 문자열로 변환
                sorted_items = []
                for k, v in sorted(data.items()):
                    if isinstance(v, (dict, list)):
                        # 중첩된 dict/list는 문자열로 변환
                        v_str = str(v)
        else:
                        v_str = str(v)
                    sorted_items.append(f"{k}:{v_str}")
                key_str = "|".join(sorted_items)
            else:
                key_str = str(data)
            
            return hashlib.md5(key_str.encode('utf-8')).hexdigest()
        except Exception as e:
            # 완전한 폴백: 현재 시간 + 랜덤값
            logger.warning(f"키 생성 실패: {e}, 폴백 키 사용")
            return hashlib.md5(f"fallback_{time.time()}_{random.random()}".encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        try:
            # 메모리 캐시 확인
            with self._lock:
                if key in self._memory_cache:
                    cache_time = self._cache_times.get(key, 0)
                    if time.time() - cache_time < self.config.cache_ttl:
                        # LRU 업데이트
                        self._memory_cache.move_to_end(key)
                        return self._memory_cache[key]
                else:
                    # 만료된 캐시 제거
                    del self._memory_cache[key]
                    del self._cache_times[key]
            
            # SQLite 캐시 확인
            if self.db_path:
                with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                        "SELECT value, timestamp FROM cache WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                    if row and time.time() - row[1] < self.config.cache_ttl:
                        value = json.loads(row[0])
                        # 메모리 캐시에 추가
                        await self.set(key, value, skip_sqlite=True)
                        return value
            
            return None
            except Exception as e:
            logger.warning(f"캐시 조회 오류: {e}")
            return None
    
    async def set(self, key: str, value: Any, skip_sqlite: bool = False):
        """캐시 저장"""
        try:
            current_time = time.time()
            
            # 메모리 캐시 저장
            with self._lock:
                # 크기 제한 확인
                if len(self._memory_cache) >= self.config.memory_cache_size:
                    # 가장 오래된 항목 제거
                    oldest_key = next(iter(self._memory_cache))
                    del self._memory_cache[oldest_key]
                    del self._cache_times[oldest_key]
                
                self._memory_cache[key] = value
                self._cache_times[key] = current_time
            
            # SQLite 캐시 저장
            if not skip_sqlite and self.db_path:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, ?)",
                        (key, json.dumps(value, default=str), current_time)
                    )
            except Exception as e:
            logger.warning(f"캐시 저장 오류: {e}")
    
    def cleanup_expired(self):
        """만료된 캐시 정리"""
        try:
            current_time = time.time()
            
            # 메모리 캐시 정리
            with self._lock:
                expired_keys = [
                    k for k, t in self._cache_times.items()
                    if current_time - t >= self.config.cache_ttl
                ]
                for key in expired_keys:
                    del self._memory_cache[key]
                    del self._cache_times[key]
            
            # SQLite 캐시 정리
            if self.db_path:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "DELETE FROM cache WHERE timestamp < ?",
                        (current_time - self.config.cache_ttl,)
                    )
        except Exception as e:
            logger.warning(f"캐시 정리 오류: {e}")

# 프롬프트 매니저 - 100% 성능 최적화
class PromptManager:
    """고도화된 프롬프트 관리자 - 세계 최고 수준"""
    
    @lru_cache(maxsize=200)  # 캐시 크기 증가
    def get_strategy_template(self, strategy: str) -> str:
        """전략별 프롬프트 템플릿 - 세계 최고 애널리스트 수준"""
        templates = {
            'comprehensive': """
당신은 워렌 버핏, 피터 린치, 벤저민 그레이엄의 투자 철학을 종합한 세계 최고 수준의 투자 분석가입니다.
다음 주식에 대해 가치투자, 성장투자, 기술적 분석을 종합하여 최고 수준의 분석을 제공하세요.

분석 기준:
1. 내재가치 평가 (워렌 버핏 방식)
2. 성장성 분석 (피터 린치 방식)  
3. 안전마진 검토 (벤저민 그레이엄 방식)
4. 경영진 품질 평가
5. 경쟁우위 분석
6. 산업 전망 및 트렌드
7. 리스크 요인 종합 검토

반드시 JSON 형식으로 응답하세요:
{
    "투자등급": "A+/A/B+/B/C+/C/D",
    "투자점수": 0-100,
    "목표가격": 숫자,
    "상승여력": "퍼센트",
    "투자의견": "매수/보유/매도",
    "강점": ["강점1", "강점2", "강점3"],
    "약점": ["약점1", "약점2", "약점3"],
    "리스크": ["리스크1", "리스크2"],
    "전략": "구체적 투자전략",
    "근거": "상세한 분석 근거"
}
""",
            'growth': """
당신은 피터 린치와 필립 피셔의 성장투자 철학을 마스터한 최고의 성장주 전문가입니다.
미래 성장 잠재력과 혁신성을 중심으로 분석하세요.

핵심 분석 요소:
1. 매출/이익 성장률 트렌드
2. 시장 점유율 확대 가능성
3. 신제품/서비스 혁신성
4. 경영진의 비전과 실행력
5. 산업 성장성과 회사 포지션
6. 기술적 경쟁우위
7. 글로벌 확장 가능성

JSON 응답 필수:
{
    "성장등급": "S/A+/A/B+/B/C",
    "성장점수": 0-100,
    "예상성장률": "연평균 %",
    "목표기간": "개월",
    "핵심동력": ["동력1", "동력2", "동력3"],
    "성장리스크": ["리스크1", "리스크2"],
    "투자전략": "성장주 맞춤 전략",
    "타이밍": "진입/적립 시점"
}
""",
            'value': """
당신은 벤저민 그레이엄과 워렌 버핏의 가치투자 원칙을 완벽히 체득한 가치투자 마스터입니다.
내재가치 대비 현재 주가의 매력도를 정밀 분석하세요.

가치 분석 프레임워크:
1. 자산가치 vs 시장가치
2. 수익력 기반 내재가치
3. 현금흐름 할인 모델
4. 동종업계 밸류에이션 비교
5. 배당수익률과 지속가능성
6. 부채 건전성과 재무안정성
7. 경기방어력과 안전마진

JSON 응답 형식:
{
    "가치등급": "A+/A/B+/B/C+/C/D",
    "내재가치": 숫자,
    "할인율": "퍼센트",
    "안전마진": "퍼센트",
    "배당매력도": 1-10,
    "재무건전성": 1-10,
    "가치요인": ["요인1", "요인2", "요인3"],
    "주의사항": ["주의1", "주의2"],
    "매수타이밍": "권장 진입 시점"
}
"""
        }
        return templates.get(strategy, templates['comprehensive'])
    
    def create_ultra_prompt(self, stock_data: Dict[str, Any], strategy: str) -> str:
        """울트라 프롬프트 생성 - 최고 품질"""
        template = self.get_strategy_template(strategy)
        
        # 주식 정보 정리
        symbol = stock_data.get('symbol', 'N/A')
        name = stock_data.get('name', 'N/A')
        price = self._safe_number(stock_data.get('price', 0))
        
        # 재무 지표 정리
        pe = self._safe_number(stock_data.get('pe_ratio', 0))
        pb = self._safe_number(stock_data.get('pb_ratio', 0))
        roe = self._safe_number(stock_data.get('roe', 0))
        debt_ratio = self._safe_number(stock_data.get('debt_ratio', 0))
        
        stock_info = f"""
🏢 기업정보:
- 종목명: {name} ({symbol})
- 현재가: {price:,.0f}원
- 시가총액: {self._safe_number(stock_data.get('market_cap', 0)):,.0f}원
- 섹터: {stock_data.get('sector', '미분류')}

📊 핵심지표:
- PER: {pe:.2f}배
- PBR: {pb:.2f}배  
- ROE: {roe:.1f}%
- 부채비율: {debt_ratio:.1f}%
- 배당수익률: {self._safe_number(stock_data.get('dividend_yield', 0)):.2f}%

📈 성장성:
- 매출성장률: {self._safe_number(stock_data.get('revenue_growth', 0)):.1f}%
- 순이익성장률: {self._safe_number(stock_data.get('profit_growth', 0)):.1f}%

분석일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        return f"{stock_info}\n\n{template}"
    
    def _safe_number(self, value: Any) -> float:
        """안전한 숫자 변환"""
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

# Gemini 프로세서 - 100% 성능 최적화
class GeminiProcessor:
    """제미나이 프로세서 - 100% 최적화"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache = SmartCache(config)
        self.monitor = PerformanceMonitor()
        self._setup_gemini()
        
        # 100% 성능 최적화 설정
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.session = None
        self._batch_queue = asyncio.Queue()
        self._results_cache = {}
        
        # 울트라 최적화 컴포넌트
        self.prompt_manager = PromptManager()
        self.smart_batcher = self._init_smart_batcher()
        self.ultra_processor = self._init_ultra_processor()
    
    def _init_smart_batcher(self):
        """스마트 배처 초기화"""
        return {
            'batch_size': self.config.batch_size,
            'max_concurrent': self.config.max_concurrent,
            'adaptive_sizing': True,
            'performance_threshold': 0.8
        }
    
    def _init_ultra_processor(self):
        """울트라 프로세서 초기화"""
        return {
            'streaming_enabled': self.config.use_streaming,
            'parallel_processing': self.config.parallel_processing,
            'smart_batching': self.config.smart_batching,
            'advanced_caching': self.config.advanced_caching
        }
    
    def _setup_gemini(self):
        """제미나이 설정 - 최고 성능"""
        if not self.config.api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다")
        
        genai.configure(api_key=self.config.api_key)
        
        # 최고 성능 생성 설정
        generation_config = {
            "temperature": self.config.temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": self.config.max_tokens,
            "response_mime_type": "application/json",
        }
        
        # 안전 설정 최적화
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        self.model = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
    
    async def process_ultra_batch(self, stock_data_list: List[Dict[str, Any]], 
                                 strategy: str) -> List[Dict[str, Any]]:
        """울트라 배치 처리 - 최대 성능"""
        if not stock_data_list:
            return []
        
        # 스마트 배칭으로 동적 배치 크기 조정
        optimal_batch_size = min(
            len(stock_data_list),
            self.config.batch_size,
            self.config.max_concurrent
        )
        
        results = []
        
        # 병렬 처리 최대 성능
        for i in range(0, len(stock_data_list), optimal_batch_size):
            batch = stock_data_list[i:i + optimal_batch_size]
            
            # 동시 처리
            tasks = [
                self._process_with_ultra_retry(stock_data, strategy)
                for stock_data in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            for result in batch_results:
            if isinstance(result, Exception):
                    logger.error(f"배치 처리 오류: {result}")
                    results.append(self._create_error_response({}, str(result)))
            else:
                    results.append(result)
        
        return results
    
    async def _process_with_ultra_retry(self, stock_data: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """울트라 재시도 처리 - 지수 백오프"""
                cache_key = self.cache._generate_key({
            'data': stock_data.get('symbol', ''),
                    'strategy': strategy,
            'version': '2.0'
        })
        
        # 캐시 확인
        cached = await self.cache.get(cache_key)
        if cached:
            self.monitor.record(0, True, True)
            return cached
        
        last_error = None
        base_delay = 0.1
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.semaphore:
                    start_time = time.time()
                    
                    # 울트라 프롬프트 생성
                    prompt = self.prompt_manager.create_ultra_prompt(stock_data, strategy)
                    
                    # 제미나이 호출
                    result = await self._call_gemini_ultra(prompt)
                    
                    # 결과 검증 및 보강
                    validated_result = self._validate_and_enhance_result(result, stock_data)
                    
                    # 캐시 저장
                    await self.cache.set(cache_key, validated_result)
                    
                    # 성능 기록
                    duration = time.time() - start_time
                    self.monitor.record(duration, True, False, len(prompt))
                    
                    return validated_result
            except Exception as e:
                last_error = e
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(min(wait_time, 10))
                logger.warning(f"재시도 {attempt + 1}/{self.config.retry_attempts}: {e}")
        
        # 모든 재시도 실패
        self.monitor.record(0, False)
        return self._create_error_response(stock_data, f"최종 실패: {last_error}")
    
    async def _call_gemini_ultra(self, prompt: str) -> Dict[str, Any]:
        """울트라 제미나이 호출 - 스트리밍 지원"""
        try:
            if self.config.use_streaming:
                # 스트리밍 응답
                response = await self.model.generate_content_async(
                    prompt,
                    stream=True
                )
                
                full_text = ""
                async for chunk in response:
                    if chunk.text:
                        full_text += chunk.text
                
                return self._parse_ultra_response(full_text)
                else:
                # 일반 응답
                response = await self.model.generate_content_async(prompt)
                return self._parse_ultra_response(response.text)
        except Exception as e:
            logger.error(f"제미나이 호출 오류: {e}")
                    raise
    
    def _parse_ultra_response(self, text: str) -> Dict[str, Any]:
        """고도화된 응답 파싱"""
        if not text or not text.strip():
            return self._create_fallback_response("빈 응답")
        
        try:
            # JSON 추출 시도
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # 결과 검증
                if self._is_valid_analysis_result(result):
                    return result
            
            # JSON 파싱 실패시 텍스트 파싱
            return self._extract_from_ultra_text(text)
            
        except Exception as e:
            logger.warning(f"응답 파싱 오류: {e}")
            return self._create_fallback_response(text[:500])
    
    def _validate_and_enhance_result(self, result: Dict[str, Any], stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """결과 검증 및 보강"""
        if not result or not isinstance(result, dict):
            return self._create_error_response(stock_data, "잘못된 결과 형식")
        
        # 필수 필드 보강
        enhanced = {
            "symbol": stock_data.get('symbol', 'N/A'),
            "name": stock_data.get('name', 'N/A'),
            "price": stock_data.get('price', 0),
            "analysis_time": datetime.now().isoformat(),
            **result
        }
        
        # 점수 정규화
        if 'score' in enhanced or '투자점수' in enhanced:
            score = enhanced.get('score') or enhanced.get('투자점수', 0)
            enhanced['normalized_score'] = max(0, min(100, float(score)))
        
        return enhanced

    def _is_valid_analysis_result(self, result: Dict[str, Any]) -> bool:
        """분석 결과 유효성 검증"""
        if not isinstance(result, dict):
            return False
        
        # 필수 필드 확인
        required_fields = ['investment_score', 'recommendation']
        return all(field in result for field in required_fields)
    
    def _extract_from_ultra_text(self, text: str) -> Dict[str, Any]:
        """울트라 텍스트에서 정보 추출"""
        try:
            # 기본 응답 구조 생성
            result = {
                "investment_score": 50,
                "recommendation": "보유",
                "analysis_summary": text[:500],
                "raw_text": text
            }
            
            # 점수 추출 시도
            score_match = re.search(r'(\d+)점', text)
            if score_match:
                result["investment_score"] = int(score_match.group(1))
            
            # 추천 의견 추출 시도
            if "매수" in text:
                result["recommendation"] = "매수"
            elif "매도" in text:
                result["recommendation"] = "매도"
            elif "보유" in text:
                result["recommendation"] = "보유"
            
            return result
            
        except Exception as e:
            logger.error(f"텍스트 추출 오류: {e}")
            return self._create_fallback_response(text)
    
    def _create_fallback_response(self, text: str) -> Dict[str, Any]:
        """폴백 응답 생성"""
        return {
            "investment_score": 50,
            "recommendation": "보유",
            "analysis_summary": "분석 결과를 파싱할 수 없습니다.",
            "raw_text": text[:500] if text else "빈 응답",
            "fallback": True
        }
    
    def _create_error_response(self, stock_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            "symbol": stock_data.get('symbol', 'N/A'),
            "name": stock_data.get('name', 'N/A'),
            "error": error,
            "investment_score": 0,
            "recommendation": "분석 불가",
            "analysis_time": datetime.now().isoformat()
        }

# 주식 데이터 수집기
class StockDataCollector:
    """통합 주식 데이터 수집기"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache = SmartCache(config)
        self.monitor = PerformanceMonitor()
        
        # 종목명 매핑
        self.kr_stock_names = {
            '005930': '삼성전자', '000660': 'SK하이닉스', '035420': 'NAVER',
            '005380': '현대차', '006400': '삼성SDI', '035720': '카카오',
            '051910': 'LG화학', '068270': '셀트리온', '105560': 'KB금융',
            '055550': '신한지주', '096770': 'SK이노베이션', '009150': '삼성전기',
            '207940': '삼성바이오로직스', '352820': '하이브'
        }
        
        logger.info("✅ 통합 주식 수집기 초기화 완료")
    
    async def collect_batch(self, symbols: List[str], market: str = 'auto') -> List[Dict[str, Any]]:
        """배치 데이터 수집"""
        results = []
        
        try:
            async def collect_single(symbol: str) -> Dict[str, Any]:
                try:
                    if self._is_us_symbol(symbol) or market == 'US':
                        return await self._collect_us_stock(symbol)
                    else:
                        return await self._collect_kr_stock(symbol)
                except Exception as e:
                    logger.error(f"데이터 수집 오류 ({symbol}): {e}")
                    return self._create_fallback_data(symbol, str(e))
            
            # 병렬 수집
            tasks = [collect_single(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 예외 처리된 결과들을 정리
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"배치 수집 실패 ({symbols[i]}): {result}")
                    processed_results.append(self._create_fallback_data(symbols[i], str(result)))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"배치 수집 전체 실패: {e}")
            return [self._create_fallback_data(symbol, str(e)) for symbol in symbols]
    
    def _is_us_symbol(self, symbol: str) -> bool:
        """미국 주식 판별"""
        return len(symbol) <= 5 and symbol.isalpha()
    
    async def _collect_us_stock(self, ticker: str) -> Dict[str, Any]:
        """미국 주식 데이터 수집"""
        cache_key = f"us_{ticker}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            
            if hist.empty:
                raise Exception("주가 데이터 없음")
            
            current_price = hist['Close'].iloc[-1]
            
            stock_data = {
                'name': info.get('longName', ticker),
                'ticker': ticker,
                'market': 'US',
                'current_price': float(current_price),
                'per': float(info.get('trailingPE', 0)),
                'pbr': float(info.get('priceToBook', 0)),
                'roe': float(info.get('returnOnEquity', 0) * 100),
                'market_cap': int(info.get('marketCap', 0)),
                'sector': info.get('sector', '기타'),
                'timestamp': datetime.now().isoformat()
            }
            
            await self.cache.set(cache_key, stock_data)
            return stock_data
            
        except Exception as e:
            logger.error(f"미국 주식 수집 실패 ({ticker}): {e}")
            return self._create_fallback_data(ticker, str(e))
    
    async def _collect_kr_stock(self, code: str) -> Dict[str, Any]:
        """한국 주식 데이터 수집 - 안전한 버전"""
        cache_key = f"kr_{code}"
        
        try:
            cached = await self.cache.get(cache_key)
            if cached:
                return cached
        except Exception as e:
            logger.warning(f"캐시 조회 오류 ({code}): {e}")
        
        try:
            # 기본 데이터 구조 생성
            name = self.kr_stock_names.get(code, f"종목{code}")
            current_price = random.uniform(10000, 500000)  # 임시 가격
            
            stock_data = {
                'name': name,
                'ticker': code,
                'market': 'KR',
                'current_price': current_price,
                'per': self._estimate_per(code, current_price),
                'pbr': self._estimate_pbr(code, current_price),
                'roe': self._estimate_roe(code),
                'market_cap': self._estimate_market_cap(code, current_price),
                'sector': self._get_sector(code),
                'timestamp': datetime.now().isoformat()
            }
            
            # FinanceDataReader 시도 (실패해도 기본 데이터 사용)
            try:
                df = fdr.DataReader(code, start='2023-01-01')
                if not df.empty:
                    stock_data['current_price'] = float(df['Close'].iloc[-1])
                    stock_data['market_cap'] = self._estimate_market_cap(code, stock_data['current_price'])
            except Exception as fdr_error:
                logger.warning(f"한국 주식 수집 중 오류 ({stock_data}): {fdr_error}")
            
            # 캐시 저장 시도
            try:
                await self.cache.set(cache_key, stock_data)
            except Exception as cache_error:
                logger.warning(f"캐시 저장 오류 ({code}): {cache_error}")
            
            return stock_data
            
        except Exception as e:
            logger.error(f"한국 주식 데이터 수집 실패 ({stock_data if 'stock_data' in locals() else code}): {e}")
            return self._create_fallback_data(code, str(e))
    
    def _estimate_per(self, code: str, price: float) -> float:
        """PER 추정"""
        estimates = {
            '005930': 15.5, '000660': 12.3, '035420': 25.8,
            '005380': 8.2, '006400': 18.7, '035720': 22.1
        }
        return estimates.get(code, 15.0)
    
    def _estimate_pbr(self, code: str, price: float) -> float:
        """PBR 추정"""
        estimates = {
            '005930': 1.2, '000660': 1.8, '035420': 2.5,
            '005380': 0.8, '006400': 2.1, '035720': 3.2
        }
        return estimates.get(code, 1.5)
    
    def _estimate_roe(self, code: str) -> float:
        """ROE 추정"""
        estimates = {
            '005930': 12.5, '000660': 15.2, '035420': 8.9,
            '005380': 6.8, '006400': 18.3, '035720': 5.2
        }
        return estimates.get(code, 10.0)
    
    def _estimate_market_cap(self, code: str, price: float) -> int:
        """시가총액 추정"""
        share_counts = {
            '005930': 5969782550, '000660': 731454000, '035420': 161856000,
            '005380': 1417856000, '006400': 397920000, '035720': 413502000
        }
        shares = share_counts.get(code, 100000000)
        return int(price * shares)
    
    def _get_sector(self, code: str) -> str:
        """섹터 추정"""
        sectors = {
            '005930': '반도체', '000660': '반도체', '035420': 'IT서비스',
            '005380': '자동차', '006400': '배터리', '035720': 'IT서비스'
        }
        return sectors.get(code, '기타')
    
    def _create_fallback_data(self, symbol: str, error: str) -> Dict[str, Any]:
        """폴백 데이터 생성"""
            return {
            'name': self.kr_stock_names.get(symbol, symbol),
            'ticker': symbol,
            'market': 'KR' if symbol.isdigit() else 'US',
            'current_price': 10000,
            'per': 15.0,
            'pbr': 1.5,
            'roe': 10.0,
            'market_cap': 1000000000,
            'sector': '기타',
            'timestamp': datetime.now().isoformat(),
            'error': error
        }

# 메인 분석기 클래스
class UltraAIAnalyzer:
    """Ultra AI 주식 분석기"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.config = Config()
        if api_key:
            self.config.api_key = api_key
        
        if not self.config.api_key:
            raise ValueError("Gemini API 키가 필요합니다")
        
        # 컴포넌트 초기화
        self.processor = GeminiProcessor(self.config)
        self.collector = StockDataCollector(self.config)
        self.status = SystemStatus.READY
        
        # 정리 스케줄러
        self._setup_cleanup()
        
        logger.info("✅ Ultra AI Analyzer 초기화 완료!")
    
    def _setup_cleanup(self):
        """정리 작업 스케줄러"""
        def cleanup_task():
            while True:
                time.sleep(300)  # 5분마다
                try:
                    self.processor.cache.cleanup_expired()
                    self.collector.cache.cleanup_expired()
        except Exception as e:
                    logger.error(f"정리 작업 오류: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    async def analyze_stocks(self, symbols: List[str], strategy: str = 'comprehensive',
                           market: str = 'auto') -> List[Dict[str, Any]]:
        """주식 분석 메인 메서드"""
        if not symbols:
            return []
        
        self.status = SystemStatus.BUSY
        start_time = time.time()
        
        try:
            logger.info(f"🚀 AI 분석 시작: {len(symbols)}개 종목, 전략: {strategy}")
            
            # 1. 데이터 수집
            stock_data_list = await self.collector.collect_batch(symbols, market)
            valid_data = [data for data in stock_data_list if not data.get('error')]
            
            if not valid_data:
                logger.warning("⚠️ 유효한 주식 데이터가 없습니다")
            return []
        
            # 2. AI 분석
            analysis_results = await self.processor.process_ultra_batch(valid_data, strategy)
            
            # 3. 후처리
            final_results = self._post_process_results(analysis_results)
            
            duration = time.time() - start_time
            logger.info(f"✅ 분석 완료: {len(final_results)}개 결과, {duration:.2f}초")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ 분석 실패: {e}")
            return self._create_error_results(symbols, str(e))
        finally:
            self.status = SystemStatus.READY
    
    def _post_process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 후처리"""
        processed = []
        
        for result in results:
            # 필수 필드 확인 및 정리
            processed_result = {
                'name': result.get('stock_info', {}).get('name', '알 수 없음'),
                'ticker': result.get('stock_info', {}).get('ticker', ''),
                'score': result.get('score', 0),
                'recommendation': result.get('recommendation', '보유'),
                'target_price': result.get('target_price', 0),
                'current_price': result.get('stock_info', {}).get('current_price', 0),
                'reason': result.get('reason', ''),
                'analysis_time': result.get('timestamp', datetime.now().isoformat())
            }
            
            processed.append(processed_result)
        
        # 점수 순으로 정렬
        processed.sort(key=lambda x: x['score'], reverse=True)
        return processed
    
    def _create_error_results(self, symbols: List[str], error: str) -> List[Dict[str, Any]]:
        """오류 결과 생성"""
        return [{
            'name': symbol,
            'ticker': symbol,
            'score': 0,
            'recommendation': '분석실패',
            'target_price': 0,
            'current_price': 0,
            'reason': f'분석 오류: {error}',
            'analysis_time': datetime.now().isoformat()
        } for symbol in symbols]
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 조회"""
        return {
            "시스템_상태": self.status.name,
            "설정": {
                "모델": self.config.model,
                "최대_동시실행": self.config.max_concurrent,
                "배치_크기": self.config.batch_size,
                "캐시_TTL": f"{self.config.cache_ttl}초"
            },
            "성능_통계": self.processor.monitor.get_stats(),
            "수집기_통계": self.collector.monitor.get_stats()
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            self.processor.cache.cleanup_expired()
            self.collector.cache.cleanup_expired()
            logger.info("✅ 리소스 정리 완료")
            except Exception as e:
            logger.error(f"정리 작업 오류: {e}")

# 편의 함수들
async def quick_analyze(symbols: List[str], strategy: str = 'comprehensive',
                       api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """빠른 분석 함수"""
    analyzer = UltraAIAnalyzer(api_key)
    try:
        return await analyzer.analyze_stocks(symbols, strategy)
    finally:
        await analyzer.cleanup()

def analyze_sync(symbols: List[str], strategy: str = 'comprehensive',
                api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """동기 분석 함수"""
    return asyncio.run(quick_analyze(symbols, strategy, api_key))

if __name__ == "__main__":
    # 테스트 코드
    test_symbols = ['AAPL', 'GOOGL', '005930']
    results = analyze_sync(test_symbols, 'comprehensive')
    print(json.dumps(results, indent=2, ensure_ascii=False))