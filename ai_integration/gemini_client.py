#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Gemini AI 클라이언트 - 100% 성능 최적화 버전
투자 분석을 위한 Gemini AI 연동 모듈
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import sqlite3
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
import os

# 설정 import 추가
from configs.settings import GEMINI_CACHE_TTL

# 환경 설정
load_dotenv()
logger = logging.getLogger(__name__)

# 100% 성능 최적화 상수
MAX_CONCURRENT = 100  # 동시 요청 수 대폭 증가
BATCH_SIZE = 50       # 배치 크기 대폭 증가
REQUEST_DELAY = 0.005 # 요청 지연 더욱 최소화
ULTRA_RETRY = 15      # 재시도 횟수 더 증가
MAX_TOKENS = 16384    # 최대 토큰 수 대폭 증가

@dataclass
class GeminiConfig:
    """Gemini 100% 최적화 설정"""
    api_key: str
    model_version: str = "gemini-1.5-flash"  # 기본값을 1.5-flash로 고정
    temperature: float = 0.05  # 일관성 극대화
    top_p: float = 0.98       # 창의성과 정확성 균형 최적화
    top_k: int = 50           # 토큰 선택 최적화
    max_output_tokens: int = MAX_TOKENS
    batch_size: int = BATCH_SIZE
    max_concurrent: int = MAX_CONCURRENT
    request_delay: float = REQUEST_DELAY
    retry_attempts: int = ULTRA_RETRY
    
    # 100% 성능 최적화 파라미터
    use_system_instruction: bool = True
    enable_safety_settings: bool = True
    response_mime_type: str = "application/json"
    candidate_count: int = 1
    
    # 새로운 울트라 최적화 설정
    enable_ultra_caching: bool = True
    smart_prompt_compression: bool = True
    adaptive_batching: bool = True
    ultra_parallel_mode: bool = True
    advanced_error_recovery: bool = True
    
    def __post_init__(self):
        """환경 변수에서 설정 읽기 - .env 파일 우선"""
        # 환경 변수에서 모델 설정 읽기 (있으면 덮어쓰기, 없으면 기본값 유지)
        env_model = os.getenv('GEMINI_MODEL')
        if env_model:
            self.model_version = env_model
        
        env_temp = os.getenv('GEMINI_TEMPERATURE')
        if env_temp:
            try:
                self.temperature = float(env_temp)
            except ValueError:
                pass  # 기본값 유지
        
        env_tokens = os.getenv('GEMINI_MAX_TOKENS')
        if env_tokens:
            try:
                self.max_output_tokens = int(env_tokens)
            except ValueError:
                pass  # 기본값 유지

class GeminiPerformanceMonitor:
    """성능 모니터링 - 고도화"""
    def __init__(self):
        self.request_times = deque(maxlen=5000)  # 더 많은 데이터 저장
        self.success_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.tokens_used = 0
        self.tokens_saved = 0  # 캐시로 절약된 토큰
        self.start_time = time.time()
        self._lock = threading.RLock()
        
        # 성능 분석용 추가 메트릭
        self.response_quality_scores = deque(maxlen=1000)
        self.model_switching_count = 0
        self.optimization_triggers = 0
        
    def record_request(self, duration: float, success: bool = True, tokens: int = 0, 
                      quality_score: float = 0.0):
        """요청 기록 - 품질 점수 포함"""
        with self._lock:
            self.request_times.append(duration)
            if success:
                self.success_count += 1
                if quality_score > 0:
                    self.response_quality_scores.append(quality_score)
            else:
                self.error_count += 1
            self.tokens_used += tokens
                
    def record_cache(self, hit: bool, tokens_saved: int = 0):
        """캐시 기록 - 절약된 토큰 포함"""
        with self._lock:
            if hit:
                self.cache_hits += 1
                self.tokens_saved += tokens_saved
            else:
                self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """고도화된 통계 조회"""
        with self._lock:
            if not self.request_times:
                return {"status": "no_data"}
            
            total = self.success_count + self.error_count
            avg_time = sum(self.request_times) / len(self.request_times)
            success_rate = (self.success_count / total * 100) if total > 0 else 0
            cache_total = self.cache_hits + self.cache_misses
            cache_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0
            uptime = time.time() - self.start_time
            
            # 품질 점수 계산
            avg_quality = sum(self.response_quality_scores) / len(self.response_quality_scores) if self.response_quality_scores else 0
            
            # 비용 효율성 계산
            cost_efficiency = (self.tokens_saved / (self.tokens_used + self.tokens_saved) * 100) if (self.tokens_used + self.tokens_saved) > 0 else 0
            
            return {
                "🚀 Gemini AI 성능": {
                    "평균응답": f"{avg_time:.3f}초",
                    "성공률": f"{success_rate:.1f}%",
                    "캐시적중": f"{cache_rate:.1f}%",
                    "응답품질": f"{avg_quality:.1f}/10",
                    "비용효율": f"{cost_efficiency:.1f}%",
                    "총요청": total,
                    "가동시간": f"{uptime:.0f}초",
                    "토큰사용": f"{self.tokens_used:,}",
                    "토큰절약": f"{self.tokens_saved:,}",
                    "예상비용": f"${(self.tokens_used * 0.00025):.4f}",
                    "절약비용": f"${(self.tokens_saved * 0.00025):.4f}"
                }
            }

class GeminiSmartCache:
    """스마트 캐싱 시스템 - 고도화"""
    def __init__(self, ttl: int = GEMINI_CACHE_TTL, max_size: int = 2000):  # 설정 파일에서 TTL 사용
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}  # 접근 횟수 추적
        self._quality_scores: Dict[str, float] = {}  # 품질 점수 추적
        self.ttl = ttl
        self.max_size = max_size
        self._lock = threading.RLock()
        
        # SQLite 캐시 초기화
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "gemini_cache_optimized.db"
        self._init_sqlite()
    
    def _init_sqlite(self):
        """SQLite 초기화 - 고도화"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gemini_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp REAL,
                    access_count INTEGER DEFAULT 1,
                    quality_score REAL DEFAULT 0.0,
                    token_count INTEGER DEFAULT 0
                )
            """)
            
            # 인덱스 추가로 성능 향상
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON gemini_cache(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quality ON gemini_cache(quality_score)")
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"SQLite 초기화 오류: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """캐시 조회 - 스마트 우선순위"""
        # 메모리 캐시 먼저 확인
        with self._lock:
            if key in self._cache:
                data = self._cache[key]
                if time.time() - data['timestamp'] < self.ttl:
                    # 접근 정보 업데이트
                    self._access_times[key] = time.time()
                    self._access_counts[key] = self._access_counts.get(key, 0) + 1
                    return data['value']
                else:
                    # 만료된 캐시 정리
                    self._cleanup_expired_key(key)
        
        # SQLite 캐시 확인
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                "SELECT value, timestamp, quality_score, token_count FROM gemini_cache WHERE key = ? AND timestamp > ?",
                (key, time.time() - self.ttl)
            )
            row = cursor.fetchone()
            
            if row:
                # 접근 횟수 업데이트
                conn.execute(
                    "UPDATE gemini_cache SET access_count = access_count + 1 WHERE key = ?",
                    (key,)
                )
                conn.commit()
            
            conn.close()
            
            if row:
                value = json.loads(row[0])
                quality_score = row[2]
                token_count = row[3]
                
                # 메모리 캐시에도 저장 (고품질 응답 우선)
                if quality_score >= 7.0:  # 높은 품질만 메모리 캐시
                    with self._lock:
                        self._cache[key] = {'value': value, 'timestamp': row[1]}
                        self._access_times[key] = time.time()
                        self._access_counts[key] = 1
                        self._quality_scores[key] = quality_score
                
                return value
        except Exception as e:
            logger.error(f"SQLite 캐시 조회 오류: {e}")
        
        return None
    
    def set(self, key: str, value: Any, quality_score: float = 0.0, token_count: int = 0):
        """캐시 저장 - 품질 기반 우선순위"""
        timestamp = time.time()
        
        # 메모리 캐시 저장 (고품질만)
        if quality_score >= 7.0:
            with self._lock:
                # 크기 제한 확인
                if len(self._cache) >= self.max_size:
                    self._evict_smart()
                
                self._cache[key] = {'value': value, 'timestamp': timestamp}
                self._access_times[key] = timestamp
                self._access_counts[key] = 1
                self._quality_scores[key] = quality_score
        
        # SQLite 캐시 저장
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """INSERT OR REPLACE INTO gemini_cache 
                   (key, value, timestamp, quality_score, token_count) 
                   VALUES (?, ?, ?, ?, ?)""",
                (key, json.dumps(value), timestamp, quality_score, token_count)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"SQLite 캐시 저장 오류: {e}")
    
    def _evict_smart(self):
        """스마트 캐시 제거 - 품질과 접근 빈도 고려"""
        if not self._cache:
            return
        
        # 품질 점수와 접근 빈도를 고려한 점수 계산
        scores = {}
        current_time = time.time()
        
        for key in self._cache.keys():
            quality = self._quality_scores.get(key, 0)
            access_count = self._access_counts.get(key, 0)
            last_access = self._access_times.get(key, 0)
            recency = max(0, 1 - (current_time - last_access) / self.ttl)
            
            # 종합 점수 (품질 50%, 접근빈도 30%, 최근성 20%)
            score = quality * 0.5 + min(access_count, 10) * 0.3 + recency * 2.0
            scores[key] = score
        
        # 가장 낮은 점수의 키 제거
        worst_key = min(scores.keys(), key=lambda k: scores[k])
        self._cleanup_expired_key(worst_key)
    
    def _cleanup_expired_key(self, key: str):
        """만료된 키 정리"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            if key in self._access_counts:
                del self._access_counts[key]
            if key in self._quality_scores:
                del self._quality_scores[key]

class GeminiClient:
    """Gemini AI 클라이언트 - 100% 성능 최적화"""
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        # 설정 초기화
        if config is None:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
            config = GeminiConfig(api_key=api_key)
        
        self.config = config
        self.monitor = GeminiPerformanceMonitor()
        self.cache = GeminiSmartCache()
        
        # Gemini 초기화 - 최적화 설정
        genai.configure(api_key=self.config.api_key)
        
        # 안전 설정 최적화
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ] if self.config.enable_safety_settings else None
        
        # 시스템 지시사항 - 투자 분석 전문가
        system_instruction = """
당신은 세계 최고 수준의 투자 분석 전문가입니다.

핵심 역량:
- 워렌 버핏, 피터 린치, 벤저민 그레이엄 수준의 분석 능력
- 정량적 분석과 정성적 분석의 완벽한 조화
- 글로벌 시장 동향과 한국 시장 특성의 깊은 이해
- 리스크 관리와 수익성 최적화의 균형

분석 원칙:
1. 데이터 기반 객관적 분석
2. 다각도 관점에서의 종합 평가
3. 투자자 유형별 맞춤 추천
4. 명확한 근거와 논리적 설명
5. 실용적이고 실행 가능한 조언

응답 형식:
- JSON 구조로 일관된 형식 제공
- 정확한 수치와 구체적인 분석 내용
- 투자 등급, 점수, 목표가격 등 명확한 결론
- 리스크 요인과 대응 방안 제시

항상 최고 품질의 분석을 제공하여 투자자의 성공적인 의사결정을 지원하세요.
""" if self.config.use_system_instruction else None
        
        # 모델 초기화
        self.model = genai.GenerativeModel(
            model_name=self.config.model_version,
            generation_config=genai.types.GenerationConfig(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_output_tokens=self.config.max_output_tokens,
                response_mime_type=self.config.response_mime_type,
                candidate_count=self.config.candidate_count
            ),
            safety_settings=safety_settings,
            system_instruction=system_instruction
        )
        
        logger.info(f"🚀 Gemini 클라이언트 최적화 완료: {self.config.model_version}")
    
    async def analyze_stock(self, prompt: str, use_cache: bool = True) -> Dict[str, Any]:
        """주식 분석 요청 - 최적화"""
        cache_key = f"stock_analysis_{hash(prompt)}"
        
        # 캐시 확인
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                # 토큰 절약 계산
                estimated_tokens = len(prompt.split()) * 1.3
                self.monitor.record_cache(True, int(estimated_tokens))
                return cached
            self.monitor.record_cache(False)
        
        # AI 분석 실행
        start_time = time.time()
        try:
            await asyncio.sleep(self.config.request_delay)
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            
            if response.text:
                # 응답 품질 평가
                quality_score = self._evaluate_response_quality(response.text, prompt)
                
                result = {
                    "analysis": response.text,
                    "timestamp": time.time(),
                    "model": self.config.model_version,
                    "quality_score": quality_score,
                    "token_count": len(response.text.split())
                }
                
                # 캐시 저장
                if use_cache:
                    self.cache.set(cache_key, result, quality_score, result["token_count"])
                
                # 성능 기록
                duration = time.time() - start_time
                self.monitor.record_request(duration, True, len(response.text.split()), quality_score)
                
                return result
            else:
                raise Exception("응답이 비어있습니다.")
                
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.record_request(duration, False)
            logger.error(f"Gemini 분석 오류: {e}")
            raise
    
    def _evaluate_response_quality(self, response: str, prompt: str) -> float:
        """응답 품질 평가 - 0-10점"""
        try:
            score = 5.0  # 기본 점수
            
            # 길이 평가 (적절한 길이)
            if 500 <= len(response) <= 3000:
                score += 1.0
            elif len(response) < 200:
                score -= 2.0
            
            # 구조화 평가 (JSON, 섹션 구분 등)
            if any(marker in response for marker in ['{', '}', '1.', '2.', '##', '**']):
                score += 1.0
            
            # 투자 관련 키워드 포함 여부
            investment_keywords = ['투자', '수익률', 'PER', 'PBR', 'ROE', '매출', '순이익', '리스크', '추천']
            keyword_count = sum(1 for keyword in investment_keywords if keyword in response)
            score += min(keyword_count * 0.3, 2.0)
            
            # 수치 정보 포함 여부 (구체적인 분석)
            import re
            numbers = re.findall(r'\d+\.?\d*%?', response)
            if len(numbers) >= 5:
                score += 1.0
            
            return min(score, 10.0)
        except:
            return 5.0
    
    async def batch_analyze(self, prompts: List[str], use_cache: bool = True) -> List[Dict[str, Any]]:
        """배치 분석 - 최적화"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def analyze_with_semaphore(prompt: str):
            async with semaphore:
                return await self.analyze_stock(prompt, use_cache)
        
        # 모든 프롬프트를 병렬 처리
        tasks = [analyze_with_semaphore(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"배치 분석 오류 (인덱스 {i}): {result}")
                processed_results.append({
                    "error": str(result),
                    "timestamp": time.time(),
                    "model": self.config.model_version
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        return self.monitor.get_stats()
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cache = GeminiSmartCache()
        logger.info("캐시가 초기화되었습니다.")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        # 정리 작업
        pass

# 편의 함수들
async def create_optimized_client(api_key: Optional[str] = None) -> GeminiClient:
    """최적화된 Gemini 클라이언트 생성"""
    if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')
    
    # 환경 변수에서 설정 읽기 (없으면 기본값 사용)
    model_version = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
    temperature = float(os.getenv('GEMINI_TEMPERATURE', '0.05'))
    max_tokens = int(os.getenv('GEMINI_MAX_TOKENS', str(MAX_TOKENS)))
    
    config = GeminiConfig(
        api_key=api_key,
        model_version=model_version,        # .env에서 읽은 모델
        temperature=temperature,            # .env에서 읽은 온도
        max_output_tokens=max_tokens,       # .env에서 읽은 토큰 수
        max_concurrent=100,                 # 동시 요청 최대화
        batch_size=50,                     # 배치 크기 최적화
        request_delay=0.005,               # 지연 최소화
        retry_attempts=15                  # 재시도 최대화
    )
    
    return GeminiClient(config)

def create_expert_prompt(stock_data: Dict[str, Any], analysis_type: str = "comprehensive") -> str:
    """전문가 수준 프롬프트 생성"""
    
    base_prompt = f"""
# 주식 투자 분석 요청

## 종목 정보
- 종목명: {stock_data.get('name', 'N/A')} ({stock_data.get('symbol', 'N/A')})
- 현재가: {stock_data.get('price', 0):,.0f}원
- 시가총액: {stock_data.get('market_cap', 0):,.0f}원
- 섹터: {stock_data.get('sector', 'N/A')}

## 재무 지표
- PER: {stock_data.get('pe_ratio', 'N/A')}
- PBR: {stock_data.get('pb_ratio', 'N/A')}
- ROE: {stock_data.get('roe', 'N/A')}%
- 부채비율: {stock_data.get('debt_ratio', 'N/A')}%
- 유동비율: {stock_data.get('current_ratio', 'N/A')}%

## 분석 요청
다음 형식의 JSON으로 응답해주세요:

{{
  "investment_grade": "투자등급 (S/A+/A/B+/B/C+/C/D)",
  "investment_score": "투자점수 (0-100)",
  "target_price": "목표가격 (12개월)",
  "upside_potential": "상승여력 (%)",
  "strengths": ["강점1", "강점2", "강점3"],
  "weaknesses": ["약점1", "약점2", "약점3"],
  "investment_strategy": "투자전략 추천",
  "risk_factors": ["리스크1", "리스크2", "리스크3"],
  "sector_outlook": "섹터 전망",
  "recommendation": "최종 추천 (매수/보유/매도)",
  "confidence_level": "신뢰도 (1-10)",
  "analysis_summary": "종합 분석 요약"
}}

세계 최고 수준의 투자 전문가로서 정확하고 실용적인 분석을 제공해주세요.
"""
    
    return base_prompt.strip() 