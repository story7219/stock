"""
🚀 Ultra 고성능 API 매니저
- 비동기 HTTP 커넥션 풀링
- 멀티레벨 캐싱 & 배치 처리
- 자동 재시도 & 회로 차단기
- 실시간 성능 모니터링
- 메모리 최적화 & 압축
"""

import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import gzip
import structlog
from concurrent.futures import ThreadPoolExecutor
import weakref
import ssl
from urllib.parse import urljoin, urlencode

from config.settings import settings
from core.cache_manager import get_cache_manager, cached
from core.performance_monitor import monitor_performance

logger = structlog.get_logger(__name__)


class RequestMethod(Enum):
    """HTTP 요청 메서드"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class CircuitState(Enum):
    """회로 차단기 상태"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class APIEndpoint:
    """API 엔드포인트 설정"""
    url: str
    method: RequestMethod = RequestMethod.GET
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retries: int = 3
    cache_ttl: int = 300
    rate_limit: int = 100  # requests per minute
    priority: int = 1  # 1=highest, 5=lowest


@dataclass
class RequestStats:
    """요청 통계"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    rate_limited: int = 0
    circuit_breaks: int = 0
    
    @property
    def success_rate(self) -> float:
        """성공률"""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """캐시 히트율"""
        total_cache_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0


@dataclass
class BatchRequest:
    """배치 요청"""
    endpoint: APIEndpoint
    params: Dict[str, Any] = field(default_factory=dict)
    data: Optional[Dict[str, Any]] = None
    callback: Optional[Callable] = None
    priority: int = 1


class CircuitBreaker:
    """회로 차단기 패턴"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable) -> Any:
        """회로 차단기를 통한 함수 호출"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """성공 시 처리"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """실패 시 처리"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class RateLimiter:
    """속도 제한기"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self) -> bool:
        """요청 허용 여부 확인"""
        now = time.time()
        # 시간 윈도우 밖의 요청 제거
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False


class UltraAPIManager:
    """🚀 Ultra 고성능 API 매니저"""
    
    def __init__(self):
        # 커넥션 풀 설정
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        # 성능 최적화
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._batch_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        
        # 캐시 및 상태 관리
        self._cache_manager = get_cache_manager()
        self._stats = RequestStats()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        
        # 백그라운드 작업
        self._workers: List[asyncio.Task] = []
        self._batch_workers: List[asyncio.Task] = []
        
        # 세션 추적 (약한 참조)
        self._active_sessions: weakref.WeakSet = weakref.WeakSet()
        
        logger.info("Ultra API 매니저 초기화")
    
    async def initialize(self) -> None:
        """API 매니저 초기화"""
        try:
            # SSL 컨텍스트 설정
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 고성능 커넥터 생성
            self._connector = aiohttp.TCPConnector(
                limit=settings.api.max_connections,
                limit_per_host=settings.api.max_connections_per_host,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
                ssl=ssl_context
            )
            
            # 세션 생성
            timeout = aiohttp.ClientTimeout(
                total=settings.api.timeout,
                connect=settings.api.connect_timeout,
                sock_read=settings.api.read_timeout
            )
            
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                headers={'User-Agent': 'UltraAPIManager/1.0'},
                json_serialize=json.dumps
            )
            
            self._active_sessions.add(self._session)
            
            # 백그라운드 워커 시작
            await self._start_workers()
            
            logger.info("Ultra API 매니저 초기화 완료")
            
        except Exception as e:
            logger.error(f"API 매니저 초기화 실패: {e}")
            raise
    
    async def _start_workers(self) -> None:
        """백그라운드 워커 시작"""
        # 요청 처리 워커
        for i in range(settings.performance.api_workers):
            worker = asyncio.create_task(self._request_worker(f"req_worker_{i}"))
            self._workers.append(worker)
        
        # 배치 처리 워커
        for i in range(2):
            batch_worker = asyncio.create_task(self._batch_worker(f"batch_worker_{i}"))
            self._batch_workers.append(batch_worker)
        
        # 통계 업데이트 워커
        stats_worker = asyncio.create_task(self._stats_worker())
        self._workers.append(stats_worker)
    
    @monitor_performance("api_request")
    async def request(
        self,
        endpoint: Union[str, APIEndpoint],
        method: RequestMethod = RequestMethod.GET,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        use_cache: bool = True,
        priority: int = 1
    ) -> Dict[str, Any]:
        """단일 API 요청"""
        
        # 엔드포인트 객체 생성
        if isinstance(endpoint, str):
            endpoint_obj = APIEndpoint(
                url=endpoint,
                method=method,
                headers=headers or {},
                timeout=timeout or settings.api.timeout,
                priority=priority
            )
        else:
            endpoint_obj = endpoint
        
        # 캐시 키 생성
        cache_key = self._generate_cache_key(endpoint_obj, params, data)
        
        # 캐시 확인
        if use_cache and endpoint_obj.method == RequestMethod.GET:
            cached_result = await self._cache_manager.get(cache_key)
            if cached_result:
                self._stats.cache_hits += 1
                return cached_result
            self._stats.cache_misses += 1
        
        # 속도 제한 확인
        rate_limiter = self._get_rate_limiter(endpoint_obj.url)
        if not await rate_limiter.acquire():
            self._stats.rate_limited += 1
            await asyncio.sleep(1)  # 잠시 대기
        
        # 회로 차단기 확인
        circuit_breaker = self._get_circuit_breaker(endpoint_obj.url)
        
        try:
            # 실제 요청 수행
            result = await circuit_breaker.call(
                lambda: self._execute_request(endpoint_obj, params, data)
            )
            
            # 캐시 저장
            if use_cache and endpoint_obj.method == RequestMethod.GET:
                await self._cache_manager.set(cache_key, result, endpoint_obj.cache_ttl)
            
            self._stats.successful_requests += 1
            return result
            
        except Exception as e:
            self._stats.failed_requests += 1
            logger.error(f"API 요청 실패 {endpoint_obj.url}: {e}")
            raise
        finally:
            self._stats.total_requests += 1
    
    async def _execute_request(
        self,
        endpoint: APIEndpoint,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """실제 HTTP 요청 실행"""
        if not self._session:
            raise RuntimeError("API 매니저가 초기화되지 않았습니다")
        
        start_time = time.time()
        
        try:
            # 요청 파라미터 준비
            request_kwargs = {
                'url': endpoint.url,
                'method': endpoint.method.value,
                'headers': endpoint.headers,
                'timeout': aiohttp.ClientTimeout(total=endpoint.timeout)
            }
            
            if params:
                request_kwargs['params'] = params
            
            if data:
                if endpoint.method in [RequestMethod.POST, RequestMethod.PUT, RequestMethod.PATCH]:
                    request_kwargs['json'] = data
            
            # HTTP 요청 실행
            async with self._session.request(**request_kwargs) as response:
                response.raise_for_status()
                
                # 응답 처리
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    result = await response.json()
                else:
                    text = await response.text()
                    result = {'data': text, 'status': response.status}
                
                # 응답 시간 업데이트
                response_time = time.time() - start_time
                self._update_response_time(response_time)
                
                return result
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP 클라이언트 오류: {e}")
            raise
        except Exception as e:
            logger.error(f"요청 실행 중 오류: {e}")
            raise
    
    async def batch_request(
        self,
        requests: List[BatchRequest],
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """배치 요청 처리"""
        
        # 우선순위별로 정렬
        sorted_requests = sorted(requests, key=lambda x: x.priority)
        
        # 세마포어로 동시 요청 수 제한
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_request(batch_req: BatchRequest) -> Dict[str, Any]:
            async with semaphore:
                try:
                    result = await self.request(
                        batch_req.endpoint,
                        params=batch_req.params,
                        data=batch_req.data,
                        priority=batch_req.priority
                    )
                    
                    # 콜백 실행
                    if batch_req.callback:
                        await self._executor.run_in_thread(batch_req.callback, result)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"배치 요청 실패: {e}")
                    return {'error': str(e)}
        
        # 모든 요청을 병렬로 처리
        tasks = [process_single_request(req) for req in sorted_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r if not isinstance(r, Exception) else {'error': str(r)} for r in results]
    
    async def _request_worker(self, worker_name: str) -> None:
        """요청 처리 워커"""
        logger.info(f"요청 워커 {worker_name} 시작")
        
        while True:
            try:
                # 큐에서 요청 가져오기
                request_data = await asyncio.wait_for(
                    self._request_queue.get(), timeout=1.0
                )
                
                # 요청 처리
                await self._process_queued_request(request_data)
                self._request_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"요청 워커 {worker_name} 오류: {e}")
                await asyncio.sleep(1)
    
    async def _batch_worker(self, worker_name: str) -> None:
        """배치 처리 워커"""
        logger.info(f"배치 워커 {worker_name} 시작")
        
        while True:
            try:
                batch_data = await asyncio.wait_for(
                    self._batch_queue.get(), timeout=5.0
                )
                
                await self._process_batch(batch_data)
                self._batch_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"배치 워커 {worker_name} 오류: {e}")
                await asyncio.sleep(1)
    
    async def _stats_worker(self) -> None:
        """통계 업데이트 워커"""
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다 통계 업데이트
                await self._update_stats()
            except Exception as e:
                logger.error(f"통계 워커 오류: {e}")
    
    def _generate_cache_key(
        self,
        endpoint: APIEndpoint,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """캐시 키 생성"""
        key_parts = [
            endpoint.url,
            endpoint.method.value,
            json.dumps(params or {}, sort_keys=True),
            json.dumps(data or {}, sort_keys=True)
        ]
        
        key_string = '|'.join(key_parts)
        return f"api_cache:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _get_circuit_breaker(self, url: str) -> CircuitBreaker:
        """회로 차단기 가져오기"""
        if url not in self._circuit_breakers:
            self._circuit_breakers[url] = CircuitBreaker(
                failure_threshold=settings.api.circuit_breaker_threshold,
                timeout=settings.api.circuit_breaker_timeout
            )
        return self._circuit_breakers[url]
    
    def _get_rate_limiter(self, url: str) -> RateLimiter:
        """속도 제한기 가져오기"""
        if url not in self._rate_limiters:
            self._rate_limiters[url] = RateLimiter(
                max_requests=settings.api.rate_limit_per_minute,
                time_window=60
            )
        return self._rate_limiters[url]
    
    def _update_response_time(self, response_time: float) -> None:
        """응답 시간 업데이트"""
        if self._stats.avg_response_time == 0:
            self._stats.avg_response_time = response_time
        else:
            # 이동 평균 계산
            self._stats.avg_response_time = (
                self._stats.avg_response_time * 0.9 + response_time * 0.1
            )
    
    async def _update_stats(self) -> None:
        """통계 정보 업데이트"""
        try:
            # 캐시에 통계 저장
            stats_data = {
                'total_requests': self._stats.total_requests,
                'success_rate': self._stats.success_rate,
                'avg_response_time': self._stats.avg_response_time,
                'cache_hit_rate': self._stats.cache_hit_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._cache_manager.set('api_stats', stats_data, 3600)
            
        except Exception as e:
            logger.error(f"통계 업데이트 실패: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """API 통계 반환"""
        return {
            'total_requests': self._stats.total_requests,
            'successful_requests': self._stats.successful_requests,
            'failed_requests': self._stats.failed_requests,
            'success_rate': self._stats.success_rate,
            'avg_response_time': self._stats.avg_response_time,
            'cache_hits': self._stats.cache_hits,
            'cache_misses': self._stats.cache_misses,
            'cache_hit_rate': self._stats.cache_hit_rate,
            'rate_limited': self._stats.rate_limited,
            'circuit_breaks': self._stats.circuit_breaks,
            'active_sessions': len(self._active_sessions),
            'circuit_breakers': len(self._circuit_breakers),
            'rate_limiters': len(self._rate_limiters)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        try:
            # 간단한 테스트 요청
            test_url = "https://httpbin.org/get"
            start_time = time.time()
            
            await self.request(test_url, use_cache=False)
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'stats': self.get_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def close(self) -> None:
        """API 매니저 종료"""
        try:
            # 워커 종료
            for worker in self._workers + self._batch_workers:
                worker.cancel()
            
            # 큐 비우기
            while not self._request_queue.empty():
                self._request_queue.get_nowait()
                self._request_queue.task_done()
            
            while not self._batch_queue.empty():
                self._batch_queue.get_nowait()
                self._batch_queue.task_done()
            
            # 세션 종료
            if self._session:
                await self._session.close()
            
            # 커넥터 종료
            if self._connector:
                await self._connector.close()
            
            # 스레드 풀 종료
            self._executor.shutdown(wait=True)
            
            logger.info("Ultra API 매니저 종료 완료")
            
        except Exception as e:
            logger.error(f"API 매니저 종료 중 오류: {e}")


# 전역 API 매니저 인스턴스
_api_manager: Optional[UltraAPIManager] = None


async def get_api_manager() -> UltraAPIManager:
    """API 매니저 싱글톤 인스턴스 반환"""
    global _api_manager
    
    if _api_manager is None:
        _api_manager = UltraAPIManager()
        await _api_manager.initialize()
    
    return _api_manager


async def initialize_api_manager() -> None:
    """API 매니저 초기화"""
    await get_api_manager()


async def cleanup_api_manager() -> None:
    """API 매니저 정리"""
    global _api_manager
    
    if _api_manager:
        await _api_manager.close()
        _api_manager = None 