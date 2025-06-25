"""
Connection pool manager for HTTP requests and database connections.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from contextlib import asynccontextmanager
import aiohttp
import ssl
from urllib.parse import urlparse

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Connection pool statistics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    last_reset: float = 0.0


class HTTPConnectionPool:
    """HTTP connection pool with rate limiting and retry logic."""
    
    def __init__(self,
                 max_connections: int = 100,
                 max_connections_per_host: int = 30,
                 timeout: int = 30,
                 rate_limit_per_minute: int = 1000,
                 retry_attempts: int = 3,
                 retry_delay: float = 1.0):
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.timeout = timeout
        self.rate_limit_per_minute = rate_limit_per_minute
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._rate_limiter: Dict[str, List[float]] = {}
        self._stats = ConnectionStats()
        self._stats.last_reset = time.time()
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        # Create SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Create connector with connection pooling
        self._connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections_per_host,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=ssl_context,
            enable_cleanup_closed=True
        )
        
        # Create session with timeout
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=timeout,
            headers={'User-Agent': 'Investment-System/1.0'}
        )
        
        logger.info(f"HTTP connection pool initialized with {self.max_connections} max connections")
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self._session:
            await self._session.close()
        if self._connector:
            await self._connector.close()
        logger.info("HTTP connection pool closed")
    
    def _check_rate_limit(self, host: str) -> bool:
        """Check if request is within rate limit."""
        current_time = time.time()
        
        if host not in self._rate_limiter:
            self._rate_limiter[host] = []
        
        # Remove requests older than 1 minute
        self._rate_limiter[host] = [
            req_time for req_time in self._rate_limiter[host]
            if current_time - req_time < 60
        ]
        
        # Check if within rate limit
        if len(self._rate_limiter[host]) >= self.rate_limit_per_minute:
            return False
        
        # Add current request
        self._rate_limiter[host].append(current_time)
        return True
    
    async def request(self,
                     method: str,
                     url: str,
                     **kwargs) -> Optional[aiohttp.ClientResponse]:
        """Make HTTP request with rate limiting and retry logic."""
        if not self._session:
            await self.initialize()
        
        parsed_url = urlparse(url)
        host = parsed_url.netloc
        
        # Check rate limit
        if not self._check_rate_limit(host):
            logger.warning(f"Rate limit exceeded for {host}")
            await asyncio.sleep(60 / self.rate_limit_per_minute)
        
        # Retry logic
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                start_time = time.time()
                
                async with self._lock:
                    self._stats.active_connections += 1
                    self._stats.total_requests += 1
                
                response = await self._session.request(method, url, **kwargs)
                
                # Update stats
                response_time = time.time() - start_time
                async with self._lock:
                    self._stats.active_connections -= 1
                    # Update average response time
                    if self._stats.avg_response_time == 0:
                        self._stats.avg_response_time = response_time
                    else:
                        self._stats.avg_response_time = (
                            self._stats.avg_response_time * 0.9 + response_time * 0.1
                        )
                
                return response
                
            except Exception as e:
                last_exception = e
                async with self._lock:
                    self._stats.active_connections -= 1
                    self._stats.failed_connections += 1
                
                if attempt < self.retry_attempts - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.retry_attempts} attempts: {e}")
        
        return None
    
    async def get(self, url: str, **kwargs) -> Optional[aiohttp.ClientResponse]:
        """Make GET request."""
        return await self.request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> Optional[aiohttp.ClientResponse]:
        """Make POST request."""
        return await self.request('POST', url, **kwargs)
    
    def get_stats(self) -> ConnectionStats:
        """Get connection pool statistics."""
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset connection pool statistics."""
        self._stats = ConnectionStats()
        self._stats.last_reset = time.time()


class DatabaseConnectionPool:
    """Database connection pool for PostgreSQL."""
    
    def __init__(self,
                 database_url: str,
                 min_size: int = 10,
                 max_size: int = 20,
                 command_timeout: int = 60):
        self.database_url = database_url
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self._pool: Optional[asyncpg.Pool] = None
        self._stats = ConnectionStats()
    
    async def initialize(self) -> bool:
        """Initialize database connection pool."""
        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available, database pool disabled")
            return False
        
        try:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=self.command_timeout
            )
            
            # Test connection
            async with self._pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            
            self._stats.total_connections = self.max_size
            logger.info(f"Database connection pool initialized with {self.max_size} max connections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            return False
    
    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire database connection from pool."""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        
        start_time = time.time()
        self._stats.active_connections += 1
        
        try:
            async with self._pool.acquire() as conn:
                yield conn
        finally:
            self._stats.active_connections -= 1
            response_time = time.time() - start_time
            
            # Update average response time
            if self._stats.avg_response_time == 0:
                self._stats.avg_response_time = response_time
            else:
                self._stats.avg_response_time = (
                    self._stats.avg_response_time * 0.9 + response_time * 0.1
                )
    
    async def execute(self, query: str, *args) -> Any:
        """Execute database query."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[Any]:
        """Fetch multiple rows from database."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[Any]:
        """Fetch single row from database."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch single value from database."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    def get_stats(self) -> ConnectionStats:
        """Get database pool statistics."""
        if self._pool:
            self._stats.idle_connections = self._pool.get_idle_size()
            self._stats.total_connections = self._pool.get_size()
        return self._stats


class ConnectionPool:
    """Main connection pool manager."""
    
    def __init__(self,
                 max_http_connections: int = 100,
                 max_http_connections_per_host: int = 30,
                 http_timeout: int = 30,
                 rate_limit_per_minute: int = 1000,
                 database_url: Optional[str] = None,
                 db_min_size: int = 10,
                 db_max_size: int = 20):
        
        self.http_pool = HTTPConnectionPool(
            max_connections=max_http_connections,
            max_connections_per_host=max_http_connections_per_host,
            timeout=http_timeout,
            rate_limit_per_minute=rate_limit_per_minute
        )
        
        self.db_pool = None
        if database_url:
            self.db_pool = DatabaseConnectionPool(
                database_url=database_url,
                min_size=db_min_size,
                max_size=db_max_size
            )
    
    async def initialize(self) -> None:
        """Initialize all connection pools."""
        await self.http_pool.initialize()
        
        if self.db_pool:
            await self.db_pool.initialize()
        
        logger.info("Connection pools initialized")
    
    async def close(self) -> None:
        """Close all connection pools."""
        await self.http_pool.close()
        
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("Connection pools closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all connection pools."""
        stats = {
            'http': self.http_pool.get_stats().__dict__
        }
        
        if self.db_pool:
            stats['database'] = self.db_pool.get_stats().__dict__
        
        return stats


# Global connection pool instance
connection_pool: Optional[ConnectionPool] = None


def get_connection_pool() -> ConnectionPool:
    """Get global connection pool instance."""
    global connection_pool
    if connection_pool is None:
        from .config import config
        connection_pool = ConnectionPool(
            max_http_connections=config.api.max_concurrent_requests,
            http_timeout=config.api.yahoo_finance_timeout,
            rate_limit_per_minute=config.api.rate_limit_per_minute,
            database_url=config.get_database_url() if config.database.host else None,
            db_min_size=config.database.pool_size // 2,
            db_max_size=config.database.pool_size
        )
    return connection_pool 