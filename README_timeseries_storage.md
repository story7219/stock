# 시계열 데이터 저장 및 관리 시스템

## 📋 개요

엔터프라이즈급 실시간 시계열 데이터 저장 및 관리 시스템입니다. TimescaleDB/InfluxDB 최적화, 계층화 저장, 인덱싱/쿼리 최적화, 자동 백업/복구, 스토리지 모니터링을 제공합니다.

## 🏗️ 아키텍처

### 시스템 구성
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Real-time     │    │   TimescaleDB   │    │   InfluxDB      │
│   Data Input    │───▶│   (Primary)     │───▶│   (Optional)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Redis       │    │   Compression   │    │   Cloud S3      │
│   (Cache)       │    │   & Archive     │    │   (Backup)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 계층화 저장 전략
1. **실시간 데이터**: Redis (메모리, 1시간 TTL)
2. **단기 데이터**: TimescaleDB (SSD, 7일)
3. **중기 데이터**: TimescaleDB 압축 (7일-180일)
4. **장기 데이터**: S3 압축 저장 (180일+)

## 🚀 주요 기능

### 1. 시계열 데이터베이스 최적화
- **TimescaleDB**: PostgreSQL 기반 시계열 확장
- **하이퍼테이블**: 자동 파티셔닝 및 압축
- **인덱싱**: 시간 기반 + 복합 인덱스
- **집계**: 사전 집계 테이블

### 2. 계층화된 저장
- **Redis**: 실시간 데이터 캐싱
- **TimescaleDB**: 주 저장소
- **S3**: 장기 백업 및 아카이브

### 3. 인덱싱 최적화
- 시간 기반 인덱스
- 복합 인덱스 (시간+종목)
- 부분 인덱스 (최근 데이터)
- 클러스터링 인덱스

### 4. 쿼리 최적화
- 사전 집계 테이블
- 캐싱 레이어
- 병렬 쿼리 처리
- 쿼리 최적화

### 5. 데이터 관리
- 자동 백업 및 복구
- 데이터 품질 모니터링
- 스토리지 사용량 관리
- 성능 튜닝 자동화

## 📦 설치

### 1. 의존성 설치

```bash
# 필수 패키지
pip install asyncpg aioredis sqlalchemy psycopg2-binary pandas numpy boto3

# 선택사항 (InfluxDB 사용 시)
pip install influxdb-client

# 개발 도구
pip install pytest pytest-asyncio black flake8 mypy
```

### 2. 데이터베이스 설정

#### TimescaleDB 설치 (Ubuntu/Debian)
```bash
# PostgreSQL 설치
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# TimescaleDB 설치
sudo apt-get install timescaledb-postgresql-14

# 설정 활성화
sudo timescaledb-tune --quiet --yes
sudo systemctl restart postgresql

# 데이터베이스 생성
sudo -u postgres psql
CREATE DATABASE timeseries;
\c timeseries
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

#### Redis 설치
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# 또는 Docker
docker run -d --name redis -p 6379:6379 redis:latest
```

#### InfluxDB 설치 (선택사항)
```bash
# Docker
docker run -d --name influxdb -p 8086:8086 influxdb:latest

# 또는 직접 설치
wget https://dl.influxdata.com/influxdb/releases/influxdb2-2.7.1-amd64.deb
sudo dpkg -i influxdb2-2.7.1-amd64.deb
sudo systemctl start influxdb
```

### 3. 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << EOF
# TimescaleDB
TIMESCALE_DSN=postgresql://user:password@localhost:5432/timeseries

# Redis
REDIS_URL=redis://localhost:6379

# InfluxDB (선택사항)
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=your_token_here
INFLUX_ORG=your_organization
INFLUX_BUCKET=trading_data

# AWS S3 (선택사항)
S3_BUCKET=timeseries-backup
S3_REGION=ap-northeast-2
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
EOF

# 환경 변수 로드
source .env
```

## ⚙️ 설정

### 1. 기본 설정 파일 사용

```bash
# 기본 설정으로 실행
python run_timeseries_storage.py
```

### 2. 커스텀 설정 파일 사용

```bash
# 설정 파일 생성
cp config/storage_config.json my_storage_config.json

# 설정 파일 편집
vim my_storage_config.json

# 커스텀 설정으로 실행
python run_timeseries_storage.py --config my_storage_config.json
```

### 3. 설정 검증

```bash
# 의존성 체크
python run_timeseries_storage.py --check-deps

# 설정 검증
python run_timeseries_storage.py --validate-config

# 시스템 테스트
python run_timeseries_storage.py --test
```

## 🔧 사용법

### 1. 시스템 시작

```python
import asyncio
from src.timeseries_storage_system import TimeSeriesStorageSystem, StorageConfig

async def main():
    # 설정 로드
    config = StorageConfig(
        timescale_dsn="postgresql://user:pass@localhost:5432/timeseries",
        redis_url="redis://localhost:6379"
    )
    
    # 시스템 생성 및 시작
    system = TimeSeriesStorageSystem(config)
    await system.initialize()
    await system.start()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 데이터 저장

```python
# 실시간 데이터 저장
data = {
    'symbol': '005930',
    'timestamp': datetime.now(),
    'price': 75000.0,
    'volume': 1000000,
    'open_price': 74800.0,
    'high_price': 75200.0,
    'low_price': 74700.0,
    'close_price': 75000.0,
    'data_type': 'price'
}

success = await system.store_data(data)
```

### 3. 데이터 쿼리

```python
# 최적화된 쿼리
from datetime import datetime, timedelta

start_time = datetime.now() - timedelta(hours=1)
end_time = datetime.now()

data = await system.query_data(
    symbol='005930',
    start_time=start_time,
    end_time=end_time,
    interval='1 minute'
)
```

### 4. 시스템 상태 모니터링

```python
# 시스템 상태 조회
status = await system.get_system_status()
print(f"스토리지 사용량: {status['storage_usage']['total_size_gb']:.2f} GB")
print(f"쿼리 성능: {len(status['query_performance']['slow_queries'])} slow queries")
```

## 📊 성능 지표

### 목표 성능
- **쓰기**: 10,000+ records/sec
- **읽기**: 100,000+ records/sec
- **쿼리**: < 100ms
- **백업**: 자동화

### 모니터링 메트릭
- 스토리지 사용량
- 쿼리 성능
- 데이터 품질
- 백업 상태

## 🔍 클래스 구조

### 1. TimeSeriesDB
```python
class TimeSeriesDB:
    """시계열 데이터베이스 최적화"""
    
    async def initialize(self):
        """데이터베이스 초기화"""
    
    async def insert_data(self, data: Dict[str, Any]) -> bool:
        """데이터 삽입"""
    
    async def query_data(self, symbol: str, start_time: datetime, 
                        end_time: datetime, data_type: str = 'price') -> List[Dict[str, Any]]:
        """데이터 쿼리"""
    
    async def get_aggregates(self, symbol: str, start_time: datetime, 
                           end_time: datetime, interval: str = '1 hour') -> List[Dict[str, Any]]:
        """집계 데이터 쿼리"""
```

### 2. DataArchiver
```python
class DataArchiver:
    """계층화된 저장 관리"""
    
    async def store_realtime(self, data: Dict[str, Any]):
        """실시간 데이터 저장 (Redis)"""
    
    async def archive_longterm(self, symbol: str, start_date: datetime, end_date: datetime):
        """장기 데이터 압축 및 클라우드 저장"""
    
    async def restore_from_s3(self, symbol: str, date: datetime) -> List[Dict[str, Any]]:
        """S3에서 데이터 복원"""
```

### 3. QueryOptimizer
```python
class QueryOptimizer:
    """쿼리 최적화"""
    
    async def optimized_query(self, symbol: str, start_time: datetime, 
                            end_time: datetime, interval: str = '1 hour') -> List[Dict[str, Any]]:
        """최적화된 쿼리"""
    
    async def parallel_query(self, symbols: List[str], start_time: datetime, 
                           end_time: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """병렬 쿼리"""
```

### 4. BackupManager
```python
class BackupManager:
    """자동 백업 및 복구"""
    
    async def create_backup(self, symbols: List[str] = None) -> bool:
        """백업 생성"""
    
    async def restore_backup(self, backup_date: datetime, symbols: List[str] = None) -> bool:
        """백업 복원"""
    
    async def schedule_backup(self):
        """정기 백업 스케줄링"""
```

### 5. StorageMonitor
```python
class StorageMonitor:
    """스토리지 사용량 및 성능 모니터링"""
    
    async def get_storage_usage(self) -> Dict[str, Any]:
        """스토리지 사용량 조회"""
    
    async def get_query_performance(self) -> Dict[str, Any]:
        """쿼리 성능 조회"""
    
    async def monitor_quality(self) -> Dict[str, Any]:
        """데이터 품질 모니터링"""
    
    async def auto_tune(self):
        """성능 튜닝 자동화"""
```

## 🛠️ 고급 설정

### 1. 성능 최적화

#### TimescaleDB 튜닝
```sql
-- 메모리 설정
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';

-- 쿼리 최적화
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- 재시작
SELECT pg_reload_conf();
```

#### Redis 튜닝
```bash
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### 2. 모니터링 설정

#### Prometheus 메트릭
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'timeseries_storage'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

#### Grafana 대시보드
```json
{
  "dashboard": {
    "title": "Timeseries Storage Dashboard",
    "panels": [
      {
        "title": "Storage Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "storage_usage_bytes",
            "legendFormat": "{{symbol}}"
          }
        ]
      }
    ]
  }
}
```

### 3. 백업 전략

#### 자동 백업 스케줄
```bash
# crontab
0 2 * * * /usr/bin/python3 /path/to/backup_script.py
0 3 * * 0 /usr/bin/python3 /path/to/full_backup_script.py
```

#### S3 라이프사이클 정책
```json
{
  "Rules": [
    {
      "ID": "ArchiveRule",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "archive/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

## 🔧 문제 해결

### 1. 일반적인 문제

#### 연결 오류
```bash
# TimescaleDB 연결 확인
psql -h localhost -U user -d timeseries -c "SELECT version();"

# Redis 연결 확인
redis-cli ping

# InfluxDB 연결 확인
curl -G http://localhost:8086/query --data-urlencode "q=SHOW DATABASES"
```

#### 성능 문제
```sql
-- 느린 쿼리 확인
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements 
WHERE query LIKE '%timeseries_data%'
ORDER BY total_time DESC
LIMIT 10;

-- 인덱스 사용률 확인
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
```

### 2. 로그 분석

#### 로그 레벨 설정
```python
import logging

# 디버그 로그 활성화
logging.getLogger('src.timeseries_storage_system').setLevel(logging.DEBUG)

# 파일 로그 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
```

#### 성능 프로파일링
```python
import cProfile
import pstats

# 프로파일링 시작
profiler = cProfile.Profile()
profiler.enable()

# 코드 실행
await system.query_data('005930', start_time, end_time)

# 프로파일링 결과
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## 📈 확장성

### 1. 수평 확장

#### 샤딩 전략
```python
# 심볼 기반 샤딩
def get_shard_key(symbol: str) -> int:
    return hash(symbol) % num_shards

# 시간 기반 샤딩
def get_time_shard(timestamp: datetime) -> int:
    return timestamp.year * 12 + timestamp.month
```

#### 로드 밸런싱
```python
# 라운드 로빈 로드 밸런서
class LoadBalancer:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.current = 0
    
    def get_next_server(self) -> str:
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server
```

### 2. 수직 확장

#### 메모리 최적화
```python
# 메모리 매핑
import mmap

class MemoryMappedStorage:
    def __init__(self, file_path: str):
        self.file = open(file_path, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), 0)
    
    def read_data(self, offset: int, size: int) -> bytes:
        return self.mmap[offset:offset + size]
```

#### CPU 최적화
```python
# 멀티프로세싱
import multiprocessing as mp

def parallel_process_data(data_chunks: List[List[Dict]]) -> List[Dict]:
    with mp.Pool() as pool:
        results = pool.map(process_chunk, data_chunks)
    return [item for sublist in results for item in sublist]
```

## 🔒 보안

### 1. 데이터 암호화

#### 전송 중 암호화
```python
import ssl

# SSL 연결 설정
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# TimescaleDB SSL 연결
dsn = "postgresql://user:pass@localhost:5432/timeseries?sslmode=require"
```

#### 저장 중 암호화
```python
from cryptography.fernet import Fernet

class EncryptedStorage:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)
    
    def encrypt_data(self, data: bytes) -> bytes:
        return self.cipher.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        return self.cipher.decrypt(encrypted_data)
```

### 2. 접근 제어

#### 역할 기반 접근 제어
```sql
-- 사용자 역할 생성
CREATE ROLE timeseries_readonly;
CREATE ROLE timeseries_writer;
CREATE ROLE timeseries_admin;

-- 권한 부여
GRANT SELECT ON ALL TABLES IN SCHEMA public TO timeseries_readonly;
GRANT INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO timeseries_writer;
GRANT ALL ON ALL TABLES IN SCHEMA public TO timeseries_admin;
```

#### API 키 인증
```python
import hashlib
import hmac

class APIKeyAuth:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()
    
    def generate_signature(self, data: str) -> str:
        return hmac.new(self.secret_key, data.encode(), hashlib.sha256).hexdigest()
    
    def verify_signature(self, data: str, signature: str) -> bool:
        expected = self.generate_signature(data)
        return hmac.compare_digest(expected, signature)
```

## 📚 추가 자료

### 1. 문서
- [TimescaleDB 공식 문서](https://docs.timescale.com/)
- [Redis 공식 문서](https://redis.io/documentation)
- [InfluxDB 공식 문서](https://docs.influxdata.com/)

### 2. 성능 벤치마크
- [TimescaleDB 벤치마크](https://blog.timescale.com/blog/timescaledb-vs-influxdb-for-time-series-data-timescale-influx-sql-nosql-36489299877/)
- [Redis 성능 가이드](https://redis.io/topics/optimization)

### 3. 모니터링 도구
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
- [Jaeger](https://www.jaegertracing.io/)

## 🤝 기여

### 1. 개발 환경 설정
```bash
# 저장소 클론
git clone https://github.com/your-repo/timeseries-storage.git
cd timeseries-storage

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 개발 서버 실행
python run_timeseries_storage.py --test
```

### 2. 테스트 실행
```bash
# 단위 테스트
pytest tests/unit/

# 통합 테스트
pytest tests/integration/

# 성능 테스트
pytest tests/performance/

# 전체 테스트
pytest tests/
```

### 3. 코드 품질
```bash
# 코드 포맷팅
black src/ tests/

# 린팅
flake8 src/ tests/

# 타입 체크
mypy src/

# 보안 검사
bandit -r src/
```

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 지원

- **이슈 리포트**: [GitHub Issues](https://github.com/your-repo/timeseries-storage/issues)
- **문서**: [Wiki](https://github.com/your-repo/timeseries-storage/wiki)
- **이메일**: support@company.com

---

**⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!** 