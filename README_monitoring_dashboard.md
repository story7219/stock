# 실시간 모니터링 대시보드

## 📋 개요

엔터프라이즈급 실시간 데이터 시스템 모니터링 대시보드입니다. 시스템 상태, 데이터 품질, 성능 메트릭, 실시간 알림 시스템을 제공하며 모바일 반응형 디자인을 지원합니다.

## 🏗️ 아키텍처

### 시스템 구성
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Real-time     │    │   Streamlit     │    │   Plotly        │
│   Data Sources  │───▶│   Dashboard     │───▶│   Charts        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AlertManager  │    │   Notification  │    │   Performance   │
│   & Thresholds  │    │   System        │    │   Dashboard     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 모니터링 구성요소
1. **실시간 시스템 상태**: CPU, 메모리, 디스크, 네트워크
2. **데이터 품질**: 커버리지, 지연, 이상치 감지
3. **성능 메트릭**: 처리량, 레이턴시, 에러율
4. **알림 시스템**: Slack, Telegram, 이메일

## 🚀 주요 기능

### 1. 시스템 상태 모니터링
- **실시간 메트릭**: CPU, 메모리, 디스크 사용률
- **네트워크 I/O**: 데이터 수신률, 처리량
- **레이턴시 히트맵**: 서비스별 응답 시간
- **연결 상태**: 데이터베이스, 캐시, 메시지 큐

### 2. 데이터 품질 대시보드
- **데이터 커버리지**: 종목별 데이터 완성도
- **이상치 감지**: 자동 이상치 탐지 및 알림
- **데이터 지연**: 실시간 지연 모니터링
- **품질 점수**: 종합 데이터 품질 평가

### 3. 성능 메트릭
- **처리량 통계**: 초당 처리 이벤트 수
- **에러율 추적**: 시스템 에러율 모니터링
- **가용성**: 시스템 가동률 측정
- **리소스 사용률**: CPU, 메모리, 디스크

### 4. 알림 시스템
- **실시간 알림**: Slack, Telegram, 이메일
- **임계값 기반**: 자동 임계값 체크
- **자동 대응**: 설정된 액션 자동 실행
- **에스컬레이션**: 중요도별 알림 정책

## 📦 설치

### 1. 의존성 설치

```bash
# 필수 패키지
pip install streamlit plotly pandas numpy psutil requests aiohttp

# 선택사항 (Prometheus 사용 시)
pip install prometheus_client

# 개발 도구
pip install pytest pytest-asyncio black flake8 mypy
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << EOF
# Slack 알림
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# Telegram 알림
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID

# 이메일 알림
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# 모니터링 설정
DASHBOARD_PORT=8501
UPDATE_INTERVAL_SECONDS=1.0
EOF

# 환경 변수 로드
source .env
```

## ⚙️ 설정

### 1. 기본 설정으로 실행

```bash
# Streamlit 대시보드 실행
streamlit run run_monitoring_dashboard.py --server.port 8501

# 또는 Python으로 실행
python run_monitoring_dashboard.py --streamlit
```

### 2. 커스텀 설정 파일 사용

```bash
# 설정 파일 생성
cp config/monitoring_config.json my_monitoring_config.json

# 설정 파일 편집
vim my_monitoring_config.json

# 커스텀 설정으로 실행
python run_monitoring_dashboard.py --config my_monitoring_config.json
```

### 3. 설정 검증

```bash
# 의존성 체크
python run_monitoring_dashboard.py --check-deps

# 설정 검증
python run_monitoring_dashboard.py --validate-config

# 시스템 테스트
python run_monitoring_dashboard.py --test
```

## 🔧 사용법

### 1. 대시보드 접속

```bash
# Streamlit 대시보드 시작
streamlit run run_monitoring_dashboard.py

# 브라우저에서 접속
# http://localhost:8501
```

### 2. 메뉴 구성

#### 시스템 상태
- 실시간 CPU, 메모리, 디스크 사용률
- 데이터 수신률 및 레이턴시
- 연결 상태 모니터링
- 실시간 차트 및 그래프

#### 데이터 품질
- 종목별 데이터 커버리지
- 이상치 감지 결과
- 데이터 지연 모니터링
- 품질 점수 추적

#### 성능 메트릭
- 처리량 통계
- 네트워크 I/O
- 에러율 추적
- 레이턴시 히트맵

#### 알림 시스템
- 실시간 알림 설정
- 알림 히스토리
- 임계값 관리
- 알림 채널 설정

#### 설정
- 모니터링 설정
- 알림 설정
- 임계값 조정
- 시스템 구성

### 3. API 사용

```python
import requests

# 메트릭 조회
response = requests.get('http://localhost:8080/api/v1/metrics')
metrics = response.json()

# 알림 전송
alert_data = {
    'level': 'warning',
    'message': 'CPU 사용률이 높습니다.',
    'metric': 'cpu_usage',
    'value': 85.5,
    'threshold': 80.0
}
response = requests.post('http://localhost:8080/api/v1/alerts', json=alert_data)
```

## 📊 성능 지표

### 목표 성능
- **실시간 업데이트**: < 1초
- **대시보드 로딩**: < 3초
- **알림 전송**: < 5초
- **데이터 처리**: 10,000+ events/sec

### 모니터링 메트릭
- 시스템 리소스 사용률
- 데이터 품질 지표
- 성능 메트릭
- 알림 통계

## 🔍 클래스 구조

### 1. RealTimeMonitor
```python
class RealTimeMonitor:
    """실시간 시스템 모니터링"""
    
    def start_monitoring(self):
        """모니터링 시작"""
    
    def stop_monitoring(self):
        """모니터링 중지"""
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """최신 메트릭 조회"""
    
    def get_metrics_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """메트릭 히스토리 조회"""
```

### 2. AlertManager
```python
class AlertManager:
    """알림 관리 시스템"""
    
    def process_alert(self, alert: Dict[str, Any]):
        """알림 처리"""
    
    def _send_slack_alert(self, alert: Dict[str, Any]):
        """Slack 알림 전송"""
    
    def _send_telegram_alert(self, alert: Dict[str, Any]):
        """Telegram 알림 전송"""
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """이메일 알림 전송"""
```

### 3. PerformanceDashboard
```python
class PerformanceDashboard:
    """성능 대시보드"""
    
    def create_system_status_dashboard(self) -> go.Figure:
        """시스템 상태 대시보드 생성"""
    
    def create_connection_status_dashboard(self) -> go.Figure:
        """연결 상태 대시보드 생성"""
    
    def create_alert_history_dashboard(self) -> go.Figure:
        """알림 히스토리 대시보드 생성"""
```

### 4. NotificationSystem
```python
class NotificationSystem:
    """알림 시스템"""
    
    def send_notification(self, message: str, level: str = 'info', 
                         channels: List[str] = None):
        """알림 전송"""
    
    def get_notification_queue(self) -> queue.Queue:
        """알림 큐 조회"""
```

## 🛠️ 고급 설정

### 1. 알림 설정

#### Slack 알림 설정
```python
# Slack Webhook URL 설정
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

# 알림 메시지 예시
{
    "text": "🚨 WARNING: CPU 사용률이 높습니다.",
    "attachments": [{
        "color": "warning",
        "fields": [
            {"title": "메트릭", "value": "cpu_usage", "short": True},
            {"title": "값", "value": "85.5%", "short": True},
            {"title": "임계값", "value": "80.0%", "short": True}
        ]
    }]
}
```

#### Telegram 알림 설정
```python
# Telegram Bot 설정
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# 알림 메시지 예시
message = f"""
🚨 WARNING
메시지: CPU 사용률이 높습니다.
메트릭: cpu_usage
값: 85.5%
임계값: 80.0%
시간: 2025-01-27 15:30:45
"""
```

### 2. 임계값 설정

```python
# 임계값 설정 예시
thresholds = {
    'cpu_threshold_percent': 80.0,
    'memory_threshold_percent': 85.0,
    'disk_threshold_percent': 90.0,
    'latency_threshold_ms': 100.0,
    'error_rate_threshold_percent': 5.0,
    'data_coverage_threshold_percent': 95.0,
    'data_delay_threshold_seconds': 60.0
}
```

### 3. 데이터 품질 모니터링

```python
# 데이터 품질 메트릭
quality_metrics = {
    'completeness': 0.98,  # 데이터 완성도
    'accuracy': 0.99,      # 데이터 정확도
    'timeliness': 0.95,    # 데이터 적시성
    'consistency': 0.97    # 데이터 일관성
}

# 이상치 감지 설정
anomaly_detection = {
    'enabled': True,
    'method': 'zscore',
    'threshold': 3.0,
    'window_size': 100
}
```

## 🔧 문제 해결

### 1. 일반적인 문제

#### 대시보드 로딩 실패
```bash
# 포트 확인
netstat -tulpn | grep 8501

# 의존성 확인
pip list | grep streamlit

# 로그 확인
tail -f monitoring_dashboard.log
```

#### 알림 전송 실패
```bash
# Slack Webhook 테스트
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"테스트 메시지"}' \
  https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# Telegram Bot 테스트
curl -X POST \
  https://api.telegram.org/botYOUR_BOT_TOKEN/sendMessage \
  -d "chat_id=YOUR_CHAT_ID" \
  -d "text=테스트 메시지"
```

#### 성능 문제
```python
# 메트릭 수집 최적화
import psutil

# CPU 사용률 최적화
cpu_percent = psutil.cpu_percent(interval=1, percpu=False)

# 메모리 사용률 최적화
memory = psutil.virtual_memory()
memory_percent = memory.percent

# 디스크 사용률 최적화
disk = psutil.disk_usage('/')
disk_percent = disk.percent
```

### 2. 로그 분석

#### 로그 레벨 설정
```python
import logging

# 디버그 로그 활성화
logging.getLogger('src.realtime_monitoring_dashboard').setLevel(logging.DEBUG)

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
monitor.start_monitoring()

# 프로파일링 결과
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## 📈 확장성

### 1. 수평 확장

#### 로드 밸런싱
```python
# Nginx 설정 예시
upstream dashboard_backend {
    server 127.0.0.1:8501;
    server 127.0.0.1:8502;
    server 127.0.0.1:8503;
}

server {
    listen 80;
    server_name dashboard.example.com;
    
    location / {
        proxy_pass http://dashboard_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Redis 캐싱
```python
import redis

# Redis 캐싱 설정
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_metrics(metrics: Dict[str, Any], ttl: int = 300):
    """메트릭 캐싱"""
    redis_client.setex('latest_metrics', ttl, json.dumps(metrics))

def get_cached_metrics() -> Dict[str, Any]:
    """캐시된 메트릭 조회"""
    cached = redis_client.get('latest_metrics')
    return json.loads(cached) if cached else {}
```

### 2. 수직 확장

#### 메모리 최적화
```python
# 메모리 사용량 모니터링
import psutil
import gc

def optimize_memory():
    """메모리 최적화"""
    # 가비지 컬렉션
    gc.collect()
    
    # 메모리 사용량 확인
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        # 오래된 데이터 정리
        cleanup_old_data()

def cleanup_old_data():
    """오래된 데이터 정리"""
    # 24시간 이상 된 메트릭 삭제
    cutoff_time = datetime.now() - timedelta(hours=24)
    # 데이터 정리 로직
```

#### CPU 최적화
```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

def parallel_metric_collection():
    """병렬 메트릭 수집"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(collect_cpu_metrics),
            executor.submit(collect_memory_metrics),
            executor.submit(collect_disk_metrics),
            executor.submit(collect_network_metrics)
        ]
        
        results = [future.result() for future in futures]
        return results
```

## 🔒 보안

### 1. 인증 및 권한

#### 기본 인증
```python
import streamlit as st

def check_authentication():
    """인증 확인"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        username = st.sidebar.text_input("사용자명")
        password = st.sidebar.text_input("비밀번호", type="password")
        
        if st.sidebar.button("로그인"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.experimental_rerun()
            else:
                st.error("인증 실패")
        
        if not st.session_state.authenticated:
            st.stop()

def authenticate_user(username: str, password: str) -> bool:
    """사용자 인증"""
    # 실제 구현에서는 데이터베이스 확인
    return username == "admin" and password == "password"
```

#### HTTPS 설정
```python
# SSL 인증서 설정
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Streamlit SSL 설정
streamlit run app.py --server.sslCertFile=cert.pem --server.sslKeyFile=key.pem
```

### 2. 데이터 암호화

#### 전송 중 암호화
```python
import ssl
import socket

def create_secure_connection(host: str, port: int):
    """보안 연결 생성"""
    context = ssl.create_default_context()
    with socket.create_connection((host, port)) as sock:
        with context.wrap_socket(sock, server_hostname=host) as ssock:
            return ssock
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

## 📚 추가 자료

### 1. 문서
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [Plotly 공식 문서](https://plotly.com/python/)
- [Prometheus 공식 문서](https://prometheus.io/docs/)

### 2. 성능 벤치마크
- [Streamlit 성능 가이드](https://docs.streamlit.io/library/advanced-features/optimize-performance)
- [Plotly 성능 최적화](https://plotly.com/python/plotly-fundamentals/)

### 3. 모니터링 도구
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
- [Datadog](https://www.datadoghq.com/)

## 🤝 기여

### 1. 개발 환경 설정
```bash
# 저장소 클론
git clone https://github.com/your-repo/monitoring-dashboard.git
cd monitoring-dashboard

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 개발 서버 실행
streamlit run run_monitoring_dashboard.py
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

- **이슈 리포트**: [GitHub Issues](https://github.com/your-repo/monitoring-dashboard/issues)
- **문서**: [Wiki](https://github.com/your-repo/monitoring-dashboard/wiki)
- **이메일**: support@company.com

---

**⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!** 