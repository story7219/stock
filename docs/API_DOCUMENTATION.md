# API Documentation

## 개요

Trading AI System의 API 문서입니다. 이 문서는 시스템의 모든 API 엔드포인트와 사용법을 설명합니다.

## 기본 정보

- **Base URL**: `http://localhost:8000`
- **API Version**: v1.0.0
- **Content Type**: `application/json`
- **Authentication**: Bearer Token

## 인증

### Bearer Token 인증

```bash
curl -H "Authorization: Bearer YOUR_API_TOKEN" \
     http://localhost:8000/api/v1/health
```

## 엔드포인트

### 1. 시스템 상태

#### GET /api/v1/health

시스템 상태를 확인합니다.

**응답 예시:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T10:30:00Z",
  "version": "1.0.0",
  "uptime": 3600
}
```

### 2. 설정 관리

#### GET /api/v1/config

현재 시스템 설정을 조회합니다.

**응답 예시:**
```json
{
  "environment": "development",
  "debug": false,
  "database": {
    "url": "sqlite:///./trading_data.db",
    "pool_size": 10
  },
  "trading": {
    "max_trades_per_day": 3,
    "stop_loss_pct": 0.05
  }
}
```

#### PUT /api/v1/config

시스템 설정을 업데이트합니다.

**요청 예시:**
```json
{
  "trading": {
    "max_trades_per_day": 5,
    "stop_loss_pct": 0.03
  }
}
```

### 3. 데이터 수집

#### GET /api/v1/data/realtime/{symbol}

실시간 주식 데이터를 조회합니다.

**경로 파라미터:**
- `symbol`: 종목 코드 (예: 005930)

**응답 예시:**
```json
{
  "symbol": "005930",
  "name": "삼성전자",
  "price": 75000,
  "change": 1500,
  "change_pct": 2.04,
  "volume": 1000000,
  "timestamp": "2025-01-27T10:30:00Z"
}
```

#### GET /api/v1/data/historical/{symbol}

과거 주식 데이터를 조회합니다.

**쿼리 파라미터:**
- `start_date`: 시작 날짜 (YYYY-MM-DD)
- `end_date`: 종료 날짜 (YYYY-MM-DD)
- `interval`: 데이터 간격 (1m, 5m, 1h, 1d)

**응답 예시:**
```json
{
  "symbol": "005930",
  "data": [
    {
      "date": "2025-01-27",
      "open": 74000,
      "high": 76000,
      "low": 73500,
      "close": 75000,
      "volume": 1000000
    }
  ]
}
```

### 4. 뉴스 데이터

#### GET /api/v1/news

뉴스 데이터를 조회합니다.

**쿼리 파라미터:**
- `category`: 뉴스 카테고리 (financial, economic, political, technological, social)
- `sentiment`: 감성 분석 (positive, neutral, negative)
- `limit`: 조회 개수 (기본값: 100)
- `offset`: 오프셋 (기본값: 0)

**응답 예시:**
```json
{
  "news": [
    {
      "id": "news_001",
      "title": "삼성전자 실적 발표",
      "content": "삼성전자가 예상치를 상회하는 실적을 발표했습니다.",
      "url": "https://example.com/news/001",
      "source": "경제일보",
      "published_at": "2025-01-27T10:00:00Z",
      "category": "financial",
      "sentiment": "positive",
      "sentiment_score": 0.8,
      "importance_score": 0.9
    }
  ],
  "total": 100,
  "offset": 0,
  "limit": 100
}
```

### 5. 트레이딩 신호

#### GET /api/v1/signals

트레이딩 신호를 조회합니다.

**쿼리 파라미터:**
- `symbol`: 종목 코드
- `strategy_type`: 전략 타입 (news_momentum, technical_pattern, theme_rotation)
- `signal_type`: 신호 타입 (buy, sell)
- `confidence_min`: 최소 신뢰도 (0.0-1.0)
- `limit`: 조회 개수 (기본값: 100)

**응답 예시:**
```json
{
  "signals": [
    {
      "id": "signal_001",
      "stock_code": "005930",
      "strategy_type": "news_momentum",
      "signal_type": "buy",
      "confidence_score": 0.8,
      "target_price": 80000,
      "stop_loss": 70000,
      "take_profit": 90000,
      "reasoning": "뉴스 모멘텀 기반 매수 신호",
      "created_at": "2025-01-27T10:30:00Z"
    }
  ],
  "total": 50,
  "offset": 0,
  "limit": 100
}
```

#### POST /api/v1/signals

새로운 트레이딩 신호를 생성합니다.

**요청 예시:**
```json
{
  "stock_code": "005930",
  "strategy_type": "news_momentum",
  "signal_type": "buy",
  "confidence_score": 0.8,
  "target_price": 80000,
  "stop_loss": 70000,
  "take_profit": 90000,
  "reasoning": "뉴스 모멘텀 기반 매수 신호"
}
```

### 6. 거래 실행

#### GET /api/v1/trades

거래 내역을 조회합니다.

**쿼리 파라미터:**
- `symbol`: 종목 코드
- `trade_type`: 거래 타입 (buy, sell)
- `start_date`: 시작 날짜
- `end_date`: 종료 날짜
- `limit`: 조회 개수

**응답 예시:**
```json
{
  "trades": [
    {
      "id": "trade_001",
      "stock_code": "005930",
      "trade_type": "buy",
      "order_type": "market",
      "quantity": 100,
      "price": 75000,
      "amount": 7500000,
      "commission": 7500,
      "tax": 0,
      "net_amount": 7492500,
      "trade_date": "2025-01-27T10:30:00Z",
      "signal_id": "signal_001",
      "strategy_type": "news_momentum"
    }
  ],
  "total": 25,
  "offset": 0,
  "limit": 100
}
```

#### POST /api/v1/trades

새로운 거래를 실행합니다.

**요청 예시:**
```json
{
  "stock_code": "005930",
  "trade_type": "buy",
  "order_type": "market",
  "quantity": 100,
  "price": 75000,
  "signal_id": "signal_001"
}
```

### 7. 포트폴리오

#### GET /api/v1/portfolio

포트폴리오 정보를 조회합니다.

**응답 예시:**
```json
{
  "id": "portfolio_001",
  "name": "메인 포트폴리오",
  "initial_capital": 100000000,
  "current_capital": 105000000,
  "total_return": 0.05,
  "daily_return": 0.02,
  "max_drawdown": -0.03,
  "sharpe_ratio": 1.2,
  "win_rate": 0.65,
  "total_trades": 25,
  "positions": [
    {
      "stock_code": "005930",
      "quantity": 100,
      "avg_price": 75000,
      "current_price": 78000,
      "unrealized_pnl": 300000,
      "unrealized_pnl_pct": 4.0
    }
  ],
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-27T10:30:00Z"
}
```

### 8. 백테스트

#### POST /api/v1/backtest

백테스트를 실행합니다.

**요청 예시:**
```json
{
  "strategy": "news_momentum",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 100000000,
  "symbols": ["005930", "000660", "035420"]
}
```

**응답 예시:**
```json
{
  "strategy_name": "news_momentum",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-12-31T00:00:00Z",
  "initial_capital": 100000000,
  "final_capital": 115000000,
  "total_return": 0.15,
  "annual_return": 0.15,
  "max_drawdown": -0.05,
  "sharpe_ratio": 1.5,
  "win_rate": 0.68,
  "total_trades": 150,
  "avg_trade_return": 0.02,
  "profit_factor": 1.8,
  "max_consecutive_losses": 3
}
```

### 9. 모니터링

#### GET /api/v1/monitoring/performance

시스템 성능 지표를 조회합니다.

**응답 예시:**
```json
{
  "cpu_usage": 45.2,
  "memory_usage": 67.8,
  "disk_usage": 23.4,
  "network_io": {
    "bytes_sent": 1024000,
    "bytes_received": 2048000
  },
  "api_requests": {
    "total": 1500,
    "success": 1450,
    "error": 50,
    "avg_response_time": 120
  },
  "trading_metrics": {
    "signals_generated": 25,
    "trades_executed": 15,
    "success_rate": 0.93
  }
}
```

#### GET /api/v1/monitoring/logs

시스템 로그를 조회합니다.

**쿼리 파라미터:**
- `level`: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `start_time`: 시작 시간
- `end_time`: 종료 시간
- `limit`: 조회 개수

**응답 예시:**
```json
{
  "logs": [
    {
      "timestamp": "2025-01-27T10:30:00Z",
      "level": "INFO",
      "message": "신호 생성: 005930, 전략: news_momentum",
      "module": "application.commands",
      "function": "execute"
    }
  ],
  "total": 1000,
  "offset": 0,
  "limit": 100
}
```

## 에러 코드

| 코드 | 설명 |
|------|------|
| 400 | Bad Request - 잘못된 요청 |
| 401 | Unauthorized - 인증 실패 |
| 403 | Forbidden - 권한 없음 |
| 404 | Not Found - 리소스 없음 |
| 422 | Unprocessable Entity - 검증 실패 |
| 500 | Internal Server Error - 서버 오류 |

## 에러 응답 형식

```json
{
  "error": {
    "code": 400,
    "message": "잘못된 요청입니다.",
    "details": {
      "field": "symbol",
      "issue": "종목 코드가 올바르지 않습니다."
    }
  }
}
```

## Rate Limiting

- **기본 제한**: 1000 requests/hour
- **API 키별 제한**: 10000 requests/hour
- **헤더 정보**:
  - `X-RateLimit-Limit`: 시간당 제한
  - `X-RateLimit-Remaining`: 남은 요청 수
  - `X-RateLimit-Reset`: 제한 초기화 시간

## WebSocket API

### 실시간 데이터 스트림

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('실시간 데이터:', data);
};

// 구독 요청
ws.send(JSON.stringify({
  "action": "subscribe",
  "symbols": ["005930", "000660"]
}));
```

### 실시간 데이터 형식

```json
{
  "type": "price_update",
  "symbol": "005930",
  "data": {
    "price": 75000,
    "change": 1500,
    "volume": 1000000,
    "timestamp": "2025-01-27T10:30:00Z"
  }
}
```

## SDK 예시

### Python SDK

```python
from trading_api import TradingAPI

api = TradingAPI(base_url="http://localhost:8000", token="YOUR_TOKEN")

# 실시간 데이터 조회
data = api.get_realtime_data("005930")
print(f"현재가: {data['price']}")

# 신호 생성
signal = api.create_signal({
    "stock_code": "005930",
    "strategy_type": "news_momentum",
    "signal_type": "buy",
    "confidence_score": 0.8
})

# 거래 실행
trade = api.execute_trade({
    "stock_code": "005930",
    "trade_type": "buy",
    "quantity": 100,
    "price": 75000
})
```

### JavaScript SDK

```javascript
import { TradingAPI } from 'trading-api-sdk';

const api = new TradingAPI({
  baseURL: 'http://localhost:8000',
  token: 'YOUR_TOKEN'
});

// 실시간 데이터 조회
const data = await api.getRealtimeData('005930');
console.log(`현재가: ${data.price}`);

// 신호 생성
const signal = await api.createSignal({
  stock_code: '005930',
  strategy_type: 'news_momentum',
  signal_type: 'buy',
  confidence_score: 0.8
});

// 거래 실행
const trade = await api.executeTrade({
  stock_code: '005930',
  trade_type: 'buy',
  quantity: 100,
  price: 75000
});
```

## 변경 이력

| 버전 | 날짜 | 변경 사항 |
|------|------|-----------|
| 1.0.0 | 2025-01-27 | 초기 버전 |

## 지원

- **문서**: [GitHub Wiki](https://github.com/your-org/trading-ai-system/wiki)
- **이슈**: [GitHub Issues](https://github.com/your-org/trading-ai-system/issues)
- **이메일**: support@trading-ai-system.com 