# 🏆 고급 스캘핑 자동매매 시스템 (v1.1)

이 프로젝트는 변동성(ATR), 모멘텀, 멀티타임프레임 분석 등 여러 기술적 지표를 종합하여 스캘핑 매매 신호를 생성하는 파이썬 기반 자동매매 시스템입니다. 리팩토링을 통해 각 분석 모듈의 책임과 역할이 명확하게 분리되었으며, 코드의 확장성과 유지보수성이 크게 향상되었습니다.

## ✨ 주요 기능

- **복합 지표 분석**: ATR, 모멘텀, 멀티타임프레임 분석을 통합하여 신호의 정확도를 높입니다.
- **모듈화된 설계**: 각 분석기(`ATRAnalyzer`, `MomentumScorer`, `MultiTimeframeAnalyzer`)가 독립적으로 작동하여 테스트 및 기능 확장이 용이합니다.
- **상태 및 설정 분리**: 데이터 클래스(`dataclass`)를 적극적으로 활용하여 시스템의 설정과 실시간 상태를 명확하게 분리하고 관리합니다.
- **비동기 및 병렬 처리**: `asyncio`와 `ThreadPoolExecutor`를 사용하여 여러 종목을 동시에, 그리고 효율적으로 분석합니다.
- **유연한 API 연동**: 특정 증권사에 종속되지 않는 추상화된 `trader` 객체를 통해 다양한 방식의 API에 연동할 수 있도록 설계되었습니다.

## 📂 프로젝트 구조

```
test_stock/
│
├── scalping_modules/
│   ├── __init__.py                 # 모듈 및 데이터 클래스 노출
│   ├── atr_analyzer.py             # ATR(변동성) 분석기
│   ├── momentum_scorer.py          # 가격/거래량 모멘텀 분석기
│   ├── multi_timeframe_analyzer.py # 다중 시간대 추세 분석기
│   └── optimized_scalping_system.py # 실제 매매 로직 실행 엔진
│
├── advanced_scalping_system.py     # 분석기들을 통합 관리하는 상위 시스템
├── main.py                         # 시스템 실행 및 테스트용 파일
├── requirements.txt                # 프로젝트 의존성 라이브러리 목록
└── README.md                       # 프로젝트 안내 파일 (현재 파일)
```

## 🛠️ 설치 및 설정

1.  **프로젝트 복제 및 이동**
    ```bash
    git clone [저장소 URL]
    cd test_stock
    ```

2.  **가상환경 생성 및 활성화 (권장)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **의존성 라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

4.  **증권사 API 연동 설정**
    이 시스템은 실제 매매를 위해 증권사 API 연동이 필요합니다. `optimized_scalping_system.py`는 `core_trader`라는 추상화된 객체를 통해 API와 통신합니다. **사용자는 사용하는 증권사에 맞춰 이 `trader` 객체를 직접 구현해야 합니다.**

    #### API 방식의 차이점:
    - **한국투자증권 (KIS)**: 최신 **REST API**와 실시간 데이터 수신을 위한 **WebSocket API**를 제공합니다. 이 방식은 OS에 구애받지 않고 `requests`, `websockets`와 같은 표준 라이브러리로 유연하게 개발할 수 있습니다.
    - **키움증권 등 기타 증권사**: PC에 **OCX 컨트롤**을 직접 다운로드하여 연동하는 전통적인 방식을 사용하는 경우가 많습니다. 이 방식은 주로 **Windows 환경에 종속**되며, `pythoncom`과 같은 라이브러리를 이용한 이벤트 기반 프로그래밍이 필요합니다.

    #### `trader` 객체 구현 예시:
    아래는 각기 다른 API 방식에 대한 `trader` 클래스의 기본 구조 예시입니다. 이 구조를 바탕으로 상세한 메서드(`get_current_price`, `place_order` 등)를 구현해야 합니다.
    ```python
    # my_trader.py (사용자 구현 예시 파일)

    class KISRestTrader:
        """한국투자증권 REST/WebSocket API 연동 클래스"""
        def __init__(self, api_key, api_secret, account_no):
            self.base_url = "https://openapivts.koreainvestment.com:29443" # 모의투자 서버
            self.api_key = api_key
            self.api_secret = api_secret
            self.account_no = account_no
            print("한국투자증권(KIS) REST Trader 초기화 완료")
        
        def get_current_price(self, symbol):
            # KIS API의 '현재가' 호출 로직 구현
            pass
        # ... 기타 필요한 메서드 구현 ...

    class KiwoomOcxTrader:
        """키움증권 OCX API 연동 클래스 (Windows 전용)"""
        def __init__(self):
            # 예: from pykiwoom.kiwoom import * 또는 pythoncom 사용
            # self.kiwoom = Kiwoom()
            # self.kiwoom.CommConnect(block=True)
            print("키움증권 OCX Trader 초기화 완료")

        def get_current_price(self, symbol):
            # OCX의 GetMasterCodeName, SetInputValue, CommRqData 등 메서드 호출 로직
            pass
        # ... 기타 필요한 메서드 구현 ...
    ```

## 🚀 실행 방법

프로젝트 루트의 `main.py` 파일을 통해 시스템을 실행할 수 있습니다. 기본적으로 `MockTrader`라는 가상 트레이더를 사용하여 실제 주문 없이 시스템의 로직을 안전하게 테스트합니다.

실제 매매를 원하시면, `main.py` 파일 내부의 `core_trader` 객체를 위 예시처럼 직접 구현하신 `KISRestTrader`나 `KiwoomOcxTrader`로 교체해야 합니다.

```bash
python main.py
```
`main.py`를 실행하면, 다음과 같은 순서로 시스템이 동작합니다.
1.  가상 트레이더(`MockTrader`)를 사용하여 시스템을 초기화합니다.
2.  `run_quick_scan`을 호출하여 시장에서 유망한 종목 5개를 탐색합니다.
3.  탐색된 종목들의 기회 점수를 출력합니다.
4.  가장 점수가 높은 종목에 대해 상세 분석을 수행하고, 그 결과를 출력합니다.
