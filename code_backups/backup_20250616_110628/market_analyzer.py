import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import asyncio
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import functools

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("google-generativeai 라이브러리가 설치되지 않았습니다. AI 기능이 비활성화됩니다.")

from kis_api import KIS_API
import yfinance as yf
import pandas as pd

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logging.warning("fredapi 라이브러리가 설치되지 않았습니다. 거시경제 지표 기능이 제한됩니다.")

@dataclass
class MarketTrendData:
    """시장 트렌드 데이터 클래스"""
    trend: str
    current_price: float
    ma20: float
    ma50: float
    ma200: float
    strength: float

@dataclass
class MacroData:
    """거시경제 데이터 클래스"""
    treasury_10y: Optional[float] = None
    fed_funds_rate: Optional[float] = None
    unemployment_rate: Optional[float] = None
    cpi: Optional[float] = None
    error: Optional[str] = None

class MarketAnalyzer:
    """
    거시 경제 지표, 시장 심리, 기술적 분석을 통합하여 시장의 '체제(Regime)'를 판단.
    선물 대가들의 리스크 관리 및 거시 분석 + 리버모어의 시장 심리 분석을 통합.
    """
    
    def __init__(self, kis_api: KIS_API, gemini_api_key: str):
        self.logger = logging.getLogger(__name__)
        self.api = kis_api
        self.gemini_api_key = gemini_api_key
        
        # 환경변수에서 모델명 로드 (기본값: gemini-1.5-flash)
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        
        # 제미나이 API 설정
        if GEMINI_AVAILABLE:
            self._initialize_gemini()
        else:
            self.model = None
            self.generation_config = None
        
        # 시장 데이터 캐시
        self.market_data_cache = {}
        self.cache_timestamp = None
        self.cache_duration = timedelta(minutes=5)
        
        # FRED API 클라이언트 초기화
        self._initialize_fred()
        
        # 상태 변수
        self.market_regime = "NEUTRAL"
        self.last_analysis_time = None
        self.last_market_analysis = None
        
        # 성능 최적화를 위한 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    def _initialize_gemini(self):
        """제미나이 API 초기화"""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.logger.info(f"🤖 제미나이 모델 '{self.model_name}'이 초기화되었습니다.")
            self._log_model_info()
            self.generation_config = self._get_optimized_config()
        except Exception as e:
            self.logger.error(f"🚫 제미나이 모델 초기화 실패: {e}")
            self._fallback_to_default_model()
    
    def _fallback_to_default_model(self):
        """기본 모델로 폴백"""
        try:
            self.model_name = 'gemini-1.5-flash'
            self.model = genai.GenerativeModel(self.model_name)
            self.generation_config = self._get_optimized_config()
            self.logger.info(f"✅ 기본 모델 '{self.model_name}'로 초기화 완료")
        except Exception as fallback_error:
            self.logger.error(f"🚫 기본 모델 초기화도 실패: {fallback_error}")
            self.model = None
            self.generation_config = None
    
    def _initialize_fred(self):
        """FRED API 초기화"""
        fred_api_key = os.getenv('FRED_API_KEY')
        if FRED_AVAILABLE and fred_api_key:
            try:
                self.fred = Fred(api_key=fred_api_key)
                self.logger.info("✅ FRED API 초기화 완료")
            except Exception as e:
                self.logger.error(f"❌ FRED API 초기화 실패: {e}")
                self.fred = None
        else:
            self.fred = None
            if not fred_api_key:
                print("⚠️ FRED_API_KEY가 설정되지 않아 거시 경제 지표 분석이 제한됩니다.")  # logging 대신 print 사용
    
    def _log_model_info(self):
        """사용 중인 모델 정보 출력"""
        model_info = {
            'gemini-1.5-flash': {
                'description': '빠른 응답 속도와 효율적인 분석',
                'best_for': '실시간 매매 신호, 빠른 시장 분석'
            },
            'gemini-1.5-pro': {
                'description': '고품질 분석과 복잡한 추론',
                'best_for': '심층 시장 분석, 복합 전략 수립'
            }
        }
        
        info = model_info.get(self.model_name, {
            'description': '사용자 지정 모델',
            'best_for': '설정된 용도에 따라'
        })
        
        self.logger.info(f"📋 모델 정보: {info['description']}")
        self.logger.info(f"   최적 용도: {info['best_for']}")

    def _get_optimized_config(self) -> genai.types.GenerationConfig:
        """모델별 최적화된 생성 설정 반환"""
        temperature = float(os.getenv('GEMINI_TEMPERATURE', '0.3'))
        max_tokens = int(os.getenv('GEMINI_MAX_TOKENS', '1000'))
        top_p = float(os.getenv('GEMINI_TOP_P', '0.8'))
        top_k = int(os.getenv('GEMINI_TOP_K', '40'))
        
        model_configs = {
            'gemini-1.5-flash': {
                'temperature': 0.3,
                'max_output_tokens': 1000,
                'top_p': 0.8,
                'top_k': 40
            },
            'gemini-1.5-pro': {
                'temperature': 0.2,
                'max_output_tokens': 2000,
                'top_p': 0.9,
                'top_k': 50
            }
        }
        
        default_config = model_configs.get(self.model_name, model_configs['gemini-1.5-flash'])
        
        config = genai.types.GenerationConfig(
            temperature=temperature if os.getenv('GEMINI_TEMPERATURE') else default_config['temperature'],
            max_output_tokens=max_tokens if os.getenv('GEMINI_MAX_TOKENS') else default_config['max_output_tokens'],
            top_p=top_p if os.getenv('GEMINI_TOP_P') else default_config['top_p'],
            top_k=top_k if os.getenv('GEMINI_TOP_K') else default_config['top_k']
        )
        
        return config

    @functools.lru_cache(maxsize=10)
    def _get_cached_stock_info(self, stock_code: str, cache_key: str) -> Optional[Dict]:
        """주식 정보 캐싱 (성능 최적화)"""
        try:
            return self.api.get_current_price(stock_code)
        except Exception as e:
            self.logger.error(f"주식 정보 조회 실패 ({stock_code}): {e}")
            return None

    async def get_market_rankings_async(self) -> Dict[str, Any]:
        """비동기 시장 랭킹 데이터 수집"""
        market_data = {
            'timestamp': datetime.now().isoformat(),
            'rankings': {}
        }
        
        try:
            # 병렬로 데이터 수집
            tasks = [
                self._get_volume_ranking_async(),
                self._get_value_ranking_async(),
                self._get_price_change_ranking_async(),
                self._get_sector_performance_async()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            market_data['rankings']['volume_top'] = results[0] if not isinstance(results[0], Exception) else []
            market_data['rankings']['value_top'] = results[1] if not isinstance(results[1], Exception) else []
            market_data['rankings']['price_change'] = results[2] if not isinstance(results[2], Exception) else {'top_gainers': [], 'top_losers': []}
            market_data['rankings']['sector_performance'] = results[3] if not isinstance(results[3], Exception) else []
            
            self.logger.info("✅ 시장 랭킹 데이터 수집 완료")
            return market_data
            
        except Exception as e:
            self.logger.error(f"❌ 시장 랭킹 데이터 수집 중 오류: {e}")
            return market_data

    async def _get_volume_ranking_async(self) -> List[Dict[str, Any]]:
        """비동기 거래량 상위 종목 조회"""
        major_stocks = ['005930', '000660', '035420', '005490', '051910', 
                       '035720', '006400', '028260', '068270', '207940']
        
        tasks = []
        for stock_code in major_stocks:
            task = asyncio.create_task(self._get_stock_info_async(stock_code))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        volume_data = []
        for result in results:
            if isinstance(result, dict) and result:
                volume_data.append(result)
        
        volume_data.sort(key=lambda x: x.get('volume', 0), reverse=True)
        return volume_data[:20]

    async def _get_stock_info_async(self, stock_code: str) -> Optional[Dict]:
        """비동기 주식 정보 조회"""
        try:
            # 동기 API를 비동기로 실행
            loop = asyncio.get_event_loop()
            price_info = await loop.run_in_executor(
                self.executor, 
                self.api.get_current_price, 
                stock_code
            )
            
            if price_info and price_info.get('rt_cd') == '0':
                output = price_info['output']
                return {
                    'code': stock_code,
                    'name': output.get('hts_kor_isnm', ''),
                    'current_price': self._safe_int(output.get('stck_prpr', 0)),
                    'change_rate': self._safe_float(output.get('prdy_ctrt', 0)),
                    'volume': self._safe_int(output.get('acml_vol', 0)),
                    'trade_value': self._safe_int(output.get('acml_tr_pbmn', 0))
                }
        except Exception as e:
            self.logger.warning(f"종목 {stock_code} 정보 조회 실패: {e}")
        return None

    def _safe_int(self, value: Union[str, int, float]) -> int:
        """안전한 정수 변환"""
        try:
            if isinstance(value, str):
                # 소수점이 있는 문자열을 float으로 먼저 변환 후 int로
                return int(float(value))
            return int(value)
        except (ValueError, TypeError):
            return 0

    def _safe_float(self, value: Union[str, int, float]) -> float:
        """안전한 실수 변환"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    async def _get_value_ranking_async(self) -> List[Dict[str, Any]]:
        """비동기 거래대금 상위 종목 조회"""
        return await self._get_volume_ranking_async()  # 동일한 로직 재사용

    async def _get_price_change_ranking_async(self) -> Dict[str, List[Dict[str, Any]]]:
        """비동기 등락률 상위/하위 종목 조회"""
        volume_data = await self._get_volume_ranking_async()
        
        if not volume_data:
            return {'top_gainers': [], 'top_losers': []}
        
        # 등락률 기준으로 정렬
        sorted_data = sorted(volume_data, key=lambda x: x.get('change_rate', 0), reverse=True)
        
        return {
            'top_gainers': sorted_data[:10],
            'top_losers': sorted_data[-10:]
        }

    async def _get_sector_performance_async(self) -> List[Dict[str, Any]]:
        """비동기 업종별 등락률 조회"""
        return [
            {'sector': '반도체', 'change_rate': 2.5, 'representative_stocks': ['005930', '000660']},
            {'sector': '인터넷', 'change_rate': 1.8, 'representative_stocks': ['035420', '035720']},
            {'sector': '철강', 'change_rate': -0.5, 'representative_stocks': ['005490']},
            {'sector': '자동차', 'change_rate': 0.3, 'representative_stocks': ['005380']},
            {'sector': '화학', 'change_rate': -1.2, 'representative_stocks': ['051910']}
        ]

    def _get_market_trend(self, kospi_data: pd.DataFrame) -> MarketTrendData:
        """KOSPI 데이터를 기반으로 시장 트렌드 분석 (Pandas Series 오류 수정)"""
        try:
            if kospi_data.empty:
                return MarketTrendData("NEUTRAL", 0, 0, 0, 0, 0)
            
            # 현재가와 이동평균선 계산
            current = float(kospi_data['Close'].iloc[-1])
            ma20_series = kospi_data['Close'].rolling(20).mean()
            ma50_series = kospi_data['Close'].rolling(50).mean()
            ma200_series = kospi_data['Close'].rolling(200).mean()
            
            # 마지막 값만 추출하여 스칼라 값으로 변환
            ma20 = float(ma20_series.iloc[-1]) if not ma20_series.empty else 0
            ma50 = float(ma50_series.iloc[-1]) if not ma50_series.empty else 0
            ma200 = float(ma200_series.iloc[-1]) if not ma200_series.empty else 0
            
            # NaN 값 체크 (스칼라 값으로 체크)
            if pd.isna(ma200) or pd.isna(ma50) or pd.isna(ma20) or ma200 == 0:
                return MarketTrendData("NEUTRAL", current, ma20, ma50, ma200, 0)
            
            # 트렌드 강도 계산
            strength = self._calculate_trend_strength(current, ma20, ma50, ma200)
            
            # 트렌드 분석 (스칼라 값으로 비교)
            if current > ma50 and ma50 > ma200:
                trend = "STRONG_UPTREND"
            elif current > ma20 and ma20 > ma50:
                trend = "UPTREND"
            elif current < ma50 and ma50 < ma200:
                trend = "STRONG_DOWNTREND"
            elif current < ma20 and ma20 < ma50:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"
            
            return MarketTrendData(trend, current, ma20, ma50, ma200, strength)
                
        except Exception as e:
            print(f"❌ 시장 트렌드 분석 중 오류: {e}")  # logging 대신 print 사용
            return MarketTrendData("NEUTRAL", 0, 0, 0, 0, 0)

    def _calculate_trend_strength(self, current: float, ma20: float, ma50: float, ma200: float) -> float:
        """트렌드 강도 계산"""
        try:
            # 이동평균선 간의 거리로 트렌드 강도 측정
            if ma200 > 0:
                strength = ((current - ma200) / ma200) * 100
                return min(max(strength, -100), 100)  # -100 ~ 100 범위로 제한
            return 0
        except:
            return 0

    def _get_macro_data(self) -> MacroData:
        """FRED에서 주요 거시 경제 지표를 가져옵니다 (개선된 버전)"""
        if not self.fred:
            return MacroData(error="FRED API key not set")
        
        try:
            # 병렬로 데이터 수집
            indicators = {
                'DGS10': 'treasury_10y',
                'FEDFUNDS': 'fed_funds_rate', 
                'UNRATE': 'unemployment_rate',
                'CPIAUCSL': 'cpi'
            }
            
            data = {}
            for fred_code, field_name in indicators.items():
                try:
                    series = self.fred.get_series(fred_code, limit=1)
                    if not series.empty:
                        data[field_name] = float(series.iloc[-1])
                except Exception as e:
                    self.logger.warning(f"FRED {fred_code} 조회 실패: {e}")
                    data[field_name] = None
            
            return MacroData(**data)
            
        except Exception as e:
            self.logger.error(f"FRED 데이터 조회 오류: {e}")
            return MacroData(error=str(e))

    async def get_market_regime_analysis(self) -> Dict:
        """
        선물 대가 & 리버모어 스타일로 시장 체제를 분석합니다.
        (공격적 성장주 투자 / 신중한 가치주 투자 / 현금 보유 및 방어)
        """
        self.logger.info("🧭 시장 체제 분석(Market Regime Analysis) 시작...")
        
        try:
            # 1. 기술적 지표 (리버모어 스타일) - 비동기로 처리
            kospi_task = asyncio.create_task(self._get_kospi_data_async())
            vix_task = asyncio.create_task(self._get_vix_data_async())
            
            kospi_data, vix_data = await asyncio.gather(kospi_task, vix_task)
            
            market_trend = self._get_market_trend(kospi_data)
            vix = vix_data if vix_data else 20.0  # 기본값
            
            # 2. 시장 심리 지표
            fear_and_greed = self._get_fear_and_greed_index()
            
            # 3. 거시 경제 지표 (선물 대가 스타일)
            macro_data = self._get_macro_data()
            
            # 4. 제미나이 AI를 통한 종합 분석
            if self.model:
                prompt = self._build_regime_analysis_prompt(market_trend, vix, fear_and_greed, macro_data)
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=self.generation_config
                )
                analysis_result = self._parse_ai_json_response(response.text)
            else:
                # AI 없이 기본 분석
                analysis_result = self._basic_regime_analysis(market_trend, vix, macro_data)
            
            if analysis_result and 'market_regime' in analysis_result:
                self.market_regime = analysis_result['market_regime']
                self.logger.info(f"✅ 시장 체제 분석 완료: {self.market_regime}")
                self.logger.info(f"   AI 분석 요약: {analysis_result.get('summary', 'N/A')}")
            else:
                print("⚠️ 시장 체제 AI 분석 실패. 기본값(NEUTRAL) 사용.")  # logging 대신 print
                analysis_result = {'market_regime': 'NEUTRAL', 'summary': 'AI 분석 실패'}
            
            self.last_market_analysis = analysis_result
            self.last_analysis_time = datetime.now()
            return analysis_result
            
        except Exception as e:
            print(f"❌ 시장 체제 분석 중 오류: {e}")  # logging 대신 print
            return {'market_regime': 'NEUTRAL', 'summary': '분석 중 오류 발생'}

    async def _get_kospi_data_async(self) -> pd.DataFrame:
        """비동기 KOSPI 데이터 조회"""
        try:
            loop = asyncio.get_event_loop()
            kospi_data = await loop.run_in_executor(
                self.executor,
                lambda: yf.download('^KS11', period='1y', progress=False)
            )
            return kospi_data
        except Exception as e:
            self.logger.error(f"KOSPI 데이터 조회 실패: {e}")
            return pd.DataFrame()

    async def _get_vix_data_async(self) -> Optional[float]:
        """비동기 VIX 데이터 조회"""
        try:
            loop = asyncio.get_event_loop()
            vix_data = await loop.run_in_executor(
                self.executor,
                lambda: yf.download('^VIX', period='5d', progress=False)
            )
            if not vix_data.empty:
                return float(vix_data['Close'].iloc[-1])
        except Exception as e:
            self.logger.error(f"VIX 데이터 조회 실패: {e}")
        return None

    def _basic_regime_analysis(self, market_trend: MarketTrendData, vix: float, macro_data: MacroData) -> Dict:
        """AI 없이 기본 시장 체제 분석"""
        try:
            # 간단한 규칙 기반 분석
            if market_trend.trend in ["STRONG_UPTREND", "UPTREND"] and vix < 25:
                regime = "AGGRESSIVE_GROWTH"
                summary = "상승 추세이고 변동성이 낮아 공격적 성장 전략 적합"
            elif market_trend.trend in ["STRONG_DOWNTREND", "DOWNTREND"] or vix > 35:
                regime = "DEFENSIVE_CASH"
                summary = "하락 추세이거나 변동성이 높아 방어적 현금 보유 전략 적합"
            else:
                regime = "CAUTIOUS_VALUE"
                summary = "혼재된 시장 상황으로 신중한 가치 투자 전략 적합"
            
            return {
                'market_regime': regime,
                'summary': summary,
                'key_indicators': {
                    'trend': market_trend.trend,
                    'vix': vix,
                    'trend_strength': market_trend.strength
                }
            }
        except Exception as e:
            self.logger.error(f"기본 체제 분석 오류: {e}")
            return {'market_regime': 'NEUTRAL', 'summary': '기본 분석 실패'}

    def _get_fear_and_greed_index(self) -> int:
        """CNN 공포탐욕지수를 가져오는 함수"""
        # 실제 구현에서는 웹 스크래핑이나 API 호출 필요
        return 50  # 중립값

    def _build_regime_analysis_prompt(self, trend: MarketTrendData, vix: float, fear_greed: int, macro_data: MacroData) -> str:
        """시장 체제 분석 프롬프트 생성"""
        return f"""
당신은 폴 튜더 존스와 같은 전설적인 거시 경제 트레이더입니다. 
제공된 데이터를 바탕으로 현재 한국 주식 시장의 '체제(Regime)'를 분석하고, 어떤 투자 전략이 가장 유리할지 판단해주세요.

## 분석 데이터

### 1. 기술적 시장 추세 (제시 리버모어 관점)
- KOSPI 추세: {trend.trend}
- 트렌드 강도: {trend.strength:.2f}%
- 현재가: {trend.current_price:.2f}
- 20일선: {trend.ma20:.2f}
- 50일선: {trend.ma50:.2f}
- 200일선: {trend.ma200:.2f}

### 2. 시장 심리 (제시 리버모어 관점)
- VIX 지수 (공포 지수): {vix:.2f} (높을수록 공포)
- 공포와 탐욕 지수: {fear_greed} (0에 가까울수록 공포, 100에 가까울수록 탐욕)

### 3. 거시 경제 지표 (선물 투자 대가 관점)
- 10년 국채 수익률: {macro_data.treasury_10y}%
- 연방기금금리: {macro_data.fed_funds_rate}%
- 실업률: {macro_data.unemployment_rate}%
- CPI: {macro_data.cpi}

## 출력 형식 (JSON)
```json
{{
  "market_regime": "AGGRESSIVE_GROWTH|CAUTIOUS_VALUE|DEFENSIVE_CASH",
  "summary": "분석 요약",
  "key_indicators": {{
    "trend": "{trend.trend}",
    "sentiment": "긍정적|중립|부정적",
    "macro": "우호적|중립|불리"
  }},
  "recommended_strategy": "추천 전략"
}}
```
"""

    def _parse_ai_json_response(self, response_text: str) -> Optional[Dict]:
        """AI 응답에서 JSON 파싱 (개선된 버전)"""
        try:
            # JSON 블록 찾기
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # JSON 블록이 없으면 전체 텍스트에서 JSON 찾기
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return None
            
            return json.loads(json_str)
            
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.error(f"AI 응답 JSON 파싱 실패: {e}")
            return None

    def shutdown(self):
        """리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.logger.info("MarketAnalyzer 리소스 정리 완료")

if __name__ == '__main__':
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    logging.info("MarketAnalyzer 모듈 테스트 완료") 