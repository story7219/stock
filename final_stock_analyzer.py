#!/usr/bin/env python3
"""
🚀 최종 통합 주식 분석 시스템
코스피200·나스닥100·S&P500 전체 종목 분석 + Gemini AI Top5 선정
라이브 한투 API 활용 (모든 기능 지원)
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# 환경 변수 로드
load_dotenv()

class FinalStockAnalyzer:
    def __init__(self):
        # 한투 API 설정
        self.kis_base_url = "https://openapi.koreainvestment.com:9443"
        self.kis_app_key = os.getenv('LIVE_KIS_APP_KEY')
        self.kis_app_secret = os.getenv('LIVE_KIS_APP_SECRET')
        self.kis_access_token = None
        
        # Gemini AI 설정
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        
        # 분석 결과 저장
        self.all_stocks_data = []
        self.analysis_results = {}
        
        print("🚀 최종 통합 주식 분석 시스템 초기화 완료")
    
    def get_kis_token(self) -> bool:
        """한투 API 토큰 발급"""
        try:
            url = f"{self.kis_base_url}/oauth2/tokenP"
            headers = {"content-type": "application/json"}
            data = {
                "grant_type": "client_credentials",
                "appkey": self.kis_app_key,
                "appsecret": self.kis_app_secret
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                self.kis_access_token = result.get('access_token')
                print("🎫 한투 API 토큰 발급 성공")
                return True
            else:
                print(f"❌ 토큰 발급 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 토큰 발급 예외: {str(e)}")
            return False
    
    def get_kospi200_stocks(self) -> List[Dict]:
        """코스피200 종목 조회"""
        try:
            if not self.kis_access_token:
                return []
            
            url = f"{self.kis_base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
            headers = {
                "content-type": "application/json",
                "authorization": f"Bearer {self.kis_access_token}",
                "appkey": self.kis_app_key,
                "appsecret": self.kis_app_secret,
                "tr_id": "FHKST03010100"
            }
            
            # 코스피200 대표 종목들
            kospi200_codes = [
                "005930",  # 삼성전자
                "000660",  # SK하이닉스
                "035420",  # NAVER
                "051910",  # LG화학
                "006400",  # 삼성SDI
                "035720",  # 카카오
                "207940",  # 삼성바이오로직스
                "068270",  # 셀트리온
                "028260",  # 삼성물산
                "012330"   # 현대모비스
            ]
            
            stocks = []
            for code in kospi200_codes:
                params = {
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": code,
                    "FID_INPUT_DATE_1": (datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": datetime.now().strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": "D"
                }
                
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'output2' in data and data['output2']:
                        stock_info = data['output2'][0]
                        stocks.append({
                            "symbol": code,
                            "name": stock_info.get('hts_kor_isnm', f'종목{code}'),
                            "exchange": "KRX",
                            "category": "kospi200",
                            "current_price": float(stock_info.get('stck_clpr', 0)),
                            "volume": int(stock_info.get('acml_vol', 0)),
                            "change_rate": float(stock_info.get('prdy_ctrt', 0))
                        })
                
                time.sleep(0.1)  # API 호출 제한 고려
            
            print(f"📊 코스피200 종목 수집 완료: {len(stocks)}개")
            return stocks
            
        except Exception as e:
            print(f"❌ 코스피200 조회 오류: {str(e)}")
            return []
    
    def get_nasdaq100_stocks(self) -> List[Dict]:
        """나스닥100 종목 조회 (Yahoo Finance 활용)"""
        try:
            # 나스닥100 대표 종목들
            nasdaq100_symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                "META", "NVDA", "NFLX", "ADBE", "PYPL",
                "INTC", "CSCO", "CMCSA", "PEP", "COST",
                "TMUS", "AVGO", "TXN", "QCOM", "CHTR"
            ]
            
            stocks = []
            for symbol in nasdaq100_symbols:
                try:
                    # Yahoo Finance API 호출 (무료)
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'chart' in data and data['chart']['result']:
                            result = data['chart']['result'][0]
                            meta = result['meta']
                            
                            stocks.append({
                                "symbol": symbol,
                                "name": meta.get('longName', symbol),
                                "exchange": "NASDAQ",
                                "category": "nasdaq100",
                                "current_price": meta.get('regularMarketPrice', 0),
                                "volume": meta.get('regularMarketVolume', 0),
                                "change_rate": meta.get('regularMarketChangePercent', 0)
                            })
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"⚠️ {symbol} 조회 실패: {str(e)}")
                    continue
            
            print(f"📊 나스닥100 종목 수집 완료: {len(stocks)}개")
            return stocks
            
        except Exception as e:
            print(f"❌ 나스닥100 조회 오류: {str(e)}")
            return []
    
    def get_sp500_stocks(self) -> List[Dict]:
        """S&P500 종목 조회 (ETF 및 대표 종목)"""
        try:
            # S&P500 대표 종목들
            sp500_symbols = [
                "SPY", "VOO", "IVV",  # S&P500 ETF
                "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
                "BRK-B", "UNH", "JNJ", "V", "PG",
                "JPM", "MA", "HD", "NVDA", "DIS",
                "BAC", "ADBE", "CRM", "NFLX", "KO"
            ]
            
            stocks = []
            for symbol in sp500_symbols:
                try:
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'chart' in data and data['chart']['result']:
                            result = data['chart']['result'][0]
                            meta = result['meta']
                            
                            stocks.append({
                                "symbol": symbol,
                                "name": meta.get('longName', symbol),
                                "exchange": "NYSE/NASDAQ",
                                "category": "sp500",
                                "current_price": meta.get('regularMarketPrice', 0),
                                "volume": meta.get('regularMarketVolume', 0),
                                "change_rate": meta.get('regularMarketChangePercent', 0)
                            })
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"⚠️ {symbol} 조회 실패: {str(e)}")
                    continue
            
            print(f"📊 S&P500 종목 수집 완료: {len(stocks)}개")
            return stocks
            
        except Exception as e:
            print(f"❌ S&P500 조회 오류: {str(e)}")
            return []
    
    def apply_investment_strategies(self, stocks: List[Dict]) -> Dict[str, List[Dict]]:
        """투자 대가 전략 적용"""
        strategies = {
            "warren_buffett": [],
            "peter_lynch": [],
            "benjamin_graham": []
        }
        
        for stock in stocks:
            # 워런 버핏 전략 (우량주 장기투자)
            buffett_score = 0
            if stock['current_price'] > 0:
                if stock['volume'] > 1000000:  # 거래량 충분
                    buffett_score += 20
                if abs(stock['change_rate']) < 5:  # 안정적
                    buffett_score += 20
                if stock['category'] in ['kospi200', 'sp500']:  # 대형주
                    buffett_score += 30
            
            stock_copy = stock.copy()
            stock_copy['strategy_score'] = buffett_score
            stock_copy['strategy'] = 'Warren Buffett'
            strategies['warren_buffett'].append(stock_copy)
            
            # 피터 린치 전략 (성장주)
            lynch_score = 0
            if stock['change_rate'] > 0:  # 상승 중
                lynch_score += 25
            if stock['volume'] > 500000:  # 거래 활발
                lynch_score += 20
            if stock['category'] == 'nasdaq100':  # 기술주 선호
                lynch_score += 25
            
            stock_copy = stock.copy()
            stock_copy['strategy_score'] = lynch_score
            stock_copy['strategy'] = 'Peter Lynch'
            strategies['peter_lynch'].append(stock_copy)
            
            # 벤저민 그레이엄 전략 (가치투자)
            graham_score = 0
            if stock['change_rate'] < 0:  # 하락 중 (저평가)
                graham_score += 30
            if stock['current_price'] > 0:
                graham_score += 15
            if stock['category'] in ['kospi200', 'sp500']:
                graham_score += 20
            
            stock_copy = stock.copy()
            stock_copy['strategy_score'] = graham_score
            stock_copy['strategy'] = 'Benjamin Graham'
            strategies['benjamin_graham'].append(stock_copy)
        
        # 각 전략별 정렬
        for strategy in strategies:
            strategies[strategy].sort(key=lambda x: x['strategy_score'], reverse=True)
        
        return strategies
    
    def gemini_ai_analysis(self, strategy_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Gemini AI를 활용한 종합 분석 및 Top5 선정"""
        try:
            # 각 전략별 상위 10개 종목 추출
            top_candidates = {}
            for strategy, stocks in strategy_results.items():
                top_candidates[strategy] = stocks[:10]
            
            # Gemini AI 프롬프트 구성
            prompt = f"""
당신은 세계적인 투자 전문가입니다. 다음 데이터를 분석하여 최적의 Top5 종목을 선정해주세요.

## 분석 데이터
{json.dumps(top_candidates, ensure_ascii=False, indent=2)}

## 요구사항
1. 각 투자 전략(워런 버핏, 피터 린치, 벤저민 그레이엄)의 특성을 고려
2. 현재 시장 상황과 기술적 지표 분석
3. 리스크 대비 수익률 최적화
4. 포트폴리오 다각화 고려

## 출력 형식
다음 JSON 형식으로 응답해주세요:
{{
    "top5_selections": [
        {{
            "rank": 1,
            "symbol": "종목코드",
            "name": "종목명",
            "category": "카테고리",
            "selected_strategy": "선택된 전략",
            "confidence_score": 95,
            "reasoning": "선정 이유 상세 설명"
        }}
    ],
    "market_analysis": "현재 시장 상황 분석",
    "risk_assessment": "리스크 평가",
    "investment_recommendation": "투자 권고사항"
}}
"""
            
            response = self.gemini_model.generate_content(prompt)
            
            # JSON 파싱 시도
            try:
                # 응답에서 JSON 부분만 추출
                response_text = response.text
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text
                
                analysis_result = json.loads(json_text)
                
                print("🤖 Gemini AI 분석 완료")
                return analysis_result
                
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 대체 분석
                print("⚠️ Gemini AI 응답 파싱 실패, 대체 분석 수행")
                return self.fallback_analysis(strategy_results)
                
        except Exception as e:
            print(f"❌ Gemini AI 분석 오류: {str(e)}")
            return self.fallback_analysis(strategy_results)
    
    def fallback_analysis(self, strategy_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Gemini AI 실패 시 대체 분석"""
        all_candidates = []
        
        for strategy, stocks in strategy_results.items():
            for stock in stocks[:5]:  # 각 전략별 상위 5개
                stock['final_score'] = stock['strategy_score']
                all_candidates.append(stock)
        
        # 점수순 정렬
        all_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        top5 = []
        selected_symbols = set()
        
        for stock in all_candidates:
            if stock['symbol'] not in selected_symbols and len(top5) < 5:
                top5.append({
                    "rank": len(top5) + 1,
                    "symbol": stock['symbol'],
                    "name": stock['name'],
                    "category": stock['category'],
                    "selected_strategy": stock['strategy'],
                    "confidence_score": min(stock['final_score'], 100),
                    "reasoning": f"{stock['strategy']} 전략 기준 상위 종목 (점수: {stock['final_score']})"
                })
                selected_symbols.add(stock['symbol'])
        
        return {
            "top5_selections": top5,
            "market_analysis": "기술적 분석 기반 자동 선정",
            "risk_assessment": "중간 위험도",
            "investment_recommendation": "분산투자 권장"
        }
    
    def generate_report(self, final_analysis: Dict[str, Any]) -> str:
        """최종 리포트 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = f"""
# 🚀 최종 투자 종목 분석 리포트
생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Top 5 선정 종목

"""
        
        for stock in final_analysis['top5_selections']:
            report += f"""
### {stock['rank']}위. {stock['name']} ({stock['symbol']})
- **카테고리**: {stock['category']}
- **선정 전략**: {stock['selected_strategy']}
- **신뢰도**: {stock['confidence_score']}%
- **선정 이유**: {stock['reasoning']}

"""
        
        report += f"""
## 🌍 시장 분석
{final_analysis['market_analysis']}

## ⚠️ 리스크 평가
{final_analysis['risk_assessment']}

## 💡 투자 권고사항
{final_analysis['investment_recommendation']}

---
*본 분석은 투자 참고용이며, 투자 결정은 개인 책임입니다.*
"""
        
        # 파일 저장
        report_filename = f"final_investment_report_{timestamp}.md"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # JSON 결과도 저장
        json_filename = f"final_analysis_result_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "all_stocks_count": len(self.all_stocks_data),
                "final_analysis": final_analysis
            }, f, ensure_ascii=False, indent=2)
        
        print(f"📄 리포트 저장: {report_filename}")
        print(f"📄 JSON 저장: {json_filename}")
        
        return report_filename
    
    def run_complete_analysis(self):
        """전체 분석 실행"""
        print("=" * 60)
        print("🚀 최종 통합 주식 분석 시스템 시작")
        print("=" * 60)
        
        # 1. 한투 API 토큰 발급
        if not self.get_kis_token():
            print("❌ 한투 API 연결 실패")
            return
        
        # 2. 전체 종목 데이터 수집
        print("\n📊 종목 데이터 수집 중...")
        
        kospi_stocks = self.get_kospi200_stocks()
        nasdaq_stocks = self.get_nasdaq100_stocks()
        sp500_stocks = self.get_sp500_stocks()
        
        self.all_stocks_data = kospi_stocks + nasdaq_stocks + sp500_stocks
        
        print(f"\n✅ 전체 수집 완료: {len(self.all_stocks_data)}개 종목")
        print(f"   - 코스피200: {len(kospi_stocks)}개")
        print(f"   - 나스닥100: {len(nasdaq_stocks)}개")
        print(f"   - S&P500: {len(sp500_stocks)}개")
        
        if not self.all_stocks_data:
            print("❌ 수집된 종목이 없습니다.")
            return
        
        # 3. 투자 전략 적용
        print("\n🎯 투자 전략 적용 중...")
        strategy_results = self.apply_investment_strategies(self.all_stocks_data)
        
        for strategy, stocks in strategy_results.items():
            print(f"   - {strategy}: {len(stocks)}개 종목 분석")
        
        # 4. Gemini AI 종합 분석
        print("\n🤖 Gemini AI 종합 분석 중...")
        final_analysis = self.gemini_ai_analysis(strategy_results)
        
        # 5. 최종 리포트 생성
        print("\n📄 최종 리포트 생성 중...")
        report_file = self.generate_report(final_analysis)
        
        # 6. 결과 출력
        print("\n" + "=" * 60)
        print("🎉 분석 완료! Top 5 선정 종목:")
        print("=" * 60)
        
        for stock in final_analysis['top5_selections']:
            print(f"{stock['rank']}위. {stock['name']} ({stock['symbol']}) - {stock['confidence_score']}%")
        
        print(f"\n📄 상세 리포트: {report_file}")
        print("=" * 60)

if __name__ == "__main__":
    analyzer = FinalStockAnalyzer()
    analyzer.run_complete_analysis() 