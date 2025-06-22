#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 제미나이 투자천재 시스템 (Gemini Investment Genius System)
- 1단계: 고정된 데이터 포맷 (stock_data.csv)
- 2단계: 퀀트 전략 수치 정의 (마법공식, 퀄리티+모멘텀)
- 3단계: 제미나이 완성형 프롬프트
- 4단계: 연속 분석 학습 시스템
- 5단계: 텔레그램 알림 연동
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os
import json
import requests
from typing import Dict, List, Tuple, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiInvestmentGenius:
    """제미나이 투자천재 시스템 메인 클래스"""
    
    def __init__(self, data_file: str = "data/stock_data.csv"):
        self.data_file = data_file
        self.results_dir = "results"
        self.config_file = "config/gemini_config.json"
        
        # 결과 저장 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs("config", exist_ok=True)
        
        # 설정 로드
        self.config = self.load_config()
        
        # 전략 정의
        self.strategies = {
            "magic_formula": self.magic_formula_strategy,
            "quality_momentum": self.quality_momentum_strategy,
            "combined": self.combined_strategy
        }
    
    def load_config(self) -> Dict:
        """설정 파일 로드"""
        default_config = {
            "telegram": {
                "bot_token": "",
                "chat_id": "",
                "enabled": False
            },
            "strategies": {
                "magic_formula_weight": 0.6,
                "quality_momentum_weight": 0.4,
                "top_n_stocks": 10
            },
            "analysis": {
                "min_market_cap": 1e10,  # 100억 이상
                "max_per": 50,           # PER 50 이하
                "min_roe": 5             # ROE 5% 이상
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 기본값과 병합
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패: {e}")
        
        # 기본 설정 저장
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        return default_config
    
    def load_stock_data(self) -> pd.DataFrame:
        """📂 1단계: 주식 데이터 로드"""
        try:
            df = pd.read_csv(self.data_file)
            logger.info(f"데이터 로드 완료: {len(df)}개 종목")
            
            # 기본 필터링
            df = df[
                (df['MarketCap'] >= self.config['analysis']['min_market_cap']) &
                (df['PER'] > 0) & (df['PER'] <= self.config['analysis']['max_per']) &
                (df['ROE'] >= self.config['analysis']['min_roe'])
            ].copy()
            
            logger.info(f"필터링 후: {len(df)}개 종목")
            return df
            
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def magic_formula_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """📘 2단계: 마법공식 전략 (조엘 그린블라트)"""
        if len(df) == 0:
            return df
        
        # PER 순위 (낮을수록 좋음)
        df['PER_Rank'] = df['PER'].rank(ascending=True)
        
        # ROIC 순위 (높을수록 좋음) - ROE로 대체
        df['ROIC_Rank'] = df['ROIC'].rank(ascending=False)
        
        # 마법공식 종합 점수 (순위 합산, 낮을수록 좋음)
        df['Magic_Score'] = df['PER_Rank'] + df['ROIC_Rank']
        
        # 정규화 (0-100점, 높을수록 좋음)
        df['Magic_Score_Normalized'] = 100 - ((df['Magic_Score'] - df['Magic_Score'].min()) / 
                                             (df['Magic_Score'].max() - df['Magic_Score'].min()) * 100)
        
        return df
    
    def quality_momentum_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """📘 2단계: 퀄리티 + 모멘텀 전략"""
        if len(df) == 0:
            return df
        
        # 정규화를 위한 스케일러
        scaler = MinMaxScaler()
        
        # 각 지표 정규화 (0-1 범위)
        quality_momentum_features = ['ROE', '6M_Return']
        volatility_feature = ['Volatility']
        
        # ROE와 6개월 수익률은 높을수록 좋음
        df[['ROE_Normalized', '6M_Return_Normalized']] = scaler.fit_transform(df[quality_momentum_features])
        
        # 변동성은 낮을수록 좋음 (역정규화)
        df['Volatility_Normalized'] = 1 - scaler.fit_transform(df[volatility_feature]).flatten()
        
        # 퀄리티 + 모멘텀 종합 점수
        df['Quality_Momentum_Score'] = (
            df['ROE_Normalized'] * 0.4 +           # ROE 40%
            df['6M_Return_Normalized'] * 0.4 +     # 6개월 수익률 40%
            df['Volatility_Normalized'] * 0.2      # 변동성 20%
        ) * 100  # 0-100점 변환
        
        return df
    
    def combined_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """📘 2단계: 통합 전략 (마법공식 + 퀄리티모멘텀)"""
        # 개별 전략 실행
        df = self.magic_formula_strategy(df)
        df = self.quality_momentum_strategy(df)
        
        # 가중 평균으로 최종 점수 계산
        magic_weight = self.config['strategies']['magic_formula_weight']
        quality_weight = self.config['strategies']['quality_momentum_weight']
        
        df['Final_Score'] = (
            df['Magic_Score_Normalized'] * magic_weight +
            df['Quality_Momentum_Score'] * quality_weight
        )
        
        return df
    
    def analyze_stocks(self, strategy: str = "combined") -> pd.DataFrame:
        """🧠 3단계: 제미나이 분석 실행"""
        # 데이터 로드
        df = self.load_stock_data()
        if len(df) == 0:
            logger.error("분석할 데이터가 없습니다.")
            return pd.DataFrame()
        
        # 전략 실행
        if strategy in self.strategies:
            df = self.strategies[strategy](df)
        else:
            logger.error(f"알 수 없는 전략: {strategy}")
            return pd.DataFrame()
        
        # 상위 종목 선정
        top_n = self.config['strategies']['top_n_stocks']
        if strategy == "combined":
            top_stocks = df.nlargest(top_n, 'Final_Score')
            score_column = 'Final_Score'
        elif strategy == "magic_formula":
            top_stocks = df.nlargest(top_n, 'Magic_Score_Normalized')
            score_column = 'Magic_Score_Normalized'
        else:  # quality_momentum
            top_stocks = df.nlargest(top_n, 'Quality_Momentum_Score')
            score_column = 'Quality_Momentum_Score'
        
        # 결과 정리
        result_columns = ['Ticker', 'Market', 'Close', 'PER', 'ROIC', 'ROE', 
                         '6M_Return', 'Volatility', 'Sector', score_column]
        
        result_df = top_stocks[result_columns].copy()
        result_df = result_df.rename(columns={score_column: 'Score'})
        result_df = result_df.round(2)
        
        return result_df
    
    def generate_gemini_prompt(self, strategy: str = "combined") -> str:
        """🧠 3단계: 제미나이용 완성형 프롬프트 생성"""
        prompt = f"""
제미나이야, stock_data.csv 파일을 기반으로 다음 조건에 맞춰 투자 유망 종목을 추출해줘.

✅ 전략 조건:
- PER 낮고 ROIC 높은 종목 우선 (마법공식) - 가중치 {self.config['strategies']['magic_formula_weight']}
- ROE 높고, 6개월 수익률 높고, 변동성 낮은 종목도 추가 점수 부여 - 가중치 {self.config['strategies']['quality_momentum_weight']}
- 총 점수로 상위 {self.config['strategies']['top_n_stocks']}개 종목을 선정

📊 필터링 조건:
- 시가총액 {self.config['analysis']['min_market_cap']/1e9:.0f}억 이상
- PER {self.config['analysis']['max_per']} 이하
- ROE {self.config['analysis']['min_roe']}% 이상

📈 출력 항목:
- Ticker, Market, Close, PER, ROIC, ROE, 6M_Return, Volatility, Sector, Score
- 결과는 top_10_stocks.csv로 저장하는 Pandas 코드를 포함해줘

📦 라이브러리: pandas, sklearn 사용

💡 분석 코드:
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 데이터 로드
df = pd.read_csv('data/stock_data.csv')

# 필터링
df = df[
    (df['MarketCap'] >= {self.config['analysis']['min_market_cap']}) &
    (df['PER'] > 0) & (df['PER'] <= {self.config['analysis']['max_per']}) &
    (df['ROE'] >= {self.config['analysis']['min_roe']})
].copy()

# 마법공식 점수
df['PER_Rank'] = df['PER'].rank(ascending=True)
df['ROIC_Rank'] = df['ROIC'].rank(ascending=False)
df['Magic_Score'] = df['PER_Rank'] + df['ROIC_Rank']
df['Magic_Score_Normalized'] = 100 - ((df['Magic_Score'] - df['Magic_Score'].min()) / 
                                     (df['Magic_Score'].max() - df['Magic_Score'].min()) * 100)

# 퀄리티+모멘텀 점수
scaler = MinMaxScaler()
df[['ROE_Normalized', '6M_Return_Normalized']] = scaler.fit_transform(df[['ROE', '6M_Return']])
df['Volatility_Normalized'] = 1 - scaler.fit_transform(df[['Volatility']])
df['Quality_Momentum_Score'] = (df['ROE_Normalized'] * 0.4 + 
                               df['6M_Return_Normalized'] * 0.4 + 
                               df['Volatility_Normalized'] * 0.2) * 100

# 최종 점수
df['Final_Score'] = (df['Magic_Score_Normalized'] * {self.config['strategies']['magic_formula_weight']} +
                    df['Quality_Momentum_Score'] * {self.config['strategies']['quality_momentum_weight']})

# 상위 종목 선정
top_stocks = df.nlargest({self.config['strategies']['top_n_stocks']}, 'Final_Score')
result = top_stocks[['Ticker', 'Market', 'Close', 'PER', 'ROIC', 'ROE', 
                    '6M_Return', 'Volatility', 'Sector', 'Final_Score']].round(2)

# CSV 저장
result.to_csv('top_10_stocks.csv', index=False)
print(result)
```
"""
        return prompt
    
    def save_results(self, result_df: pd.DataFrame, strategy: str = "combined") -> str:
        """💻 4단계: 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/top_stocks_{strategy}_{timestamp}.csv"
        
        try:
            result_df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"결과 저장 완료: {filename}")
            
            # 최신 결과도 별도 저장 (텔레그램용)
            latest_filename = f"{self.results_dir}/latest_top_stocks.csv"
            result_df.to_csv(latest_filename, index=False, encoding='utf-8-sig')
            
            return filename
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            return ""
    
    def send_telegram_alert(self, result_df: pd.DataFrame) -> bool:
        """📤 5단계: 텔레그램 알림"""
        if not self.config['telegram']['enabled'] or not self.config['telegram']['bot_token']:
            logger.info("텔레그램 알림이 비활성화되어 있습니다.")
            return False
        
        try:
            # 메시지 생성
            message = "🧠 제미나이 투자천재 분석 결과\n"
            message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            message += "🏆 TOP 10 추천 종목:\n"
            
            for idx, row in result_df.head(10).iterrows():
                market_flag = "🇺🇸" if row['Market'] == 'US' else "🇰🇷"
                message += f"{idx+1}. {market_flag} {row['Ticker']}\n"
                message += f"   💰 {row['Close']:,.0f} | 📊 {row['Score']:.1f}점\n"
                message += f"   PER: {row['PER']:.1f} | ROE: {row['ROE']:.1f}%\n\n"
            
            message += "💡 투자는 본인 책임하에 신중히 결정하세요!"
            
            # 텔레그램 전송
            url = f"https://api.telegram.org/bot{self.config['telegram']['bot_token']}/sendMessage"
            payload = {
                'chat_id': self.config['telegram']['chat_id'],
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info("텔레그램 알림 전송 완료")
                return True
            else:
                logger.error(f"텔레그램 전송 실패: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"텔레그램 알림 오류: {e}")
            return False
    
    def run_analysis(self, strategy: str = "combined", send_alert: bool = True) -> pd.DataFrame:
        """전체 분석 실행"""
        logger.info(f"🧠 제미나이 투자천재 분석 시작 - 전략: {strategy}")
        
        # 분석 실행
        result_df = self.analyze_stocks(strategy)
        
        if len(result_df) == 0:
            logger.error("분석 결과가 없습니다.")
            return pd.DataFrame()
        
        # 결과 저장
        filename = self.save_results(result_df, strategy)
        
        # 텔레그램 알림
        if send_alert:
            self.send_telegram_alert(result_df)
        
        # 결과 출력
        print("\n🏆 제미나이 투자천재 분석 결과:")
        print("=" * 80)
        print(result_df.to_string(index=False))
        print("=" * 80)
        print(f"📁 결과 파일: {filename}")
        
        return result_df
    
    def setup_telegram(self, bot_token: str, chat_id: str):
        """텔레그램 설정"""
        self.config['telegram']['bot_token'] = bot_token
        self.config['telegram']['chat_id'] = chat_id
        self.config['telegram']['enabled'] = True
        
        # 설정 저장
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        logger.info("텔레그램 설정 완료")
    
    def get_gemini_learning_prompt(self) -> str:
        """🧠 4단계: 제미나이 학습용 연속 분석 프롬프트"""
        return """
제미나이야, 이전 분석과 동일한 방식으로 이번 주의 stock_data.csv를 다시 분석해줘.

📋 분석 조건 (이전과 동일):
- 마법공식 (PER↓ + ROIC↑) 60% 가중치
- 퀄리티+모멘텀 (ROE↑ + 6M수익률↑ + 변동성↓) 40% 가중치
- 상위 10개 종목 선정

📊 출력 형식 (지난번과 동일하게):
- Ticker, Market, Close, PER, ROIC, ROE, 6M_Return, Volatility, Sector, Score
- CSV 저장 코드 포함
- 일관성 있는 판단 기준 유지

💡 학습 포인트:
- 동일한 로직으로 반복 분석
- 시간에 따른 종목 변화 추적 가능
- 전략의 일관성 검증
"""

def main():
    """메인 실행 함수"""
    # 제미나이 투자천재 시스템 초기화
    genius = GeminiInvestmentGenius()
    
    print("🧠 제미나이 투자천재 시스템")
    print("=" * 50)
    
    while True:
        print("\n📋 메뉴:")
        print("1. 통합 전략 분석")
        print("2. 마법공식 전략")
        print("3. 퀄리티+모멘텀 전략")
        print("4. 제미나이 프롬프트 생성")
        print("5. 텔레그램 설정")
        print("6. 자동 분석 (텔레그램 알림)")
        print("0. 종료")
        
        choice = input("\n선택하세요: ").strip()
        
        if choice == "1":
            genius.run_analysis("combined", send_alert=False)
        elif choice == "2":
            genius.run_analysis("magic_formula", send_alert=False)
        elif choice == "3":
            genius.run_analysis("quality_momentum", send_alert=False)
        elif choice == "4":
            prompt = genius.generate_gemini_prompt()
            print("\n🧠 제미나이용 프롬프트:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)
        elif choice == "5":
            bot_token = input("텔레그램 봇 토큰: ").strip()
            chat_id = input("채팅 ID: ").strip()
            genius.setup_telegram(bot_token, chat_id)
        elif choice == "6":
            genius.run_analysis("combined", send_alert=True)
        elif choice == "0":
            print("👋 제미나이 투자천재 시스템을 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main() 