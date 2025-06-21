"""
🚀 척후병 전략 파라미터 최적화 백테스터 (v2.0)
- 지정된 파라미터 범위에 대해 모든 조합을 생성하고 병렬로 백테스트를 실행합니다.
- 각 조합의 성과(수익률, MDD, 승률 등)를 측정하고 최적의 파라미터를 찾습니다.
- 과거 데이터를 효율적으로 로드하고 캐싱하여 반복 테스트 속도를 향상시킵니다.
"""
import logging
import pandas as pd
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from gspread_dataframe import set_with_dataframe
import yfinance as yf

# --- 시스템 모듈 임포트 ---
# (CoreTrader는 실제 API 호출을 하므로, 시뮬레이션에 필요한 데이터만 가져오도록 모킹/수정 필요)
# from core_trader import CoreTrader 
from google_sheet_logger import GoogleSheetLogger # 구글 시트 로거 임포트

# --- 로거 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 데이터 관리 ---
class HistoricalDataManager:
    """과거 데이터를 효율적으로 관리 (파일 캐싱)"""
    def __init__(self, cache_dir="historical_data_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # self.trader = CoreTrader() # 실제 API 호출 대신 파일 I/O로 대체

    def get_data(self, symbol, start_date, end_date) -> Optional[pd.DataFrame]:
        """파일 캐시에서 데이터를 읽거나, 없을 경우 Yahoo Finance에서 다운로드합니다."""
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}.pkl")
        if os.path.exists(cache_file):
            logger.debug(f"캐시에서 {symbol} 데이터 로드.")
            return pd.read_pickle(cache_file)
        
        logger.info(f"캐시 없음: Yahoo Finance에서 {symbol} 데이터 다운로드 중...")
        try:
            # 코스피/코스닥 종목에 .KS/.KQ 접미사 추가 (yfinance 형식)
            if len(symbol) == 6 and symbol.isdigit():
                # 간단한 예시로 .KS를 붙이지만, 실제로는 종목 시장 정보 필요
                 ticker = f"{symbol}.KS"
            else:
                 ticker = symbol

            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df.empty:
                logger.warning(f"{symbol}에 대한 데이터를 찾을 수 없습니다.")
                return None

            # 우리 시스템에서 사용하는 컬럼명으로 변경
            df.rename(columns={
                'Open': 'stck_oprc',
                'High': 'stck_hgpr',
                'Low': 'stck_lwpr',
                'Close': 'stck_clpr',
                'Volume': 'acml_vol'
            }, inplace=True)
            
            # 필요한 경우 'Adj Close' 등 다른 데이터도 활용 가능
            df['volume_ma20'] = df['acml_vol'].rolling(window=20).mean()

            df.to_pickle(cache_file)
            logger.info(f"{symbol} 데이터 다운로드 및 캐시 저장 완료.")
            return df

        except Exception as e:
            logger.error(f"{symbol} 데이터 다운로드 중 오류 발생: {e}", exc_info=True)
            return None

# --- 백테스팅 시뮬레이터 ---
def run_simulation(args: Dict) -> Dict:
    """단일 파라미터 조합으로 백테스트를 실행하는 함수 (병렬 처리용)"""
    params = args['params']
    symbol = args['symbol']
    data = args['data'].copy()
    
    initial_cash = 10_000_000
    cash = initial_cash
    holdings = None  # 보유 주식 정보 초기화
    
    trade_log = []
    daily_portfolio_value = []
    profit_percentages = []
    
    # 시장 상황별 성과 분석을 위한 변수들
    market_stats = {
        '상승장': {'trades': 0, 'wins': 0, 'total_return': 0.0},
        '하락장': {'trades': 0, 'wins': 0, 'total_return': 0.0},
        '횡보장': {'trades': 0, 'wins': 0, 'total_return': 0.0}
    }
    
    # 거래량 기반 성과 분석
    volume_stats = {
        '증가': {'trades': 0, 'wins': 0, 'total_return': 0.0},
        '감소': {'trades': 0, 'wins': 0, 'total_return': 0.0}
    }
    
    # 이동평균선 추가 (추세 강도 판단용)
    data['ma5'] = data['stck_clpr'].rolling(window=5).mean()
    data['ma20'] = data['stck_clpr'].rolling(window=20).mean()
    data['volume_ma5'] = data['acml_vol'].rolling(window=5).mean()
    data['volume_ma20'] = data['acml_vol'].rolling(window=20).mean()
    
    for date, row in data.iterrows():
        # 시장 상황 판단
        market_condition = '상승장' if row['ma5'] > row['ma20'] else '하락장' if row['ma5'] < row['ma20'] else '횡보장'
        volume_condition = '증가' if row['acml_vol'] > row['volume_ma5'] else '감소'
        
        # 매수/매도 로직
        if holdings:
            current_profit_percent = ((row['stck_clpr'] - holdings['purchase_price']) / holdings['purchase_price']) * 100
            
            # 매도 조건 확인
            if current_profit_percent <= -params['initial_stop_loss']:
                # 손절
                profit = (row['stck_clpr'] - holdings['purchase_price']) * holdings['quantity']
                
                # 시장 상황별 통계 업데이트
                market_stats[market_condition]['trades'] += 1
                if profit > 0:
                    market_stats[market_condition]['wins'] += 1
                market_stats[market_condition]['total_return'] += current_profit_percent
                
                # 거래량 기반 통계 업데이트
                volume_stats[volume_condition]['trades'] += 1
                if profit > 0:
                    volume_stats[volume_condition]['wins'] += 1
                volume_stats[volume_condition]['total_return'] += current_profit_percent
                
                trade_log.append({
                    'date': date,
                    'type': 'SELL',
                    'price': row['stck_clpr'],
                    'profit_percent': current_profit_percent,
                    'market_condition': market_condition,
                    'volume_condition': volume_condition
                })
                
                holdings = None
                cash += row['stck_clpr'] * holdings['quantity']
                
            elif current_profit_percent >= params['trailing_activation']:
                # 트레일링 스탑 로직
                stop_price = holdings['high_price'] * (1 - params['trailing_stop'] / 100)
                if row['stck_clpr'] <= stop_price:
                    profit = (row['stck_clpr'] - holdings['purchase_price']) * holdings['quantity']
                    
                    # 통계 업데이트
                    market_stats[market_condition]['trades'] += 1
                    if profit > 0:
                        market_stats[market_condition]['wins'] += 1
                    market_stats[market_condition]['total_return'] += current_profit_percent
                    
                    volume_stats[volume_condition]['trades'] += 1
                    if profit > 0:
                        volume_stats[volume_condition]['wins'] += 1
                    volume_stats[volume_condition]['total_return'] += current_profit_percent
                    
                    trade_log.append({
                        'date': date,
                        'type': 'SELL',
                        'price': row['stck_clpr'],
                        'profit_percent': current_profit_percent,
                        'market_condition': market_condition,
                        'volume_condition': volume_condition
                    })
                    
                    holdings = None
                    cash += row['stck_clpr'] * holdings['quantity']
                else:
                    holdings['high_price'] = max(holdings['high_price'], row['stck_hgpr'])
        
        # 매수 조건 확인
        elif row['ai_score'] >= 85 and cash >= row['stck_clpr']:
            quantity = int(cash / row['stck_clpr'])
            if quantity > 0:
                holdings = {
                    'purchase_price': row['stck_clpr'],
                    'quantity': quantity,
                    'high_price': row['stck_clpr'],
                    'purchase_date': date
                }
                cash -= row['stck_clpr'] * quantity
                
                trade_log.append({
                    'date': date,
                    'type': 'BUY',
                    'price': row['stck_clpr'],
                    'market_condition': market_condition,
                    'volume_condition': volume_condition
                })
        
        # 포트폴리오 가치 업데이트
        current_value = cash
        if holdings:
            current_value += holdings['quantity'] * row['stck_clpr']
        daily_portfolio_value.append(current_value)
    
    # 최종 결과 계산
    final_value = daily_portfolio_value[-1] if daily_portfolio_value else initial_cash
    total_return = ((final_value / initial_cash) - 1) * 100
    
    # 승률 계산
    for condition in market_stats:
        if market_stats[condition]['trades'] > 0:
            market_stats[condition]['win_rate'] = (market_stats[condition]['wins'] / market_stats[condition]['trades']) * 100
            market_stats[condition]['avg_return'] = market_stats[condition]['total_return'] / market_stats[condition]['trades']
    
    for condition in volume_stats:
        if volume_stats[condition]['trades'] > 0:
            volume_stats[condition]['win_rate'] = (volume_stats[condition]['wins'] / volume_stats[condition]['trades']) * 100
            volume_stats[condition]['avg_return'] = volume_stats[condition]['total_return'] / volume_stats[condition]['trades']
    
    return {
        'params': params,
        'total_return': total_return,
        'market_stats': market_stats,
        'volume_stats': volume_stats,
        'trade_log': trade_log
    }

# --- 메인 컨트롤러 ---
class ParameterOptimizer:
    """백테스팅을 총괄하고 파라미터를 최적화하는 컨트롤러"""
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data_manager = HistoricalDataManager()
        self.market_analyzer = MarketAnalyzer(start_date, end_date)

    def define_parameter_space(self):
        """최적화할 파라미터의 범위와 간격을 정의합니다."""
        param_space = {
            'initial_stop_loss': np.arange(2.0, 6.5, 0.5),      # 2.0% ~ 6.0%, 0.5% 간격
            'trailing_activation': np.arange(3.0, 8.5, 0.5),   # 3.0% ~ 8.0%, 0.5% 간격
            'trailing_stop': np.arange(1.5, 5.0, 0.5),         # 1.5% ~ 4.5%, 0.5% 간격
        }
        keys = param_space.keys()
        values = param_space.values()
        # 모든 파라미터 조합 생성
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        logger.info(f"총 {len(param_combinations)}개의 파라미터 조합 생성됨.")
        return param_combinations

    def run_optimization(self):
        param_combinations = self.define_parameter_space()
        
        all_results = []
        for symbol in self.symbols:
            logger.info(f"\n===== {symbol} 종목에 대한 최적화 시작 =====")
            data = self.data_manager.get_data(symbol, self.start_date, self.end_date)
            if data is None or data.empty:
                logger.warning(f"{symbol} 데이터가 없어 건너뜁니다.")
                continue

            # --- 병렬 처리 비활성화 (디버깅용) ---
            symbol_results = []
            logger.info(f"{symbol} 최적화 중 (순차 처리)...")
            for params in tqdm(param_combinations, desc=f"{symbol} 최적화 중"):
                try:
                    task = {'params': params, 'symbol': symbol, 'data': data}
                    result = self.run_simulation(task)
                    symbol_results.append(result)
                except Exception as e:
                    logger.error(f"파라미터 {params} 처리 중 오류: {e}", exc_info=True)
            # --- 병렬 처리 원본 코드 ---
            # with ProcessPoolExecutor() as executor:
            #     futures = {executor.submit(run_simulation, task): task for task in tasks}
            #     for future in tqdm(as_completed(futures), total=len(futures), desc=f"{symbol} 최적화 중"):
            #         try:
            #             result = future.result()
            #             symbol_results.append(result)
            #         except Exception as e:
            #             logger.error(f"파라미터 {futures[future]['params']} 처리 중 오류: {e}", exc_info=True)
            
            if symbol_results:
                best_for_symbol = max(symbol_results, key=lambda x: x['total_return'])
                logger.info(f"✅ {symbol} 최적 파라미터(수익률 기준): {best_for_symbol['params']}, 수익률: {best_for_symbol['total_return']:.2f}%")
                all_results.extend(symbol_results)
        
        logger.info("\n===== 전체 종목 통합 결과 분석 =====")
        self.display_results(all_results)

    def display_results(self, results: List[Dict]):
        """최적화 결과를 분석하고 출력합니다."""
        if not results:
            logger.warning("결과가 없습니다. 최적화를 실행할 수 없습니다.")
            return

        results_df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("🏆 백테스팅 파라미터 최적화 결과 🏆")
        print("="*80)
        
        print("\n---  수익률(Total Return) 기준 TOP 5 ---")
        top_by_return = results_df.sort_values(by='total_return', ascending=False).head(5)
        print(top_by_return)
        
        print("\n--- 최대 낙폭(MDD) 방어 TOP 5 (낮을수록 좋음) ---")
        top_by_mdd = results_df.sort_values(by='mdd', ascending=True).head(5)
        print(top_by_mdd)
        
        print("\n--- 승률(Win Rate) 기준 TOP 5 ---")
        top_by_win_rate = results_df.sort_values(by='win_rate', ascending=False).head(5)
        print(top_by_win_rate)
        
        # 이익실현 통계 추가
        print("\n--- 이익실현 통계 (수익률 TOP 5 기준) ---")
        for idx, row in top_by_return.iterrows():
            print(f"\n[파라미터 조합 {idx+1}]")
            print(f"평균 이익실현 수익률: {row['avg_profit_taking']:.2f}%")
            print(f"최대 이익실현 수익률: {row['max_profit_taking']:.2f}%")
            print(f"최소 이익실현 수익률: {row['min_profit_taking']:.2f}%")
            print(f"이익실현 횟수: {row['profit_taking_points']}회")
            print(f"평균 보유기간: {row['avg_holding_period']:.1f}일")
            print(f"최대 보유기간: {row['max_holding_period']}일")
        
        best_params = top_by_return.iloc[0]['params']
        print("\n" + "*"*80)
        print(f"🎯 최종 추천 파라미터 (수익률 기준):")
        print(json.dumps(best_params, indent=2))
        print("*"*80)

        # --- 결과 저장 및 시각화 ---
        self.save_and_visualize_results(results_df.copy(), top_by_return.iloc[0])

    def save_and_visualize_results(self, results_df: pd.DataFrame, best_result: pd.Series):
        """결과를 파일로 저장하고 시각화합니다."""
        logger.info("백테스팅 결과 저장 및 시각화 시작...")

        # 결과 저장 디렉토리 생성
        output_dir = "backtest_results"
        charts_dir = os.path.join(output_dir, "charts")
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
            
        # 1. CSV 파일로 전체 결과 저장
        csv_path = os.path.join(output_dir, f"backtest_results_{self.start_date}_{self.end_date}.csv")
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"전체 결과가 {csv_path}에 저장되었습니다.")

        # 2. 구글 시트에 결과 업로드
        try:
            logger.info("구글 시트에 결과 업로드 시도...")
            spreadsheet_key = GoogleSheetLogger.load_spreadsheet_key('spreadsheet_key.txt')
            
            # 키 파일이나 인증 파일이 없으면 g_logger는 초기화에 실패하고 self.initialized = False가 됨
            g_logger = GoogleSheetLogger(
                credentials_path='credentials.json',
                spreadsheet_key=spreadsheet_key
            )

            if g_logger.initialized:
                sheet_name = f"백테스트_{self.start_date}_{self.end_date}"
                worksheet = g_logger.get_or_create_worksheet(sheet_name)
                
                if worksheet:
                    # 구글 시트 업로드용 데이터프레임 복사 및 수정
                    upload_df = results_df.copy()
                    upload_df['params'] = upload_df['params'].astype(str)
                    
                    set_with_dataframe(worksheet, upload_df, row=1, col=1, include_index=False, include_column_header=True, resize=True)
                    logger.info(f"'{sheet_name}' 구글 시트에 결과 업로드를 완료했습니다.")
                else:
                    logger.warning("결과를 업로드할 구글 워크시트를 가져오지 못했습니다.")
            else:
                # 이전에 FileNotFoundError 등으로 g_logger 초기화가 실패한 경우
                logger.warning("구글 시트 로거가 초기화되지 않았습니다. 업로드를 건너뜁니다.")

        except Exception as e:
            logger.error(f"구글 시트 업로드 중 예상치 못한 오류 발생: {e}", exc_info=True)

        # 3. 파라미터-수익률 관계 시각화
        try:
            # 원본 results_df의 'params'는 딕셔너리 형태이므로 eval 필요 없음
            params_df = pd.json_normalize(results_df['params'])
            
            # 3-1. 3D 산점도 (Stop Loss, Trailing Activation, Return)
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(params_df['initial_stop_loss'], params_df['trailing_activation'], results_df['total_return'], c=results_df['total_return'], cmap='viridis')
            ax.set_xlabel('Initial Stop Loss (%)')
            ax.set_ylabel('Trailing Activation (%)')
            ax.set_zlabel('Total Return (%)')
            ax.set_title('Stop Loss vs. Trailing Activation vs. Return')
            plt.colorbar(sc, label='Total Return (%)')
            plot_path = os.path.join(charts_dir, '3d_scatter_return.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"3D 산점도 저장: {plot_path}")

            # 3-2. 파라미터별 Heatmap
            plt.figure(figsize=(10, 8))
            pivot_table = pd.pivot_table(pd.concat([params_df, results_df['total_return']], axis=1), 
                                         values='total_return', 
                                         index=['initial_stop_loss'], 
                                         columns=['trailing_activation'], 
                                         aggfunc='mean')
            sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu")
            plt.title('Heatmap of Return by Stop Loss and Trailing Activation')
            plot_path = os.path.join(charts_dir, 'heatmap_return.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"수익률 히트맵 저장: {plot_path}")
            
        except Exception as e:
            logger.error(f"시각화 그래프 생성 중 오류 발생: {e}", exc_info=True)

    def run_simulation(self, args: Dict) -> Dict:
        params = args['params']
        symbol = args['symbol']
        data = args['data'].copy()
        
        initial_cash = 10_000_000
        cash = initial_cash
        holdings = None  # 보유 주식 정보 초기화
        
        trade_log = []
        daily_portfolio_value = []
        profit_percentages = []

        # 이동평균선 추가 (추세 강도 판단용)
        data['ma5'] = data['stck_clpr'].rolling(window=5).mean()
        data['ma20'] = data['stck_clpr'].rolling(window=20).mean()
        data['volume_ma5'] = data['acml_vol'].rolling(window=5).mean()
        
        # AI 점수 계산 (거래량 + 이동평균선 기반)
        volume_surge = data['acml_vol'] > data['volume_ma5'] * 1.5
        price_trend = (data['stck_clpr'] > data['ma5']) & (data['ma5'] > data['ma20'])
        data['ai_score'] = np.where(volume_surge & price_trend, 90, 
                                   np.where(volume_surge | price_trend, 80, 70))

        def get_dynamic_trailing_stop(current_profit_percent: float, volume_trend: bool) -> float:
            """수익률과 거래량 추세에 따라 동적으로 트레일링 스탑 비율 결정"""
            base_stop = params['trailing_stop']
            
            # 1. 수익 구간별 기본 비율
            if current_profit_percent >= 7.0:
                stop_ratio = base_stop * 2.0  # 수익 7% 이상: 넓은 스탑
            elif current_profit_percent >= 3.0:
                stop_ratio = base_stop * 1.5  # 수익 3~7%: 중간 스탑
            else:
                stop_ratio = base_stop        # 수익 0~3%: 기본 스탑
            
            # 2. 거래량 증가 시 추가 여유
            if volume_trend:
                stop_ratio *= 1.2
            
            return stop_ratio

        # 추가 통계 데이터 저장용 변수들
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        total_profit = 0
        total_loss = 0
        morning_trades = []  # 09:00-12:00
        lunch_trades = []    # 12:00-13:00
        afternoon_trades = [] # 13:00-15:30
        
        profit_factors = {
            'morning': {'profit': 0, 'loss': 0},
            'lunch': {'profit': 0, 'loss': 0},
            'afternoon': {'profit': 0, 'loss': 0}
        }

        for date, row in data.iterrows():
            time = pd.to_datetime(date).time()
            is_morning = time < pd.to_datetime('13:00').time()
            
            # 1. 매도 조건 확인
            if holdings:
                current_profit_percent = ((row['stck_clpr'] - holdings['purchase_price']) / holdings['purchase_price']) * 100
                volume_trend = row['acml_vol'] > row['volume_ma5']
                
                # 트레일링 스탑 로직
                if holdings['trailing_activated']:
                    # 동적 트레일링 스탑 계산
                    stop_ratio = get_dynamic_trailing_stop(current_profit_percent, volume_trend)
                    stop_price = holdings['high_price'] * (1 - stop_ratio / 100)
                    
                    if row['stck_lwpr'] <= stop_price:
                        sell_price = max(stop_price, row['stck_clpr'])
                        profit_percent = ((sell_price - holdings['purchase_price']) / holdings['purchase_price']) * 100
                        profit = (sell_price - holdings['purchase_price']) * holdings['quantity']
                        cash += sell_price * holdings['quantity']
                        
                        # 연속 승/패 기록
                        if profit > 0:
                            consecutive_wins += 1
                            consecutive_losses = 0
                            total_profit += profit
                            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                        else:
                            consecutive_losses += 1
                            consecutive_wins = 0
                            total_loss += abs(profit)
                            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        
                        # 시간대별 수익/손실 기록
                        trade_time = pd.to_datetime(date).time()
                        if trade_time < pd.to_datetime('12:00').time():
                            morning_trades.append(profit_percent)
                            if profit > 0:
                                profit_factors['morning']['profit'] += profit
                            else:
                                profit_factors['morning']['loss'] += abs(profit)
                        elif trade_time < pd.to_datetime('13:00').time():
                            lunch_trades.append(profit_percent)
                            if profit > 0:
                                profit_factors['lunch']['profit'] += profit
                            else:
                                profit_factors['lunch']['loss'] += abs(profit)
                        else:
                            afternoon_trades.append(profit_percent)
                            if profit > 0:
                                profit_factors['afternoon']['profit'] += profit
                            else:
                                profit_factors['afternoon']['loss'] += abs(profit)
                        
                        trade_info = {
                            'date': date,
                            'type': 'SELL',
                            'price': sell_price,
                            'profit': profit,
                            'profit_percent': profit_percent,
                            'high_price_percent': ((holdings['high_price'] - holdings['purchase_price']) / holdings['purchase_price']) * 100,
                            'reason': f'트레일링 스탑 (스탑률: {stop_ratio:.1f}%)',
                            'holding_period': (pd.to_datetime(date) - holdings['purchase_date']).days,
                            'time_of_day': trade_time.strftime('%H:%M'),
                            'volume_ratio': row['acml_vol'] / row['volume_ma5'],
                            'price_to_ma20': row['stck_clpr'] / row['ma20'],
                            'consecutive_wins': consecutive_wins if profit > 0 else 0,
                            'consecutive_losses': consecutive_losses if profit <= 0 else 0
                        }
                        
                        trade_log.append(trade_info)
                        
                        if profit > 0:
                            profit_percentages.append(profit_percent)
                        
                        holdings = None
                        continue
                
                # 초기 손절 로직 (장 초반에는 손절을 더 타이트하게)
                else:
                    stop_loss = params['initial_stop_loss'] * (0.8 if is_morning else 1.0)
                    stop_price = holdings['purchase_price'] * (1 - stop_loss / 100)
                    if row['stck_lwpr'] <= stop_price:
                        sell_price = max(stop_price, row['stck_clpr'])
                        profit_percent = ((sell_price - holdings['purchase_price']) / holdings['purchase_price']) * 100
                        profit = (sell_price - holdings['purchase_price']) * holdings['quantity']
                        cash += sell_price * holdings['quantity']
                        
                        trade_info = {
                            'date': date,
                            'type': 'SELL',
                            'price': sell_price,
                            'profit': profit,
                            'profit_percent': profit_percent,
                            'reason': f'손절 (손절률: {stop_loss:.1f}%)',
                            'holding_period': (pd.to_datetime(date) - holdings['purchase_date']).days,
                            'time_of_day': time.strftime('%H:%M'),
                            'volume_ratio': row['acml_vol'] / row['volume_ma5'],
                            'price_to_ma20': row['stck_clpr'] / row['ma20'],
                            'consecutive_wins': 0,
                            'consecutive_losses': 1
                        }
                        
                        trade_log.append(trade_info)
                        holdings = None
                        continue
                
                # 보유 중인 경우 고가 갱신 및 트레일링 활성화 체크
                if holdings:
                    holdings['high_price'] = max(holdings['high_price'], row['stck_hgpr'])
                    # 장 후반부에는 트레일링 활성화 기준을 낮춤 (수익 기회 확대)
                    activation_threshold = params['trailing_activation'] * (0.8 if not is_morning else 1.0)
                    if not holdings['trailing_activated'] and holdings['high_price'] >= holdings['purchase_price'] * (1 + activation_threshold / 100):
                        holdings['trailing_activated'] = True

            # 2. 매수 조건 확인
            if not holdings and row['ai_score'] >= 85:
                buy_price = row['stck_clpr']
                quantity = 1
                if cash >= buy_price * quantity:
                    cash -= buy_price * quantity
                    holdings = {
                        'purchase_price': buy_price,
                        'quantity': quantity,
                        'high_price': buy_price,
                        'trailing_activated': False,
                        'purchase_date': pd.to_datetime(date)  # 보유 기간 계산용
                    }
                    trade_log.append({
                        'date': date,
                        'type': 'BUY',
                        'price': buy_price,
                        'time': time.strftime('%H:%M')
                    })
            
            current_value = cash + (holdings['quantity'] * row['stck_clpr'] if holdings else 0)
            daily_portfolio_value.append(current_value)

        # 통계 계산
        final_value = daily_portfolio_value[-1]
        total_return = (final_value / initial_cash - 1) * 100
        
        roll_max = pd.Series(daily_portfolio_value).cummax()
        daily_drawdown = (pd.Series(daily_portfolio_value) / roll_max) - 1.0
        mdd = daily_drawdown.min() * 100
        
        wins = [t for t in trade_log if t.get('profit', 0) > 0]
        sells = [t for t in trade_log if t['type'] == 'SELL']
        win_rate = (len(wins) / len(sells)) * 100 if sells else 0
        
        # 보유 기간 통계
        holding_periods = [t.get('holding_period', 0) for t in trade_log if t['type'] == 'SELL']
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        max_holding_period = np.max(holding_periods) if holding_periods else 0
        
        # 이익실현 통계
        avg_profit_taking = np.mean(profit_percentages) if profit_percentages else 0
        max_profit_taking = np.max(profit_percentages) if profit_percentages else 0
        min_profit_taking = np.min(profit_percentages) if profit_percentages else 0

        # 각 거래마다 시장 상황과 종목 상황 분석 추가
        for date, row in data.iterrows():
            market_condition = self.market_analyzer.analyze_market_condition(date)
            stock_condition = StockAnalyzer.analyze_stock_condition(data, date)
            
            if holdings and (market_condition is not None) and (stock_condition is not None):
                trade_info.update({
                    'market_condition': market_condition,
                    'stock_condition': stock_condition
                })
            
            # ... rest of the simulation logic ...
        
        # 시장 상황별 성과 분석
        market_performance = {
            '상승장': {'trades': 0, 'wins': 0, 'total_return': 0},
            '하락장': {'trades': 0, 'wins': 0, 'total_return': 0},
            '횡보장': {'trades': 0, 'wins': 0, 'total_return': 0}
        }
        
        for trade in trade_log:
            if 'market_condition' in trade:
                kospi_trend = trade['market_condition']['KOSPI']['추세']
                if trade['type'] == 'SELL':
                    market_performance[kospi_trend]['trades'] += 1
                    if trade.get('profit', 0) > 0:
                        market_performance[kospi_trend]['wins'] += 1
                    market_performance[kospi_trend]['total_return'] += trade.get('profit_percent', 0)
        
        # 결과에 시장 상황별 성과 추가
        for trend, stats in market_performance.items():
            if stats['trades'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
                stats['avg_return'] = stats['total_return'] / stats['trades']
            else:
                stats['win_rate'] = 0
                stats['avg_return'] = 0
        
        results = {
            "params": params,
            "total_return": total_return,
            "mdd": abs(mdd),
            "win_rate": win_rate,
            "trades": len(sells),
            "avg_profit_taking": avg_profit_taking,
            "max_profit_taking": max_profit_taking,
            "min_profit_taking": min_profit_taking,
            "profit_taking_points": len(profit_percentages),
            "avg_holding_period": avg_holding_period,
            "max_holding_period": max_holding_period,
            "market_performance": market_performance,
            "trade_details": [
                {
                    'date': t['date'],
                    'type': t['type'],
                    'price': t['price'],
                    'profit_percent': t.get('profit_percent', 0),
                    'market_trend': t.get('market_condition', {}).get('KOSPI', {}).get('추세', 'N/A'),
                    'stock_trend': t.get('stock_condition', {}).get('추세', 'N/A'),
                    'volume_trend': t.get('stock_condition', {}).get('거래량', {}).get('추세', 'N/A')
                }
                for t in trade_log if t['type'] == 'SELL'
            ]
        }
        
        return results

class MarketAnalyzer:
    """시장 상황 분석기"""
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.kospi_data = None
        self.kosdaq_data = None
        self.market_trend = {}
        self._load_market_data()
    
    def _load_market_data(self):
        """코스피, 코스닥 지수 데이터 로드"""
        try:
            # 코스피 지수 데이터 다운로드
            self.kospi_data = yf.download('^KS11', self.start_date, self.end_date)
            # 코스닥 지수 데이터 다운로드
            self.kosdaq_data = yf.download('^KQ11', self.start_date, self.end_date)
            
            # 이동평균선 계산
            for df in [self.kospi_data, self.kosdaq_data]:
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA60'] = df['Close'].rolling(window=60).mean()
                
                # 변동성 계산
                df['Daily_Return'] = df['Close'].pct_change()
                df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
            
            logger.info("시장 데이터 로드 완료")
        except Exception as e:
            logger.error(f"시장 데이터 로드 중 오류 발생: {e}")
    
    def analyze_market_condition(self, date):
        """특정 날짜의 시장 상황 분석"""
        if self.kospi_data is None or self.kosdaq_data is None:
            return None
        
        date = pd.to_datetime(date)
        market_condition = {}
        
        for market_name, df in [("KOSPI", self.kospi_data), ("KOSDAQ", self.kosdaq_data)]:
            if date not in df.index:
                continue
                
            # 20일 데이터 슬라이스
            hist_data = df[:date].tail(20)
            
            if len(hist_data) < 20:
                continue
            
            current_price = hist_data['Close'].iloc[-1]
            ma5 = hist_data['MA5'].iloc[-1]
            ma20 = hist_data['MA20'].iloc[-1]
            ma60 = hist_data['MA60'].iloc[-1]
            
            # 추세 판단
            trend = "상승" if ma5 > ma20 > ma60 else \
                   "하락" if ma5 < ma20 < ma60 else "횡보"
            
            # 변동성
            volatility = hist_data['Volatility'].iloc[-1]
            
            # 모멘텀 (20일 수익률)
            momentum = ((current_price / hist_data['Close'].iloc[0]) - 1) * 100
            
            market_condition[market_name] = {
                "추세": trend,
                "변동성": volatility,
                "모멘텀": momentum,
                "이격도": {
                    "MA5": (current_price / ma5 - 1) * 100,
                    "MA20": (current_price / ma20 - 1) * 100,
                    "MA60": (current_price / ma60 - 1) * 100
                }
            }
        
        return market_condition

class StockAnalyzer:
    """개별 종목 분석기"""
    @staticmethod
    def analyze_stock_condition(data: pd.DataFrame, current_date: pd.Timestamp) -> Dict:
        """특정 시점의 종목 상태 분석"""
        # 분석 구간 설정 (20일)
        hist_data = data[:current_date].tail(20)
        if len(hist_data) < 20:
            return None
            
        current_price = hist_data['stck_clpr'].iloc[-1]
        
        # 1. 추세 분석
        ma5 = hist_data['stck_clpr'].rolling(5).mean().iloc[-1]
        ma20 = hist_data['stck_clpr'].rolling(20).mean().iloc[-1]
        ma60 = hist_data['stck_clpr'].rolling(60).mean().iloc[-1] if len(data[:current_date]) >= 60 else None
        
        # 추세 판단
        if ma60 is not None:
            trend = "상승" if ma5 > ma20 > ma60 else \
                   "하락" if ma5 < ma20 < ma60 else "횡보"
        else:
            trend = "상승" if ma5 > ma20 else \
                   "하락" if ma5 < ma20 else "횡보"
        
        # 2. 거래량 분석
        volume_ma5 = hist_data['acml_vol'].rolling(5).mean().iloc[-1]
        volume_ma20 = hist_data['acml_vol'].rolling(20).mean().iloc[-1]
        current_volume = hist_data['acml_vol'].iloc[-1]
        
        volume_trend = "증가" if current_volume > volume_ma5 > volume_ma20 else \
                      "감소" if current_volume < volume_ma5 < volume_ma20 else "평균"
        
        # 3. 변동성 분석
        returns = hist_data['stck_clpr'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # 연간화된 변동성
        
        # 4. 모멘텀 지표
        momentum_1d = returns.iloc[-1] * 100  # 1일 수익률
        momentum_5d = (current_price / hist_data['stck_clpr'].iloc[-5] - 1) * 100  # 5일 수익률
        momentum_20d = (current_price / hist_data['stck_clpr'].iloc[0] - 1) * 100  # 20일 수익률
        
        # 5. 거래량 프로파일
        volume_profile = {
            "평균대비": current_volume / volume_ma20,
            "5일평균대비": current_volume / volume_ma5
        }
        
        return {
            "추세": trend,
            "이격도": {
                "MA5": (current_price / ma5 - 1) * 100,
                "MA20": (current_price / ma20 - 1) * 100,
                "MA60": (current_price / ma60 - 1) * 100 if ma60 else None
            },
            "거래량": {
                "추세": volume_trend,
                "프로파일": volume_profile
            },
            "변동성": volatility,
            "모멘텀": {
                "1일": momentum_1d,
                "5일": momentum_5d,
                "20일": momentum_20d
            }
        }

if __name__ == "__main__":
    # 테스트할 종목 리스트와 기간
    test_symbols = ["005930", "035720", "000660"] # 삼성전자, 카카오, SK하이닉스
    start_date = "2020-01-01"
    end_date = "2024-05-31"

    optimizer = ParameterOptimizer(symbols=test_symbols, start_date=start_date, end_date=end_date)
    optimizer.run_optimization() 