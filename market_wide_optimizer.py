"""
시장 전체 최적화 백테스터 (v1.0)
- 여러 종목에 걸쳐 범용적으로 가장 우수한 '황금 비율'을 찾아냅니다.
- 실시간 상승률 상위 종목들을 표본으로 삼아, 가장 안정적인 파라미터를 도출합니다.
"""
import logging
import pandas as pd
from itertools import product
from datetime import datetime, timedelta
from core_trader import CoreTrader
from optimization_tester import OptimizationBacktester # 기존 옵티마이저를 재사용

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_market_wide_optimization(param_grid, num_top_stocks=10):
    """
    시장 전체를 대상으로 최적의 파라미터를 찾습니다.
    """
    trader = CoreTrader()
    
    # 1. 테스트 대상 선정
    logger.info(f"실시간 상승률 상위 {num_top_stocks}개 종목을 테스트 대상으로 선정합니다...")
    top_stocks = trader.get_top_ranking_stocks()
    if not top_stocks:
        logger.error("테스트 대상을 선정하지 못했습니다. API 연결을 확인하세요.")
        return
        
    target_stocks = [stock['symbol'] for stock in top_stocks[:num_top_stocks]]
    logger.info(f"✅ 테스트 대상: {target_stocks}")

    # 2. 모든 파라미터 조합의 누적 성과 기록용 딕셔너리
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    # 각 파라미터를 문자열 키로 사용하여 수익률과 테스트 횟수를 저장
    performance_aggregator = {str(params): {'total_return': 0, 'count': 0} for params in param_combinations}

    # 3. 교차 검증 실행
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)
    TEST_START_DATE = start_date.strftime('%Y-%m-%d')
    TEST_END_DATE = end_date.strftime('%Y-%m-%d')
    
    total_tests = len(target_stocks)
    for i, stock_code in enumerate(target_stocks):
        logger.info("\n" + "="*50)
        logger.info(f"({i+1}/{total_tests}) '{stock_code}' 종목 최적화 시작...")
        
        # 각 종목에 대해 최적화 백테스터 실행
        optimizer = OptimizationBacktester(
            stock_code=stock_code,
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE
        )
        
        if optimizer.historical_data is None:
            logger.warning(f"'{stock_code}' 데이터가 없어 건너뜁니다.")
            continue

        # 모든 파라미터 조합으로 시뮬레이션
        for params in param_combinations:
            result = optimizer._run_simulation(params)
            if result:
                # 결과 집계
                key = str(params)
                performance_aggregator[key]['total_return'] += result['total_return']
                performance_aggregator[key]['count'] += 1
    
    # 4. 최종 결과 분석 및 출력
    final_results = []
    for params_str, data in performance_aggregator.items():
        if data['count'] > 0:
            # 평균 수익률 계산
            avg_return = data['total_return'] / data['count']
            final_results.append({'params': params_str, 'avg_return': avg_return, 'tested_stocks': data['count']})

    if not final_results:
        logger.error("유효한 분석 결과가 없습니다.")
        return

    results_df = pd.DataFrame(final_results)
    results_df = results_df.sort_values(by="avg_return", ascending=False).reset_index(drop=True)
    
    print("\n\n" + "="*80)
    print(f"🏆 시장 전체 최적화 결과 (Top 5) - {len(target_stocks)}개 종목 평균")
    print(results_df.head(5).to_string())
    print("="*80)
    
    # 문자열로 변환된 파라미터를 다시 dict로 복원
    best_params_str = results_df.iloc[0]['params']
    best_params = eval(best_params_str)
    best_return = results_df.iloc[0]['avg_return']

    print(f"\n✅ 시장 전체에 가장 유효한 '범용' 파라미터 조합: {best_params}")
    print(f"✅ 예상 평균 수익률: {best_return:.2f}%")


if __name__ == "__main__":
    # 테스트할 파라미터 범위 정의
    PARAM_GRID = {
        'initial_stop_loss': [2, 3, 4],
        'trailing_activation': [4, 5, 6],
        'trailing_stop': [2, 3]
    }
    
    run_market_wide_optimization(param_grid=PARAM_GRID, num_top_stocks=10) 