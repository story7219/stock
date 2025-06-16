"""
기관투자자급 트레이딩 시스템
- 최신 퀀트 기법 적용
- 리스크 관리 고도화
- 포트폴리오 최적화
- 실시간 알파 발굴
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# 최신 라이브러리들
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from scipy import optimize
from scipy.stats import norm, jarque_bera
import cvxpy as cp
from arch import arch_model
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InstitutionalConfig:
    """기관투자자급 설정"""
    # 포트폴리오 설정
    max_position_size: float = 0.05  # 개별 종목 최대 5%
    max_sector_exposure: float = 0.20  # 섹터별 최대 20%
    target_volatility: float = 0.15  # 목표 변동성 15%
    
    # 리스크 관리
    var_confidence: float = 0.05  # VaR 95% 신뢰구간
    max_drawdown: float = 0.10  # 최대 낙폭 10%
    correlation_threshold: float = 0.7  # 상관관계 임계값
    
    # 알파 발굴
    min_alpha_threshold: float = 0.02  # 최소 알파 2%
    lookback_period: int = 252  # 1년 룩백
    rebalancing_frequency: str = "weekly"  # 주간 리밸런싱
    
    # 실행 설정
    slippage_bps: float = 5.0  # 슬리피지 5bp
    commission_bps: float = 3.0  # 수수료 3bp
    market_impact_factor: float = 0.1  # 마켓 임팩트

class AdvancedFactorModel:
    """고급 팩터 모델 (Fama-French 5팩터 + 모멘텀 + 품질)"""
    
    def __init__(self):
        self.factors = [
            'market_beta',      # 시장 베타
            'size_factor',      # 규모 팩터 (SMB)
            'value_factor',     # 가치 팩터 (HML)
            'profitability',    # 수익성 팩터 (RMW)
            'investment',       # 투자 팩터 (CMA)
            'momentum',         # 모멘텀 팩터
            'quality',          # 품질 팩터
            'low_volatility',   # 저변동성 팩터
            'dividend_yield'    # 배당수익률 팩터
        ]
        
    def calculate_factors(self, price_data: pd.DataFrame, 
                         fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """팩터 계산"""
        factors_df = pd.DataFrame(index=price_data.index)
        
        # 1. 시장 베타 (60일 롤링)
        market_returns = price_data.pct_change().mean(axis=1)
        for stock in price_data.columns:
            stock_returns = price_data[stock].pct_change()
            rolling_beta = stock_returns.rolling(60).cov(market_returns) / market_returns.rolling(60).var()
            factors_df[f'{stock}_market_beta'] = rolling_beta
        
        # 2. 규모 팩터 (시가총액 기반)
        market_cap = fundamental_data['market_cap']
        size_factor = np.log(market_cap) / np.log(market_cap.median())
        factors_df = factors_df.join(size_factor.rename('size_factor'))
        
        # 3. 가치 팩터 (PBR, PER 기반)
        pbr = fundamental_data['pbr']
        per = fundamental_data['per']
        value_factor = (1/pbr + 1/per) / 2
        factors_df = factors_df.join(value_factor.rename('value_factor'))
        
        # 4. 모멘텀 팩터 (12-1개월)
        momentum_12m = price_data.pct_change(252).shift(21)  # 12개월 전 대비, 1개월 지연
        factors_df = factors_df.join(momentum_12m.add_suffix('_momentum'))
        
        # 5. 품질 팩터 (ROE, 부채비율 기반)
        roe = fundamental_data['roe']
        debt_ratio = fundamental_data['debt_ratio']
        quality_factor = roe / (1 + debt_ratio)
        factors_df = factors_df.join(quality_factor.rename('quality_factor'))
        
        # 6. 저변동성 팩터
        volatility_60d = price_data.pct_change().rolling(60).std()
        low_vol_factor = 1 / volatility_60d
        factors_df = factors_df.join(low_vol_factor.add_suffix('_low_vol'))
        
        return factors_df

class RiskModel:
    """고급 리스크 모델"""
    
    def __init__(self, config: InstitutionalConfig):
        self.config = config
        
    def calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Value at Risk 계산 (Historical Simulation + Monte Carlo)"""
        # Historical VaR
        hist_var = np.percentile(returns.dropna(), confidence * 100)
        
        # Parametric VaR (정규분포 가정)
        mean_return = returns.mean()
        std_return = returns.std()
        param_var = norm.ppf(confidence, mean_return, std_return)
        
        # Monte Carlo VaR
        np.random.seed(42)
        simulated_returns = np.random.normal(mean_return, std_return, 10000)
        mc_var = np.percentile(simulated_returns, confidence * 100)
        
        # 가중평균 (Historical 50%, Parametric 30%, MC 20%)
        combined_var = 0.5 * hist_var + 0.3 * param_var + 0.2 * mc_var
        
        return combined_var
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Expected Shortfall (CVaR) 계산"""
        var = self.calculate_var(returns, confidence)
        es = returns[returns <= var].mean()
        return es
    
    def calculate_maximum_drawdown(self, prices: pd.Series) -> float:
        """최대 낙폭 계산"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def estimate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """GARCH 모델을 이용한 변동성 예측"""
        try:
            # GARCH(1,1) 모델
            model = arch_model(returns.dropna() * 100, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # 조건부 변동성 예측
            forecast = fitted_model.forecast(horizon=1)
            conditional_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
            
            return conditional_vol
            
        except Exception as e:
            logger.warning(f"GARCH 모델 실패, 단순 변동성 사용: {e}")
            return returns.rolling(30).std().iloc[-1]

class PortfolioOptimizer:
    """포트폴리오 최적화 (Black-Litterman + Risk Parity)"""
    
    def __init__(self, config: InstitutionalConfig):
        self.config = config
        
    def black_litterman_optimization(self, 
                                   expected_returns: pd.Series,
                                   cov_matrix: pd.DataFrame,
                                   market_caps: pd.Series,
                                   views: Dict[str, float] = None,
                                   view_confidence: float = 0.5) -> pd.Series:
        """Black-Litterman 모델 최적화"""
        
        # 1. 시장 균형 수익률 계산
        risk_aversion = 3.0  # 위험회피계수
        market_weights = market_caps / market_caps.sum()
        equilibrium_returns = risk_aversion * cov_matrix.dot(market_weights)
        
        # 2. 투자자 견해 반영
        if views:
            # 견해 행렬 P와 견해 벡터 Q 구성
            P = np.zeros((len(views), len(expected_returns)))
            Q = np.zeros(len(views))
            
            for i, (asset, view_return) in enumerate(views.items()):
                if asset in expected_returns.index:
                    asset_idx = expected_returns.index.get_loc(asset)
                    P[i, asset_idx] = 1
                    Q[i] = view_return
            
            # 견해의 불확실성 행렬 Omega
            omega = np.diag(np.diag(P @ cov_matrix @ P.T)) / view_confidence
            
            # Black-Litterman 공식
            tau = 0.05  # 불확실성 매개변수
            M1 = np.linalg.inv(tau * cov_matrix)
            M2 = P.T @ np.linalg.inv(omega) @ P
            M3 = np.linalg.inv(tau * cov_matrix) @ equilibrium_returns
            M4 = P.T @ np.linalg.inv(omega) @ Q
            
            bl_returns = np.linalg.inv(M1 + M2) @ (M3 + M4)
            bl_returns = pd.Series(bl_returns, index=expected_returns.index)
        else:
            bl_returns = equilibrium_returns
        
        # 3. 최적 포트폴리오 계산
        weights = self.mean_variance_optimization(bl_returns, cov_matrix)
        
        return weights
    
    def risk_parity_optimization(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """리스크 패리티 최적화"""
        n_assets = len(cov_matrix)
        
        def risk_budget_objective(weights, cov_matrix):
            """리스크 기여도 균등화 목적함수"""
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            contrib = weights * marginal_contrib
            return np.sum((contrib - contrib.mean()) ** 2)
        
        # 제약조건: 가중치 합 = 1, 모든 가중치 >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        ]
        bounds = [(0.01, self.config.max_position_size) for _ in range(n_assets)]
        
        # 초기값: 동일가중
        x0 = np.array([1/n_assets] * n_assets)
        
        # 최적화 실행
        result = optimize.minimize(
            risk_budget_objective,
            x0,
            args=(cov_matrix.values,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = pd.Series(result.x, index=cov_matrix.index)
            return weights / weights.sum()  # 정규화
        else:
            logger.warning("리스크 패리티 최적화 실패, 동일가중 사용")
            return pd.Series([1/n_assets] * n_assets, index=cov_matrix.index)
    
    def mean_variance_optimization(self, 
                                 expected_returns: pd.Series,
                                 cov_matrix: pd.DataFrame,
                                 target_return: float = None) -> pd.Series:
        """평균-분산 최적화 (CVXPY 사용)"""
        
        n_assets = len(expected_returns)
        weights = cp.Variable(n_assets)
        
        # 목적함수: 포트폴리오 분산 최소화
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        objective = cp.Minimize(portfolio_variance)
        
        # 제약조건
        constraints = [
            cp.sum(weights) == 1,  # 가중치 합 = 1
            weights >= 0.01,       # 최소 가중치
            weights <= self.config.max_position_size  # 최대 가중치
        ]
        
        # 목표 수익률 제약 (선택적)
        if target_return:
            portfolio_return = weights.T @ expected_returns.values
            constraints.append(portfolio_return >= target_return)
        
        # 최적화 문제 정의 및 해결
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            optimal_weights = pd.Series(weights.value, index=expected_returns.index)
            return optimal_weights / optimal_weights.sum()
        else:
            logger.warning("평균-분산 최적화 실패, 동일가중 사용")
            return pd.Series([1/n_assets] * n_assets, index=expected_returns.index)

class AlphaModel:
    """알파 발굴 모델 (ML + 전통적 기법 결합)"""
    
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        }
        self.scaler = RobustScaler()
        
    def create_features(self, price_data: pd.DataFrame, 
                       volume_data: pd.DataFrame = None,
                       fundamental_data: pd.DataFrame = None) -> pd.DataFrame:
        """피처 생성 (기술적 + 기본적 + 대안적 지표)"""
        
        features = pd.DataFrame(index=price_data.index)
        
        for stock in price_data.columns:
            stock_price = price_data[stock]
            
            # 기술적 지표
            features[f'{stock}_rsi'] = ta.momentum.RSIIndicator(stock_price).rsi()
            features[f'{stock}_macd'] = ta.trend.MACD(stock_price).macd()
            features[f'{stock}_bb_upper'] = ta.volatility.BollingerBands(stock_price).bollinger_hband()
            features[f'{stock}_bb_lower'] = ta.volatility.BollingerBands(stock_price).bollinger_lband()
            features[f'{stock}_atr'] = ta.volatility.AverageTrueRange(stock_price, stock_price, stock_price).average_true_range()
            
            # 모멘텀 지표
            features[f'{stock}_mom_1m'] = stock_price.pct_change(21)
            features[f'{stock}_mom_3m'] = stock_price.pct_change(63)
            features[f'{stock}_mom_6m'] = stock_price.pct_change(126)
            features[f'{stock}_mom_12m'] = stock_price.pct_change(252)
            
            # 변동성 지표
            features[f'{stock}_vol_5d'] = stock_price.pct_change().rolling(5).std()
            features[f'{stock}_vol_20d'] = stock_price.pct_change().rolling(20).std()
            features[f'{stock}_vol_60d'] = stock_price.pct_change().rolling(60).std()
            
            # 가격 패턴
            features[f'{stock}_price_position'] = (stock_price - stock_price.rolling(252).min()) / (stock_price.rolling(252).max() - stock_price.rolling(252).min())
            features[f'{stock}_ma_ratio'] = stock_price / stock_price.rolling(20).mean()
            
            # 볼륨 지표 (있는 경우)
            if volume_data is not None and stock in volume_data.columns:
                volume = volume_data[stock]
                features[f'{stock}_volume_ratio'] = volume / volume.rolling(20).mean()
                features[f'{stock}_price_volume'] = stock_price.pct_change() * volume
        
        # 시장 전체 지표
        market_return = price_data.pct_change().mean(axis=1)
        features['market_momentum'] = market_return.rolling(20).mean()
        features['market_volatility'] = market_return.rolling(20).std()
        features['market_trend'] = (price_data.mean(axis=1) > price_data.mean(axis=1).rolling(50).mean()).astype(int)
        
        return features.fillna(method='ffill').fillna(0)
    
    def train_ensemble_model(self, features: pd.DataFrame, 
                           returns: pd.DataFrame,
                           lookback_days: int = 252) -> Dict:
        """앙상블 모델 훈련"""
        
        trained_models = {}
        
        for stock in returns.columns:
            logger.info(f"📈 {stock} 알파 모델 훈련 중...")
            
            # 타겟 변수: 미래 수익률 (5일 후)
            target = returns[stock].shift(-5)
            
            # 피처 선택 (해당 종목 + 시장 지표)
            stock_features = features.filter(regex=f'{stock}_|market_')
            
            # 데이터 정리
            valid_idx = ~(stock_features.isna().any(axis=1) | target.isna())
            X = stock_features[valid_idx]
            y = target[valid_idx]
            
            if len(X) < lookback_days:
                logger.warning(f"{stock}: 데이터 부족으로 스킵")
                continue
            
            # 시계열 분할
            tscv = TimeSeriesSplit(n_splits=5)
            
            stock_models = {}
            for model_name, model in self.models.items():
                try:
                    # 피처 스케일링
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # 모델 훈련
                    model.fit(X_scaled, y)
                    
                    # 교차검증 점수
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(X_scaled):
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        model.fit(X_train, y_train)
                        score = model.score(X_val, y_val)
                        cv_scores.append(score)
                    
                    avg_score = np.mean(cv_scores)
                    stock_models[model_name] = {
                        'model': model,
                        'score': avg_score,
                        'scaler': self.scaler
                    }
                    
                    logger.info(f"  {model_name}: R² = {avg_score:.4f}")
                    
                except Exception as e:
                    logger.error(f"  {model_name} 훈련 실패: {e}")
            
            trained_models[stock] = stock_models
        
        return trained_models
    
    def predict_alpha(self, models: Dict, features: pd.DataFrame) -> pd.Series:
        """알파 예측 (앙상블)"""
        
        predictions = {}
        
        for stock, stock_models in models.items():
            if not stock_models:
                continue
                
            # 해당 종목 피처 추출
            stock_features = features.filter(regex=f'{stock}_|market_').iloc[-1:].fillna(0)
            
            # 각 모델의 예측값
            model_predictions = []
            model_weights = []
            
            for model_name, model_info in stock_models.items():
                try:
                    model = model_info['model']
                    scaler = model_info['scaler']
                    score = model_info['score']
                    
                    # 피처 스케일링 및 예측
                    X_scaled = scaler.transform(stock_features)
                    pred = model.predict(X_scaled)[0]
                    
                    model_predictions.append(pred)
                    model_weights.append(max(score, 0))  # 음수 점수는 0으로
                    
                except Exception as e:
                    logger.warning(f"{stock} {model_name} 예측 실패: {e}")
            
            if model_predictions:
                # 가중평균 앙상블
                weights = np.array(model_weights)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    ensemble_pred = np.average(model_predictions, weights=weights)
                    predictions[stock] = ensemble_pred
        
        return pd.Series(predictions)

class ExecutionModel:
    """실행 모델 (TWAP, VWAP, Implementation Shortfall)"""
    
    def __init__(self, config: InstitutionalConfig):
        self.config = config
        
    def calculate_optimal_execution(self, 
                                  target_positions: pd.Series,
                                  current_positions: pd.Series,
                                  market_data: pd.DataFrame,
                                  execution_horizon: int = 5) -> pd.DataFrame:
        """최적 실행 전략 계산"""
        
        # 거래해야 할 수량 계산
        trade_sizes = target_positions - current_positions
        
        execution_schedule = pd.DataFrame(
            index=range(execution_horizon),
            columns=trade_sizes.index
        )
        
        for stock in trade_sizes.index:
            if abs(trade_sizes[stock]) < 0.001:  # 거래량이 너무 작으면 스킵
                execution_schedule[stock] = 0
                continue
            
            # 시장 임팩트 고려한 최적 실행
            total_size = abs(trade_sizes[stock])
            
            # TWAP 기반 (시간 가중 평균가격)
            if total_size < 0.02:  # 작은 거래량
                # 균등 분할
                daily_size = trade_sizes[stock] / execution_horizon
                execution_schedule[stock] = daily_size
            else:
                # 시장 임팩트 최소화 (앞쪽에 더 많이)
                weights = np.array([0.4, 0.25, 0.15, 0.12, 0.08])[:execution_horizon]
                weights = weights / weights.sum()
                
                for i in range(execution_horizon):
                    execution_schedule.loc[i, stock] = trade_sizes[stock] * weights[i]
        
        return execution_schedule.fillna(0)
    
    def estimate_transaction_costs(self, 
                                 execution_schedule: pd.DataFrame,
                                 market_data: pd.DataFrame) -> pd.Series:
        """거래비용 추정"""
        
        total_costs = {}
        
        for stock in execution_schedule.columns:
            daily_trades = execution_schedule[stock]
            
            # 기본 비용
            commission_cost = abs(daily_trades).sum() * self.config.commission_bps / 10000
            
            # 슬리피지 비용
            slippage_cost = abs(daily_trades).sum() * self.config.slippage_bps / 10000
            
            # 마켓 임팩트 (거래량에 비례)
            avg_volume = market_data.get(f'{stock}_volume', pd.Series([1000000])).mean()
            impact_cost = (abs(daily_trades).sum() ** 1.5) * self.config.market_impact_factor / avg_volume
            
            total_costs[stock] = commission_cost + slippage_cost + impact_cost
        
        return pd.Series(total_costs)

class InstitutionalTradingSystem:
    """기관투자자급 통합 트레이딩 시스템"""
    
    def __init__(self, config: InstitutionalConfig = None):
        self.config = config or InstitutionalConfig()
        
        # 서브시스템 초기화
        self.factor_model = AdvancedFactorModel()
        self.risk_model = RiskModel(self.config)
        self.optimizer = PortfolioOptimizer(self.config)
        self.alpha_model = AlphaModel()
        self.execution_model = ExecutionModel(self.config)
        
        # 상태 변수
        self.current_positions = pd.Series(dtype=float)
        self.trained_models = {}
        self.last_rebalance = None
        
        logger.info("🏦 기관투자자급 트레이딩 시스템 초기화 완료")
    
    async def run_daily_process(self, 
                              price_data: pd.DataFrame,
                              volume_data: pd.DataFrame = None,
                              fundamental_data: pd.DataFrame = None) -> Dict:
        """일일 프로세스 실행"""
        
        logger.info("🚀 일일 트레이딩 프로세스 시작")
        
        results = {
            'timestamp': datetime.now(),
            'signals': {},
            'positions': {},
            'risk_metrics': {},
            'execution_plan': {}
        }
        
        try:
            # 1. 팩터 분석
            logger.info("📊 팩터 분석 중...")
            if fundamental_data is not None:
                factors = self.factor_model.calculate_factors(price_data, fundamental_data)
            else:
                factors = pd.DataFrame()  # 기본적 분석 없이 진행
            
            # 2. 알파 신호 생성
            logger.info("🔍 알파 신호 생성 중...")
            features = self.alpha_model.create_features(price_data, volume_data, fundamental_data)
            
            if not self.trained_models:
                logger.info("🤖 ML 모델 훈련 중...")
                returns = price_data.pct_change()
                self.trained_models = self.alpha_model.train_ensemble_model(features, returns)
            
            alpha_signals = self.alpha_model.predict_alpha(self.trained_models, features)
            results['signals'] = alpha_signals.to_dict()
            
            # 3. 리스크 분석
            logger.info("⚠️ 리스크 분석 중...")
            returns = price_data.pct_change().dropna()
            
            risk_metrics = {}
            for stock in returns.columns:
                stock_returns = returns[stock]
                risk_metrics[stock] = {
                    'var_95': self.risk_model.calculate_var(stock_returns, 0.05),
                    'expected_shortfall': self.risk_model.calculate_expected_shortfall(stock_returns, 0.05),
                    'max_drawdown': self.risk_model.calculate_maximum_drawdown(price_data[stock]),
                    'volatility': self.risk_model.estimate_garch_volatility(stock_returns)
                }
            
            results['risk_metrics'] = risk_metrics
            
            # 4. 포트폴리오 최적화
            logger.info("⚖️ 포트폴리오 최적화 중...")
            
            # 예상 수익률 (알파 신호 + 리스크 조정)
            expected_returns = alpha_signals.copy()
            for stock in expected_returns.index:
                # 리스크 조정 수익률
                vol = risk_metrics[stock]['volatility']
                expected_returns[stock] = expected_returns[stock] / (1 + vol * 2)
            
            # 공분산 행렬
            cov_matrix = returns.cov() * 252  # 연율화
            
            # Black-Litterman 최적화
            if fundamental_data is not None:
                market_caps = fundamental_data.get('market_cap', pd.Series([1] * len(expected_returns), index=expected_returns.index))
            else:
                market_caps = pd.Series([1] * len(expected_returns), index=expected_returns.index)
            
            optimal_weights = self.optimizer.black_litterman_optimization(
                expected_returns, cov_matrix, market_caps
            )
            
            # 리스크 패리티와 결합 (50:50)
            risk_parity_weights = self.optimizer.risk_parity_optimization(cov_matrix)
            combined_weights = 0.5 * optimal_weights + 0.5 * risk_parity_weights
            
            results['positions'] = combined_weights.to_dict()
            
            # 5. 실행 계획
            logger.info("📋 실행 계획 수립 중...")
            execution_schedule = self.execution_model.calculate_optimal_execution(
                combined_weights, self.current_positions, price_data
            )
            
            transaction_costs = self.execution_model.estimate_transaction_costs(
                execution_schedule, price_data
            )
            
            results['execution_plan'] = {
                'schedule': execution_schedule.to_dict(),
                'estimated_costs': transaction_costs.to_dict()
            }
            
            # 포지션 업데이트
            self.current_positions = combined_weights
            self.last_rebalance = datetime.now()
            
            logger.info("✅ 일일 프로세스 완료")
            
        except Exception as e:
            logger.error(f"❌ 일일 프로세스 실패: {e}")
            results['error'] = str(e)
        
        return results
    
    def generate_performance_report(self, 
                                  price_data: pd.DataFrame,
                                  benchmark_data: pd.DataFrame = None) -> Dict:
        """성과 리포트 생성"""
        
        if self.current_positions.empty:
            return {"error": "포지션 데이터 없음"}
        
        # 포트폴리오 수익률 계산
        returns = price_data.pct_change().dropna()
        portfolio_returns = (returns * self.current_positions).sum(axis=1)
        
        # 성과 지표 계산
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 최대 낙폭
        cumulative_returns = (1 + portfolio_returns).cumprod()
        max_drawdown = self.risk_model.calculate_maximum_drawdown(cumulative_returns)
        
        # 벤치마크 대비 성과 (있는 경우)
        alpha = 0
        beta = 1
        information_ratio = 0
        
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change().dropna()
            
            # 공통 기간 추출
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            port_ret = portfolio_returns.loc[common_dates]
            bench_ret = benchmark_returns.loc[common_dates]
            
            if len(common_dates) > 30:
                # 베타 계산
                covariance = np.cov(port_ret, bench_ret)[0, 1]
                benchmark_variance = np.var(bench_ret)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
                
                # 알파 계산
                alpha = annualized_return - beta * (bench_ret.mean() * 252)
                
                # 정보비율
                active_returns = port_ret - bench_ret
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        report = {
            'performance_metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio
            },
            'risk_metrics': {
                'var_95': self.risk_model.calculate_var(portfolio_returns, 0.05),
                'expected_shortfall': self.risk_model.calculate_expected_shortfall(portfolio_returns, 0.05),
                'current_volatility': self.risk_model.estimate_garch_volatility(portfolio_returns)
            },
            'portfolio_composition': self.current_positions.to_dict(),
            'last_rebalance': self.last_rebalance
        }
        
        return report

# 사용 예시
async def main():
    """메인 실행 함수"""
    
    # 설정
    config = InstitutionalConfig(
        max_position_size=0.05,
        target_volatility=0.15,
        max_drawdown=0.10
    )
    
    # 시스템 초기화
    trading_system = InstitutionalTradingSystem(config)
    
    # 샘플 데이터 (실제로는 데이터 피드에서 가져옴)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    # 가격 데이터 시뮬레이션
    np.random.seed(42)
    price_data = pd.DataFrame(
        np.random.randn(len(dates), len(stocks)).cumsum() + 100,
        index=dates,
        columns=stocks
    )
    
    # 일일 프로세스 실행
    results = await trading_system.run_daily_process(price_data)
    
    print("🏦 기관투자자급 트레이딩 시스템 결과:")
    print(f"📊 알파 신호: {results['signals']}")
    print(f"⚖️ 최적 포지션: {results['positions']}")
    print(f"📋 실행 계획: {results['execution_plan']['estimated_costs']}")
    
    # 성과 리포트
    performance = trading_system.generate_performance_report(price_data)
    print(f"📈 성과 지표: {performance['performance_metrics']}")

if __name__ == "__main__":
    asyncio.run(main()) 