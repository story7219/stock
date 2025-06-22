"""
중기투자 분석기 (Mid-Term Investment Analyzer)
한국(KOSPI) + 미국(NASDAQ, S&P500) 주식 대상 퀀트 분석
PER↓ + ROE↑ + 모멘텀↑ + 변동성↓ 조건 적용
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings
import os
import yfinance as yf
from matplotlib import rcParams
import seaborn as sns
warnings.filterwarnings('ignore')

class MidTermInvestmentAnalyzer:
    """
    중기투자(3~6개월) 유망종목 분석기
    - Value + Quality + Momentum + Low Volatility 통합 분석
    - 한국/미국 시장별 상위 10개 종목 추출
    - 종합점수 계산 및 CSV 저장 기능
    """
    
    def __init__(self):
        """분석기 초기화"""
        self.name = "중기투자 퀀트 분석기"
        self.version = "1.0"
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 분석 기준 설정
        self.criteria = {
            'min_market_cap': 1e11,  # 최소 시가총액 (1000억)
            'max_per': 50,           # 최대 PER
            'min_roe': 5,            # 최소 ROE
            'min_3m_return': -0.2,   # 최소 3개월 수익률 (-20%)
            'min_6m_return': -0.3,   # 최소 6개월 수익률 (-30%)
            'max_volatility': 0.6,   # 최대 변동성 (60%)
            'top_n_per_market': 10   # 시장별 선택 종목 수
        }
    
    def load_data(self, csv_path='data/stock_data.csv'):
        """
        주식 데이터 로드
        
        Args:
            csv_path: CSV 파일 경로
            
        Returns:
            pandas.DataFrame: 주식 데이터
        """
        try:
            data = pd.read_csv(csv_path)
            print(f"✅ 데이터 로드 완료: {len(data)}개 종목")
            print(f"📊 한국 종목: {len(data[data['Market']=='KR'])}개")
            print(f"🇺🇸 미국 종목: {len(data[data['Market']=='US'])}개")
            return data
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None
    
    def calculate_composite_score(self, data):
        """
        종합점수 계산 (Value + Quality + Momentum + Low Volatility)
        
        Args:
            data: 주식 데이터 DataFrame
            
        Returns:
            pandas.DataFrame: 점수가 추가된 데이터
        """
        df = data.copy()
        scaler = MinMaxScaler()
        
        # 1단계: 기본 필터링
        basic_filter = (
            (df['PER'] > 0) & (df['PER'] <= self.criteria['max_per']) &
            (df['ROE'] >= self.criteria['min_roe']) &
            (df['MarketCap'] >= self.criteria['min_market_cap']) &
            (df['3M_Return'] >= self.criteria['min_3m_return']) &
            (df['6M_Return'] >= self.criteria['min_6m_return']) &
            (df['Volatility'] <= self.criteria['max_volatility'])
        )
        
        filtered_df = df[basic_filter].copy()
        
        if len(filtered_df) == 0:
            print("⚠️ 필터링 조건을 만족하는 종목이 없습니다.")
            return df
        
        # 2단계: 각 팩터별 점수 계산 (0-1 정규화)
        
        # Value 점수 (PER 낮을수록 좋음)
        per_data = filtered_df['PER'].values.reshape(-1, 1)
        per_normalized = scaler.fit_transform(per_data).flatten()
        filtered_df['value_score'] = 1 - per_normalized  # 역정규화
        
        # Quality 점수 (ROE 높을수록 좋음)
        roe_data = np.clip(filtered_df['ROE'].values, 0, 50).reshape(-1, 1)  # ROE 50% 상한
        roe_normalized = scaler.fit_transform(roe_data).flatten()
        filtered_df['quality_score'] = roe_normalized
        
        # Momentum 점수 (3M, 6M 수익률 높을수록 좋음)
        momentum_3m = scaler.fit_transform(filtered_df[['3M_Return']]).flatten()
        momentum_6m = scaler.fit_transform(filtered_df[['6M_Return']]).flatten()
        filtered_df['momentum_score'] = (momentum_3m + momentum_6m) / 2
        
        # Low Volatility 점수 (변동성 낮을수록 좋음)
        vol_data = filtered_df['Volatility'].values.reshape(-1, 1)
        vol_normalized = scaler.fit_transform(vol_data).flatten()
        filtered_df['stability_score'] = 1 - vol_normalized  # 역정규화
        
        # 3단계: 종합점수 계산 (가중평균)
        weights = {
            'value': 0.25,      # 25% - PER 기반 가치
            'quality': 0.30,    # 30% - ROE 기반 품질
            'momentum': 0.30,   # 30% - 수익률 기반 모멘텀
            'stability': 0.15   # 15% - 변동성 기반 안정성
        }
        
        filtered_df['composite_score'] = (
            filtered_df['value_score'] * weights['value'] +
            filtered_df['quality_score'] * weights['quality'] +
            filtered_df['momentum_score'] * weights['momentum'] +
            filtered_df['stability_score'] * weights['stability']
        )
        
        # 4단계: 100점 만점으로 변환
        filtered_df['final_score'] = filtered_df['composite_score'] * 100
        
        # 원본 데이터에 점수 병합
        score_columns = ['value_score', 'quality_score', 'momentum_score', 
                        'stability_score', 'composite_score', 'final_score']
        
        for col in score_columns:
            df[col] = 0
        
        df.loc[filtered_df.index, score_columns] = filtered_df[score_columns]
        
        return df
    
    def select_top_stocks(self, data):
        """
        시장별 상위 종목 선택
        
        Args:
            data: 점수가 계산된 데이터
            
        Returns:
            dict: 시장별 상위 종목 딕셔너리
        """
        results = {}
        
        # 한국 시장 상위 종목
        kr_stocks = data[data['Market'] == 'KR'].copy()
        kr_filtered = kr_stocks[kr_stocks['final_score'] > 0]
        kr_top = kr_filtered.nlargest(self.criteria['top_n_per_market'], 'final_score')
        results['한국'] = kr_top
        
        # 미국 시장 상위 종목
        us_stocks = data[data['Market'] == 'US'].copy()
        us_filtered = us_stocks[us_stocks['final_score'] > 0]
        us_top = us_filtered.nlargest(self.criteria['top_n_per_market'], 'final_score')
        results['미국'] = us_top
        
        return results
    
    def create_recommendation_report(self, top_stocks):
        """
        추천 보고서 생성
        
        Args:
            top_stocks: 시장별 상위 종목 딕셔너리
            
        Returns:
            pandas.DataFrame: 통합 추천 종목 데이터
        """
        all_recommendations = []
        
        for market, stocks in top_stocks.items():
            if len(stocks) > 0:
                market_data = stocks[['Ticker', 'Market', 'Close', 'PER', 'ROE', 
                                   '3M_Return', '6M_Return', 'Volatility', 'MarketCap',
                                   'value_score', 'quality_score', 'momentum_score', 
                                   'stability_score', 'final_score']].copy()
                
                # 수익률을 퍼센트로 변환
                market_data['3M_Return_pct'] = market_data['3M_Return'] * 100
                market_data['6M_Return_pct'] = market_data['6M_Return'] * 100
                market_data['Volatility_pct'] = market_data['Volatility'] * 100
                
                # 시가총액을 억원/억달러 단위로 변환
                if market == '한국':
                    market_data['MarketCap_display'] = (market_data['MarketCap'] / 1e8).round(0).astype(int)
                    market_data['MarketCap_unit'] = '억원'
                else:
                    market_data['MarketCap_display'] = (market_data['MarketCap'] / 1e8).round(0).astype(int)
                    market_data['MarketCap_unit'] = '억달러'
                
                all_recommendations.append(market_data)
        
        if all_recommendations:
            final_df = pd.concat(all_recommendations, ignore_index=True)
            return final_df.sort_values('final_score', ascending=False)
        else:
            return pd.DataFrame()
    
    def save_to_csv(self, recommendations, filename='mid_term_recommend.csv'):
        """
        추천 종목을 CSV로 저장
        
        Args:
            recommendations: 추천 종목 DataFrame
            filename: 저장할 파일명
        """
        if len(recommendations) == 0:
            print("❌ 저장할 추천 종목이 없습니다.")
            return
        
        # CSV 저장용 컬럼 선택 및 정리
        csv_columns = {
            'Ticker': '종목코드',
            'Market': '시장',
            'Close': '현재가',
            'PER': 'PER',
            'ROE': 'ROE',
            '3M_Return_pct': '3개월수익률(%)',
            '6M_Return_pct': '6개월수익률(%)',
            'Volatility_pct': '변동성(%)',
            'MarketCap_display': '시가총액',
            'MarketCap_unit': '단위',
            'final_score': '종합점수'
        }
        
        save_df = recommendations[list(csv_columns.keys())].copy()
        save_df.columns = list(csv_columns.values())
        
        # 수치 반올림
        save_df['PER'] = save_df['PER'].round(1)
        save_df['ROE'] = save_df['ROE'].round(1)
        save_df['3개월수익률(%)'] = save_df['3개월수익률(%)'].round(1)
        save_df['6개월수익률(%)'] = save_df['6개월수익률(%)'].round(1)
        save_df['변동성(%)'] = save_df['변동성(%)'].round(1)
        save_df['종합점수'] = save_df['종합점수'].round(1)
        
        save_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 저장 완료: {filename}")
        print(f"📊 총 {len(save_df)}개 종목 저장")
    
    def create_visualizations(self, recommendations):
        """📊 HTS 스타일 전문 차트 생성"""
        try:
            # 결과 디렉토리 생성
            viz_dir = "results/visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            
            # 한국/미국 데이터 분리
            kr_data = recommendations[recommendations['Market'] == 'KR'].head(10)
            us_data = recommendations[recommendations['Market'] == 'US'].head(10)
            
            # HTS 스타일 색상 설정
            colors = {
                'bg': '#1e1e1e',           # 어두운 배경
                'grid': '#404040',         # 격자
                'text': '#ffffff',         # 텍스트
                'red': '#ff4444',          # 상승/매수
                'blue': '#4488ff',         # 하락/매도
                'green': '#44ff44',        # 중립
                'yellow': '#ffff44'        # 강조
            }
            
            plt.style.use('dark_background')
            
            # 1. HTS 스타일 종목 현황판
            fig = plt.figure(figsize=(20, 12))
            fig.patch.set_facecolor(colors['bg'])
            
            # 메인 타이틀
            fig.suptitle('📊 중기투자 퀀트 분석 시스템 (HTS Style)', 
                        fontsize=20, fontweight='bold', color=colors['text'], y=0.95)
            
            # 2x2 레이아웃으로 구성
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # 좌상: 종목 순위표 (HTS 호가창 스타일)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_facecolor(colors['bg'])
            
            # 테이블 데이터 준비
            table_data = []
            headers = ['순위', '종목', '시장', '현재가', 'PER', 'ROE', '6M수익률', '점수']
            
            # 한국 데이터 추가
            for i, (_, row) in enumerate(kr_data.iterrows()):
                table_data.append([
                    f'{i+1:2d}',
                    f'{row["Ticker"][:6]}',
                    'KR',
                    f'{row["Close"]:,.0f}',
                    f'{row["PER"]:.1f}',
                    f'{row["ROE"]:.1f}%',
                    f'{row["6M_Return_pct"]:.1f}%',
                    f'{row["final_score"]:.0f}'
                ])
            
            # 미국 데이터 추가
            for i, (_, row) in enumerate(us_data.iterrows()):
                table_data.append([
                    f'{i+11:2d}',
                    f'{row["Ticker"][:6]}',
                    'US',
                    f'${row["Close"]:.2f}',
                    f'{row["PER"]:.1f}',
                    f'{row["ROE"]:.1f}%',
                    f'{row["6M_Return_pct"]:.1f}%',
                    f'{row["final_score"]:.0f}'
                ])
            
            # 테이블 생성
            table = ax1.table(cellText=table_data, colLabels=headers,
                             cellLoc='center', loc='center',
                             colWidths=[0.08, 0.15, 0.08, 0.15, 0.1, 0.1, 0.12, 0.1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # 테이블 스타일링 (HTS 스타일)
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#2d2d2d')
                table[(0, i)].set_text_props(weight='bold', color=colors['yellow'])
            
            for i in range(1, len(table_data) + 1):
                for j in range(len(headers)):
                    if j == 6:  # 수익률 컬럼
                        value = float(table_data[i-1][j].replace('%', ''))
                        if value > 0:
                            table[(i, j)].set_text_props(color=colors['red'])
                        else:
                            table[(i, j)].set_text_props(color=colors['blue'])
                    elif j == 7:  # 점수 컬럼
                        table[(i, j)].set_text_props(color=colors['yellow'], weight='bold')
                    
                    # 행 배경색 교대로 설정
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#2a2a2a')
                    else:
                        table[(i, j)].set_facecolor('#1a1a1a')
            
            ax1.axis('off')
            ax1.set_title('📈 추천 종목 현황판', fontsize=14, fontweight='bold', 
                         color=colors['text'], pad=20)
            
            # 우상: 점수 분포 차트 (HTS 캔들차트 스타일)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.set_facecolor(colors['bg'])
            
            # 점수별 막대 그래프
            all_scores = recommendations['final_score'].head(20)
            bars = ax2.bar(range(len(all_scores)), all_scores, 
                          color=[colors['red'] if score > 70 else colors['blue'] if score > 50 else colors['green'] 
                                for score in all_scores],
                          alpha=0.8, edgecolor='white', linewidth=0.5)
            
            ax2.set_title('🎯 종합점수 분포', fontsize=14, fontweight='bold', color=colors['text'])
            ax2.set_xlabel('종목 순위', color=colors['text'])
            ax2.set_ylabel('투자 점수', color=colors['text'])
            ax2.grid(True, color=colors['grid'], alpha=0.3)
            ax2.tick_params(colors=colors['text'])
            
            # 점수 기준선 표시
            ax2.axhline(y=70, color=colors['red'], linestyle='--', alpha=0.7, label='우수 (70점+)')
            ax2.axhline(y=50, color=colors['yellow'], linestyle='--', alpha=0.7, label='양호 (50점+)')
            ax2.legend(loc='upper right')
            
            # 좌하: PER vs ROE 산점도 (HTS 기술적 분석 스타일)
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.set_facecolor(colors['bg'])
            
            # 한국 종목
            if len(kr_data) > 0:
                kr_scatter = ax3.scatter(kr_data['PER'], kr_data['ROE'], 
                                       s=kr_data['final_score']*3, c=colors['red'], 
                                       alpha=0.7, label='한국', edgecolors='white', marker='o')
            
            # 미국 종목
            if len(us_data) > 0:
                us_scatter = ax3.scatter(us_data['PER'], us_data['ROE'], 
                                       s=us_data['final_score']*3, c=colors['blue'], 
                                       alpha=0.7, label='미국', edgecolors='white', marker='s')
            
            ax3.set_title('💎 밸류에이션 맵 (PER vs ROE)', fontsize=14, fontweight='bold', color=colors['text'])
            ax3.set_xlabel('PER (배)', color=colors['text'])
            ax3.set_ylabel('ROE (%)', color=colors['text'])
            ax3.grid(True, color=colors['grid'], alpha=0.3)
            ax3.tick_params(colors=colors['text'])
            ax3.legend()
            
            # 우하: 수익률 트렌드 (HTS 주가차트 스타일)
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.set_facecolor(colors['bg'])
            
            # 상위 10개 종목의 수익률 추세
            top_10 = recommendations.head(10)
            
            for i, (_, row) in enumerate(top_10.iterrows()):
                color = colors['red'] if row['6M_Return_pct'] > 0 else colors['blue']
                market_marker = 'o' if row['Market'] == 'KR' else 's'
                
                ax4.plot([0, 3, 6], [0, row['3M_Return_pct'], row['6M_Return_pct']], 
                        color=color, linewidth=2, alpha=0.8,
                        marker=market_marker, markersize=6,
                        label=f'{row["Ticker"][:6]} ({row["6M_Return_pct"]:.1f}%)')
            
            ax4.set_title('📊 수익률 추세 분석', fontsize=14, fontweight='bold', color=colors['text'])
            ax4.set_xlabel('기간 (개월)', color=colors['text'])
            ax4.set_ylabel('수익률 (%)', color=colors['text'])
            ax4.set_xticks([0, 3, 6])
            ax4.set_xticklabels(['현재', '3개월', '6개월'])
            ax4.grid(True, color=colors['grid'], alpha=0.3)
            ax4.tick_params(colors=colors['text'])
            ax4.axhline(y=0, color='white', linestyle='-', alpha=0.5)
            
            # 범례를 차트 외부에 배치
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/hts_style_analysis.png", dpi=300, bbox_inches='tight',
                       facecolor=colors['bg'])
            # GUI 환경에서는 plt.show() 제거 (스레드 충돌 방지)
            # plt.show()
            plt.close()  # 메모리 정리
            
            # 2. HTS 스타일 개별 종목 상세 차트
            self.create_individual_stock_charts(recommendations.head(5), viz_dir, colors)
            
            print(f"\n📊 HTS 스타일 차트 저장 완료: {viz_dir}/")
            
        except Exception as e:
            print(f"❌ 시각화 생성 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def create_individual_stock_charts(self, top_stocks, viz_dir, colors):
        """개별 종목 상세 분석 차트 (HTS 종목 상세 스타일)"""
        for idx, (_, stock) in enumerate(top_stocks.iterrows()):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.patch.set_facecolor(colors['bg'])
            
            # 종목 정보 헤더
            fig.suptitle(f'📊 {stock["Ticker"]} 종목 분석 리포트', 
                        fontsize=16, fontweight='bold', color=colors['text'])
            
            # 1. 종목 기본 정보 (HTS 종목 정보창 스타일)
            ax1.set_facecolor(colors['bg'])
            info_text = f"""
종목코드: {stock['Ticker']}
시장: {'한국 (KOSPI/KOSDAQ)' if stock['Market'] == 'KR' else '미국 (NASDAQ/NYSE)'}
현재가: {stock['Close']:,.2f} {'원' if stock['Market'] == 'KR' else '달러'}
시가총액: {stock['MarketCap_display']:,} {stock['MarketCap_unit']}

📊 밸류에이션 지표
PER: {stock['PER']:.1f} 배
ROE: {stock['ROE']:.1f} %

📈 수익률 현황
3개월: {stock['3M_Return_pct']:.1f} %
6개월: {stock['6M_Return_pct']:.1f} %
변동성: {stock['Volatility_pct']:.1f} %

🎯 투자 점수: {stock['final_score']:.1f} / 100
"""
            ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, 
                    fontsize=11, verticalalignment='top', color=colors['text'],
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#2d2d2d', alpha=0.8))
            ax1.axis('off')
            
            # 2. 점수 구성 요소 (레이더 차트)
            ax2.set_facecolor(colors['bg'])
            categories = ['Value\n(PER)', 'Quality\n(ROE)', 'Momentum\n(수익률)', 'Stability\n(안정성)']
            values = [
                stock['value_score'] * 100,
                stock['quality_score'] * 100,
                stock['momentum_score'] * 100,
                stock['stability_score'] * 100
            ]
            
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            ax2 = plt.subplot(2, 2, 2, projection='polar')
            ax2.set_facecolor(colors['bg'])
            ax2.plot(angles, values, 'o-', linewidth=2, color=colors['yellow'])
            ax2.fill(angles, values, alpha=0.25, color=colors['yellow'])
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(categories, color=colors['text'])
            ax2.set_ylim(0, 100)
            ax2.grid(True, color=colors['grid'], alpha=0.3)
            ax2.set_title('점수 구성 요소', fontsize=12, fontweight='bold', 
                         color=colors['text'], pad=20)
            
            # 3. 수익률 추세 (HTS 주가 차트 스타일)
            ax3.set_facecolor(colors['bg'])
            months = ['현재', '1개월', '2개월', '3개월', '4개월', '5개월', '6개월']
            # 가상의 월별 수익률 데이터 (실제로는 API에서 가져와야 함)
            returns = [0, stock['3M_Return_pct']/3, stock['3M_Return_pct']/1.5, 
                      stock['3M_Return_pct'], stock['6M_Return_pct']/1.5, 
                      stock['6M_Return_pct']/1.2, stock['6M_Return_pct']]
            
            colors_line = [colors['red'] if r > 0 else colors['blue'] for r in returns]
            ax3.plot(months, returns, 'o-', linewidth=3, color=colors['yellow'], markersize=8)
            ax3.fill_between(months, returns, alpha=0.3, color=colors['yellow'])
            ax3.set_title('수익률 추세', fontsize=12, fontweight='bold', color=colors['text'])
            ax3.set_ylabel('수익률 (%)', color=colors['text'])
            ax3.grid(True, color=colors['grid'], alpha=0.3)
            ax3.tick_params(colors=colors['text'])
            ax3.axhline(y=0, color='white', linestyle='-', alpha=0.5)
            
            # 4. 리스크 분석 (변동성 vs 수익률)
            ax4.set_facecolor(colors['bg'])
            
            # 현재 종목 위치 표시
            ax4.scatter(stock['Volatility_pct'], stock['6M_Return_pct'], 
                       s=300, c=colors['red'], marker='*', 
                       edgecolors='white', linewidth=2, label='현재 종목')
            
            # 비교군 표시 (같은 시장의 다른 종목들)
            market_stocks = top_stocks[top_stocks['Market'] == stock['Market']]
            ax4.scatter(market_stocks['Volatility_pct'], market_stocks['6M_Return_pct'], 
                       s=100, c=colors['blue'], alpha=0.6, label='동일 시장')
            
            ax4.set_title('리스크-수익률 분석', fontsize=12, fontweight='bold', color=colors['text'])
            ax4.set_xlabel('변동성 (%)', color=colors['text'])
            ax4.set_ylabel('6개월 수익률 (%)', color=colors['text'])
            ax4.grid(True, color=colors['grid'], alpha=0.3)
            ax4.tick_params(colors=colors['text'])
            ax4.legend()
            
            # 효율적 투자 영역 표시
            ax4.axhline(y=0, color='white', linestyle='-', alpha=0.5)
            ax4.axvline(x=30, color=colors['green'], linestyle='--', alpha=0.5, label='적정 변동성')
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/stock_detail_{stock['Ticker']}.png", 
                       dpi=300, bbox_inches='tight', facecolor=colors['bg'])
            plt.close()
    
    def print_summary_report(self, top_stocks, recommendations):
        """
        요약 보고서 출력
        
        Args:
            top_stocks: 시장별 상위 종목
            recommendations: 전체 추천 종목
        """
        print("\n" + "="*80)
        print("📊 중기투자 퀀트 분석 결과 요약")
        print("="*80)
        
        for market, stocks in top_stocks.items():
            if len(stocks) > 0:
                print(f"\n🎯 {market} 시장 상위 {len(stocks)}개 종목:")
                print("-" * 60)
                
                for idx, (_, stock) in enumerate(stocks.head().iterrows(), 1):
                    print(f"{idx:2d}. {stock['Ticker']:>8} | "
                          f"점수: {stock['final_score']:5.1f} | "
                          f"PER: {stock['PER']:5.1f} | "
                          f"ROE: {stock['ROE']:5.1f}% | "
                          f"6M수익률: {stock['6M_Return']*100:6.1f}%")
        
        if len(recommendations) > 0:
            print(f"\n📈 전체 통계:")
            print(f"- 총 추천 종목: {len(recommendations)}개")
            print(f"- 평균 종합점수: {recommendations['final_score'].mean():.1f}점")
            print(f"- 평균 PER: {recommendations['PER'].mean():.1f}")
            print(f"- 평균 ROE: {recommendations['ROE'].mean():.1f}%")
            print(f"- 평균 6개월 수익률: {recommendations['6M_Return_pct'].mean():.1f}%")
            print(f"- 평균 변동성: {recommendations['Volatility_pct'].mean():.1f}%")
        
        print("\n" + "="*80)
    
    def run_analysis(self, csv_path='data/stock_data.csv'):
        """
        전체 분석 실행
        
        Args:
            csv_path: 데이터 파일 경로
            
        Returns:
            pandas.DataFrame: 최종 추천 종목
        """
        print(f"🚀 {self.name} 시작")
        print(f"📅 분석 시점: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1단계: 데이터 로드
        data = self.load_data(csv_path)
        if data is None:
            return None
        
        # 2단계: 종합점수 계산
        print("\n⚙️ 종합점수 계산 중...")
        scored_data = self.calculate_composite_score(data)
        
        # 3단계: 상위 종목 선택
        print("🎯 상위 종목 선택 중...")
        top_stocks = self.select_top_stocks(scored_data)
        
        # 4단계: 추천 보고서 생성
        recommendations = self.create_recommendation_report(top_stocks)
        
        # 5단계: 결과 출력
        self.print_summary_report(top_stocks, recommendations)
        
        # 6단계: CSV 저장
        self.save_to_csv(recommendations)
        
        # 7단계: 시각화 (선 그래프 중심)
        print("\n📊 시각화 차트 생성 중...")
        self.create_visualizations(recommendations)
        
        return recommendations

# 사용 예시 및 실행 코드
if __name__ == "__main__":
    # 분석기 초기화 및 실행
    analyzer = MidTermInvestmentAnalyzer()
    
    # 전체 분석 실행
    results = analyzer.run_analysis('data/stock_data.csv')
    
    if results is not None and len(results) > 0:
        print("\n✅ 분석 완료!")
        print("📁 생성된 파일:")
        print("- mid_term_recommend.csv (추천 종목 목록)")
        print("- mid_term_analysis_chart.png (분석 차트)")
        
        # 상위 5개 종목 간단 출력
        print("\n🏆 종합점수 상위 5개 종목:")
        print("-" * 50)
        for idx, (_, stock) in enumerate(results.head().iterrows(), 1):
            print(f"{idx}. {stock['종목코드']} ({stock['시장']}) - {stock['종합점수']:.1f}점")
    else:
        print("❌ 분석 결과가 없습니다.") 