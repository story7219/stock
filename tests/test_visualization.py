"""
📊 시각화 기능 테스트 파일
선 그래프 중심의 시각화가 제대로 작동하는지 테스트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import seaborn as sns
from datetime import datetime
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def create_sample_data():
    """📊 테스트용 샘플 데이터 생성"""
    
    # 한국 주식 샘플
    kr_stocks = {
        'Ticker': ['005930', '000660', '035420', '051910', '068270'],
        'Name': ['삼성전자', 'SK하이닉스', 'NAVER', 'LG화학', '셀트리온'],
        'Market': ['KR'] * 5,
        'PER': [15.2, 8.5, 22.1, 12.3, 18.7],
        'ROE': [12.5, 18.3, 15.2, 9.8, 22.1],
        '6개월수익률': [8.5, -2.3, 12.7, 5.2, -1.8],
        '변동성': [25.3, 32.1, 28.9, 22.5, 35.2]
    }
    
    # 미국 주식 샘플
    us_stocks = {
        'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'Name': ['Apple', 'Microsoft', 'Alphabet', 'Amazon', 'Tesla'],
        'Market': ['US'] * 5,
        'PER': [28.5, 32.1, 25.8, 45.2, 52.3],
        'ROE': [28.2, 35.1, 18.9, 12.5, 15.8],
        '6개월수익률': [15.2, 22.1, 8.9, -5.2, 35.8],
        '변동성': [28.5, 25.1, 30.2, 35.8, 55.2]
    }
    
    # 데이터프레임 생성
    kr_df = pd.DataFrame(kr_stocks)
    us_df = pd.DataFrame(us_stocks)
    
    # 합치기
    df = pd.concat([kr_df, us_df], ignore_index=True)
    
    return df

def test_line_charts(data):
    """📈 선 그래프 테스트"""
    
    print("📊 선 그래프 시각화 테스트 시작...")
    
    # 결과 디렉토리 생성
    test_dir = "./test_charts"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. 기본 선 그래프 테스트
    plt.figure(figsize=(15, 10))
    
    # 서브플롯 1: 시장별 PER 비교
    plt.subplot(2, 2, 1)
    kr_data = data[data['Market'] == 'KR']
    us_data = data[data['Market'] == 'US']
    
    plt.plot(range(len(kr_data)), kr_data['PER'], 'o-', 
            label='🇰🇷 한국 PER', linewidth=3, markersize=10, color='red')
    plt.plot(range(len(us_data)), us_data['PER'], 's-', 
            label='🇺🇸 미국 PER', linewidth=3, markersize=10, color='blue')
    
    plt.title('💰 시장별 PER 비교', fontsize=14, fontweight='bold')
    plt.xlabel('종목 순서')
    plt.ylabel('PER (배)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: ROE 추세
    plt.subplot(2, 2, 2)
    plt.plot(range(len(kr_data)), kr_data['ROE'], 'o-', 
            label='🇰🇷 한국 ROE', linewidth=3, markersize=10, color='green')
    plt.plot(range(len(us_data)), us_data['ROE'], 's-', 
            label='🇺🇸 미국 ROE', linewidth=3, markersize=10, color='orange')
    
    plt.title('🏆 시장별 ROE 비교', fontsize=14, fontweight='bold')
    plt.xlabel('종목 순서')
    plt.ylabel('ROE (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 3: 6개월 수익률 추세
    plt.subplot(2, 2, 3)
    plt.plot(range(len(kr_data)), kr_data['6개월수익률'], 'o-', 
            label='🇰🇷 한국 수익률', linewidth=3, markersize=10, color='navy')
    plt.plot(range(len(us_data)), us_data['6개월수익률'], 's-', 
            label='🇺🇸 미국 수익률', linewidth=3, markersize=10, color='darkred')
    
    plt.title('📈 6개월 수익률 비교', fontsize=14, fontweight='bold')
    plt.xlabel('종목 순서')
    plt.ylabel('수익률 (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 서브플롯 4: 변동성 비교
    plt.subplot(2, 2, 4)
    plt.plot(range(len(kr_data)), kr_data['변동성'], 'o-', 
            label='🇰🇷 한국 변동성', linewidth=3, markersize=10, color='purple')
    plt.plot(range(len(us_data)), us_data['변동성'], 's-', 
            label='🇺🇸 미국 변동성', linewidth=3, markersize=10, color='brown')
    
    plt.title('📊 변동성 비교', fontsize=14, fontweight='bold')
    plt.xlabel('종목 순서')
    plt.ylabel('변동성 (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{test_dir}/basic_line_charts_{timestamp}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 기본 선 그래프 테스트 완료: {test_dir}/basic_line_charts_{timestamp}.png")

def test_advanced_line_charts(data):
    """📊 고급 선 그래프 테스트"""
    
    print("📈 고급 선 그래프 시각화 테스트 시작...")
    
    test_dir = "./test_charts"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. 시장별 평균 지표 비교 (정규화 선 그래프)
    plt.figure(figsize=(14, 8))
    
    indicators = ['PER', 'ROE', '6개월수익률', '변동성']
    kr_data = data[data['Market'] == 'KR']
    us_data = data[data['Market'] == 'US']
    
    kr_means = [kr_data[ind].mean() for ind in indicators]
    us_means = [us_data[ind].mean() for ind in indicators]
    
    # 정규화 (0-100 스케일)
    kr_normalized = []
    us_normalized = []
    
    for i, (kr_val, us_val) in enumerate(zip(kr_means, us_means)):
        if indicators[i] in ['PER', '변동성']:  # 낮을수록 좋음
            max_val = max(kr_val, us_val)
            kr_normalized.append((max_val - kr_val) / max_val * 100)
            us_normalized.append((max_val - us_val) / max_val * 100)
        else:  # 높을수록 좋음
            max_val = max(kr_val, us_val)
            kr_normalized.append(kr_val / max_val * 100 if max_val > 0 else 0)
            us_normalized.append(us_val / max_val * 100 if max_val > 0 else 0)
    
    # 선 그래프 생성
    x_pos = range(len(indicators))
    plt.plot(x_pos, kr_normalized, 'o-', label='🇰🇷 한국 시장', 
            linewidth=4, markersize=12, color='red')
    plt.plot(x_pos, us_normalized, 's-', label='🇺🇸 미국 시장', 
            linewidth=4, markersize=12, color='blue')
    
    # 수치 표시
    for i, (kr_val, us_val) in enumerate(zip(kr_normalized, us_normalized)):
        plt.annotate(f'{kr_val:.1f}', (i, kr_val), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontweight='bold', color='red', fontsize=11)
        plt.annotate(f'{us_val:.1f}', (i, us_val), textcoords="offset points", 
                    xytext=(0,-20), ha='center', fontweight='bold', color='blue', fontsize=11)
    
    plt.title('🌍 시장별 투자 지표 비교 (정규화 점수)', fontsize=16, fontweight='bold')
    plt.xlabel('투자 지표', fontsize=12)
    plt.ylabel('정규화 점수 (0-100)', fontsize=12)
    plt.xticks(x_pos, indicators)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(f'{test_dir}/advanced_line_chart_{timestamp}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 고급 선 그래프 테스트 완료: {test_dir}/advanced_line_chart_{timestamp}.png")

def test_individual_stock_analysis(data):
    """📊 개별 종목 분석 선 그래프"""
    
    print("🏆 개별 종목 분석 시각화 테스트 시작...")
    
    test_dir = "./test_charts"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 상위 5개 종목 (간단한 점수 계산)
    data_copy = data.copy()
    data_copy['종합점수'] = (
        (100 - data_copy['PER']) * 0.3 +  # PER 역순
        data_copy['ROE'] * 0.4 +
        (data_copy['6개월수익률'] + 50) * 0.3  # 음수 보정
    )
    
    top5 = data_copy.nlargest(5, '종합점수')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🏆 TOP 5 종목 상세 분석', fontsize=16, fontweight='bold')
    
    for i, (idx, stock) in enumerate(top5.iterrows()):
        if i >= 5:
            break
            
        row = i // 3
        col = i % 3
        
        # 각 종목의 지표들
        indicators = ['PER점수', 'ROE', '6M수익률', '변동성점수']
        values = [
            100 - stock['PER'],  # PER 역순
            stock['ROE'],
            stock['6개월수익률'],
            100 - stock['변동성']  # 변동성 역순
        ]
        
        # 선 그래프로 표시
        axes[row, col].plot(indicators, values, 'o-', linewidth=3, markersize=8)
        axes[row, col].fill_between(indicators, values, alpha=0.3)
        
        # 수치 표시
        for j, (ind, val) in enumerate(zip(indicators, values)):
            axes[row, col].annotate(f'{val:.1f}', (j, val), 
                                   textcoords="offset points", xytext=(0,10), 
                                   ha='center', fontweight='bold')
        
        axes[row, col].set_title(f'{stock["Name"]} ({stock["Market"]})', 
                               fontweight='bold')
        axes[row, col].set_ylim(-10, 110)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].tick_params(axis='x', rotation=45)
    
    # 빈 서브플롯 제거
    if len(top5) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(f'{test_dir}/individual_analysis_{timestamp}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 개별 종목 분석 테스트 완료: {test_dir}/individual_analysis_{timestamp}.png")

def main():
    """🚀 시각화 테스트 메인 함수"""
    
    print("="*60)
    print("📊 선 그래프 중심 시각화 기능 테스트")
    print("="*60)
    
    # 샘플 데이터 생성
    print("📊 테스트 데이터 생성 중...")
    data = create_sample_data()
    print(f"✅ {len(data)}개 종목 샘플 데이터 생성 완료")
    
    try:
        # 1. 기본 선 그래프 테스트
        test_line_charts(data)
        
        # 2. 고급 선 그래프 테스트
        test_advanced_line_charts(data)
        
        # 3. 개별 종목 분석 테스트
        test_individual_stock_analysis(data)
        
        print("\n" + "="*60)
        print("🎉 모든 시각화 테스트 완료!")
        print("📁 생성된 차트 파일들을 ./test_charts/ 폴더에서 확인하세요")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 시각화 테스트 오류: {e}")
        print("💡 matplotlib 설치 확인: pip install matplotlib seaborn")

if __name__ == "__main__":
    main() 