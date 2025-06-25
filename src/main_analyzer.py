#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 통합 투자 분석 시스템 - 메인 실행기
======================================
코스피200·나스닥100·S&P500 전체 데이터 수집 + Gemini AI 분석
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# 로컬 모듈 import
try:
    from kis_stock_analyzer import KISStockAnalyzer
except ImportError:
    try:
        from src.kis_stock_analyzer import KISStockAnalyzer
    except ImportError:
        print("❌ KIS 분석기를 찾을 수 없습니다")
        KISStockAnalyzer = None

try:
    from gemini_ai_analyzer import GeminiAIAnalyzer
except ImportError:
    try:
        from src.gemini_ai_analyzer import GeminiAIAnalyzer
    except ImportError:
        print("❌ Gemini AI 분석기를 찾을 수 없습니다")
        GeminiAIAnalyzer = None

try:
    from modules.us_realtime_derivatives import USRealtimeDerivatives
except ImportError:
    try:
        from src.modules.us_realtime_derivatives import USRealtimeDerivatives
    except ImportError:
        print("⚠️ 미국 실시간 파생상품 모듈을 찾을 수 없습니다 (선택 기능)")
        USRealtimeDerivatives = None

# 환경 변수 로드
load_dotenv()

class MainAnalyzer:
    """메인 통합 분석 시스템"""
    
    def __init__(self):
        print("🚀 통합 투자 분석 시스템 초기화")
        
        # 각 분석기 초기화
        if KISStockAnalyzer:
            self.kis_analyzer = KISStockAnalyzer()
        else:
            print("❌ KIS 분석기 초기화 실패")
            self.kis_analyzer = None
            
        if GeminiAIAnalyzer:
            self.gemini_analyzer = GeminiAIAnalyzer()
        else:
            print("❌ Gemini AI 분석기 초기화 실패")
            self.gemini_analyzer = None
            
        if USRealtimeDerivatives:
            self.us_derivatives = USRealtimeDerivatives()
        else:
            print("⚠️ 미국 실시간 파생상품 분석기 없음 (선택 기능)")
            self.us_derivatives = None
        
        print("✅ 분석기 초기화 완료")
    
    def collect_all_market_data(self) -> Dict:
        """전체 시장 데이터 수집"""
        print("\n🌍 전체 시장 데이터 수집 시작")
        
        if not self.kis_analyzer:
            print("❌ KIS 분석기가 없어서 데이터 수집 불가")
            return {}
        
        all_data = {}
        
        try:
            # 1. S&P500 선물 데이터 수집
            print("\n📈 S&P500 선물 데이터 수집 중...")
            sp500_futures = self.kis_analyzer.collect_sp500_futures_data()
            all_data['sp500_futures'] = sp500_futures
            print(f"✅ S&P500 선물 데이터 {len(sp500_futures)}개 수집 완료")
            
            # 2. 나스닥100 데이터 수집
            print("\n📊 나스닥100 데이터 수집 중...")
            nasdaq_data = self.kis_analyzer.collect_nasdaq100_data()
            all_data['nasdaq100'] = nasdaq_data
            print(f"✅ 나스닥100 데이터 {len(nasdaq_data)}개 수집 완료")
            
            # 3. 코스피200 관련 데이터 수집
            print("\n🇰🇷 코스피200 관련 데이터 수집 중...")
            kospi_data = self.kis_analyzer.collect_kospi200_related_data()
            all_data['kospi200_related'] = kospi_data
            print(f"✅ 코스피200 관련 데이터 {len(kospi_data)}개 수집 완료")
            
            # 4. 미국 실시간 파생상품 데이터 (보조)
            if self.us_derivatives:
                print("\n🇺🇸 미국 실시간 파생상품 데이터 수집 중...")
                try:
                    us_data = asyncio.run(self.us_derivatives.get_market_overview())
                    if us_data:
                        all_data['us_derivatives'] = us_data
                        print("✅ 미국 실시간 파생상품 데이터 수집 완료")
                    else:
                        print("⚠️ 미국 실시간 파생상품 데이터 수집 실패")
                except Exception as e:
                    print(f"⚠️ 미국 실시간 파생상품 데이터 수집 오류: {e}")
            
            # 데이터 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'market_data_{timestamp}.json'
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n💾 전체 데이터 저장 완료: {filename}")
            return all_data
            
        except Exception as e:
            print(f"❌ 데이터 수집 중 오류 발생: {e}")
            return {}
    
    def analyze_with_investment_strategies(self, market_data: Dict) -> Dict:
        """투자 대가 전략으로 분석"""
        print("\n🧠 투자 대가 전략 분석 시작")
        
        if not self.gemini_analyzer:
            print("❌ Gemini AI 분석기가 없어서 분석 불가")
            return {}
        
        # 유효한 데이터만 필터링
        valid_data = {}
        for category, data_list in market_data.items():
            if isinstance(data_list, list) and data_list:
                # 성공적으로 수집된 데이터만 포함
                valid_stocks = []
                for stock in data_list:
                    # 데이터가 있고 오류가 없는 경우만 포함
                    if (stock.get('data') and 
                        not isinstance(stock.get('data'), dict) or 
                        not stock.get('data', {}).get('rt_cd') == '2'):
                        valid_stocks.append(stock)
                
                if valid_stocks:
                    valid_data[category] = valid_stocks
                    print(f"📊 {category}: {len(valid_stocks)}개 유효 종목")
        
        if not valid_data:
            print("❌ 분석할 유효한 데이터가 없습니다")
            return {}
        
        # Gemini AI 종합 분석
        analysis_results = self.gemini_analyzer.comprehensive_analysis(valid_data)
        
        return analysis_results
    
    def select_top5_and_generate_report(self, analysis_results: Dict) -> Dict:
        """Top5 종목 선정 및 리포트 생성"""
        print("\n🏆 Top5 종목 선정 및 리포트 생성")
        
        # Top5 선정
        top5_result = self.gemini_analyzer.select_top5_stocks(analysis_results)
        
        # 투자 리포트 생성
        report = self.gemini_analyzer.generate_investment_report(top5_result)
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        json_filename = f'final_analysis_{timestamp}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_results': analysis_results,
                'top5_result': top5_result
            }, f, ensure_ascii=False, indent=2, default=str)
        
        # 리포트 저장
        report_filename = f'investment_report_{timestamp}.md'
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # HTML 리포트도 생성
        html_report = self.generate_html_report(top5_result, report)
        html_filename = f'investment_report_{timestamp}.html'
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"\n💾 최종 결과 저장 완료:")
        print(f"  📊 JSON: {json_filename}")
        print(f"  📄 마크다운: {report_filename}")
        print(f"  🌐 HTML: {html_filename}")
        
        return top5_result
    
    def generate_html_report(self, top5_result: Dict, markdown_report: str) -> str:
        """HTML 형태의 투자 리포트 생성"""
        
        html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Gemini AI 투자 분석 리포트</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #2980b9;
        }}
        .stock-card {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stock-symbol {{
            font-size: 1.5em;
            font-weight: bold;
            color: #e74c3c;
        }}
        .allocation {{
            background: #27ae60;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            display: inline-block;
            margin: 5px 0;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
        }}
        .warning {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        pre {{
            background: #2c3e50;
            color: white;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Gemini AI 투자 분석 리포트</h1>
        <p class="timestamp">생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="warning">
            ⚠️ <strong>투자 주의사항:</strong> 본 리포트는 AI 기반 분석 결과이며, 실제 투자 결정 시에는 반드시 전문가와 상담하시기 바랍니다.
        </div>
        
        <h2>📊 Top5 선정 종목</h2>
"""
        
        # Top5 종목 카드 생성
        for i, stock in enumerate(top5_result.get('top5_stocks', []), 1):
            symbol = stock.get('symbol', 'N/A')
            category = stock.get('category', 'N/A')
            allocation = stock.get('allocation', 'N/A')
            reasoning = stock.get('reasoning', 'N/A')
            
            html_template += f"""
        <div class="stock-card">
            <div class="stock-symbol">{i}. {symbol}</div>
            <p><strong>카테고리:</strong> {category}</p>
            <div class="allocation">투자 비율: {allocation}</div>
            <p><strong>선정 이유:</strong> {reasoning}</p>
        </div>
"""
        
        # 추가 분석 정보
        selection_reasoning = top5_result.get('selection_reasoning', {})
        if isinstance(selection_reasoning, dict):
            html_template += f"""
        <h2>🎯 포트폴리오 전략</h2>
        <p>{selection_reasoning.get('portfolio_strategy', 'N/A')}</p>
        
        <h2>📈 시장 전망</h2>
        <p>{selection_reasoning.get('market_outlook', 'N/A')}</p>
        
        <h2>⚠️ 리스크 관리</h2>
        <p>{selection_reasoning.get('risk_management', 'N/A')}</p>
"""
        
        html_template += """
        <h2>📄 상세 분석 리포트</h2>
        <pre>""" + markdown_report.replace('<', '&lt;').replace('>', '&gt;') + """</pre>
        
        <div class="timestamp">
            <p>🤖 Powered by Gemini AI | 📊 Data from KIS API</p>
        </div>
    </div>
</body>
</html>"""
        
        return html_template
    
    def run_full_analysis(self):
        """전체 분석 프로세스 실행"""
        print("🎯 통합 투자 분석 시스템 시작")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # 1단계: 전체 시장 데이터 수집
            print("\n🔥 1단계: 전체 시장 데이터 수집")
            market_data = self.collect_all_market_data()
            
            if not market_data:
                print("❌ 데이터 수집 실패로 분석을 중단합니다")
                return
            
            # 2단계: 투자 대가 전략 분석
            print("\n🔥 2단계: 투자 대가 전략 분석")
            analysis_results = self.analyze_with_investment_strategies(market_data)
            
            if not analysis_results:
                print("❌ 분석 실패로 Top5 선정을 중단합니다")
                return
            
            # 3단계: Top5 선정 및 리포트 생성
            print("\n🔥 3단계: Top5 선정 및 리포트 생성")
            top5_result = self.select_top5_and_generate_report(analysis_results)
            
            # 완료 시간 계산
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "=" * 60)
            print("🎉 통합 투자 분석 완료!")
            print(f"⏰ 총 소요 시간: {duration}")
            print(f"📊 분석된 종목 수: {sum(len(stocks) for stocks in market_data.values() if isinstance(stocks, list))}")
            print(f"🏆 최종 선정: Top5 종목")
            
            # Top5 종목 요약 출력
            if top5_result.get('top5_stocks'):
                print("\n🏆 선정된 Top5 종목:")
                for i, stock in enumerate(top5_result['top5_stocks'], 1):
                    symbol = stock.get('symbol', 'N/A')
                    allocation = stock.get('allocation', 'N/A')
                    print(f"  {i}. {symbol} ({allocation})")
            
            print("\n💡 생성된 파일들을 확인하여 상세한 분석 결과를 확인하세요!")
            
        except Exception as e:
            print(f"\n❌ 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

def main():
    """메인 실행 함수"""
    analyzer = MainAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main() 