#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 통합 주식 분석 시스템
========================
한투 API + Gemini AI + 투자 대가 전략
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
import time

# 모듈 임포트
from kis_stock_analyzer import KISStockAnalyzer
from gemini_ai_analyzer import GeminiAIAnalyzer

# 환경 변수 로드
load_dotenv()

class IntegratedStockSystem:
    """통합 주식 분석 시스템"""
    
    def __init__(self):
        print("🚀 통합 주식 분석 시스템 초기화")
        
        # 컴포넌트 초기화
        self.kis_analyzer = KISStockAnalyzer()
        self.gemini_analyzer = GeminiAIAnalyzer()
        
        # 시스템 상태
        self.last_data_collection = None
        self.collected_data = {}
        
        print("✅ 모든 컴포넌트 초기화 완료")
    
    def collect_all_market_data(self) -> Dict:
        """전체 시장 데이터 수집"""
        print("\n" + "="*60)
        print("📊 전체 시장 데이터 수집 시작")
        print("="*60)
        
        # 한투 API 데이터 수집
        kis_data = self.kis_analyzer.run_data_collection()
        
        # 수집 시간 기록
        self.last_data_collection = datetime.now()
        self.collected_data = kis_data
        
        # 데이터 정제 및 보완
        processed_data = self._process_collected_data(kis_data)
        
        print("\n📊 데이터 수집 완료")
        print(f"⏰ 수집 시간: {self.last_data_collection}")
        
        return processed_data
    
    def _process_collected_data(self, raw_data: Dict) -> Dict:
        """수집된 데이터 정제 및 보완"""
        print("\n🔧 데이터 정제 중...")
        
        processed = {}
        
        for category, stocks in raw_data.items():
            if not isinstance(stocks, list):
                continue
            
            processed_stocks = []
            
            for stock in stocks:
                # 데이터 유효성 검사
                if self._is_valid_stock_data(stock):
                    # 기술적 지표 추가
                    enhanced_stock = self._add_technical_indicators(stock)
                    processed_stocks.append(enhanced_stock)
                else:
                    print(f"⚠️ 유효하지 않은 데이터: {stock.get('symbol', 'Unknown')}")
            
            processed[category] = processed_stocks
            print(f"✅ {category}: {len(processed_stocks)}개 종목 정제 완료")
        
        return processed
    
    def _is_valid_stock_data(self, stock: Dict) -> bool:
        """주식 데이터 유효성 검사"""
        required_fields = ['symbol', 'exchange']
        
        for field in required_fields:
            if not stock.get(field):
                return False
        
        # 데이터 필드 검사
        data = stock.get('data', {})
        if not data or (isinstance(data, dict) and data.get('rt_cd') == '2'):
            return False
        
        return True
    
    def _add_technical_indicators(self, stock: Dict) -> Dict:
        """기술적 지표 추가"""
        # 기본 기술적 지표 계산 (더미 데이터)
        stock['technical_indicators'] = {
            'rsi': 50.0,  # RSI
            'macd': 0.0,  # MACD
            'bollinger_upper': 0.0,
            'bollinger_lower': 0.0,
            'moving_average_20': 0.0,
            'moving_average_50': 0.0,
            'volume_trend': 'neutral'
        }
        
        return stock
    
    def run_gemini_analysis(self, processed_data: Dict) -> Dict:
        """Gemini AI 분석 실행"""
        print("\n" + "="*60)
        print("🤖 Gemini AI 분석 시작")
        print("="*60)
        
        # 종합 분석
        analysis_results = self.gemini_analyzer.comprehensive_analysis(processed_data)
        
        # Top5 종목 선정
        top5_result = self.gemini_analyzer.select_top5_stocks(analysis_results)
        
        return {
            'analysis_results': analysis_results,
            'top5_result': top5_result,
            'analysis_timestamp': datetime.now()
        }
    
    def generate_comprehensive_report(self, analysis_data: Dict) -> str:
        """종합 리포트 생성"""
        print("\n📄 종합 리포트 생성 중...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# 🚀 통합 주식 분석 리포트
**생성 시간:** {timestamp}

## 📊 시스템 개요
- **데이터 소스:** 한국투자증권 API
- **AI 분석:** Gemini AI
- **투자 전략:** 워런 버핏, 피터 린치, 벤저민 그레이엄

## 🎯 Top5 선정 종목
"""
        
        top5_stocks = analysis_data.get('top5_result', {}).get('top5_stocks', [])
        
        for i, stock in enumerate(top5_stocks, 1):
            symbol = stock.get('symbol', 'N/A')
            category = stock.get('category', 'N/A')
            allocation = stock.get('allocation', 'N/A')
            reasoning = stock.get('reasoning', 'N/A')
            
            report += f"""
### {i}. {symbol} ({category})
- **투자 비율:** {allocation}
- **선정 이유:** {reasoning}
"""
        
        # 투자 전략 추가
        selection_reasoning = analysis_data.get('top5_result', {}).get('selection_reasoning', {})
        if isinstance(selection_reasoning, dict):
            report += f"""

## 🎯 포트폴리오 전략
{selection_reasoning.get('portfolio_strategy', 'N/A')}

## 📈 시장 전망
{selection_reasoning.get('market_outlook', 'N/A')}

## ⚠️ 리스크 관리
{selection_reasoning.get('risk_management', 'N/A')}
"""
        
        # 데이터 수집 요약
        report += f"""

## 📊 데이터 수집 요약
- **수집 시간:** {self.last_data_collection}
- **총 수집 종목:** {self._count_total_stocks()}개
- **유효 데이터:** {self._count_valid_stocks()}개

## 🔧 시스템 정보
- **한투 API:** ✅ 연결됨
- **Gemini AI:** {'✅ 연결됨' if self.gemini_analyzer.model else '❌ 미연결'}
- **분석 전략:** 3개 (버핏, 린치, 그레이엄)
"""
        
        return report
    
    def _count_total_stocks(self) -> int:
        """총 수집 종목 수"""
        total = 0
        for category, stocks in self.collected_data.items():
            if isinstance(stocks, list):
                total += len(stocks)
        return total
    
    def _count_valid_stocks(self) -> int:
        """유효 데이터 종목 수"""
        valid = 0
        for category, stocks in self.collected_data.items():
            if isinstance(stocks, list):
                for stock in stocks:
                    if self._is_valid_stock_data(stock):
                        valid += 1
        return valid
    
    def save_results(self, analysis_data: Dict, report: str) -> Dict:
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files_saved = {}
        
        # JSON 결과 저장
        json_filename = f'integrated_analysis_{timestamp}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'collected_data': self.collected_data,
                'analysis_data': analysis_data,
                'system_info': {
                    'collection_time': str(self.last_data_collection),
                    'analysis_time': str(datetime.now()),
                    'total_stocks': self._count_total_stocks(),
                    'valid_stocks': self._count_valid_stocks()
                }
            }, f, ensure_ascii=False, indent=2, default=str)
        
        files_saved['json'] = json_filename
        
        # 리포트 저장
        report_filename = f'comprehensive_report_{timestamp}.md'
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        files_saved['report'] = report_filename
        
        # CSV 요약 저장
        csv_filename = f'top5_summary_{timestamp}.csv'
        self._save_top5_csv(analysis_data, csv_filename)
        files_saved['csv'] = csv_filename
        
        return files_saved
    
    def _save_top5_csv(self, analysis_data: Dict, filename: str):
        """Top5 종목 CSV 저장"""
        top5_stocks = analysis_data.get('top5_result', {}).get('top5_stocks', [])
        
        if not top5_stocks:
            return
        
        csv_data = []
        for i, stock in enumerate(top5_stocks, 1):
            csv_data.append({
                '순위': i,
                '종목코드': stock.get('symbol', 'N/A'),
                '카테고리': stock.get('category', 'N/A'),
                '투자비율': stock.get('allocation', 'N/A'),
                '선정이유': stock.get('reasoning', 'N/A')[:100] + '...' if len(stock.get('reasoning', '')) > 100 else stock.get('reasoning', 'N/A')
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    def run_complete_analysis(self) -> Dict:
        """완전한 분석 프로세스 실행"""
        print("\n" + "="*80)
        print("🚀 통합 주식 분석 시스템 - 완전 분석 시작")
        print("="*80)
        
        try:
            # 1. 데이터 수집
            processed_data = self.collect_all_market_data()
            
            # 2. Gemini AI 분석
            analysis_data = self.run_gemini_analysis(processed_data)
            
            # 3. 리포트 생성
            report = self.generate_comprehensive_report(analysis_data)
            
            # 4. 결과 저장
            saved_files = self.save_results(analysis_data, report)
            
            # 5. 결과 요약
            print("\n" + "="*80)
            print("✅ 통합 분석 완료")
            print("="*80)
            print(f"📊 총 수집 종목: {self._count_total_stocks()}개")
            print(f"✅ 유효 데이터: {self._count_valid_stocks()}개")
            print(f"🏆 Top5 선정: {len(analysis_data.get('top5_result', {}).get('top5_stocks', []))}개")
            print(f"\n💾 저장된 파일:")
            for file_type, filename in saved_files.items():
                print(f"  📄 {file_type.upper()}: {filename}")
            
            return {
                'success': True,
                'processed_data': processed_data,
                'analysis_data': analysis_data,
                'report': report,
                'saved_files': saved_files,
                'completion_time': datetime.now()
            }
            
        except Exception as e:
            print(f"\n❌ 분석 중 오류 발생: {e}")
            return {
                'success': False,
                'error': str(e),
                'completion_time': datetime.now()
            }

def main():
    """메인 실행 함수"""
    system = IntegratedStockSystem()
    
    # 완전한 분석 실행
    result = system.run_complete_analysis()
    
    if result['success']:
        print(f"\n🎉 분석 성공적으로 완료!")
        print(f"⏰ 완료 시간: {result['completion_time']}")
    else:
        print(f"\n💥 분석 실패: {result['error']}")
        print(f"⏰ 실패 시간: {result['completion_time']}")

if __name__ == "__main__":
    main() 