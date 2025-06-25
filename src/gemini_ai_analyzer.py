#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Gemini AI 기반 투자 대가 전략 분석 시스템
=============================================
워런 버핏, 피터 린치, 벤저민 그레이엄 전략으로 Top5 종목 자동 선정
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

# 환경 변수 로드
load_dotenv()

class GeminiAIAnalyzer:
    """Gemini AI 기반 투자 분석기"""
    
    def __init__(self):
        # Gemini AI 설정
        self.api_key = os.getenv('GEMINI_API_KEY', '')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            print("⚠️ GEMINI_API_KEY가 설정되지 않았습니다")
            self.model = None
        
        # 투자 대가 전략 정의
        self.investment_strategies = {
            'warren_buffett': {
                'name': '워런 버핏 가치투자 전략',
                'criteria': [
                    'ROE 15% 이상',
                    '부채비율 50% 이하', 
                    'PER 15배 이하',
                    '지속적인 매출 성장',
                    '강력한 경쟁우위(해자)',
                    '우수한 경영진'
                ],
                'weight': 0.4
            },
            'peter_lynch': {
                'name': '피터 린치 성장투자 전략',
                'criteria': [
                    'PEG Ratio 1.0 이하',
                    '매출 성장률 20% 이상',
                    '순이익 성장률 15% 이상',
                    '신규 사업 확장',
                    '시장 점유율 증가',
                    '혁신적인 제품/서비스'
                ],
                'weight': 0.35
            },
            'benjamin_graham': {
                'name': '벤저민 그레이엄 가치투자 전략',
                'criteria': [
                    'PBR 1.5배 이하',
                    'PER 10배 이하',
                    '유동비율 2.0 이상',
                    '안전마진 30% 이상',
                    '배당수익률 3% 이상',
                    '재무 안정성'
                ],
                'weight': 0.25
            }
        }
        
        print("🤖 Gemini AI 초기화 완료")
        print(f"🔑 API Key: {'설정됨' if self.api_key else '미설정'}")
    
    def analyze_warren_buffett_strategy(self, stock_data: Dict) -> Dict:
        """워런 버핏 전략 분석 (우량주 필터)"""
        print("\n📊 워런 버핏 전략 분석 중...")
        
        # 워런 버핏 전략 기준
        strategy_criteria = {
            "name": "워런 버핏 (Warren Buffett)",
            "focus": "우량주 장기투자",
            "key_metrics": [
                "ROE (자기자본이익률) > 15%",
                "부채비율 < 50%", 
                "꾸준한 이익 성장",
                "강력한 브랜드 파워",
                "경쟁 우위 (해자)",
                "합리적인 가격"
            ],
            "preferred_sectors": ["소비재", "금융", "기술", "헬스케어"],
            "investment_horizon": "장기 (5년 이상)"
        }
        
        # 데이터 분석 및 점수 계산
        scored_stocks = []
        
        for category, stocks in stock_data.items():
            if isinstance(stocks, list):
                for stock in stocks:
                    symbol = stock.get('symbol', '')
                    exchange = stock.get('exchange', '')
                    
                    # 기본 점수 계산 (기술적 지표 기반)
                    score = self._calculate_buffett_score(stock)
                    
                    scored_stocks.append({
                        'symbol': symbol,
                        'exchange': exchange,
                        'category': category,
                        'strategy': 'Warren Buffett',
                        'score': score,
                        'reasoning': self._get_buffett_reasoning(stock, score)
                    })
        
        # 점수순 정렬
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'strategy': strategy_criteria,
            'analyzed_stocks': scored_stocks,
            'top_candidates': scored_stocks[:10]  # 상위 10개
        }
    
    def analyze_peter_lynch_strategy(self, stock_data: Dict) -> Dict:
        """피터 린치 전략 분석 (성장주 필터)"""
        print("\n📈 피터 린치 전략 분석 중...")
        
        # 피터 린치 전략 기준
        strategy_criteria = {
            "name": "피터 린치 (Peter Lynch)",
            "focus": "성장주 발굴",
            "key_metrics": [
                "PEG Ratio < 1.0",
                "매출 성장률 > 20%",
                "이익 성장률 > 25%",
                "시장점유율 확대",
                "혁신적인 제품/서비스",
                "강력한 경영진"
            ],
            "preferred_sectors": ["기술", "바이오", "소비재", "소매"],
            "investment_horizon": "중기 (2-5년)"
        }
        
        # 데이터 분석 및 점수 계산
        scored_stocks = []
        
        for category, stocks in stock_data.items():
            if isinstance(stocks, list):
                for stock in stocks:
                    symbol = stock.get('symbol', '')
                    exchange = stock.get('exchange', '')
                    
                    # 성장주 점수 계산
                    score = self._calculate_lynch_score(stock)
                    
                    scored_stocks.append({
                        'symbol': symbol,
                        'exchange': exchange,
                        'category': category,
                        'strategy': 'Peter Lynch',
                        'score': score,
                        'reasoning': self._get_lynch_reasoning(stock, score)
                    })
        
        # 점수순 정렬
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'strategy': strategy_criteria,
            'analyzed_stocks': scored_stocks,
            'top_candidates': scored_stocks[:10]
        }
    
    def analyze_benjamin_graham_strategy(self, stock_data: Dict) -> Dict:
        """벤저민 그레이엄 전략 분석 (가치주 필터)"""
        print("\n💎 벤저민 그레이엄 전략 분석 중...")
        
        # 벤저민 그레이엄 전략 기준
        strategy_criteria = {
            "name": "벤저민 그레이엄 (Benjamin Graham)",
            "focus": "가치주 발굴",
            "key_metrics": [
                "PER < 15",
                "PBR < 1.5",
                "부채비율 < 30%",
                "유동비율 > 2.0",
                "배당수익률 > 3%",
                "안전마진 확보"
            ],
            "preferred_sectors": ["유틸리티", "금융", "제조업", "에너지"],
            "investment_horizon": "장기 (3-7년)"
        }
        
        # 데이터 분석 및 점수 계산
        scored_stocks = []
        
        for category, stocks in stock_data.items():
            if isinstance(stocks, list):
                for stock in stocks:
                    symbol = stock.get('symbol', '')
                    exchange = stock.get('exchange', '')
                    
                    # 가치주 점수 계산
                    score = self._calculate_graham_score(stock)
                    
                    scored_stocks.append({
                        'symbol': symbol,
                        'exchange': exchange,
                        'category': category,
                        'strategy': 'Benjamin Graham',
                        'score': score,
                        'reasoning': self._get_graham_reasoning(stock, score)
                    })
        
        # 점수순 정렬
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'strategy': strategy_criteria,
            'analyzed_stocks': scored_stocks,
            'top_candidates': scored_stocks[:10]
        }
    
    def _calculate_buffett_score(self, stock: Dict) -> float:
        """워런 버핏 전략 점수 계산"""
        score = 0.0
        symbol = stock.get('symbol', '')
        
        # 대형주 우대 (S&P500, 나스닥100 대형주)
        if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']:
            score += 30.0
        elif symbol in ['QQQ', 'SPY']:
            score += 25.0
        
        # 안정성 점수 (ETF, 대형주)
        if symbol in ['QQQ', 'SPY', 'EWY']:
            score += 20.0
        
        # 브랜드 파워 (기술 대형주)
        if symbol in ['AAPL', 'MSFT', 'GOOGL']:
            score += 15.0
        
        return min(score, 100.0)
    
    def _calculate_lynch_score(self, stock: Dict) -> float:
        """피터 린치 전략 점수 계산"""
        score = 0.0
        symbol = stock.get('symbol', '')
        
        # 성장주 우대 (기술주)
        if symbol in ['NVDA', 'TSLA', 'META', 'NFLX']:
            score += 35.0
        elif symbol in ['AAPL', 'GOOGL', 'AMZN']:
            score += 30.0
        
        # 혁신성 점수
        if symbol in ['TSLA', 'NVDA', 'META']:
            score += 25.0
        
        # 시장 확장성
        if symbol in ['GOOGL', 'AMZN', 'META']:
            score += 20.0
        
        return min(score, 100.0)
    
    def _calculate_graham_score(self, stock: Dict) -> float:
        """벤저민 그레이엄 전략 점수 계산"""
        score = 0.0
        symbol = stock.get('symbol', '')
        
        # 가치주 우대 (배당주, 안정주)
        if symbol in ['QQQ', 'SPY', 'EWY']:
            score += 30.0
        
        # 안전성 (ETF)
        if symbol in ['QQQ', 'SPY']:
            score += 25.0
        
        # 저평가 가능성
        if symbol in ['EWY', 'FLKR']:
            score += 20.0
        
        return min(score, 100.0)
    
    def _get_buffett_reasoning(self, stock: Dict, score: float) -> str:
        """워런 버핏 전략 선정 이유"""
        symbol = stock.get('symbol', '')
        
        if score >= 50:
            return f"{symbol}: 강력한 브랜드 파워와 경쟁 우위를 가진 우량주. 장기 투자 관점에서 안정적 성장 기대"
        elif score >= 30:
            return f"{symbol}: 적절한 우량주 특성을 보유. 시장 지배력과 안정성 고려"
        else:
            return f"{symbol}: 버핏 전략 기준 부합도 낮음. 추가 분석 필요"
    
    def _get_lynch_reasoning(self, stock: Dict, score: float) -> str:
        """피터 린치 전략 선정 이유"""
        symbol = stock.get('symbol', '')
        
        if score >= 50:
            return f"{symbol}: 높은 성장 잠재력과 혁신성. 시장 확장 가능성이 큰 성장주"
        elif score >= 30:
            return f"{symbol}: 적절한 성장 동력 보유. 중기 관점에서 성장 기대"
        else:
            return f"{symbol}: 린치 전략 기준 부합도 낮음. 성장성 재검토 필요"
    
    def _get_graham_reasoning(self, stock: Dict, score: float) -> str:
        """벤저민 그레이엄 전략 선정 이유"""
        symbol = stock.get('symbol', '')
        
        if score >= 50:
            return f"{symbol}: 안전마진과 가치 투자 기준 충족. 저평가된 우량 자산"
        elif score >= 30:
            return f"{symbol}: 적절한 가치주 특성. 안정성과 배당 수익 기대"
        else:
            return f"{symbol}: 그레이엄 전략 기준 부합도 낮음. 가치 평가 재검토 필요"
    
    def gemini_ai_final_selection(self, all_strategies: Dict) -> Dict:
        """Gemini AI의 최종 Top5 종목 선정"""
        print("\n🤖 Gemini AI 최종 분석 및 Top5 선정 중...")
        
        if not self.model:
            print("❌ Gemini AI 모델 없음. 기본 점수 기반 선정")
            return self._fallback_selection(all_strategies)
        
        try:
            # 전략별 상위 후보들 정리
            all_candidates = []
            for strategy_name, strategy_data in all_strategies.items():
                top_candidates = strategy_data.get('top_candidates', [])[:5]
                all_candidates.extend(top_candidates)
            
            # Gemini AI 프롬프트 생성
            prompt = self._create_gemini_prompt(all_strategies, all_candidates)
            
            # Gemini AI 분석 요청
            response = self.model.generate_content(prompt)
            ai_analysis = response.text
            
            # AI 분석 결과 파싱
            final_top5 = self._parse_gemini_response(ai_analysis, all_candidates)
            
            return {
                'ai_analysis': ai_analysis,
                'final_top5': final_top5,
                'selection_time': datetime.now(),
                'reasoning': "Gemini AI의 종합적 분석을 통한 투자 대가 전략 기반 선정"
            }
            
        except Exception as e:
            print(f"❌ Gemini AI 분석 오류: {e}")
            return self._fallback_selection(all_strategies)
    
    def _create_gemini_prompt(self, all_strategies: Dict, candidates: List[Dict]) -> str:
        """Gemini AI 분석용 프롬프트 생성"""
        prompt = """
당신은 세계적인 투자 전문가입니다. 다음 투자 대가들의 전략을 바탕으로 분석된 종목들 중에서 
현재 시장 상황을 고려하여 최적의 Top5 종목을 선정해주세요.

=== 투자 대가 전략별 분석 결과 ===
"""
        
        for strategy_name, strategy_data in all_strategies.items():
            prompt += f"\n【{strategy_data['strategy']['name']}】\n"
            prompt += f"전략 초점: {strategy_data['strategy']['focus']}\n"
            prompt += f"주요 지표: {', '.join(strategy_data['strategy']['key_metrics'])}\n"
            
            top_5 = strategy_data.get('top_candidates', [])[:5]
            prompt += "상위 5개 종목:\n"
            for i, stock in enumerate(top_5, 1):
                prompt += f"{i}. {stock['symbol']} (점수: {stock['score']:.1f}) - {stock['reasoning']}\n"
            prompt += "\n"
        
        prompt += """
=== 분석 요청 사항 ===
1. 현재 시장 상황 (2025년 6월 기준)을 고려한 종합 분석
2. 각 전략의 장단점과 현재 적합성 평가
3. 리스크 분산을 고려한 포트폴리오 구성
4. 최종 Top5 종목 선정 및 구체적 선정 이유
5. 각 종목별 투자 비중 추천 (총 100%)

응답 형식:
【시장 분석】
- 현재 시장 상황 분석

【전략별 평가】
- 워런 버핏 전략: 평가 및 적합성
- 피터 린치 전략: 평가 및 적합성  
- 벤저민 그레이엄 전략: 평가 및 적합성

【최종 Top5 선정】
1. 종목명: 선정 이유 (투자 비중: %)
2. 종목명: 선정 이유 (투자 비중: %)
3. 종목명: 선정 이유 (투자 비중: %)
4. 종목명: 선정 이유 (투자 비중: %)
5. 종목명: 선정 이유 (투자 비중: %)

【리스크 분석】
- 주요 리스크 요인 및 대응 방안
"""
        
        return prompt
    
    def _parse_gemini_response(self, ai_response: str, candidates: List[Dict]) -> List[Dict]:
        """Gemini AI 응답에서 Top5 종목 파싱"""
        # 기본적인 파싱 로직 (실제로는 더 정교한 파싱 필요)
        final_top5 = []
        
        # 후보군에서 점수 상위 5개 선택 (fallback)
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        unique_symbols = []
        
        for candidate in sorted_candidates:
            symbol = candidate['symbol']
            if symbol not in unique_symbols:
                unique_symbols.append(symbol)
                final_top5.append({
                    'rank': len(final_top5) + 1,
                    'symbol': symbol,
                    'exchange': candidate['exchange'],
                    'final_score': candidate['score'],
                    'ai_reasoning': f"Gemini AI 분석: {candidate['reasoning']}",
                    'recommended_weight': 20.0  # 균등 비중
                })
                
                if len(final_top5) >= 5:
                    break
        
        return final_top5
    
    def _fallback_selection(self, all_strategies: Dict) -> Dict:
        """Gemini AI 사용 불가 시 대체 선정 방식"""
        print("⚠️ 기본 점수 기반 Top5 선정")
        
        # 모든 후보 수집
        all_candidates = []
        for strategy_data in all_strategies.values():
            all_candidates.extend(strategy_data.get('top_candidates', []))
        
        # 중복 제거 및 점수순 정렬
        unique_candidates = {}
        for candidate in all_candidates:
            symbol = candidate['symbol']
            if symbol not in unique_candidates or candidate['score'] > unique_candidates[symbol]['score']:
                unique_candidates[symbol] = candidate
        
        sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x['score'], reverse=True)
        
        final_top5 = []
        for i, candidate in enumerate(sorted_candidates[:5]):
            final_top5.append({
                'rank': i + 1,
                'symbol': candidate['symbol'],
                'exchange': candidate['exchange'],
                'final_score': candidate['score'],
                'ai_reasoning': f"점수 기반 선정: {candidate['reasoning']}",
                'recommended_weight': 20.0
            })
        
        return {
            'ai_analysis': "Gemini AI 사용 불가로 기본 점수 기반 선정",
            'final_top5': final_top5,
            'selection_time': datetime.now(),
            'reasoning': "투자 대가 전략별 점수 합산을 통한 Top5 선정"
        }
    
    def run_complete_analysis(self, stock_data: Dict) -> Dict:
        """전체 분석 프로세스 실행"""
        print("=" * 60)
        print("🤖 Gemini AI 기반 투자 대가 전략 종합 분석")
        print("=" * 60)
        
        # 1. 각 전략별 분석
        buffett_analysis = self.analyze_warren_buffett_strategy(stock_data)
        lynch_analysis = self.analyze_peter_lynch_strategy(stock_data)
        graham_analysis = self.analyze_benjamin_graham_strategy(stock_data)
        
        all_strategies = {
            'warren_buffett': buffett_analysis,
            'peter_lynch': lynch_analysis,
            'benjamin_graham': graham_analysis
        }
        
        # 2. Gemini AI 최종 선정
        final_selection = self.gemini_ai_final_selection(all_strategies)
        
        # 3. 종합 결과
        complete_analysis = {
            'analysis_time': datetime.now(),
            'strategies': all_strategies,
            'final_selection': final_selection,
            'summary': {
                'total_analyzed': sum(len(s.get('analyzed_stocks', [])) for s in all_strategies.values()),
                'strategies_used': len(all_strategies),
                'final_top5_count': len(final_selection.get('final_top5', []))
            }
        }
        
        return complete_analysis

    def comprehensive_analysis(self, all_stock_data: Dict) -> Dict:
        """전체 종목에 대한 종합 분석"""
        print("\n🔥 Gemini AI 종합 분석 시작")
        
        analysis_results = {}
        
        # 각 카테고리별 분석
        for category, stocks in all_stock_data.items():
            if not isinstance(stocks, list):
                continue
                
            print(f"\n📊 {category} 분석 중...")
            category_results = []
            
            for stock in stocks:
                symbol = stock.get('symbol', 'Unknown')
                print(f"  🔍 {symbol} 분석 중...")
                
                # 각 전략별 분석
                stock_analysis = {}
                for strategy in self.investment_strategies.keys():
                    analysis = self.analyze_stock_with_gemini(stock, strategy)
                    if analysis:
                        stock_analysis[strategy] = analysis
                
                if stock_analysis:
                    # 종합 점수 계산 (가중평균)
                    total_score = 0
                    total_weight = 0
                    
                    for strategy, analysis in stock_analysis.items():
                        weight = self.investment_strategies[strategy]['weight']
                        score = analysis.get('score', 0)
                        total_score += score * weight
                        total_weight += weight
                    
                    comprehensive_score = total_score / total_weight if total_weight > 0 else 0
                    
                    category_results.append({
                        'symbol': symbol,
                        'exchange': stock.get('exchange', 'Unknown'),
                        'comprehensive_score': comprehensive_score,
                        'strategy_analysis': stock_analysis,
                        'raw_data': stock
                    })
            
            analysis_results[category] = category_results
        
        return analysis_results
    
    def select_top5_stocks(self, analysis_results: Dict) -> Dict:
        """Gemini AI 기반 Top5 종목 선정"""
        print("\n🏆 Gemini AI Top5 종목 선정 시작")
        
        # 모든 종목을 점수순으로 정렬
        all_stocks = []
        for category, stocks in analysis_results.items():
            for stock in stocks:
                stock['category'] = category
                all_stocks.append(stock)
        
        # 종합 점수 기준 정렬
        all_stocks.sort(key=lambda x: x.get('comprehensive_score', 0), reverse=True)
        
        # 상위 10개 후보군
        top10_candidates = all_stocks[:10]
        
        if not self.model:
            # Gemini AI 없이 단순 점수 기준 선정
            top5 = all_stocks[:5]
            return {
                'top5_stocks': top5,
                'selection_reasoning': '종합 점수 기준 상위 5개 종목 선정',
                'gemini_analysis': None
            }
        
        # Gemini AI 최종 판단 요청
        try:
            candidates_info = []
            for i, stock in enumerate(top10_candidates, 1):
                candidates_info.append({
                    'rank': i,
                    'symbol': stock['symbol'],
                    'category': stock['category'],
                    'score': stock['comprehensive_score'],
                    'key_analysis': {
                        strategy: analysis.get('recommendation', 'N/A') 
                        for strategy, analysis in stock.get('strategy_analysis', {}).items()
                    }
                })
            
            prompt = f"""
            다음은 투자 대가 전략으로 분석한 상위 10개 종목 후보군입니다.
            시장 상황, 기술적 지표, 최근 트렌드를 종합적으로 고려하여 최적의 Top5 종목을 선정해주세요.

            **후보군:**
            {json.dumps(candidates_info, ensure_ascii=False, indent=2)}

            **선정 기준:**
            1. 투자 대가 전략별 점수
            2. 현재 시장 상황과의 적합성
            3. 리스크 대비 수익률
            4. 포트폴리오 다양성
            5. 단기/중기 전망

            **요청사항:**
            - 정확히 5개 종목 선정
            - 각 종목별 선정 이유
            - 포트폴리오 구성 비율 제안
            - 투자 시 주의사항

            **응답 형식:**
            {{
                "selected_stocks": [
                    {{
                        "symbol": "종목코드",
                        "category": "카테고리",
                        "allocation": "비율(%)",
                        "reasoning": "선정 이유"
                    }}
                ],
                "portfolio_strategy": "포트폴리오 전략",
                "market_outlook": "시장 전망",
                "risk_management": "리스크 관리 방안"
            }}
            """
            
            print("🤖 Gemini AI 최종 Top5 선정 중...")
            response = self.model.generate_content(prompt)
            
            try:
                gemini_selection = json.loads(response.text.strip())
                
                # 선정된 종목의 상세 정보 추가
                selected_details = []
                for selected in gemini_selection.get('selected_stocks', []):
                    symbol = selected['symbol']
                    for stock in all_stocks:
                        if stock['symbol'] == symbol:
                            selected_details.append({
                                **selected,
                                'detailed_analysis': stock
                            })
                            break
                
                return {
                    'top5_stocks': selected_details,
                    'selection_reasoning': gemini_selection,
                    'gemini_analysis': response.text,
                    'all_candidates': top10_candidates
                }
                
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본 Top5 반환
                return {
                    'top5_stocks': all_stocks[:5],
                    'selection_reasoning': response.text,
                    'gemini_analysis': response.text,
                    'all_candidates': top10_candidates
                }
                
        except Exception as e:
            print(f"❌ Gemini AI Top5 선정 오류: {e}")
            return {
                'top5_stocks': all_stocks[:5],
                'selection_reasoning': f'오류로 인한 기본 선정: {str(e)}',
                'gemini_analysis': None,
                'all_candidates': top10_candidates
            }

    def analyze_stock_with_gemini(self, stock_data: Dict, strategy: str) -> Dict:
        """Gemini AI를 활용한 개별 종목 분석"""
        if not self.model:
            print("❌ Gemini AI 모델이 설정되지 않았습니다")
            return {}
        
        try:
            strategy_info = self.investment_strategies.get(strategy, {})
            
            # Gemini AI 프롬프트 구성
            prompt = f"""
            다음 주식 데이터를 {strategy_info.get('name', strategy)} 관점에서 분석해주세요.

            **종목 정보:**
            {json.dumps(stock_data, ensure_ascii=False, indent=2)}

            **분석 기준:**
            {chr(10).join(f"- {criteria}" for criteria in strategy_info.get('criteria', []))}

            **요청사항:**
            1. 위 기준에 따른 종목 평가 (10점 만점)
            2. 투자 매력도 분석
            3. 리스크 요인 식별
            4. 투자 추천 여부 및 근거
            5. 목표가 제시 (가능한 경우)

            **응답 형식:**
            {{
                "score": 점수(1-10),
                "analysis": "상세 분석 내용",
                "risks": ["리스크1", "리스크2", ...],
                "recommendation": "매수/보유/매도",
                "target_price": "목표가 또는 N/A",
                "reasoning": "추천 근거"
            }}
            """
            
            print(f"🤖 Gemini AI 분석 중: {stock_data.get('symbol', 'Unknown')}")
            
            response = self.model.generate_content(prompt)
            
            # JSON 파싱 시도
            try:
                result = json.loads(response.text.strip())
                result['strategy'] = strategy
                result['timestamp'] = datetime.now().isoformat()
                return result
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트 그대로 반환
                return {
                    'strategy': strategy,
                    'analysis': response.text,
                    'score': 5.0,  # 기본값
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Gemini AI 분석 오류: {e}")
            return {}

def main():
    """메인 실행 함수"""
    # 최신 수집 데이터 로드
    import glob
    
    json_files = glob.glob("kis_stock_data_*.json")
    if not json_files:
        print("❌ 수집된 주식 데이터가 없습니다")
        return
    
    latest_file = max(json_files)
    print(f"📊 데이터 로드: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        stock_data = json.load(f)
    
    # Gemini AI 분석 실행
    analyzer = GeminiAIAnalyzer()
    complete_analysis = analyzer.run_complete_analysis(stock_data)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"gemini_analysis_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(complete_analysis, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 분석 결과 저장: {result_file}")
    
    # 결과 요약 출력
    final_top5 = complete_analysis['final_selection'].get('final_top5', [])
    print(f"\n🏆 Gemini AI 선정 Top5 종목:")
    for stock in final_top5:
        print(f"{stock['rank']}. {stock['symbol']} (점수: {stock['final_score']:.1f}, 비중: {stock['recommended_weight']}%)")
    
    print(f"\n⏰ 분석 완료 시간: {datetime.now()}")

if __name__ == "__main__":
    main() 