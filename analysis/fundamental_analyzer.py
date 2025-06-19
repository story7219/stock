"""
📈 펀더멘털 분석기
- 재무 데이터, 공시 정보 등을 바탕으로 기업의 내재 가치를 분석합니다.
- DART에서 수집된 데이터를 주로 활용합니다.
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

def calculate_financial_ratios(financial_data: Dict[str, Any]) -> Dict[str, float]:
    """재무 데이터를 기반으로 주요 재무 비율을 계산합니다."""
    ratios = {}
    if not financial_data:
        return ratios
        
    try:
        latest_year = max(financial_data.keys())
        latest_data = financial_data[latest_year]
        
        net_income = latest_data.get('당기순이익', 0)
        total_equity = latest_data.get('자본총계', 1)
        total_debt = latest_data.get('부채총계', 0)
        revenue = latest_data.get('매출액', 1)
        operating_income = latest_data.get('영업이익', 0)

        if total_equity > 0:
            ratios['roe'] = (net_income / total_equity) * 100
            ratios['debt_ratio'] = (total_debt / total_equity) * 100
        
        if revenue > 0:
            ratios['operating_margin'] = (operating_income / revenue) * 100
            
    except Exception as e:
        logger.warning(f"⚠️ 재무 비율 계산 실패: {e}")
    
    return ratios

def analyze_financial_trends(financial_data: Dict[str, Any]) -> Dict[str, float]:
    """재무 데이터의 추세를 분석합니다."""
    trends = {}
    if not financial_data or len(financial_data) < 2:
        return trends
        
    try:
        years = sorted(financial_data.keys())
        latest_data = financial_data[years[-1]]
        prev_data = financial_data[years[-2]]

        if prev_data.get('매출액', 0) > 0:
            growth = (latest_data.get('매출액', 0) - prev_data['매출액']) / prev_data['매출액']
            trends['revenue_growth'] = growth * 100

        if abs(prev_data.get('당기순이익', 0)) > 0:
            growth = (latest_data.get('당기순이익', 0) - prev_data['당기순이익']) / abs(prev_data['당기순이익'])
            trends['net_income_growth'] = growth * 100
            
    except Exception as e:
        logger.warning(f"⚠️ 재무 트렌드 분석 실패: {e}")
        
    return trends

def calculate_dart_analysis_score(dart_data: Dict[str, Any]) -> float:
    """DART 데이터를 종합하여 펀더멘털 점수를 계산합니다."""
    score = 50.0  # 기본 점수
    if not dart_data:
        return score

    try:
        # 재무제표 점수 (가중치: 50)
        financials = dart_data.get('financial_statements', {})
        if financials:
            ratios = calculate_financial_ratios(financials)
            trends = analyze_financial_trends(financials)
            
            # ROE
            if ratios.get('roe', 0) > 15: score += 15
            elif ratios.get('roe', 0) > 8: score += 7
            
            # 부채비율
            if ratios.get('debt_ratio', 100) < 50: score += 15
            elif ratios.get('debt_ratio', 100) < 100: score += 7
                
            # 성장성
            if trends.get('revenue_growth', 0) > 10: score += 10
            if trends.get('net_income_growth', 0) > 5: score += 10

        # 공시 점수 (가중치: 20)
        disclosures = dart_data.get('recent_disclosures', [])
        if disclosures:
            if any(is_negative_disclosure(d.get('report_nm', '')) for d in disclosures):
                score -= 20 # 부정적 공시 감점
            elif len(disclosures) > 5: # 너무 잦은 공시
                score -= 10
        else: # 최근 공시 없음 (안정적)
            score += 5
            
        # 기업 정보 점수 (가중치: 15)
        company_info = dart_data.get('company_info')
        if company_info and company_info.get('est_dt'):
            try:
                est_year = int(company_info['est_dt'][:4])
                if (datetime.now().year - est_year) > 10:
                    score += 15 # 업력 10년 이상
            except (ValueError, TypeError):
                pass

        # 주주 구성 점수 (가중치: 15)
        shareholders = dart_data.get('major_shareholders', {}).get('shareholders', [])
        if shareholders:
            total_stake = sum(s.get('stocks_rt', 0) for s in shareholders)
            if total_stake > 40:
                score += 15 # 지배주주 지분율 40% 이상
            elif total_stake > 20:
                score += 7

        return min(100, max(0, score))

    except Exception as e:
        logger.warning(f"⚠️ DART 분석 점수 계산 중 오류: {e}")
        return 50.0

def assess_financial_health(ratios: Dict) -> str:
    """재무 비율을 바탕으로 재무 건전성을 평가합니다."""
    if not ratios:
        return "정보 없음"
    
    debt_ratio = ratios.get('debt_ratio', 100)
    roe = ratios.get('roe', 0)
    
    if debt_ratio < 40 and roe > 15:
        return "매우 우수"
    elif debt_ratio < 70 and roe > 8:
        return "양호"
    elif debt_ratio < 100:
        return "보통"
    elif roe < 0:
        return "위험"
    else:
        return "주의 필요"

def is_negative_disclosure(report_name: str) -> bool:
    """부정적인 내용의 공시인지 판별합니다."""
    negative_keywords = ['정정', '취소', '연기', '중단', '손실', '적자', '감축', '회생', '파산']
    return any(keyword in report_name for keyword in negative_keywords) 