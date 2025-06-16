import yfinance as yf
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
from gspread.exceptions import SpreadsheetNotFound
from datetime import datetime
import os
import warnings
import logging
from typing import Dict

# pandas의 FutureWarning를 무시하도록 설정
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 한국어 주석 ---

class ONeilScanner:
    """
    윌리엄 오닐의 CAN SLIM 투자 기법을 기반으로 주식을 스크리닝합니다.
    정량적으로 분석 가능한 C, A, S, L, I 지표를 중심으로 평가합니다.
    """

    def __init__(self, kis_api):
        """ONeil 스캐너 초기화"""
        self.api = kis_api  # kis_api 객체를 그대로 저장
        self.logger = logging.getLogger(__name__)
        
        # CAN SLIM 기준 설정
        self.can_slim_criteria = {
            'C': {'min_eps_growth': 25, 'weight': 0.2},
            'A': {'min_annual_growth': 25, 'weight': 0.15},
            'N': {'new_high_days': 30, 'weight': 0.15},
            'S': {'min_supply_demand': 1.5, 'weight': 0.15},
            'L': {'market_leader': True, 'weight': 0.1},
            'I': {'institutional_support': True, 'weight': 0.1},
            'M': {'market_direction': True, 'weight': 0.15}
        }
        
        self.logger.info("ONeil CAN SLIM 스캐너가 초기화되었습니다.")

    async def analyze_stock(self, symbol: str) -> Dict:
        """종목에 대한 CAN SLIM 분석 수행"""
        try:
            # symbol이 문자열인지 확인하고 처리
            if not isinstance(symbol, str):
                symbol = str(symbol)
            
            # 여기서 .upper() 호출 시 symbol에 대해서만 호출
            symbol = symbol.upper()  # self.api.upper() 대신 symbol.upper() 사용
            
            self.logger.info(f"📊 {symbol}에 대한 CAN SLIM 분석 시작...")
            
            # 기본 주가 정보 조회
            price_info = self.api.get_current_price(symbol)
            if not price_info or price_info.get('rt_cd') != '0':
                self.logger.warning(f"{symbol} 가격 정보 조회 실패")
                return {}
            
            # CAN SLIM 각 항목 분석
            can_slim_scores = {}
            
            # C - Current Earnings (현재 수익)
            can_slim_scores['C'] = await self._analyze_current_earnings(symbol)
            
            # A - Annual Earnings (연간 수익)
            can_slim_scores['A'] = await self._analyze_annual_earnings(symbol)
            
            # N - New Products/Services (신제품/서비스)
            can_slim_scores['N'] = await self._analyze_new_highs(symbol)
            
            # S - Supply and Demand (수급)
            can_slim_scores['S'] = await self._analyze_supply_demand(symbol)
            
            # L - Leader or Laggard (선도주 vs 후행주)
            can_slim_scores['L'] = await self._analyze_market_leadership(symbol)
            
            # I - Institutional Sponsorship (기관 후원)
            can_slim_scores['I'] = await self._analyze_institutional_support(symbol)
            
            # M - Market Direction (시장 방향)
            can_slim_scores['M'] = await self._analyze_market_direction()
            
            # 종합 점수 계산
            total_score = 0
            for key, score in can_slim_scores.items():
                weight = self.can_slim_criteria[key]['weight']
                total_score += score * weight
            
            # 결과 정리
            result = {
                'symbol': symbol,
                'can_slim_score': round(total_score * 10, 2),  # 10점 만점으로 환산
                'individual_scores': can_slim_scores,
                'analysis_time': datetime.now().isoformat(),
                'recommendation': 'BUY' if total_score > 0.7 else 'HOLD' if total_score > 0.4 else 'SELL'
            }
            
            self.logger.info(f"✅ {symbol} CAN SLIM 분석 완료 - 점수: {result['can_slim_score']}/10")
            return result
            
        except Exception as e:
            self.logger.error(f"{symbol} CAN SLIM 분석 중 오류: {e}")
            return {}

    def check_current_earnings(self, min_growth_pct=25):
        """(C) 최근 분기 주당 순이익(EPS) 또는 순이익 증가율 체크"""
        if self.quarterly_financials is None or self.quarterly_financials.empty:
            return False, "분기 재무 데이터 없음"

        try:
            q_financials = self.quarterly_financials.transpose()
            # 가장 최근 분기와 4분기 전(1년 전)의 순이익 비교
            net_income_q_curr = q_financials['Net Income'].iloc[0]
            net_income_q_prev = q_financials['Net Income'].iloc[4]

            if net_income_q_prev <= 0:
                growth = float('inf') if net_income_q_curr > 0 else 0
            else:
                growth = ((net_income_q_curr / net_income_q_prev) - 1) * 100

            is_passed = growth >= min_growth_pct
            if is_passed: self.passed_criteria += 1
            return is_passed, f"{growth:.2f}%"
        except (IndexError, KeyError):
            return False, "데이터 부족"

    def check_annual_earnings(self, min_growth_pct=25):
        """(A) 연간 순이익 성장률 체크 (최근 3년)"""
        if self.financials is None or self.financials.empty:
            return False, "연간 재무 데이터 없음"
        
        try:
            a_financials = self.financials.transpose()
            # 최근 3년간의 순이익
            net_incomes = a_financials['Net Income'].iloc[:3]
            
            # 전년 대비 성장률 계산
            yearly_growth = net_incomes.pct_change(periods=-1).dropna() * 100
            
            # 3년 평균 성장률이 기준을 넘고, 적자가 없는지 확인
            is_passed = (yearly_growth.mean() >= min_growth_pct) and (net_incomes > 0).all()
            if is_passed: self.passed_criteria += 1
            return is_passed, f"평균 {yearly_growth.mean():.2f}%"
        except (IndexError, KeyError):
            return False, "데이터 부족"
            
    def check_leader_or_laggard(self, near_high_pct=15):
        """(L) 선도주 vs 후발주 (52주 신고가 근접 여부로 판단)"""
        try:
            high_52week = self.history['High'].max()
            current_price = self.history['Close'].iloc[-1]
            
            distance_from_high = (1 - (current_price / high_52week)) * 100
            is_passed = distance_from_high <= near_high_pct
            if is_passed: self.passed_criteria += 1
            return is_passed, f"52주 신고가 대비 {-distance_from_high:.2f}%"
        except (IndexError, KeyError):
            return False, "데이터 부족"

    def check_supply_and_demand(self, volume_increase_pct=40):
        """(S) 수급: 최근 거래량이 평균 거래량을 상회하는지 체크"""
        try:
            avg_volume = self.history['Volume'].rolling(window=50).mean().iloc[-2] # 어제까지의 50일 평균
            latest_volume = self.history['Volume'].iloc[-1]

            volume_change = ((latest_volume / avg_volume) - 1) * 100
            is_passed = volume_change >= volume_increase_pct
            if is_passed: self.passed_criteria += 1
            return is_passed, f"50일 평균 대비 {volume_change:.2f}%"
        except (IndexError, KeyError, ZeroDivisionError):
            return False, "데이터 부족"

    def check_institutional_sponsorship(self, min_own_pct=10):
        """(I) 기관의 뒷받침: 기관 투자자 보유 비중 체크"""
        try:
            # yfinance는 institutional_holders를 제공하지만, major_holders가 더 안정적일 수 있음
            inst_own_pct = self.major_holders.iloc[0, 0] * 100
            is_passed = inst_own_pct >= min_own_pct
            if is_passed: self.passed_criteria += 1
            return is_passed, f"{inst_own_pct:.2f}%"
        except (IndexError, KeyError, TypeError):
            return False, "데이터 부족"

    def run_scan(self):
        """모든 CAN SLIM 기준을 실행하고 결과를 리포트합니다."""
        if self.info is None:
            return None # 데이터 로딩 실패 시 스캔 중단
        
        self.report['결과']['C (분기실적)'] = self.check_current_earnings()
        self.report['결과']['A (연간실적)'] = self.check_annual_earnings()
        self.report['결과']['L (선도주)'] = self.check_leader_or_laggard()
        self.report['결과']['S (수급/거래량)'] = self.check_supply_and_demand()
        self.report['결과']['I (기관보유)'] = self.check_institutional_sponsorship()
        
        self.report['종합점수'] = f"{self.passed_criteria} / {self.total_criteria}"
        self.report['최종판단'] = "관심" if self.passed_criteria >= 4 else "보류"
        
        return self.report

def check_market_direction(market_index_symbol='^KS11'):
    """(M) 시장 방향성 체크. KOSPI 지수가 50일 이동평균선 위에 있는지 확인."""
    market = yf.Ticker(market_index_symbol)
    hist = market.history(period="3mo")
    
    if hist.empty:
        print("경고: 시장 지수 데이터를 가져올 수 없어 시장 방향성을 판단할 수 없습니다.")
        return False, None
        
    sma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
    current_price = hist['Close'].iloc[-1]
    
    is_uptrend = current_price > sma50
    
    market_status_text = (
        f"--- 시장 방향성 (M) 체크 ({market_index_symbol}) ---\n"
        f"현재 지수: {current_price:,.2f} | 50일 이동평균: {sma50:,.2f}\n"
        f"시장 추세: {'상승장' if is_uptrend else '하락장 또는 조정장'}\n"
        f"{'-' * 35}\n"
    )
    print(market_status_text)
    return is_uptrend, market_status_text

def upload_to_gsheet(spreadsheet_name, philosophy_text, market_status_text, results_df):
    """스캔 결과와 투자 철학을 구글 스프레드시트에 업로드합니다."""
    print("\n--- 구글 스프레드시트 업로드 시작 ---")
    try:
        # ==================== 핵심 수정 사항 ====================
        # 어떤 컴퓨터에서든 바탕화면의 '네이버블로그' 폴더를 자동으로 찾도록 경로 수정
        home_dir = os.path.expanduser('~')
        credentials_path = os.path.join(home_dir, 'Desktop', '네이버블로그', 'g-credentials.json')
        # =======================================================
        
        gc = gspread.service_account(filename=credentials_path)
        spreadsheet = gc.open(spreadsheet_name)
        print(f"'{spreadsheet_name}' 스프레드시트를 성공적으로 열었습니다.")

        # 1. 투자 철학 시트
        full_philosophy_text = philosophy_text + "\n\n" + market_status_text
        try:
            ws_philosophy = spreadsheet.worksheet('오닐 투자철학')
            spreadsheet.del_worksheet(ws_philosophy)
        except gspread.WorksheetNotFound:
            pass
        ws_philosophy = spreadsheet.add_worksheet(title='오닐 투자철학', rows=100, cols=1)
        ws_philosophy.update('A1', full_philosophy_text)
        ws_philosophy.format('A1', {'wrapStrategy': 'WRAP'})
        print("- '오닐 투자철학' 시트 업로드 완료.")

        # 2. 스캔 결과 시트
        sheet_title = f'CAN SLIM 결과 ({datetime.now().strftime("%Y-%m-%d")})'
        try:
            ws_results = spreadsheet.worksheet(sheet_title)
            spreadsheet.del_worksheet(ws_results)
        except gspread.WorksheetNotFound:
            pass
        ws_results = spreadsheet.add_worksheet(title=sheet_title, rows=len(results_df)+1, cols=len(results_df.columns))
        set_with_dataframe(ws_results, results_df)
        print(f"- '{sheet_title}' 시트 업로드 완료.")

        print("--- 모든 데이터 업로드 성공! ---")

    except FileNotFoundError:
        print(f"\n오류: '{credentials_path}' 경로에서 'g-credentials.json' 파일을 찾을 수 없습니다.")
        print("파일이 해당 위치에 정확히 있는지, 파일 이름이 올바른지 다시 확인해주세요.")
    except SpreadsheetNotFound:
        print(f"\n오류: '{spreadsheet_name}' 스프레드시트를 찾을 수 없습니다. 구글 드라이브에서 파일을 생성하고 서비스 계정에 '편집자'로 공유했는지 확인해주세요.")
    except Exception as e:
        print(f"\n오류: 구글 시트 업로드 중 예상치 못한 문제 발생 - {e}")

if __name__ == '__main__':
    # 윌리엄 오닐의 투자 철학 (사용자 질문 내용)
    oneil_philosophy = """
# 윌리엄 오닐(William O'Neil)의 투자 기법과 철학

# 🎯 윌리엄 오닐 소개
윌리엄 제임스 오닐(William James O'Neil, 1933-2021)은 미국의 전설적인 투자자이자 기업가입니다. 그는 투자 정보 회사인 인베스터스 비즈니스 데일리(Investor's Business Daily)를 창립했으며, 성장주 투자의 대가로 불립니다.

# 📈 CAN SLIM 투자 기법
오닐의 대표적인 투자 방법론으로, 7가지 핵심 요소를 체크하는 시스템입니다:

## C - Current Earnings (현재 수익)
- 최근 분기 주당순이익(EPS)이 전년 동기 대비 25% 이상 증가
- 연속적인 수익 증가 추세 확인

## A - Annual Earnings (연간 수익)
- 과거 3년간 연평균 25% 이상의 수익 성장률
- 지속적이고 안정적인 성장 패턴

## N - New Products, Services, Management (새로운 요소들)
- 혁신적인 신제품이나 서비스 출시
- 새로운 경영진 또는 경영 전략
- 새로운 시장 진출이나 업계 환경 변화

## S - Supply and Demand (수급)
- 발행주식수가 적거나 자사주 매입으로 감소
- 기관투자가들의 관심과 매수세 증가
- 유동주식 비중이 적을수록 유리
- 대량 거래량 증가 시점 포착

## L - Leader or Laggard (선도주 vs 후발주)
- 해당 업종의 1위 또는 2위 기업
- 상대강도지수(RS Rating) 80 이상
- 시장 대비 상대적 성과가 우수

## I - Institutional Sponsorship (기관투자가 후원)
- 뮤추얼펀드, 연기금 등 기관투자가들의 꾸준한 매수
- 최근 분기 기관 보유 비중 증가
- 우수한 펀드매니저들의 관심 종목
- 기관 보유 비중 40-60%가 이상적

## M - Market Direction (시장 방향)
- 전체 주식시장이 상승 추세에 있어야 함
- 주요 지수들의 기술적 분석 결과 긍정적
- 75%의 개별 종목이 시장 방향을 따름
- 약세장에서는 현금 보유 비중 확대

# 🔍 핵심 투자 철학

## 1. 성장주 투자
저평가된 가치주보다는 빠르게 성장하는 기업에 투자하는 것을 선호합니다. 수익이 지속적으로 증가하고 혁신적인 제품이나 서비스를 가진 기업을 찾습니다.

## 2. 기술적 분석과 펀더멘털 분석의 결합
재무제표 분석(펀더멘털)과 차트 분석(기술적 분석)을 모두 활용합니다. 좋은 기업이라도 차트 패턴이 나쁘면 투자하지 않습니다.

## 3. 손절매의 중요성
매수가 대비 7-8% 하락하면 무조건 손절매를 실행합니다. '작은 손실은 친구, 큰 손실은 적'이라는 철학을 가지고 있습니다.

# 📊 차트 패턴 분석

## 컵 앤 핸들 (Cup and Handle) 패턴
오닐이 가장 선호하는 차트 패턴입니다:
- 7주 이상의 컵 모양 조정 (12-65% 하락)
- 1-5주간의 핸들 형성 (8-12% 추가 조정)
- 저항선 돌파 시 매수 신호
- 돌파 시점에 거래량 급증 필수

# ⚡ 매매 규칙

## 매수 규칙
- 차트 패턴의 돌파점에서 매수
- 거래량이 평소보다 40-50% 이상 증가할 때
- CAN SLIM 조건을 모두 만족하는 종목
- 시장이 상승 추세일 때만 매수

## 매도 규칙
**손절매: 매수가 대비 7-8% 하락 시 무조건 손절**
**이익실현: 20-25% 상승 시 일부 매도 고려**
**시장 전반이 약세로 전환될 때 전량 매도**

💎 핵심 포인트: 윌리엄 오닐의 투자법은 체계적이고 규칙 기반의 접근법입니다. 감정을 배제하고 데이터와 패턴을 중시하며, 빠른 손절매를 통해 리스크를 관리하는 것이 특징입니다. 성장주에 집중하되 기술적 분석을 통해 적절한 진입 시점을 찾는 것이 핵심입니다.
"""

    # 1. 시장 방향성(M) 먼저 체크
    is_uptrend, market_status_text = check_market_direction(market_index_symbol='^GSPC')
    
    if not is_uptrend:
        print("시장 전체가 하락/조정 추세이므로, 보수적인 접근이 필요합니다. 스캔을 종료합니다.")
    else:
        stocks_to_scan = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
        print(f"총 {len(stocks_to_scan)}개 종목에 대한 CAN SLIM 스캔을 시작합니다...\n")
        
        all_results = []
        for stock_symbol in stocks_to_scan:
            scanner = ONeilScanner(stock_symbol)
            result = scanner.run_scan()
            
            if result:
                # 콘솔 출력용
                print(f"===== {result['종목']} 리포트 =====")
                for key, (passed, value) in result['결과'].items():
                    print(f"  - {key}: {'통과' if passed else '실패'} ({value})")
                print(f"  ▶ 종합 점수: {result['종합점수']}")
                print(f"  ▶ 최종 판단: {result['최종판단']}")
                print("=" * 28 + "\n")
                
                # 데이터프레임 저장용 데이터 가공
                flat_result = {'종목': result['종목']}
                for key, (passed, value) in result['결과'].items():
                    flat_result[key] = f"{'통과' if passed else '실패'} ({value})"
                flat_result['종합점수'] = result['종합점수']
                flat_result['최종판단'] = result['최종판단']
                all_results.append(flat_result)

        if all_results:
            results_df = pd.DataFrame(all_results)
            passed_stocks = results_df[results_df['최종판단'] == '관심']['종목'].tolist()
            
            print("\n--- 최종 스캔 결과 ---")
            if passed_stocks:
                print(f"관심 종목 리스트: {', '.join(passed_stocks)}")
            else:
                print("현재 기준에 부합하는 관심 종목을 찾지 못했습니다.")

            # 구글 시트에 결과 업로드
            upload_to_gsheet(
                spreadsheet_name='주식 분석 리포트',
                philosophy_text=oneil_philosophy,
                market_status_text=market_status_text,
                results_df=results_df
            )
        else:
            print("\n분석할 데이터가 없어 구글 시트 업로드를 건너뜁니다.")
            
    print("\n[알림] 본 스캐너는 윌리엄 오닐의 투자 철학 중 정량적 부분을 구현한 보조 도구이며, 투자 추천이 아닙니다.") 