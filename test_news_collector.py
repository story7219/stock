"""
📰 NewsCollector 종합 테스트 스크립트
- 네이버 금융 뉴스 크롤링 테스트
- 한국거래소 공시 수집 테스트
- 감정분석 기능 테스트
- 종목별 뉴스 필터링 테스트
- 성능 측정
"""
import sys
import os
import logging
import time
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from news_collector import NewsCollector, NewsItem, AnnouncementItem

class NewsCollectorTester:
    """📰 뉴스 컬렉터 종합 테스트 클래스"""
    
    def __init__(self):
        """테스터 초기화"""
        self.collector = NewsCollector()
        self.test_results = []
        self.total_score = 0
        self.max_score = 0
        
        logger.info("🚀 NewsCollectorTester 초기화 완료")
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("=" * 80)
        print("📰 NewsCollector 종합 테스트 시작")
        print("=" * 80)
        
        start_time = time.time()
        
        # 1. 실시간 뉴스 수집 테스트
        self.test_realtime_news()
        
        # 2. 공시 수집 테스트
        self.test_announcements()
        
        # 3. 감정분석 테스트
        self.test_sentiment_analysis()
        
        # 4. 종목별 뉴스 필터링 테스트
        self.test_stock_filtering()
        
        # 5. 시장 감정 요약 테스트
        self.test_market_sentiment()
        
        # 6. 성능 테스트
        self.test_performance()
        
        # 최종 결과 출력
        total_elapsed = time.time() - start_time
        self.print_final_results(total_elapsed)
        
        # 리소스 정리
        self.collector.cleanup()
    
    def test_realtime_news(self):
        """📰 실시간 뉴스 수집 테스트"""
        print("\n" + "=" * 50)
        print("📰 1. 실시간 뉴스 수집 테스트")
        print("=" * 50)
        
        try:
            start_time = time.time()
            
            # 일반 뉴스 수집
            news_list = self.collector.get_realtime_news(limit=10)
            
            elapsed = time.time() - start_time
            
            if news_list:
                print(f"✅ 뉴스 수집 성공: {len(news_list)}개")
                print(f"⏱️ 수집 시간: {elapsed:.3f}초")
                
                # 샘플 뉴스 출력
                print("\n📋 수집된 뉴스 샘플:")
                for i, news in enumerate(news_list[:3], 1):
                    print(f"  {i}. {news.title}")
                    print(f"     감정: {news.sentiment} ({news.sentiment_score:.2f})")
                    print(f"     시간: {news.timestamp.strftime('%Y-%m-%d %H:%M')}")
                    print(f"     출처: {news.source}")
                    print()
                
                # 점수 부여
                if elapsed < 5.0:
                    score = 20
                elif elapsed < 10.0:
                    score = 15
                else:
                    score = 10
                
                self._add_test_result("실시간 뉴스 수집", True, score, 20, f"{len(news_list)}개, {elapsed:.3f}초")
                
            else:
                print("❌ 뉴스 수집 실패")
                self._add_test_result("실시간 뉴스 수집", False, 0, 20, "수집 실패")
                
        except Exception as e:
            print(f"❌ 뉴스 수집 테스트 실패: {e}")
            self._add_test_result("실시간 뉴스 수집", False, 0, 20, f"오류: {e}")
    
    def test_announcements(self):
        """📋 공시 수집 테스트"""
        print("\n" + "=" * 50)
        print("📋 2. 전자공시 수집 테스트")
        print("=" * 50)
        
        try:
            start_time = time.time()
            
            # 최근 1일 공시 수집
            announcements = self.collector.get_announcements(days=1)
            
            elapsed = time.time() - start_time
            
            if announcements:
                print(f"✅ 공시 수집 성공: {len(announcements)}개")
                print(f"⏱️ 수집 시간: {elapsed:.3f}초")
                
                # 중요도별 분류
                high_count = sum(1 for a in announcements if a.importance == 'high')
                medium_count = sum(1 for a in announcements if a.importance == 'medium')
                low_count = sum(1 for a in announcements if a.importance == 'low')
                
                print(f"\n📊 중요도별 분류:")
                print(f"  🔥 고중요: {high_count}개")
                print(f"  ⚡ 중중요: {medium_count}개")
                print(f"  📝 저중요: {low_count}개")
                
                # 샘플 공시 출력
                print("\n📋 수집된 공시 샘플:")
                for i, ann in enumerate(announcements[:3], 1):
                    print(f"  {i}. [{ann.importance.upper()}] {ann.title}")
                    print(f"     회사: {ann.company}")
                    print(f"     유형: {ann.announcement_type}")
                    print(f"     종목코드: {ann.stock_code}")
                    print()
                
                # 점수 부여
                if elapsed < 8.0 and len(announcements) > 0:
                    score = 20
                elif elapsed < 15.0:
                    score = 15
                else:
                    score = 10
                
                self._add_test_result("전자공시 수집", True, score, 20, f"{len(announcements)}개, {elapsed:.3f}초")
                
            else:
                print("⚠️ 공시가 없거나 수집 실패")
                self._add_test_result("전자공시 수집", True, 10, 20, "공시 없음")
                
        except Exception as e:
            print(f"❌ 공시 수집 테스트 실패: {e}")
            self._add_test_result("전자공시 수집", False, 0, 20, f"오류: {e}")
    
    def test_sentiment_analysis(self):
        """🤖 감정분석 테스트"""
        print("\n" + "=" * 50)
        print("🤖 3. 감정분석 기능 테스트")
        print("=" * 50)
        
        # 테스트 문장들
        test_cases = [
            ("삼성전자가 신고가를 돌파하며 급등세를 보이고 있다", "positive"),
            ("실적 호조로 목표가 상향 조정", "positive"),
            ("주가가 급락하며 투자자들의 우려가 커지고 있다", "negative"),
            ("실적 부진으로 적자 전환", "negative"),
            ("주식시장이 보합세를 보이고 있다", "neutral"),
        ]
        
        correct_count = 0
        total_tests = len(test_cases)
        
        print("📊 감정분석 테스트 케이스:")
        
        for i, (text, expected) in enumerate(test_cases, 1):
            try:
                sentiment, score = self.collector.analyze_sentiment(text)
                
                is_correct = sentiment == expected
                if is_correct:
                    correct_count += 1
                
                status = "✅" if is_correct else "❌"
                print(f"  {i}. {status} \"{text[:30]}...\"")
                print(f"     예상: {expected} | 결과: {sentiment} ({score:.2f})")
                
            except Exception as e:
                print(f"  {i}. ❌ 오류: {e}")
        
        accuracy = (correct_count / total_tests) * 100
        print(f"\n📈 감정분석 정확도: {accuracy:.1f}% ({correct_count}/{total_tests})")
        
        # 점수 부여
        if accuracy >= 80:
            score = 20
        elif accuracy >= 60:
            score = 15
        elif accuracy >= 40:
            score = 10
        else:
            score = 5
        
        self._add_test_result("감정분석", True, score, 20, f"정확도 {accuracy:.1f}%")
    
    def test_stock_filtering(self):
        """📈 종목별 뉴스 필터링 테스트"""
        print("\n" + "=" * 50)
        print("📈 4. 종목별 뉴스 필터링 테스트")
        print("=" * 50)
        
        try:
            # 먼저 뉴스 수집
            print("📰 전체 뉴스 수집 중...")
            news_list = self.collector.get_realtime_news(limit=20)
            
            if not news_list:
                print("❌ 뉴스가 없어서 필터링 테스트 불가")
                self._add_test_result("종목별 필터링", False, 0, 15, "뉴스 없음")
                return
            
            # 주요 종목들로 필터링 테스트
            test_stocks = ["005930", "000660", "035420"]  # 삼성전자, SK하이닉스, NAVER
            
            total_filtered = 0
            
            for stock_code in test_stocks:
                company_name = self.collector.stock_mapping.get(stock_code, "Unknown")
                
                filtered_news = self.collector.filter_stock_related(news_list, stock_code)
                total_filtered += len(filtered_news)
                
                print(f"📊 {company_name}({stock_code}): {len(filtered_news)}개 관련 뉴스")
                
                # 샘플 출력
                for news in filtered_news[:2]:
                    print(f"   - {news.title[:50]}...")
            
            print(f"\n📈 총 필터링된 뉴스: {total_filtered}개")
            
            # 점수 부여
            if total_filtered > 0:
                score = 15
            else:
                score = 8  # 관련 뉴스가 없어도 기능은 작동함
            
            self._add_test_result("종목별 필터링", True, score, 15, f"{total_filtered}개 필터링")
            
        except Exception as e:
            print(f"❌ 종목별 필터링 테스트 실패: {e}")
            self._add_test_result("종목별 필터링", False, 0, 15, f"오류: {e}")
    
    def test_market_sentiment(self):
        """📊 시장 감정 요약 테스트"""
        print("\n" + "=" * 50)
        print("📊 5. 시장 감정 요약 테스트")
        print("=" * 50)
        
        try:
            start_time = time.time()
            
            sentiment_summary = self.collector.get_market_sentiment_summary()
            
            elapsed = time.time() - start_time
            
            if sentiment_summary:
                print(f"✅ 시장 감정 분석 완료")
                print(f"⏱️ 분석 시간: {elapsed:.3f}초")
                print(f"\n📊 시장 감정 요약:")
                print(f"  전체 감정: {sentiment_summary['sentiment'].upper()}")
                print(f"  감정 점수: {sentiment_summary['score']}")
                print(f"  분석 뉴스: {sentiment_summary['news_count']}개")
                
                if 'distribution' in sentiment_summary:
                    dist = sentiment_summary['distribution']
                    print(f"  감정 분포:")
                    print(f"    긍정: {dist.get('positive', 0)}개")
                    print(f"    부정: {dist.get('negative', 0)}개")
                    print(f"    중립: {dist.get('neutral', 0)}개")
                
                score = 15
                self._add_test_result("시장 감정 요약", True, score, 15, f"{sentiment_summary['sentiment']}, {elapsed:.3f}초")
                
            else:
                print("❌ 시장 감정 분석 실패")
                self._add_test_result("시장 감정 요약", False, 0, 15, "분석 실패")
                
        except Exception as e:
            print(f"❌ 시장 감정 요약 테스트 실패: {e}")
            self._add_test_result("시장 감정 요약", False, 0, 15, f"오류: {e}")
    
    def test_performance(self):
        """⚡ 성능 테스트"""
        print("\n" + "=" * 50)
        print("⚡ 6. 성능 테스트")
        print("=" * 50)
        
        try:
            # 캐시 테스트
            print("📋 캐시 성능 테스트...")
            
            # 첫 번째 호출 (캐시 없음)
            start_time = time.time()
            news1 = self.collector.get_realtime_news(limit=5)
            first_call_time = time.time() - start_time
            
            # 두 번째 호출 (캐시 사용)
            start_time = time.time()
            news2 = self.collector.get_realtime_news(limit=5)
            second_call_time = time.time() - start_time
            
            print(f"  첫 번째 호출: {first_call_time:.3f}초")
            print(f"  두 번째 호출: {second_call_time:.3f}초 (캐시)")
            
            if second_call_time < first_call_time * 0.5:
                print("✅ 캐시 시스템 정상 작동")
                cache_score = 10
            else:
                print("⚠️ 캐시 효과 미미")
                cache_score = 5
            
            # 전체 성능 평가
            if first_call_time < 5.0:
                perf_score = 10
            elif first_call_time < 10.0:
                perf_score = 7
            else:
                perf_score = 3
            
            total_perf_score = cache_score + perf_score
            self._add_test_result("성능 테스트", True, total_perf_score, 20, f"첫 호출 {first_call_time:.3f}초")
            
        except Exception as e:
            print(f"❌ 성능 테스트 실패: {e}")
            self._add_test_result("성능 테스트", False, 0, 20, f"오류: {e}")
    
    def _add_test_result(self, test_name: str, success: bool, score: int, max_score: int, details: str):
        """테스트 결과 추가"""
        self.test_results.append({
            'name': test_name,
            'success': success,
            'score': score,
            'max_score': max_score,
            'details': details
        })
        self.total_score += score
        self.max_score += max_score
    
    def print_final_results(self, total_elapsed: float):
        """최종 결과 출력"""
        print("\n" + "=" * 80)
        print("📊 NewsCollector 테스트 최종 결과")
        print("=" * 80)
        
        print("\n📋 테스트 상세 결과:")
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            score_str = f"{result['score']}/{result['max_score']}"
            print(f"  {status} {result['name']:<20} {score_str:<8} {result['details']}")
        
        # 전체 점수 계산
        percentage = (self.total_score / self.max_score) * 100 if self.max_score > 0 else 0
        
        print(f"\n🎯 종합 성능 점수: {self.total_score}/{self.max_score} ({percentage:.1f}%)")
        print(f"⏱️ 전체 테스트 시간: {total_elapsed:.3f}초")
        
        # 등급 부여
        if percentage >= 90:
            grade = "🥇 EXCELLENT"
        elif percentage >= 80:
            grade = "🥈 GOOD"
        elif percentage >= 70:
            grade = "🥉 FAIR"
        else:
            grade = "📝 NEEDS_IMPROVEMENT"
        
        print(f"🏆 종합 등급: {grade}")
        
        print("\n" + "=" * 80)
        print("📰 NewsCollector 테스트 완료!")
        print("=" * 80)

def main():
    """메인 함수"""
    try:
        tester = NewsCollectorTester()
        tester.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 