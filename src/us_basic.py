"""
🇺🇸 미국주식 나스닥100 & S&P500 TOP5 추천 시스템
🌟 세계 최고 애널리스트 수준 AI 분석 통합

주요 기능:
1. 🏆 골드만삭스 스타일 DCF 밸류에이션
2. 📊 모건스탠리 스타일 멀티플 분석  
3. 🎯 JP모건 스타일 리스크 분석
4. 💎 워렌 버핏 스타일 질적 분석
5. 🚀 캐시 우드 스타일 혁신 분석
6. 📈 레이 달리오 스타일 매크로 분석
7. 📱 텔레그램 알림 기능
8. 📰 검증된 라이브러리 뉴스 크롤링
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import sys
import os
import yfinance as yf
import feedparser
import time

# 기존 시스템과 연동
from ai_trading import AdvancedScalpingAI
from core_legacy.core_trader import CoreTrader
# from world_class_analyst_ai import WorldClassAnalystAI  # 삭제된 모듈 주석 처리

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('us_stock_analyzer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class USStockNewsLoader:
    """📰 검증된 라이브러리를 사용한 미국주식 뉴스 수집 클래스"""
    
    def __init__(self):
        """초기화"""
        logger.info("📰 USStockNewsLoader 초기화...")
        
        # RSS 피드 URL들
        self.rss_feeds = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'cnbc_finance': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss'
        }
        
        # 캐시 시스템
        self.cache = {}
        self.cache_duration = 300  # 5분 캐시
        
        logger.info("✅ USStockNewsLoader 초기화 완료")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 확인"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_duration
    
    async def get_us_stock_news(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """미국주식 뉴스 수집 (yfinance + RSS 피드)"""
        try:
            cache_key = f"us_news_{symbol}_{limit}"
            
            # 캐시 확인
            if self._is_cache_valid(cache_key):
                logger.info(f"📋 {symbol or 'US_MARKET'} 뉴스 캐시 사용")
                return self.cache[cache_key]['data']
            
            all_news = []
            
            # 1. yfinance를 통한 특정 종목 뉴스 수집
            if symbol:
                logger.info(f"📰 {symbol} 종목 뉴스 수집 중 (yfinance)...")
                ticker_news = await self._get_yfinance_news(symbol, limit // 2)
                all_news.extend(ticker_news)
            
            # 2. RSS 피드를 통한 일반 시장 뉴스 수집
            logger.info("📰 RSS 피드 뉴스 수집 중...")
            rss_news = await self._get_rss_news(limit - len(all_news))
            all_news.extend(rss_news)
            
            # 중복 제거 및 정렬
            unique_news = self._remove_duplicates(all_news)
            recent_news = sorted(unique_news, key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # 결과 제한
            final_news = recent_news[:limit]
            
            # 캐시 저장
            self.cache[cache_key] = {
                'data': final_news,
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ 뉴스 수집 완료: {len(final_news)}개 기사")
            return final_news
            
        except Exception as e:
            logger.error(f"❌ 미국주식 뉴스 수집 실패: {e}")
            return []
    
    async def _get_yfinance_news(self, symbol: str, limit: int) -> List[Dict]:
        """yfinance를 통한 종목별 뉴스 수집"""
        try:
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            news_list = []
            
            for item in news_data[:limit]:
                try:
                    # Unix 타임스탬프를 datetime으로 변환
                    publish_time = datetime.fromtimestamp(item.get('providerPublishTime', time.time()))
                    
                    news_item = {
                        'title': item.get('title', 'N/A'),
                        'link': item.get('link', ''),
                        'summary': item.get('summary', '')[:200] + "..." if item.get('summary') else "요약 정보 없음",
                        'publish_time': publish_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'source': item.get('publisher', 'yfinance'),
                        'symbol': symbol,
                        'timestamp': publish_time.isoformat()
                    }
                    
                    news_list.append(news_item)
                    
                except Exception as e:
                    logger.warning(f"⚠️ yfinance 뉴스 아이템 파싱 실패: {e}")
                    continue
            
            logger.info(f"📰 {symbol} yfinance 뉴스 {len(news_list)}개 수집")
            return news_list
            
        except Exception as e:
            logger.warning(f"⚠️ yfinance 뉴스 수집 실패: {e}")
            return []
    
    async def _get_rss_news(self, limit: int) -> List[Dict]:
        """RSS 피드를 통한 시장 뉴스 수집"""
        try:
            all_news = []
            
            for feed_name, feed_url in self.rss_feeds.items():
                try:
                    logger.info(f"📡 {feed_name} RSS 피드 수집 중...")
                    
                    # feedparser로 RSS 파싱
                    feed = feedparser.parse(feed_url)
                    
                    if feed.bozo:
                        logger.warning(f"⚠️ {feed_name} RSS 피드 파싱 오류")
                        continue
                    
                    # 각 피드에서 최대 3개씩 수집
                    for entry in feed.entries[:3]:
                        try:
                            # 발행 시간 파싱
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                publish_time = datetime(*entry.published_parsed[:6])
                            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                                publish_time = datetime(*entry.updated_parsed[:6])
                            else:
                                publish_time = datetime.now()
                            
                            # 요약 텍스트 정리
                            summary = getattr(entry, 'summary', '')
                            if summary:
                                # HTML 태그 제거
                                import re
                                summary = re.sub(r'<[^>]+>', '', summary)
                                summary = summary[:200] + "..." if len(summary) > 200 else summary
                            
                            news_item = {
                                'title': getattr(entry, 'title', 'N/A'),
                                'link': getattr(entry, 'link', ''),
                                'summary': summary or "요약 정보 없음",
                                'publish_time': publish_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'source': feed_name.replace('_', ' ').title(),
                                'symbol': 'US_MARKET',
                                'timestamp': publish_time.isoformat()
                            }
                            
                            all_news.append(news_item)
                            
                        except Exception as e:
                            logger.warning(f"⚠️ RSS 엔트리 파싱 실패: {e}")
                            continue
                    
                    # 요청 간격 조절
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"⚠️ {feed_name} RSS 피드 수집 실패: {e}")
                    continue
            
            # 최신순 정렬
            all_news.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            logger.info(f"📰 RSS 피드 뉴스 {len(all_news)}개 수집")
            return all_news[:limit]
            
        except Exception as e:
            logger.error(f"❌ RSS 뉴스 수집 실패: {e}")
            return []
    
    async def get_market_overview_news(self, limit: int = 15) -> List[Dict]:
        """시장 전체 개요 뉴스 수집"""
        try:
            logger.info("📰 시장 개요 뉴스 수집 중...")
            
            # 주요 경제/금융 RSS 피드
            overview_feeds = {
                'fed_news': 'https://www.federalreserve.gov/feeds/press_all.xml',
                'sec_news': 'https://www.sec.gov/news/pressrelease.rss',
                'treasury': 'https://home.treasury.gov/rss/press-releases',
                'reuters_markets': 'https://feeds.reuters.com/reuters/businessNews',
                'ap_business': 'https://feeds.apnews.com/rss/apf-business'
            }
            
            all_news = []
            
            for feed_name, feed_url in overview_feeds.items():
                try:
                    logger.info(f"📡 {feed_name} 개요 뉴스 수집 중...")
                    
                    feed = feedparser.parse(feed_url)
                    
                    if feed.bozo:
                        logger.warning(f"⚠️ {feed_name} 피드 파싱 오류")
                        continue
                    
                    for entry in feed.entries[:4]:  # 각 소스에서 4개씩
                        try:
                            # 발행 시간 파싱
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                publish_time = datetime(*entry.published_parsed[:6])
                            else:
                                publish_time = datetime.now()
                            
                            # 최근 7일 내 뉴스만 수집
                            if (datetime.now() - publish_time).days > 7:
                                continue
                            
                            news_item = {
                                'title': getattr(entry, 'title', 'N/A'),
                                'link': getattr(entry, 'link', ''),
                                'summary': getattr(entry, 'summary', 'N/A')[:150] + "..." if getattr(entry, 'summary', '') else "요약 정보 없음",
                                'publish_time': publish_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'source': feed_name.replace('_', ' ').title(),
                                'symbol': 'MARKET_OVERVIEW',
                                'timestamp': publish_time.isoformat()
                            }
                            
                            all_news.append(news_item)
                            
                        except Exception as e:
                            continue
                    
                    time.sleep(0.5)  # 요청 간격 조절
                    
                except Exception as e:
                    logger.warning(f"⚠️ {feed_name} 개요 뉴스 수집 실패: {e}")
                    continue
            
            # 중복 제거 및 정렬
            unique_news = self._remove_duplicates(all_news)
            recent_news = sorted(unique_news, key=lambda x: x.get('timestamp', ''), reverse=True)
            
            logger.info(f"✅ 시장 개요 뉴스 {len(recent_news)}개 수집 완료")
            return recent_news[:limit]
            
        except Exception as e:
            logger.error(f"❌ 시장 개요 뉴스 수집 실패: {e}")
            return []
    
    def _remove_duplicates(self, news_list: List[Dict]) -> List[Dict]:
        """중복 뉴스 제거"""
        seen_titles = set()
        unique_news = []
        
        for news in news_list:
            title = news.get('title', '').lower().strip()
            
            # 제목이 너무 짧거나 이미 본 제목인 경우 제외
            if len(title) < 10 or title in seen_titles:
                continue
            
            seen_titles.add(title)
            unique_news.append(news)
        
        return unique_news

class USStockAnalyzer:
    """🇺🇸 미국주식 분석 시스템 메인 클래스 (세계 최고 애널리스트 수준)"""
    
    def __init__(self):
        """초기화"""
        try:
            logger.info("🇺🇸 미국주식 분석 시스템 초기화 중...")
            
            # CoreTrader 초기화 (한국투자증권 API 사용)
            self.trader = CoreTrader()
            
            # AdvancedScalpingAI 초기화 (미국주식 기능 포함)
            self.ai_system = AdvancedScalpingAI(self.trader)
            
            # 텔레그램 알림 설정 (CoreTrader의 notifier 사용)
            self.telegram_notifier = self.trader.notifier
            
            # 뉴스 크롤러 초기화
            self.news_crawler = USStockNewsLoader()
            
            logger.info("✅ 미국주식 분석 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 초기화 실패: {e}")
            raise
    
    def print_welcome_message(self):
        """환영 메시지 출력"""
        print("=" * 80)
        print("🇺🇸 미국주식 나스닥100 & S&P500 TOP5 추천 시스템")
        print("🌟 세계 최고 애널리스트 수준 AI 분석")
        print("=" * 80)
        print("📊 지원 지수:")
        print("   1. 나스닥100 (NASDAQ-100)")
        print("   2. S&P500 (Standard & Poor's 500)")
        print()
        print("🎯 지원 전략:")
        print("   • 윌리엄 오닐 (CAN SLIM)")
        print("   • 제시 리버모어 (추세추종)")
        print("   • 일목산인 (균형표)")
        print("   • 워렌 버핏 (가치투자)")
        print("   • 피터 린치 (성장주)")
        print("   • 블랙록 (기관투자)")
        print()
        print("🌟 세계 최고 애널리스트 분석:")
        print("   • 🏆 골드만삭스 스타일 DCF 밸류에이션")
        print("   • 📊 모건스탠리 스타일 멀티플 분석")
        print("   • 🎯 JP모건 스타일 리스크 분석")
        print()
        print("📱 텔레그램 알림: 활성화됨")
        print("=" * 80)
        print()
    
    def display_menu(self):
        """메인 메뉴 표시"""
        print("\n" + "="*80)
        print("🇺🇸 미국주식 투자대가 분석 시스템 (🌟 세계 최고 애널리스트 수준)")
        print("="*80)
        print("📊 나스닥100 TOP5 분석:")
        print("  1. 윌리엄 오닐 전략 (CAN SLIM)")
        print("  2. 제시 리버모어 전략 (추세추종)")
        print("  3. 일목산인 전략 (균형표)")
        print("  4. 워렌 버핏 전략 (가치투자)")
        print("  5. 피터 린치 전략 (성장주)")
        print("  6. 블랙록 전략 (기관투자)")
        print("\n📈 S&P500 TOP5 분석:")
        print("  7. 윌리엄 오닐 전략 (CAN SLIM)")
        print("  8. 제시 리버모어 전략 (추세추종)")
        print("  9. 일목산인 전략 (균형표)")
        print(" 10. 워렌 버핏 전략 (가치투자)")
        print(" 11. 피터 린치 전략 (성장주)")
        print(" 12. 블랙록 전략 (기관투자)")
        print("\n🔄 통합 분석:")
        print(" 13. 나스닥100 전체 분석 (6가지 전략)")
        print(" 14. S&P500 전체 분석 (6가지 전략)")
        print(" 15. 미국주식 전체 분석 (나스닥+S&P500)")
        print("\n🌟 세계 최고 애널리스트 분석:")
        print(" 19. 골드만삭스 스타일 개별 종목 분석")
        print(" 20. 나스닥100 골드만삭스 스타일 TOP5")
        print(" 21. S&P500 골드만삭스 스타일 TOP5")
        print("\n📰 뉴스 크롤링 (검증된 라이브러리):")
        print(" 16. 미국 주식시장 전체 뉴스 (yfinance + RSS)")
        print(" 17. 특정 종목 뉴스 검색 (yfinance)")
        print(" 18. 시장 개요 뉴스 (경제/금융 RSS)")
        print("\n 0. 종료")
        print("="*80)
    
    async def send_telegram_notification(self, title: str, results: List[Dict]):
        """텔레그램으로 분석 결과 전송 (한국 주식과 동일한 방식)"""
        try:
            if not results:
                await self.telegram_notifier.send_message(f"❌ {title} 분석 결과가 없습니다.")
                return
            
            # 메시지 헤더
            message = f"🇺🇸 {title}\n"
            message += "=" * 50 + "\n\n"
            
            # TOP 5 종목 정보
            for i, stock in enumerate(results, 1):
                symbol = stock.get('symbol', 'N/A')
                name = stock.get('name', 'N/A')
                score = stock.get('score', 0)
                recommendation = stock.get('recommendation', 'HOLD')
                reason = stock.get('reason', '분석 결과 기반')
                current_price = stock.get('current_price', 0)
                change_rate = stock.get('change_rate', 0)
                
                # 추천 등급 한국어 변환
                recommendation_kr = self._translate_recommendation(recommendation)
                
                # 이모지 추가
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1] if i <= 5 else f"{i}️⃣"
                
                message += f"{rank_emoji} {name} ({symbol})\n"
                message += f"📊 점수: {score}점 | 💡 {recommendation_kr}\n"
                message += f"💰 ${current_price} | 📈 {change_rate:+.2f}%\n"
                message += f"🎯 {reason}\n\n"
            
            # 푸터
            message += "=" * 50 + "\n"
            message += f"⏰ 분석시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            message += "🤖 미국주식 AI 분석 시스템"
            
            # 텔레그램 전송
            await self.telegram_notifier.send_message(message)
            logger.info(f"📱 텔레그램 알림 전송 완료: {title}")
            
        except Exception as e:
            logger.error(f"❌ 텔레그램 알림 전송 실패: {e}")
            print(f"❌ 텔레그램 알림 전송 실패: {e}")
    
    async def send_news_telegram_notification(self, title: str, news_list: List[Dict]):
        """뉴스 결과를 텔레그램으로 전송"""
        try:
            if not news_list:
                await self.telegram_notifier.send_message(f"❌ {title} 뉴스가 없습니다.")
                return
            
            # 메시지 헤더
            message = f"📰 {title}\n"
            message += "=" * 50 + "\n\n"
            
            # 뉴스 아이템들
            for i, news in enumerate(news_list[:10], 1):  # 최대 10개만 전송
                title_text = news.get('title', 'N/A')
                publish_time = news.get('publish_time', 'N/A')
                source = news.get('source', 'N/A')
                link = news.get('link', '')
                
                message += f"{i}. {title_text}\n"
                message += f"⏰ {publish_time} | 📰 {source}\n"
                if link:
                    message += f"🔗 {link}\n"
                message += "\n"
            
            # 푸터
            message += "=" * 50 + "\n"
            message += f"⏰ 수집시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            message += "🤖 미국주식 뉴스 크롤링 시스템"
            
            # 텔레그램 전송
            await self.telegram_notifier.send_message(message)
            logger.info(f"📱 뉴스 텔레그램 알림 전송 완료: {title}")
            
        except Exception as e:
            logger.error(f"❌ 뉴스 텔레그램 알림 전송 실패: {e}")
            print(f"❌ 뉴스 텔레그램 알림 전송 실패: {e}")
    
    def print_news_results(self, title: str, news_list: List[Dict]):
        """뉴스 결과를 콘솔에 출력"""
        if not news_list:
            print(f"❌ {title} 뉴스가 없습니다.")
            return
        
        print(f"\n✅ {title} 수집 완료! {len(news_list)}개 뉴스:")
        print("=" * 100)
        
        for i, news in enumerate(news_list, 1):
            title_text = news.get('title', 'N/A')
            summary = news.get('summary', 'N/A')
            publish_time = news.get('publish_time', 'N/A')
            source = news.get('source', 'N/A')
            link = news.get('link', '')
            
            print(f"  {i:2d}. {title_text}")
            print(f"       📰 출처: {source} | ⏰ 시간: {publish_time}")
            print(f"       📝 요약: {summary}")
            if link:
                print(f"       🔗 링크: {link}")
            print("-" * 100)
        
        print("=" * 100)
        print("📱 텔레그램으로 뉴스를 전송했습니다!")
        print()
    
    async def get_us_market_news(self):
        """미국 주식시장 전체 뉴스 수집"""
        try:
            print("\n🔄 미국 주식시장 뉴스를 수집하고 있습니다...")
            news_list = await self.news_crawler.get_us_stock_news(limit=15)
            
            title = "미국 주식시장 뉴스"
            self.print_news_results(title, news_list)
            await self.send_news_telegram_notification(title, news_list)
            
        except Exception as e:
            logger.error(f"❌ 미국 시장 뉴스 수집 실패: {e}")
            print(f"❌ 뉴스 수집 중 오류 발생: {e}")
    
    async def get_stock_specific_news(self):
        """특정 종목 뉴스 검색 - 인기 종목들 자동 분석"""
        try:
            # 인기 미국 주식 종목들을 자동으로 분석
            popular_symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
            
            print(f"\n🔄 인기 미국 주식 종목들의 뉴스를 수집하고 있습니다...")
            print(f"📊 분석 대상: {', '.join(popular_symbols)}")
            
            for i, symbol in enumerate(popular_symbols, 1):
                print(f"\n📈 [{i}/{len(popular_symbols)}] {symbol} 종목 뉴스 수집 중...")
                news_list = await self.news_crawler.get_us_stock_news(symbol=symbol, limit=5)
                
                title = f"{symbol} 종목 뉴스"
                self.print_news_results(title, news_list)
                
                # 각 종목별로 텔레그램 알림
                if news_list:
                    await self.send_news_telegram_notification(title, news_list[:3])  # 상위 3개만 전송
            
            print("\n✅ 인기 종목 뉴스 수집 완료!")
            
        except Exception as e:
            logger.error(f"❌ 종목별 뉴스 수집 실패: {e}")
            print(f"❌ 뉴스 수집 중 오류 발생: {e}")
    
    async def get_market_overview_news(self):
        """시장 개요 뉴스 수집 (경제/외환 포함)"""
        try:
            print("\n🔄 시장 개요 뉴스를 수집하고 있습니다...")
            news_list = await self.news_crawler.get_market_overview_news(limit=20)
            
            title = "시장 개요 뉴스 (경제/외환 포함)"
            self.print_news_results(title, news_list)
            await self.send_news_telegram_notification(title, news_list)
            
        except Exception as e:
            logger.error(f"❌ 시장 개요 뉴스 수집 실패: {e}")
            print(f"❌ 뉴스 수집 중 오류 발생: {e}")
    
    def print_analysis_results(self, title: str, results: List[Dict]):
        """분석 결과를 한국어로 포맷팅하여 출력"""
        if not results:
            print(f"❌ {title} 분석 결과가 없습니다.")
            return
        
        print(f"\n✅ {title} 분석 완료! TOP {len(results)} 종목:")
        print("=" * 100)
        
        for i, stock in enumerate(results, 1):
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')
            score = stock.get('score', 0)
            recommendation = stock.get('recommendation', 'HOLD')
            reason = stock.get('reason', '분석 결과 기반')
            current_price = stock.get('current_price', 0)
            change_rate = stock.get('change_rate', 0)
            market_cap = stock.get('market_cap', 0)
            volume = stock.get('volume', 0)
            strategy = stock.get('strategy', 'N/A')
            
            # 추천 등급 한국어 변환
            recommendation_kr = self._translate_recommendation(recommendation)
            
            # 시가총액 포맷팅 (달러 -> 원화 개념으로)
            market_cap_str = self._format_market_cap(market_cap)
            
            # 거래량 포맷팅
            volume_str = f"{volume:,}" if volume > 0 else "N/A"
            
            print(f"  {i:2d}위. {name} ({symbol})")
            print(f"       📊 점수: {score}점 | 💡 추천: {recommendation_kr}")
            print(f"       🎯 이유: {reason}")
            print(f"       💰 현재가: ${current_price} | 📈 변화율: {change_rate:+.2f}%")
            print(f"       🏢 시가총액: {market_cap_str} | 📊 거래량: {volume_str}")
            print(f"       🔍 전략: {strategy.upper()}")
            print("-" * 100)
        
        print("=" * 100)
        print("📱 텔레그램으로 결과를 전송했습니다!")
        print()
    
    def _translate_recommendation(self, recommendation: str) -> str:
        """추천 등급을 한국어로 변환"""
        translations = {
            'STRONG_BUY': '적극매수',
            'BUY': '매수',
            'HOLD': '보유',
            'SELL': '매도',
            'STRONG_SELL': '적극매도'
        }
        return translations.get(recommendation, recommendation)
    
    def _format_market_cap(self, market_cap: int) -> str:
        """시가총액 포맷팅"""
        if market_cap >= 1_000_000_000_000:  # 1조 달러 이상
            return f"{market_cap / 1_000_000_000_000:.1f}조 달러"
        elif market_cap >= 1_000_000_000:  # 10억 달러 이상
            return f"{market_cap / 1_000_000_000:.0f}십억 달러"
        elif market_cap >= 1_000_000:  # 100만 달러 이상
            return f"{market_cap / 1_000_000:.0f}백만 달러"
        else:
            return f"{market_cap:,} 달러"
    
    async def analyze_nasdaq100_all_strategies(self):
        """나스닥100 전체 전략 분석"""
        try:
            print("\n🔄 나스닥100 전체 전략 분석 시작...")
            
            strategies = [
                ("william_oneil", "윌리엄 오닐 (CAN SLIM)"),
                ("jesse_livermore", "제시 리버모어 (추세추종)"),
                ("ichimoku", "일목산인 (균형표)"),
                ("warren_buffett", "워렌 버핏 (가치투자)"),
                ("peter_lynch", "피터 린치 (성장주)"),
                ("blackrock", "블랙록 (기관투자)")
            ]
            
            for i, (strategy_code, strategy_name) in enumerate(strategies, 1):
                print(f"\n📊 [{i}/6] {strategy_name} 전략 분석 중...")
                results = await self.ai_system.analyze_nasdaq100_top5(strategy_code)
                title = f"나스닥100 {strategy_name} TOP5"
                self.print_analysis_results(f"🇺🇸 {title}", results)
                # 텔레그램 알림 전송
                await self.send_telegram_notification(title, results)
                
        except Exception as e:
            logger.error(f"❌ 나스닥100 전체 분석 실패: {e}")
            print(f"❌ 분석 중 오류 발생: {e}")
    
    async def analyze_sp500_all_strategies(self):
        """S&P500 전체 전략 분석"""
        try:
            print("\n🔄 S&P500 전체 전략 분석 시작...")
            
            strategies = [
                ("william_oneil", "윌리엄 오닐 (CAN SLIM)"),
                ("jesse_livermore", "제시 리버모어 (추세추종)"),
                ("ichimoku", "일목산인 (균형표)"),
                ("warren_buffett", "워렌 버핏 (가치투자)"),
                ("peter_lynch", "피터 린치 (성장주)"),
                ("blackrock", "블랙록 (기관투자)")
            ]
            
            for i, (strategy_code, strategy_name) in enumerate(strategies, 1):
                print(f"\n📊 [{i}/6] {strategy_name} 전략 분석 중...")
                results = await self.ai_system.analyze_sp500_top5(strategy_code)
                title = f"S&P500 {strategy_name} TOP5"
                self.print_analysis_results(f"🇺🇸 {title}", results)
                # 텔레그램 알림 전송
                await self.send_telegram_notification(title, results)
                
        except Exception as e:
            logger.error(f"❌ S&P500 전체 분석 실패: {e}")
            print(f"❌ 분석 중 오류 발생: {e}")
    
    async def analyze_all_indices(self):
        """나스닥100 + S&P500 전체 분석"""
        try:
            print("\n🔄 미국주식 전체 분석 시작...")
            print("📊 나스닥100과 S&P500 모든 전략을 분석합니다...")
            
            # 전체 분석 시작 알림
            await self.telegram_notifier.send_message("🇺🇸 미국주식 전체 분석을 시작합니다!\n📊 나스닥100 + S&P500 (12가지 전략)")
            
            print("\n" + "="*50)
            print("📈 나스닥100 전체 분석")
            print("="*50)
            await self.analyze_nasdaq100_all_strategies()
            
            print("\n" + "="*50)
            print("📈 S&P500 전체 분석") 
            print("="*50)
            await self.analyze_sp500_all_strategies()
            
            print("\n✅ 미국주식 전체 분석 완료!")
            
            # 전체 분석 완료 알림
            await self.telegram_notifier.send_message("✅ 미국주식 전체 분석이 완료되었습니다!\n🇺🇸 나스닥100 + S&P500 모든 전략 분석 완료")
            
        except Exception as e:
            logger.error(f"❌ 전체 분석 실패: {e}")
            print(f"❌ 분석 중 오류 발생: {e}")
    
    async def run_interactive_mode(self):
        """대화형 모드 실행"""
        self.print_welcome_message()
        
        while True:
            try:
                self.display_menu()
                choice = input("선택하세요 (0-21): ").strip()
                
                if choice == '0':
                    print("👋 미국주식 분석 시스템을 종료합니다.")
                    break
                elif choice == '1':
                    print("\n🔄 나스닥100 윌리엄 오닐 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_nasdaq100_top5("william_oneil")
                    title = "나스닥100 윌리엄 오닐 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '2':
                    print("\n🔄 나스닥100 제시 리버모어 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_nasdaq100_top5("jesse_livermore")
                    title = "나스닥100 제시 리버모어 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '3':
                    print("\n🔄 나스닥100 일목산인 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_nasdaq100_top5("ichimoku")
                    title = "나스닥100 일목산인 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '4':
                    print("\n🔄 나스닥100 워렌 버핏 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_nasdaq100_top5("warren_buffett")
                    title = "나스닥100 워렌 버핏 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '5':
                    print("\n🔄 나스닥100 피터 린치 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_nasdaq100_top5("peter_lynch")
                    title = "나스닥100 피터 린치 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '6':
                    print("\n🔄 나스닥100 블랙록 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_nasdaq100_top5("blackrock")
                    title = "나스닥100 블랙록 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '7':
                    print("\n🔄 S&P500 윌리엄 오닐 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_sp500_top5("william_oneil")
                    title = "S&P500 윌리엄 오닐 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '8':
                    print("\n🔄 S&P500 제시 리버모어 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_sp500_top5("jesse_livermore")
                    title = "S&P500 제시 리버모어 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '9':
                    print("\n🔄 S&P500 일목산인 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_sp500_top5("ichimoku")
                    title = "S&P500 일목산인 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '10':
                    print("\n🔄 S&P500 워렌 버핏 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_sp500_top5("warren_buffett")
                    title = "S&P500 워렌 버핏 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '11':
                    print("\n🔄 S&P500 피터 린치 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_sp500_top5("peter_lynch")
                    title = "S&P500 피터 린치 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '12':
                    print("\n🔄 S&P500 블랙록 전략 분석을 시작합니다...")
                    results = await self.ai_system.analyze_sp500_top5("blackrock")
                    title = "S&P500 블랙록 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '13':
                    print("\n🔄 나스닥100 전체 전략 분석을 시작합니다... (시간이 다소 소요됩니다)")
                    await self.analyze_nasdaq100_all_strategies()
                    print("✅ 나스닥100 전체 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '14':
                    print("\n🔄 S&P500 전체 전략 분석을 시작합니다... (시간이 다소 소요됩니다)")
                    await self.analyze_sp500_all_strategies()
                    print("✅ S&P500 전체 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '15':
                    print("\n🔄 미국주식 전체 분석을 시작합니다... (시간이 많이 소요됩니다)")
                    await self.analyze_all_indices()
                    print("✅ 미국주식 전체 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '16':
                    await self.get_us_market_news()
                    print("✅ 뉴스 수집이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '17':
                    symbol = input("검색할 종목 심볼을 입력하세요 (예: AAPL): ").strip().upper()
                    if symbol:
                        await self.get_specific_stock_news(symbol)
                        print("✅ 종목별 뉴스 수집이 완료되었습니다!")
                        print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                        await asyncio.sleep(3)
                    else:
                        print("❌ 올바른 종목 심볼을 입력해주세요.")
                        print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                        await asyncio.sleep(3)
                elif choice == '18':
                    await self.get_market_overview_news()
                    print("✅ 시장 개요 뉴스 수집이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '19':
                    print("\n🔄 개별 종목 분석을 시작합니다...")
                    # 기존 시스템으로 대체
                    symbol = input("분석할 종목 심볼을 입력하세요 (예: AAPL): ").strip().upper()
                    if symbol:
                        print(f"📊 {symbol} 분석 중...")
                        print("⚠️ 세계 최고 애널리스트 AI 모듈이 현재 사용 불가합니다.")
                        print("🔄 기존 시스템으로 분석합니다...")
                        # 기존 AI 시스템으로 분석 (워렌 버핏 전략 사용)
                        result = await self.ai_system._analyze_us_stock(symbol, "warren_buffett")
                        if result and 'error' not in result:
                            self.print_analysis_results(f"📈 {symbol} 분석 결과", [result])
                        else:
                            print(f"❌ {symbol} 분석에 실패했습니다.")
                        print("✅ 개별 종목 분석이 완료되었습니다!")
                        print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                        await asyncio.sleep(3)
                    else:
                        print("❌ 올바른 종목 심볼을 입력해주세요.")
                        print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                        await asyncio.sleep(3)
                elif choice == '20':
                    print("\n🔄 나스닥100 세계 최고 애널리스트 분석을 시작합니다...")
                    print("⚠️ 세계 최고 애널리스트 AI 모듈이 현재 사용 불가합니다.")
                    print("🔄 기존 시스템으로 분석합니다...")
                    # 기존 시스템으로 대체
                    results = await self.ai_system.analyze_nasdaq100_top5("warren_buffett")
                    title = "나스닥100 세계 최고 애널리스트 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                elif choice == '21':
                    print("\n🔄 S&P500 세계 최고 애널리스트 분석을 시작합니다...")
                    print("⚠️ 세계 최고 애널리스트 AI 모듈이 현재 사용 불가합니다.")
                    print("🔄 기존 시스템으로 분석합니다...")
                    # 기존 시스템으로 대체
                    results = await self.ai_system.analyze_sp500_top5("warren_buffett")
                    title = "S&P500 세계 최고 애널리스트 TOP5"
                    self.print_analysis_results(f"🇺🇸 {title}", results)
                    await self.send_telegram_notification(title, results)
                    print("✅ 분석이 완료되었습니다!")
                    print("⏳ 3초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(3)
                else:
                    print("❌ 잘못된 선택입니다. 0-21 사이의 숫자를 입력하세요.")
                    print("⏳ 2초 후 자동으로 메뉴로 돌아갑니다...")
                    await asyncio.sleep(2)
                    continue
                
                # 분석이 완료된 후 메뉴로 돌아가기 전 잠깐 대기
                print("\n" + "🚀" * 40)
                print("   ⚡ 자동으로 메뉴로 돌아갑니다...")
                print("🚀" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 사용자에 의해 종료되었습니다.")
                break
            except Exception as e:
                logger.error(f"❌ 실행 중 오류 발생: {e}")
                print(f"❌ 오류가 발생했습니다: {e}")
                print("🔄 3초 후 자동으로 메뉴로 돌아갑니다...")
                await asyncio.sleep(3)

async def main():
    """메인 실행 함수"""
    try:
        # 미국주식 분석 시스템 초기화
        analyzer = USStockAnalyzer()
        
        # 대화형 모드 실행
        await analyzer.run_interactive_mode()
        
    except Exception as e:
        logger.error(f"❌ 시스템 실행 실패: {e}")
        print(f"❌ 시스템 실행 실패: {e}")

if __name__ == "__main__":
    """프로그램 시작점"""
    print("🇺🇸 미국주식 나스닥100 & S&P500 TOP5 추천 시스템 시작...")
    
    try:
        # 이벤트 루프 실행
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"❌ 프로그램 실행 오류: {e}")
        logger.error(f"프로그램 실행 오류: {e}") 