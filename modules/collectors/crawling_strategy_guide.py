#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: crawling_strategy_guide.py
목적: World-Class 크롤링 라이브러리 선택 가이드 및 최적화 전략
Author: World-Class Trading AI System
Created: 2025-07-12
Version: 1.0.0

🎯 크롤링 라이브러리 선택 가이드:
- 정적 페이지: requests + BeautifulSoup (빠르고 간단)
- 동적 페이지: Selenium (JavaScript 렌더링 필수)
- 대규모 크롤링: Scrapy (비동기 처리, 확장성)
- 보안 우회: cloudscraper (Cloudflare 등 우회)
- 고성능 비동기: aiohttp (동시 100개+ 요청)
"""

from __future__ import annotations
import asyncio
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict
import List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import field
from enum import Enum

import requests
from bs4 import BeautifulSoup
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import aiohttp
import cloudscraper
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PageType(Enum):
    """페이지 타입 분류"""
    STATIC = "static"           # 정적 HTML
    DYNAMIC = "dynamic"         # JavaScript 렌더링
    PROTECTED = "protected"     # Cloudflare 등 보호
    API = "api"                 # REST API
    MASS = "mass"               # 대규모 수집

class CrawlingLibrary(Enum):
    """크롤링 라이브러리 분류"""
    REQUESTS_BS4 = "requests_beautifulsoup"
    SELENIUM = "selenium"
    CLOUDSCRAPER = "cloudscraper"
    AIOHTTP = "aiohttp"
    SCRAPY = "scrapy"

@dataclass
class LibraryStrategy:
    """라이브러리 선택 전략"""
    primary: CrawlingLibrary
    fallback: Optional[CrawlingLibrary] = None
    performance_mode: str = "single"  # single, async, multiprocessing
    anti_bot: List[str] = field(default_factory=lambda: ["headers_randomization"])

class CrawlingStrategyGuide:
    """World-Class 크롤링 전략 가이드"""

    def __init__(self):
        self.strategies = self._initialize_strategies()
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        ]

    def _initialize_strategies(self) -> Dict[PageType, LibraryStrategy]:
        """전략 초기화"""
        return {
            PageType.STATIC: LibraryStrategy(
                primary=CrawlingLibrary.REQUESTS_BS4,
                fallback=CrawlingLibrary.AIOHTTP,
                performance_mode="async",
                anti_bot=["headers_randomization", "rate_limiting"]
            ),
            PageType.DYNAMIC: LibraryStrategy(
                primary=CrawlingLibrary.SELENIUM,
                fallback=CrawlingLibrary.CLOUDSCRAPER,
                performance_mode="single",
                anti_bot=["user_agent_rotation", "proxy_rotation", "headless_mode"]
            ),
            PageType.PROTECTED: LibraryStrategy(
                primary=CrawlingLibrary.CLOUDSCRAPER,
                fallback=CrawlingLibrary.SELENIUM,
                performance_mode="single",
                anti_bot=["cloudflare_bypass", "proxy_rotation", "session_management"]
            ),
            PageType.API: LibraryStrategy(
                primary=CrawlingLibrary.REQUESTS_BS4,
                fallback=CrawlingLibrary.AIOHTTP,
                performance_mode="async",
                anti_bot=["rate_limiting", "retry_logic"]
            ),
            PageType.MASS: LibraryStrategy(
                primary=CrawlingLibrary.AIOHTTP,
                fallback=CrawlingLibrary.SCRAPY,
                performance_mode="multiprocessing",
                anti_bot=["proxy_pool", "session_rotation", "distributed_crawling"]
            )
        }

    def select_strategy(self, url: str, page_type: PageType, volume: str = "low") -> LibraryStrategy:
        """용도별 최적 전략 선택"""

        # URL 패턴 분석
        if "cloudflare" in url.lower() or "krx.co.kr" in url:
            page_type = PageType.PROTECTED
        elif "api" in url.lower() or "json" in url.lower():
            page_type = PageType.API
        elif volume == "high" or "mass" in url.lower():
            page_type = PageType.MASS

        strategy = self.strategies[page_type]

        # 볼륨에 따른 성능 모드 조정
        if volume == "high":
            strategy.performance_mode = "multiprocessing"
        elif volume == "medium":
            strategy.performance_mode = "async"

        return strategy

    def get_implementation_guide(self, strategy: LibraryStrategy) -> Dict[str, Any]:
        """구현 가이드 제공"""

        guides = {
            CrawlingLibrary.REQUESTS_BS4: {
                "description": "정적 페이지 크롤링 - 빠르고 간단",
                "pros": ["빠른 속도", "간단한 사용법", "안정성"],
                "cons": ["동적 콘텐츠 불가", "JavaScript 렌더링 없음"],
                "best_for": ["API 호출", "정적 HTML", "간단한 크롤링"],
                "code_example": """

def crawl_static_page(url: str) -> str:
    headers = {'User-Agent': 'Mozilla/5.0...'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()
                """
            },
            CrawlingLibrary.SELENIUM: {
                "description": "동적 페이지 크롤링 - 완전한 브라우저 제어",
                "pros": ["JavaScript 렌더링", "완전한 제어", "복잡한 상호작용"],
                "cons": ["느린 속도", "높은 리소스 사용", "복잡한 설정"],
                "best_for": ["SPA", "복잡한 OTP", "동적 콘텐츠"],
                "code_example": """

def crawl_dynamic_page(url: str) -> str:
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    content = driver.page_source
    driver.quit()
    return content
                """
            },
            CrawlingLibrary.CLOUDSCRAPER: {
                "description": "보안 우회 크롤링 - Cloudflare 등 우회",
                "pros": ["보안 우회", "requests와 유사한 API", "안정성"],
                "cons": ["일부 사이트에서만 효과", "속도 제한"],
                "best_for": ["Cloudflare 보호 사이트", "KRX", "금융 데이터"],
                "code_example": """

def crawl_protected_page(url: str) -> str:
    scraper = cloudscraper.create_scraper()
    response = scraper.get(url)
    return response.text
                """
            },
            CrawlingLibrary.AIOHTTP: {
                "description": "비동기 고성능 크롤링 - 동시 100개+ 요청",
                "pros": ["매우 빠른 속도", "확장성", "비동기 처리"],
                "cons": ["복잡한 설정", "동적 콘텐츠 제한"],
                "best_for": ["대량 API 호출", "고성능 크롤링", "실시간 데이터"],
                "code_example": """

async def crawl_async(urls: List[str]) -> List[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [await resp.text() for resp in responses]
                """
            },
            CrawlingLibrary.SCRAPY: {
                "description": "대규모 크롤링 프레임워크 - 체계적 관리",
                "pros": ["확장성", "체계적 관리", "고급 기능"],
                "cons": ["높은 학습곡선", "복잡한 설정"],
                "best_for": ["대규모 프로젝트", "자동화", "엔터프라이즈"],
                "code_example": """
# scrapy spider 예시
class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['http://example.com']

    def parse(self, response):
        yield {'title': response.css('h1::text').get()}
                """
            }
        }

        return guides.get(strategy.primary, {})

    def get_anti_bot_strategies(self, strategy: LibraryStrategy) -> Dict[str, str]:
        """안티봇 우회 전략"""

        strategies = {
            "headers_randomization": """
def get_random_headers() -> Dict[str, str]:
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
        'Referer': 'http://example.com/',
        'Cache-Control': 'no-cache'
    }
            """,
            "rate_limiting": """
def rate_limited_request(url: str, delay: float = 1.0) -> requests.Response:
    time.sleep(delay)
    return requests.get(url, headers=get_random_headers())
            """,
            "proxy_rotation": """
class ProxyPool:
    def __init__(self, proxies: List[str]):
        self.proxies = proxies
        self.current = 0

    def get_proxy(self) -> str:
        proxy = self.proxies[self.current]
        self.current = (self.current + 1) % len(self.proxies)
        return proxy
            """,
            "session_management": """
def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(get_random_headers())
    return session
            """
        }

        return {strategy: strategies.get(strategy, "") for strategy in strategy.anti_bot}

    def get_performance_optimization(self, strategy: LibraryStrategy) -> Dict[str, str]:
        """성능 최적화 가이드"""

        optimizations = {
            "single": """
# 단일 스레드 최적화
def single_thread_optimization():
    # 메모리 효율적인 처리
    for item in large_dataset:
        process_item(item)
        del item  # 메모리 해제
            """,
            "async": """
# 비동기 최적화
async def async_optimization():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results
            """,
            "multiprocessing": """
# 멀티프로세싱 최적화
from concurrent.futures import ProcessPoolExecutor

def multiprocessing_optimization():
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_data, chunk) for chunk in data_chunks]
        results = [f.result() for f in futures]
        return results
            """
        }

        return optimizations.get(strategy.performance_mode, {})

class WorldClassCrawler:
    """World-Class 크롤링 시스템"""

    def __init__(self):
        self.strategy_guide = CrawlingStrategyGuide()
        self.session = None
        self.driver = None

    def crawl(self, url: str, page_type: PageType = PageType.STATIC,
              volume: str = "low") -> Optional[str]:
        """통합 크롤링 메서드"""

        strategy = self.strategy_guide.select_strategy(url, page_type, volume)
        guide = self.strategy_guide.get_implementation_guide(strategy)

        logger.info(f"선택된 전략: {strategy.primary.value}")
        logger.info(f"페이지 타입: {page_type.value}")
        logger.info(f"볼륨: {volume}")

        try:
            if strategy.primary == CrawlingLibrary.REQUESTS_BS4:
                return self._crawl_with_requests(url)
            elif strategy.primary == CrawlingLibrary.SELENIUM:
                return self._crawl_with_selenium(url)
            elif strategy.primary == CrawlingLibrary.CLOUDSCRAPER:
                return self._crawl_with_cloudscraper(url)
            elif strategy.primary == CrawlingLibrary.AIOHTTP:
                return asyncio.run(self._crawl_with_aiohttp(url))
            else:
                logger.warning(f"지원하지 않는 라이브러리: {strategy.primary}")
                return None

        except Exception as e:
            logger.error(f"크롤링 실패: {e}")
            # Fallback 전략 시도
            if strategy.fallback:
                logger.info(f"Fallback 전략 시도: {strategy.fallback}")
                return self._try_fallback(url, strategy.fallback)
            return None

    def _crawl_with_requests(self, url: str) -> Optional[str]:
        """requests + BeautifulSoup 크롤링"""
        try:
            headers = {
                'User-Agent': random.choice(self.strategy_guide.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"requests 크롤링 실패: {e}")
            return None

    def _crawl_with_selenium(self, url: str) -> Optional[str]:
        """Selenium 크롤링"""
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')

            driver = webdriver.Chrome(options=options)
            driver.get(url)
            content = driver.page_source
            driver.quit()
            return content
        except Exception as e:
            logger.error(f"Selenium 크롤링 실패: {e}")
            return None

    def _crawl_with_cloudscraper(self, url: str) -> Optional[str]:
        """cloudscraper 크롤링"""
        try:
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url)
            return response.text
        except Exception as e:
            logger.error(f"cloudscraper 크롤링 실패: {e}")
            return None

    async def _crawl_with_aiohttp(self, url: str) -> Optional[str]:
        """aiohttp 비동기 크롤링"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()
        except Exception as e:
            logger.error(f"aiohttp 크롤링 실패: {e}")
            return None

    def _try_fallback(self, url: str, fallback_library: CrawlingLibrary) -> Optional[str]:
        """Fallback 전략 시도"""
        try:
            if fallback_library == CrawlingLibrary.REQUESTS_BS4:
                return self._crawl_with_requests(url)
            elif fallback_library == CrawlingLibrary.SELENIUM:
                return self._crawl_with_selenium(url)
            elif fallback_library == CrawlingLibrary.CLOUDSCRAPER:
                return self._crawl_with_cloudscraper(url)
            else:
                return None
        except Exception as e:
            logger.error(f"Fallback 전략 실패: {e}")
            return None

def main():
    """메인 실행 함수 - 크롤링 전략 가이드"""

    crawler = WorldClassCrawler()

    # 테스트 URL들
    test_urls = [
        ("https://example.com", PageType.STATIC, "low"),
        ("https://krx.co.kr", PageType.PROTECTED, "medium"),
        ("https://api.example.com/data", PageType.API, "high"),
    ]

    for url, page_type, volume in test_urls:
        print(f"\n{'='*50}")
        print(f"URL: {url}")
        print(f"페이지 타입: {page_type.value}")
        print(f"볼륨: {volume}")

        strategy = crawler.strategy_guide.select_strategy(url, page_type, volume)
        guide = crawler.strategy_guide.get_implementation_guide(strategy)

        print(f"선택된 라이브러리: {strategy.primary.value}")
        print(f"성능 모드: {strategy.performance_mode}")
        print(f"안티봇 전략: {strategy.anti_bot}")
        print(f"설명: {guide.get('description', 'N/A')}")
        print(f"장점: {guide.get('pros', [])}")
        print(f"단점: {guide.get('cons', [])}")

if __name__ == "__main__":
    main()
