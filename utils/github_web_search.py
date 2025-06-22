#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 GitHub 웹 검색 모듈
API 토큰 없이도 웹 검색을 통해 GitHub 정보를 수집

지원 기능:
1. 웹 검색을 통한 GitHub 저장소 찾기
2. 공개 저장소 정보 스크래핑
3. 코드 검색 및 분석
"""

import requests
import json
import re
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import quote
from bs4 import BeautifulSoup
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubWebSearcher:
    """GitHub 웹 검색 클래스"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def search_repositories_web(self, query: str, language: str = None, 
                               per_page: int = 10) -> List[Dict[str, Any]]:
        """웹 검색을 통한 GitHub 저장소 검색"""
        logger.info(f"🌐 웹 검색으로 GitHub 저장소 검색: {query}")
        
        # Google 검색을 통한 GitHub 저장소 찾기
        search_query = f"site:github.com {query}"
        if language:
            search_query += f" language:{language}"
        
        try:
            # Google 검색 결과
            google_results = self._search_google(search_query, per_page)
            
            # GitHub 저장소 정보 추출
            repositories = []
            for result in google_results:
                if 'github.com' in result.get('url', ''):
                    repo_info = self._extract_repo_info_from_url(result['url'])
                    if repo_info:
                        repo_info.update({
                            'title': result.get('title', ''),
                            'description': result.get('description', ''),
                            'search_source': 'google'
                        })
                        repositories.append(repo_info)
            
            logger.info(f"✅ 웹 검색 완료: {len(repositories)}개 저장소 발견")
            return repositories
            
        except Exception as e:
            logger.warning(f"웹 검색 실패: {e}")
            return self._get_popular_repositories(query)
    
    def _search_google(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Google 검색"""
        # DuckDuckGo를 사용한 검색 (더 안정적)
        return self._search_duckduckgo(query, num_results)
    
    def _search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """DuckDuckGo 검색"""
        try:
            url = "https://html.duckduckgo.com/html/"
            params = {'q': query}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # 검색 결과 파싱
            for result in soup.find_all('div', class_='result')[:num_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'url': title_elem.get('href', ''),
                        'description': snippet_elem.get_text(strip=True) if snippet_elem else ''
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"DuckDuckGo 검색 실패: {e}")
            return []
    
    def _extract_repo_info_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """GitHub URL에서 저장소 정보 추출"""
        # GitHub 저장소 URL 패턴 매칭
        pattern = r'https://github\.com/([^/]+)/([^/]+)'
        match = re.match(pattern, url)
        
        if not match:
            return None
        
        owner, repo = match.groups()
        
        # 공개 저장소 정보 수집
        try:
            repo_info = self._scrape_repo_info(owner, repo)
            return repo_info
        except Exception as e:
            logger.warning(f"저장소 정보 수집 실패 ({owner}/{repo}): {e}")
            return {
                'name': repo,
                'full_name': f'{owner}/{repo}',
                'owner': {'login': owner},
                'html_url': url,
                'description': '',
                'stars': 0,
                'forks': 0,
                'language': '',
                'web_scraped': True
            }
    
    def _scrape_repo_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """GitHub 저장소 페이지 스크래핑"""
        url = f"https://github.com/{owner}/{repo}"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 기본 정보
        repo_info = {
            'name': repo,
            'full_name': f'{owner}/{repo}',
            'html_url': url,
            'owner': {'login': owner},
            'web_scraped': True
        }
        
        # 설명 추출
        description_elem = soup.find('p', class_='f4 my-3')
        if description_elem:
            repo_info['description'] = description_elem.get_text(strip=True)
        else:
            repo_info['description'] = ''
        
        # 스타, 포크 수 추출
        stats = soup.find_all('a', class_='Link--muted')
        for stat in stats:
            text = stat.get_text(strip=True)
            if 'star' in stat.get('href', '').lower():
                repo_info['stars'] = self._parse_number(text)
            elif 'fork' in stat.get('href', '').lower():
                repo_info['forks'] = self._parse_number(text)
        
        # 언어 정보 추출
        lang_elem = soup.find('span', class_='color-fg-default text-bold mr-1')
        if lang_elem:
            repo_info['language'] = lang_elem.get_text(strip=True)
        else:
            repo_info['language'] = ''
        
        # 토픽 추출
        topics = []
        topic_elems = soup.find_all('a', class_='topic-tag')
        for topic in topic_elems:
            topics.append(topic.get_text(strip=True))
        repo_info['topics'] = topics
        
        return repo_info
    
    def _parse_number(self, text: str) -> int:
        """숫자 문자열 파싱 (1.2k -> 1200)"""
        text = text.replace(',', '').strip()
        
        if 'k' in text.lower():
            return int(float(text.lower().replace('k', '')) * 1000)
        elif 'm' in text.lower():
            return int(float(text.lower().replace('m', '')) * 1000000)
        else:
            try:
                return int(text)
            except:
                return 0
    
    def _get_popular_repositories(self, query: str) -> List[Dict[str, Any]]:
        """인기 있는 저장소 목록 (폴백)"""
        logger.info(f"🔄 인기 저장소 목록 생성: {query}")
        
        # 주식/금융 관련 인기 저장소들
        popular_repos = [
            {
                'name': 'yfinance',
                'full_name': 'ranaroussi/yfinance',
                'description': 'Download market data from Yahoo! Finance\'s API',
                'html_url': 'https://github.com/ranaroussi/yfinance',
                'stars': 8000,
                'forks': 1500,
                'language': 'Python',
                'owner': {'login': 'ranaroussi'},
                'topics': ['finance', 'yahoo-finance', 'stock-data'],
                'fallback': True
            },
            {
                'name': 'zipline',
                'full_name': 'quantopian/zipline',
                'description': 'Zipline, a Pythonic Algorithmic Trading Library',
                'html_url': 'https://github.com/quantopian/zipline',
                'stars': 15000,
                'forks': 4000,
                'language': 'Python',
                'owner': {'login': 'quantopian'},
                'topics': ['algorithmic-trading', 'finance', 'backtesting'],
                'fallback': True
            },
            {
                'name': 'pandas-datareader',
                'full_name': 'pydata/pandas-datareader',
                'description': 'Up to date remote data access for pandas',
                'html_url': 'https://github.com/pydata/pandas-datareader',
                'stars': 2500,
                'forks': 600,
                'language': 'Python',
                'owner': {'login': 'pydata'},
                'topics': ['pandas', 'data-reader', 'finance'],
                'fallback': True
            },
            {
                'name': 'backtrader',
                'full_name': 'mementum/backtrader',
                'description': 'Python Backtesting library for trading strategies',
                'html_url': 'https://github.com/mementum/backtrader',
                'stars': 9000,
                'forks': 2500,
                'language': 'Python',
                'owner': {'login': 'mementum'},
                'topics': ['backtesting', 'trading', 'finance'],
                'fallback': True
            },
            {
                'name': 'FinanceDatabase',
                'full_name': 'JerBouma/FinanceDatabase',
                'description': 'This is a database of 300.000+ symbols containing Equities, ETFs, Funds, Indices, Currencies, Cryptocurrencies and Money Markets.',
                'html_url': 'https://github.com/JerBouma/FinanceDatabase',
                'stars': 2000,
                'forks': 400,
                'language': 'Python',
                'owner': {'login': 'JerBouma'},
                'topics': ['finance', 'database', 'stocks'],
                'fallback': True
            }
        ]
        
        # 쿼리와 관련된 저장소 필터링
        query_lower = query.lower()
        relevant_repos = []
        
        for repo in popular_repos:
            # 이름, 설명, 토픽에서 쿼리 관련성 확인
            relevance_score = 0
            
            if any(word in repo['name'].lower() for word in query_lower.split()):
                relevance_score += 3
            if any(word in repo['description'].lower() for word in query_lower.split()):
                relevance_score += 2
            if any(word in ' '.join(repo['topics']).lower() for word in query_lower.split()):
                relevance_score += 1
            
            if relevance_score > 0:
                repo['relevance_score'] = relevance_score
                relevant_repos.append(repo)
        
        # 관련성 점수로 정렬
        relevant_repos.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_repos[:5]  # 상위 5개만 반환
    
    def search_code_web(self, query: str, language: str = None) -> List[Dict[str, Any]]:
        """웹 검색을 통한 코드 검색"""
        logger.info(f"🔍 웹 검색으로 코드 검색: {query}")
        
        search_query = f"site:github.com {query}"
        if language:
            search_query += f" filetype:{language}"
        
        try:
            results = self._search_duckduckgo(search_query, 10)
            code_results = []
            
            for result in results:
                if 'github.com' in result['url'] and '/blob/' in result['url']:
                    code_info = self._extract_code_info_from_url(result['url'])
                    if code_info:
                        code_info.update({
                            'title': result['title'],
                            'description': result['description'],
                            'search_source': 'web'
                        })
                        code_results.append(code_info)
            
            logger.info(f"✅ 코드 검색 완료: {len(code_results)}개 결과")
            return code_results
            
        except Exception as e:
            logger.warning(f"코드 검색 실패: {e}")
            return []
    
    def _extract_code_info_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """GitHub 코드 URL에서 정보 추출"""
        # GitHub blob URL 패턴 매칭
        pattern = r'https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)'
        match = re.match(pattern, url)
        
        if not match:
            return None
        
        owner, repo, branch, file_path = match.groups()
        
        return {
            'name': file_path.split('/')[-1],
            'path': file_path,
            'html_url': url,
            'repository': {
                'name': repo,
                'full_name': f'{owner}/{repo}',
                'html_url': f'https://github.com/{owner}/{repo}'
            },
            'branch': branch,
            'web_scraped': True
        }
    
    def get_trending_repositories(self, language: str = None, 
                                 period: str = 'daily') -> List[Dict[str, Any]]:
        """트렌딩 저장소 조회"""
        logger.info(f"📈 트렌딩 저장소 조회: {language} ({period})")
        
        try:
            url = "https://github.com/trending"
            if language:
                url += f"/{language}"
            
            params = {'since': period}
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            repositories = []
            
            # 트렌딩 저장소 파싱
            for article in soup.find_all('article', class_='Box-row')[:10]:
                repo_link = article.find('h2').find('a')
                if repo_link:
                    repo_name = repo_link.get('href').strip('/')
                    
                    # 설명
                    desc_elem = article.find('p', class_='col-9')
                    description = desc_elem.get_text(strip=True) if desc_elem else ''
                    
                    # 언어
                    lang_elem = article.find('span', {'itemprop': 'programmingLanguage'})
                    language = lang_elem.get_text(strip=True) if lang_elem else ''
                    
                    # 스타 수
                    star_elem = article.find('a', href=f'/{repo_name}/stargazers')
                    stars = 0
                    if star_elem:
                        star_text = star_elem.get_text(strip=True)
                        stars = self._parse_number(star_text)
                    
                    repositories.append({
                        'name': repo_name.split('/')[-1],
                        'full_name': repo_name,
                        'description': description,
                        'html_url': f'https://github.com/{repo_name}',
                        'language': language,
                        'stars': stars,
                        'trending': True,
                        'period': period
                    })
            
            logger.info(f"✅ 트렌딩 저장소 조회 완료: {len(repositories)}개")
            return repositories
            
        except Exception as e:
            logger.warning(f"트렌딩 저장소 조회 실패: {e}")
            return []
    
    def test_web_search(self) -> Dict[str, Any]:
        """웹 검색 테스트"""
        logger.info("🌐 웹 검색 테스트 시작")
        
        results = {
            'web_search_working': False,
            'repository_search_working': False,
            'code_search_working': False,
            'trending_working': False,
            'test_results': {}
        }
        
        # 저장소 검색 테스트
        try:
            repos = self.search_repositories_web("python", per_page=2)
            if repos:
                results['repository_search_working'] = True
                results['test_results']['repositories'] = len(repos)
        except Exception as e:
            logger.warning(f"저장소 검색 테스트 실패: {e}")
        
        # 코드 검색 테스트
        try:
            codes = self.search_code_web("def main", "python")
            if codes:
                results['code_search_working'] = True
                results['test_results']['code_results'] = len(codes)
        except Exception as e:
            logger.warning(f"코드 검색 테스트 실패: {e}")
        
        # 트렌딩 저장소 테스트
        try:
            trending = self.get_trending_repositories("python")
            if trending:
                results['trending_working'] = True
                results['test_results']['trending'] = len(trending)
        except Exception as e:
            logger.warning(f"트렌딩 저장소 테스트 실패: {e}")
        
        # 전체 웹 검색 상태
        results['web_search_working'] = any([
            results['repository_search_working'],
            results['code_search_working'],
            results['trending_working']
        ])
        
        logger.info(f"✅ 웹 검색 테스트 완료: {results}")
        return results

# 전역 인스턴스
github_web_searcher = GitHubWebSearcher()

def search_github_repos_web(query: str, **kwargs) -> List[Dict[str, Any]]:
    """GitHub 저장소 웹 검색 (편의 함수)"""
    return github_web_searcher.search_repositories_web(query, **kwargs)

def search_github_code_web(query: str, **kwargs) -> List[Dict[str, Any]]:
    """GitHub 코드 웹 검색 (편의 함수)"""
    return github_web_searcher.search_code_web(query, **kwargs)

def get_github_trending(language: str = None, period: str = 'daily') -> List[Dict[str, Any]]:
    """GitHub 트렌딩 저장소 (편의 함수)"""
    return github_web_searcher.get_trending_repositories(language, period)

def test_github_web_search() -> Dict[str, Any]:
    """GitHub 웹 검색 테스트 (편의 함수)"""
    return github_web_searcher.test_web_search()

# 테스트 실행
if __name__ == "__main__":
    print("🌐 GitHub 웹 검색 모듈 테스트")
    
    # 웹 검색 테스트
    web_test = test_github_web_search()
    print(f"웹 검색 테스트 결과: {json.dumps(web_test, indent=2, ensure_ascii=False)}")
    
    # 저장소 검색 테스트
    print("\n📋 저장소 웹 검색 테스트")
    repos = search_github_repos_web("python stock analysis", per_page=3)
    for repo in repos:
        print(f"  • {repo['full_name']}: ⭐{repo.get('stars', 0)} - {repo.get('description', '')[:100]}")
    
    # 트렌딩 저장소 테스트
    print("\n📈 트렌딩 저장소 테스트")
    trending = get_github_trending("python", "daily")
    for repo in trending[:3]:
        print(f"  • {repo['full_name']}: ⭐{repo.get('stars', 0)} - {repo.get('description', '')[:100]}")
    
    print("\n✅ 모든 웹 검색 테스트 완료!") 