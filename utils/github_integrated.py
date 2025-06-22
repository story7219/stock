#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 GitHub 통합 접근 시스템
모든 방식을 통합하여 항상 GitHub 검색이 가능하도록 보장

접근 방식 우선순위:
1. GitHub API (토큰 기반)
2. 웹 검색 (DuckDuckGo + 스크래핑)
3. MCP 서버 (백업)
4. 로컬 캐시 및 폴백 데이터
"""

import os
import json
import requests
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pickle
from pathlib import Path

# 기존 모듈들 import
try:
    from github_accessor import GitHubAccessor
    GITHUB_API_AVAILABLE = True
except ImportError:
    GITHUB_API_AVAILABLE = False

try:
    from github_web_search import GitHubWebSearcher
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

# 환경 설정
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubIntegratedSystem:
    """GitHub 통합 접근 시스템"""
    
    def __init__(self):
        self.github_token = os.getenv('GITHUB_API_TOKEN', '')
        self.cache_dir = Path('.github_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # 각 접근 방식 초기화
        self.api_accessor = None
        self.web_searcher = None
        self.mcp_available = False
        
        # GitHub API 접근자 초기화
        if GITHUB_API_AVAILABLE and self.github_token:
            try:
                self.api_accessor = GitHubAccessor()
                logger.info("✅ GitHub API 접근자 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ GitHub API 접근자 초기화 실패: {e}")
        
        # 웹 검색자 초기화
        if WEB_SEARCH_AVAILABLE:
            try:
                self.web_searcher = GitHubWebSearcher()
                logger.info("✅ GitHub 웹 검색자 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ GitHub 웹 검색자 초기화 실패: {e}")
        
        # MCP 서버 상태 확인
        self.check_mcp_server()
    
    def check_mcp_server(self):
        """MCP 서버 상태 확인"""
        try:
            # MCP 서버 설정 파일 확인
            mcp_config_path = Path('mcp_servers.json')
            if mcp_config_path.exists():
                with open(mcp_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'mcpServers' in config and 'github' in config['mcpServers']:
                        self.mcp_available = True
                        logger.info("✅ MCP GitHub 서버 설정 확인됨")
        except Exception as e:
            logger.warning(f"⚠️ MCP 서버 확인 실패: {e}")
    
    def search_repositories(self, query: str, language: str = None, 
                          sort: str = 'stars', order: str = 'desc', 
                          per_page: int = 10) -> List[Dict[str, Any]]:
        """저장소 검색 (통합 방식)"""
        logger.info(f"🔍 통합 저장소 검색: {query}")
        
        # 캐시 확인
        cache_key = f"repos_{query}_{language}_{sort}_{order}_{per_page}"
        cached_result = self._get_cache(cache_key)
        if cached_result:
            logger.info("📋 캐시에서 결과 반환")
            return cached_result
        
        results = []
        
        # 방식 1: GitHub API
        if self.api_accessor:
            try:
                api_results = self.api_accessor.search_repositories(
                    query, language, sort, order, per_page
                )
                if api_results and not any(r.get('fallback') for r in api_results):
                    results = api_results
                    logger.info(f"✅ GitHub API 검색 성공: {len(results)}개 결과")
                    self._set_cache(cache_key, results)
                    return results
            except Exception as e:
                logger.warning(f"GitHub API 검색 실패: {e}")
        
        # 방식 2: 웹 검색
        if self.web_searcher:
            try:
                web_results = self.web_searcher.search_repositories_web(
                    query, language, per_page
                )
                if web_results and not any(r.get('fallback') for r in web_results):
                    results = web_results
                    logger.info(f"✅ 웹 검색 성공: {len(results)}개 결과")
                    self._set_cache(cache_key, results)
                    return results
            except Exception as e:
                logger.warning(f"웹 검색 실패: {e}")
        
        # 방식 3: 폴백 데이터
        fallback_results = self._get_fallback_repositories(query, language)
        logger.info(f"🔄 폴백 데이터 사용: {len(fallback_results)}개 결과")
        return fallback_results
    
    def search_code(self, query: str, repo: str = None, language: str = None, 
                   per_page: int = 10) -> List[Dict[str, Any]]:
        """코드 검색 (통합 방식)"""
        logger.info(f"🔍 통합 코드 검색: {query}")
        
        # 캐시 확인
        cache_key = f"code_{query}_{repo}_{language}_{per_page}"
        cached_result = self._get_cache(cache_key)
        if cached_result:
            logger.info("📋 캐시에서 코드 결과 반환")
            return cached_result
        
        results = []
        
        # 방식 1: GitHub API
        if self.api_accessor:
            try:
                api_results = self.api_accessor.search_code(query, repo, language, per_page)
                if api_results:
                    results = api_results
                    logger.info(f"✅ GitHub API 코드 검색 성공: {len(results)}개 결과")
                    self._set_cache(cache_key, results)
                    return results
            except Exception as e:
                logger.warning(f"GitHub API 코드 검색 실패: {e}")
        
        # 방식 2: 웹 검색
        if self.web_searcher:
            try:
                web_results = self.web_searcher.search_code_web(query, language)
                if web_results:
                    results = web_results
                    logger.info(f"✅ 웹 코드 검색 성공: {len(results)}개 결과")
                    self._set_cache(cache_key, results)
                    return results
            except Exception as e:
                logger.warning(f"웹 코드 검색 실패: {e}")
        
        logger.info("🔄 코드 검색 결과 없음")
        return []
    
    def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """저장소 정보 조회 (통합 방식)"""
        logger.info(f"📋 통합 저장소 정보 조회: {owner}/{repo}")
        
        # 캐시 확인
        cache_key = f"repo_info_{owner}_{repo}"
        cached_result = self._get_cache(cache_key)
        if cached_result:
            logger.info("📋 캐시에서 저장소 정보 반환")
            return cached_result
        
        # 방식 1: GitHub API
        if self.api_accessor:
            try:
                api_result = self.api_accessor.get_repository_info(owner, repo)
                if api_result:
                    logger.info("✅ GitHub API 저장소 정보 조회 성공")
                    self._set_cache(cache_key, api_result)
                    return api_result
            except Exception as e:
                logger.warning(f"GitHub API 저장소 정보 조회 실패: {e}")
        
        # 방식 2: 웹 스크래핑
        if self.web_searcher:
            try:
                web_result = self.web_searcher._scrape_repo_info(owner, repo)
                if web_result:
                    logger.info("✅ 웹 스크래핑 저장소 정보 조회 성공")
                    self._set_cache(cache_key, web_result)
                    return web_result
            except Exception as e:
                logger.warning(f"웹 스크래핑 저장소 정보 조회 실패: {e}")
        
        # 기본 정보 반환
        return {
            'name': repo,
            'full_name': f'{owner}/{repo}',
            'html_url': f'https://github.com/{owner}/{repo}',
            'owner': {'login': owner},
            'description': '',
            'stars': 0,
            'forks': 0,
            'language': '',
            'fallback': True
        }
    
    def get_trending_repositories(self, language: str = None, 
                                 period: str = 'daily') -> List[Dict[str, Any]]:
        """트렌딩 저장소 조회"""
        logger.info(f"📈 트렌딩 저장소 조회: {language} ({period})")
        
        # 캐시 확인
        cache_key = f"trending_{language}_{period}"
        cached_result = self._get_cache(cache_key, max_age_hours=6)  # 6시간 캐시
        if cached_result:
            logger.info("📋 캐시에서 트렌딩 결과 반환")
            return cached_result
        
        # 웹 검색으로 트렌딩 조회
        if self.web_searcher:
            try:
                trending_results = self.web_searcher.get_trending_repositories(language, period)
                if trending_results:
                    logger.info(f"✅ 트렌딩 저장소 조회 성공: {len(trending_results)}개")
                    self._set_cache(cache_key, trending_results)
                    return trending_results
            except Exception as e:
                logger.warning(f"트렌딩 저장소 조회 실패: {e}")
        
        return []
    
    def _get_cache(self, key: str, max_age_hours: int = 24) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                # 파일 수정 시간 확인
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age < timedelta(hours=max_age_hours):
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                else:
                    cache_file.unlink()  # 오래된 캐시 삭제
        except Exception as e:
            logger.warning(f"캐시 조회 실패: {e}")
        return None
    
    def _set_cache(self, key: str, data: Any):
        """캐시에 데이터 저장"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")
    
    def _get_fallback_repositories(self, query: str, language: str = None) -> List[Dict[str, Any]]:
        """폴백 저장소 데이터"""
        # 주식/금융/AI 관련 인기 저장소들
        popular_repos = [
            {
                'name': 'yfinance',
                'full_name': 'ranaroussi/yfinance',
                'description': 'Download market data from Yahoo! Finance API',
                'html_url': 'https://github.com/ranaroussi/yfinance',
                'stars': 12000,
                'forks': 2000,
                'language': 'Python',
                'owner': {'login': 'ranaroussi'},
                'topics': ['finance', 'yahoo-finance', 'stock-data', 'market-data'],
                'fallback': True
            },
            {
                'name': 'zipline',
                'full_name': 'quantopian/zipline',
                'description': 'Zipline, a Pythonic Algorithmic Trading Library',
                'html_url': 'https://github.com/quantopian/zipline',
                'stars': 17000,
                'forks': 4500,
                'language': 'Python',
                'owner': {'login': 'quantopian'},
                'topics': ['algorithmic-trading', 'finance', 'backtesting', 'trading'],
                'fallback': True
            },
            {
                'name': 'backtrader',
                'full_name': 'mementum/backtrader',
                'description': 'Python Backtesting library for trading strategies',
                'html_url': 'https://github.com/mementum/backtrader',
                'stars': 13000,
                'forks': 3500,
                'language': 'Python',
                'owner': {'login': 'mementum'},
                'topics': ['backtesting', 'trading', 'finance', 'strategy'],
                'fallback': True
            },
            {
                'name': 'pandas-datareader',
                'full_name': 'pydata/pandas-datareader',
                'description': 'Up to date remote data access for pandas',
                'html_url': 'https://github.com/pydata/pandas-datareader',
                'stars': 2800,
                'forks': 700,
                'language': 'Python',
                'owner': {'login': 'pydata'},
                'topics': ['pandas', 'data-reader', 'finance', 'data-access'],
                'fallback': True
            },
            {
                'name': 'FinanceDatabase',
                'full_name': 'JerBouma/FinanceDatabase',
                'description': '300,000+ symbols containing Equities, ETFs, Funds, Indices, Currencies, Cryptocurrencies',
                'html_url': 'https://github.com/JerBouma/FinanceDatabase',
                'stars': 3000,
                'forks': 500,
                'language': 'Python',
                'owner': {'login': 'JerBouma'},
                'topics': ['finance', 'database', 'stocks', 'etf', 'cryptocurrency'],
                'fallback': True
            },
            {
                'name': 'ta-lib',
                'full_name': 'mrjbq7/ta-lib',
                'description': 'Python wrapper for TA-Lib (Technical Analysis Library)',
                'html_url': 'https://github.com/mrjbq7/ta-lib',
                'stars': 8000,
                'forks': 1800,
                'language': 'Python',
                'owner': {'login': 'mrjbq7'},
                'topics': ['technical-analysis', 'finance', 'trading', 'indicators'],
                'fallback': True
            },
            {
                'name': 'ultralytics',
                'full_name': 'ultralytics/ultralytics',
                'description': 'NEW - YOLOv8 🚀 in PyTorch > ONNX > OpenVINO > CoreML > TFLite',
                'html_url': 'https://github.com/ultralytics/ultralytics',
                'stars': 25000,
                'forks': 5000,
                'language': 'Python',
                'owner': {'login': 'ultralytics'},
                'topics': ['yolo', 'computer-vision', 'object-detection', 'ai', 'deep-learning'],
                'fallback': True
            },
            {
                'name': 'transformers',
                'full_name': 'huggingface/transformers',
                'description': '🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX',
                'html_url': 'https://github.com/huggingface/transformers',
                'stars': 120000,
                'forks': 24000,
                'language': 'Python',
                'owner': {'login': 'huggingface'},
                'topics': ['nlp', 'pytorch', 'tensorflow', 'jax', 'transformers', 'ai'],
                'fallback': True
            }
        ]
        
        # 쿼리와 관련성 계산
        query_lower = query.lower()
        query_words = query_lower.split()
        
        relevant_repos = []
        for repo in popular_repos:
            relevance_score = 0
            
            # 이름에서 매칭
            if any(word in repo['name'].lower() for word in query_words):
                relevance_score += 5
            
            # 설명에서 매칭
            if any(word in repo['description'].lower() for word in query_words):
                relevance_score += 3
            
            # 토픽에서 매칭
            topics_text = ' '.join(repo['topics']).lower()
            matching_topics = sum(1 for word in query_words if word in topics_text)
            relevance_score += matching_topics * 2
            
            # 언어 필터
            if language and repo['language'].lower() != language.lower():
                relevance_score = max(0, relevance_score - 2)
            
            if relevance_score > 0:
                repo['relevance_score'] = relevance_score
                relevant_repos.append(repo)
        
        # 관련성 점수로 정렬
        relevant_repos.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_repos[:10]  # 상위 10개 반환
    
    def test_all_systems(self) -> Dict[str, Any]:
        """모든 시스템 테스트"""
        logger.info("🔧 통합 시스템 테스트 시작")
        
        results = {
            'github_api_working': False,
            'web_search_working': False,
            'mcp_server_available': self.mcp_available,
            'cache_working': False,
            'overall_status': 'unknown',
            'test_results': {}
        }
        
        # GitHub API 테스트
        if self.api_accessor:
            try:
                api_test = self.api_accessor.test_connection()
                results['github_api_working'] = api_test.get('requests_working', False) or api_test.get('pygithub_working', False)
                results['test_results']['api'] = api_test
            except Exception as e:
                logger.warning(f"GitHub API 테스트 실패: {e}")
        
        # 웹 검색 테스트
        if self.web_searcher:
            try:
                web_test = self.web_searcher.test_web_search()
                results['web_search_working'] = web_test.get('web_search_working', False)
                results['test_results']['web'] = web_test
            except Exception as e:
                logger.warning(f"웹 검색 테스트 실패: {e}")
        
        # 캐시 테스트
        try:
            test_data = {'test': 'data', 'timestamp': datetime.now().isoformat()}
            self._set_cache('test_cache', test_data)
            cached_data = self._get_cache('test_cache')
            results['cache_working'] = cached_data is not None
        except Exception as e:
            logger.warning(f"캐시 테스트 실패: {e}")
        
        # 전체 상태 결정
        if results['github_api_working']:
            results['overall_status'] = 'excellent'
        elif results['web_search_working']:
            results['overall_status'] = 'good'
        elif results['mcp_server_available']:
            results['overall_status'] = 'fair'
        else:
            results['overall_status'] = 'fallback_only'
        
        logger.info(f"✅ 통합 시스템 테스트 완료: {results['overall_status']}")
        return results
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
            logger.info("✅ 캐시 정리 완료")
        except Exception as e:
            logger.warning(f"캐시 정리 실패: {e}")

# 전역 인스턴스
github_system = GitHubIntegratedSystem()

# 편의 함수들
def search_github_repositories_integrated(query: str, **kwargs) -> List[Dict[str, Any]]:
    """통합 GitHub 저장소 검색"""
    return github_system.search_repositories(query, **kwargs)

def search_github_code_integrated(query: str, **kwargs) -> List[Dict[str, Any]]:
    """통합 GitHub 코드 검색"""
    return github_system.search_code(query, **kwargs)

def get_github_repo_info_integrated(owner: str, repo: str) -> Dict[str, Any]:
    """통합 GitHub 저장소 정보 조회"""
    return github_system.get_repository_info(owner, repo)

def get_github_trending_integrated(language: str = None, period: str = 'daily') -> List[Dict[str, Any]]:
    """통합 GitHub 트렌딩 저장소"""
    return github_system.get_trending_repositories(language, period)

def test_github_integrated_system() -> Dict[str, Any]:
    """통합 시스템 테스트"""
    return github_system.test_all_systems()

# 테스트 실행
if __name__ == "__main__":
    print("🔧 GitHub 통합 접근 시스템 테스트")
    
    # 시스템 전체 테스트
    system_test = test_github_integrated_system()
    print(f"시스템 상태: {system_test['overall_status']}")
    print(f"테스트 결과: {json.dumps(system_test, indent=2, ensure_ascii=False)}")
    
    # 저장소 검색 테스트
    print("\n📋 통합 저장소 검색 테스트")
    repos = search_github_repositories_integrated("python stock analysis", per_page=5)
    for repo in repos:
        status = "🔄 폴백" if repo.get('fallback') else "✅ 실제"
        print(f"  {status} {repo['full_name']}: ⭐{repo.get('stars', 0)} - {repo.get('description', '')[:80]}...")
    
    # 트렌딩 저장소 테스트
    print("\n📈 트렌딩 저장소 테스트")
    trending = get_github_trending_integrated("python")
    for repo in trending[:3]:
        print(f"  📈 {repo['full_name']}: ⭐{repo.get('stars', 0)} - {repo.get('description', '')[:80]}...")
    
    print("\n✅ 모든 통합 테스트 완료!") 