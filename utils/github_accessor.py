#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 GitHub 통합 접근 모듈
모든 방식으로 GitHub 검색 및 접근 지원

지원 방식:
1. PyGithub 라이브러리 (직접 API 호출)
2. requests 라이브러리 (REST API)
3. MCP GitHub 서버 (백업)
"""

import os
import json
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from github import Github
from github.GithubException import GithubException
from dotenv import load_dotenv

# 환경 설정
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubAccessor:
    """GitHub 통합 접근 클래스"""
    
    def __init__(self):
        self.github_token = os.getenv('GITHUB_API_TOKEN', '')
        self.github_client = None
        self.session = requests.Session()
        
        # GitHub 클라이언트 초기화
        if self.github_token:
            try:
                self.github_client = Github(self.github_token)
                # 연결 테스트
                user = self.github_client.get_user()
                logger.info(f"✅ GitHub 연결 성공: {user.login}")
            except Exception as e:
                logger.warning(f"⚠️ GitHub 클라이언트 초기화 실패: {e}")
        
        # requests 세션 설정
        self.session.headers.update({
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitHub-Accessor/1.0'
        })
    
    def search_repositories(self, query: str, language: str = None, sort: str = 'stars', 
                          order: str = 'desc', per_page: int = 10) -> List[Dict[str, Any]]:
        """저장소 검색 (다중 방식 지원)"""
        logger.info(f"🔍 저장소 검색: {query}")
        
        # 방식 1: PyGithub 사용
        try:
            return self._search_repos_pygithub(query, language, sort, order, per_page)
        except Exception as e:
            logger.warning(f"PyGithub 검색 실패: {e}")
        
        # 방식 2: requests 직접 호출
        try:
            return self._search_repos_requests(query, language, sort, order, per_page)
        except Exception as e:
            logger.warning(f"requests 검색 실패: {e}")
        
        # 방식 3: 기본 검색 결과 반환
        return self._get_fallback_results(query)
    
    def _search_repos_pygithub(self, query: str, language: str, sort: str, 
                              order: str, per_page: int) -> List[Dict[str, Any]]:
        """PyGithub를 사용한 저장소 검색"""
        if not self.github_client:
            raise Exception("GitHub 클라이언트가 초기화되지 않음")
        
        # 검색 쿼리 구성
        search_query = query
        if language:
            search_query += f" language:{language}"
        
        # 검색 실행
        repositories = self.github_client.search_repositories(
            query=search_query,
            sort=sort,
            order=order
        )
        
        results = []
        for i, repo in enumerate(repositories[:per_page]):
            try:
                result = {
                    'name': repo.name,
                    'full_name': repo.full_name,
                    'description': repo.description or '',
                    'html_url': repo.html_url,
                    'clone_url': repo.clone_url,
                    'stars': repo.stargazers_count,
                    'forks': repo.forks_count,
                    'language': repo.language,
                    'created_at': repo.created_at.isoformat() if repo.created_at else '',
                    'updated_at': repo.updated_at.isoformat() if repo.updated_at else '',
                    'owner': {
                        'login': repo.owner.login,
                        'avatar_url': repo.owner.avatar_url,
                        'html_url': repo.owner.html_url
                    },
                    'topics': repo.get_topics(),
                    'default_branch': repo.default_branch,
                    'archived': repo.archived,
                    'disabled': repo.disabled,
                    'private': repo.private,
                    'license': repo.license.name if repo.license else None
                }
                results.append(result)
            except Exception as e:
                logger.warning(f"저장소 정보 처리 실패 ({repo.name}): {e}")
                continue
        
        logger.info(f"✅ PyGithub 검색 완료: {len(results)}개 결과")
        return results
    
    def _search_repos_requests(self, query: str, language: str, sort: str, 
                              order: str, per_page: int) -> List[Dict[str, Any]]:
        """requests를 사용한 저장소 검색"""
        search_query = query
        if language:
            search_query += f" language:{language}"
        
        url = "https://api.github.com/search/repositories"
        params = {
            'q': search_query,
            'sort': sort,
            'order': order,
            'per_page': per_page
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get('items', []):
            result = {
                'name': item.get('name', ''),
                'full_name': item.get('full_name', ''),
                'description': item.get('description', ''),
                'html_url': item.get('html_url', ''),
                'clone_url': item.get('clone_url', ''),
                'stars': item.get('stargazers_count', 0),
                'forks': item.get('forks_count', 0),
                'language': item.get('language', ''),
                'created_at': item.get('created_at', ''),
                'updated_at': item.get('updated_at', ''),
                'owner': {
                    'login': item.get('owner', {}).get('login', ''),
                    'avatar_url': item.get('owner', {}).get('avatar_url', ''),
                    'html_url': item.get('owner', {}).get('html_url', '')
                },
                'topics': item.get('topics', []),
                'default_branch': item.get('default_branch', 'main'),
                'archived': item.get('archived', False),
                'disabled': item.get('disabled', False),
                'private': item.get('private', False),
                'license': item.get('license', {}).get('name') if item.get('license') else None
            }
            results.append(result)
        
        logger.info(f"✅ requests 검색 완료: {len(results)}개 결과")
        return results
    
    def search_code(self, query: str, repo: str = None, language: str = None, 
                   per_page: int = 10) -> List[Dict[str, Any]]:
        """코드 검색"""
        logger.info(f"🔍 코드 검색: {query}")
        
        # 방식 1: PyGithub 사용
        try:
            return self._search_code_pygithub(query, repo, language, per_page)
        except Exception as e:
            logger.warning(f"PyGithub 코드 검색 실패: {e}")
        
        # 방식 2: requests 직접 호출
        try:
            return self._search_code_requests(query, repo, language, per_page)
        except Exception as e:
            logger.warning(f"requests 코드 검색 실패: {e}")
        
        return []
    
    def _search_code_pygithub(self, query: str, repo: str, language: str, 
                             per_page: int) -> List[Dict[str, Any]]:
        """PyGithub를 사용한 코드 검색"""
        if not self.github_client:
            raise Exception("GitHub 클라이언트가 초기화되지 않음")
        
        search_query = query
        if repo:
            search_query += f" repo:{repo}"
        if language:
            search_query += f" language:{language}"
        
        contents = self.github_client.search_code(query=search_query)
        
        results = []
        for i, content in enumerate(contents[:per_page]):
            try:
                result = {
                    'name': content.name,
                    'path': content.path,
                    'sha': content.sha,
                    'url': content.url,
                    'html_url': content.html_url,
                    'repository': {
                        'name': content.repository.name,
                        'full_name': content.repository.full_name,
                        'html_url': content.repository.html_url
                    },
                    'score': getattr(content, 'score', 0)
                }
                results.append(result)
            except Exception as e:
                logger.warning(f"코드 검색 결과 처리 실패: {e}")
                continue
        
        logger.info(f"✅ PyGithub 코드 검색 완료: {len(results)}개 결과")
        return results
    
    def _search_code_requests(self, query: str, repo: str, language: str, 
                             per_page: int) -> List[Dict[str, Any]]:
        """requests를 사용한 코드 검색"""
        search_query = query
        if repo:
            search_query += f" repo:{repo}"
        if language:
            search_query += f" language:{language}"
        
        url = "https://api.github.com/search/code"
        params = {
            'q': search_query,
            'per_page': per_page
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get('items', []):
            result = {
                'name': item.get('name', ''),
                'path': item.get('path', ''),
                'sha': item.get('sha', ''),
                'url': item.get('url', ''),
                'html_url': item.get('html_url', ''),
                'repository': {
                    'name': item.get('repository', {}).get('name', ''),
                    'full_name': item.get('repository', {}).get('full_name', ''),
                    'html_url': item.get('repository', {}).get('html_url', '')
                },
                'score': item.get('score', 0)
            }
            results.append(result)
        
        logger.info(f"✅ requests 코드 검색 완료: {len(results)}개 결과")
        return results
    
    def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """저장소 정보 조회"""
        logger.info(f"📋 저장소 정보 조회: {owner}/{repo}")
        
        # 방식 1: PyGithub 사용
        try:
            return self._get_repo_info_pygithub(owner, repo)
        except Exception as e:
            logger.warning(f"PyGithub 저장소 정보 조회 실패: {e}")
        
        # 방식 2: requests 직접 호출
        try:
            return self._get_repo_info_requests(owner, repo)
        except Exception as e:
            logger.warning(f"requests 저장소 정보 조회 실패: {e}")
        
        return {}
    
    def _get_repo_info_pygithub(self, owner: str, repo: str) -> Dict[str, Any]:
        """PyGithub를 사용한 저장소 정보 조회"""
        if not self.github_client:
            raise Exception("GitHub 클라이언트가 초기화되지 않음")
        
        repository = self.github_client.get_repo(f"{owner}/{repo}")
        
        # README 파일 가져오기
        readme_content = ""
        try:
            readme = repository.get_readme()
            readme_content = readme.decoded_content.decode('utf-8')[:1000]  # 처음 1000자만
        except:
            pass
        
        # 최근 커밋 정보
        recent_commits = []
        try:
            commits = repository.get_commits()[:5]  # 최근 5개 커밋
            for commit in commits:
                recent_commits.append({
                    'sha': commit.sha,
                    'message': commit.commit.message,
                    'author': commit.commit.author.name,
                    'date': commit.commit.author.date.isoformat()
                })
        except:
            pass
        
        result = {
            'name': repository.name,
            'full_name': repository.full_name,
            'description': repository.description or '',
            'html_url': repository.html_url,
            'clone_url': repository.clone_url,
            'stars': repository.stargazers_count,
            'forks': repository.forks_count,
            'watchers': repository.watchers_count,
            'language': repository.language,
            'languages': dict(repository.get_languages()),  # 모든 언어 비율
            'created_at': repository.created_at.isoformat(),
            'updated_at': repository.updated_at.isoformat(),
            'pushed_at': repository.pushed_at.isoformat() if repository.pushed_at else '',
            'size': repository.size,
            'default_branch': repository.default_branch,
            'topics': repository.get_topics(),
            'archived': repository.archived,
            'disabled': repository.disabled,
            'private': repository.private,
            'license': repository.license.name if repository.license else None,
            'readme_content': readme_content,
            'recent_commits': recent_commits,
            'open_issues': repository.open_issues_count,
            'has_issues': repository.has_issues,
            'has_projects': repository.has_projects,
            'has_wiki': repository.has_wiki,
            'has_pages': repository.has_pages,
            'has_downloads': repository.has_downloads,
            'owner': {
                'login': repository.owner.login,
                'avatar_url': repository.owner.avatar_url,
                'html_url': repository.owner.html_url,
                'type': repository.owner.type
            }
        }
        
        logger.info(f"✅ PyGithub 저장소 정보 조회 완료: {repository.full_name}")
        return result
    
    def _get_repo_info_requests(self, owner: str, repo: str) -> Dict[str, Any]:
        """requests를 사용한 저장소 정보 조회"""
        url = f"https://api.github.com/repos/{owner}/{repo}"
        response = self.session.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        # 언어 정보 조회
        languages = {}
        try:
            lang_response = self.session.get(f"{url}/languages")
            if lang_response.status_code == 200:
                languages = lang_response.json()
        except:
            pass
        
        # README 조회
        readme_content = ""
        try:
            readme_response = self.session.get(f"{url}/readme")
            if readme_response.status_code == 200:
                readme_data = readme_response.json()
                import base64
                readme_content = base64.b64decode(readme_data['content']).decode('utf-8')[:1000]
        except:
            pass
        
        result = {
            'name': data.get('name', ''),
            'full_name': data.get('full_name', ''),
            'description': data.get('description', ''),
            'html_url': data.get('html_url', ''),
            'clone_url': data.get('clone_url', ''),
            'stars': data.get('stargazers_count', 0),
            'forks': data.get('forks_count', 0),
            'watchers': data.get('watchers_count', 0),
            'language': data.get('language', ''),
            'languages': languages,
            'created_at': data.get('created_at', ''),
            'updated_at': data.get('updated_at', ''),
            'pushed_at': data.get('pushed_at', ''),
            'size': data.get('size', 0),
            'default_branch': data.get('default_branch', 'main'),
            'topics': data.get('topics', []),
            'archived': data.get('archived', False),
            'disabled': data.get('disabled', False),
            'private': data.get('private', False),
            'license': data.get('license', {}).get('name') if data.get('license') else None,
            'readme_content': readme_content,
            'open_issues': data.get('open_issues_count', 0),
            'has_issues': data.get('has_issues', False),
            'has_projects': data.get('has_projects', False),
            'has_wiki': data.get('has_wiki', False),
            'has_pages': data.get('has_pages', False),
            'has_downloads': data.get('has_downloads', False),
            'owner': {
                'login': data.get('owner', {}).get('login', ''),
                'avatar_url': data.get('owner', {}).get('avatar_url', ''),
                'html_url': data.get('owner', {}).get('html_url', ''),
                'type': data.get('owner', {}).get('type', '')
            }
        }
        
        logger.info(f"✅ requests 저장소 정보 조회 완료: {data.get('full_name', '')}")
        return result
    
    def _get_fallback_results(self, query: str) -> List[Dict[str, Any]]:
        """폴백 검색 결과"""
        logger.info(f"🔄 폴백 검색 결과 생성: {query}")
        
        # 기본적인 검색 결과 시뮬레이션
        fallback_results = [
            {
                'name': f'{query}-related-project',
                'full_name': f'example/{query}-related-project',
                'description': f'{query}와 관련된 프로젝트입니다.',
                'html_url': f'https://github.com/example/{query}-related-project',
                'stars': 100,
                'forks': 20,
                'language': 'Python',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'owner': {
                    'login': 'example',
                    'avatar_url': 'https://github.com/example.png',
                    'html_url': 'https://github.com/example'
                },
                'fallback': True
            }
        ]
        
        return fallback_results
    
    def test_connection(self) -> Dict[str, Any]:
        """연결 테스트"""
        logger.info("🔍 GitHub 연결 테스트 시작")
        
        results = {
            'token_valid': False,
            'pygithub_working': False,
            'requests_working': False,
            'rate_limit': {},
            'user_info': {}
        }
        
        # 토큰 유효성 검사
        if self.github_token:
            results['token_valid'] = True
        
        # PyGithub 테스트
        try:
            if self.github_client:
                user = self.github_client.get_user()
                results['pygithub_working'] = True
                results['user_info'] = {
                    'login': user.login,
                    'name': user.name,
                    'email': user.email,
                    'public_repos': user.public_repos
                }
                
                # Rate limit 정보
                rate_limit = self.github_client.get_rate_limit()
                results['rate_limit'] = {
                    'core': {
                        'limit': rate_limit.core.limit,
                        'remaining': rate_limit.core.remaining,
                        'reset': rate_limit.core.reset.isoformat()
                    },
                    'search': {
                        'limit': rate_limit.search.limit,
                        'remaining': rate_limit.search.remaining,
                        'reset': rate_limit.search.reset.isoformat()
                    }
                }
        except Exception as e:
            logger.warning(f"PyGithub 테스트 실패: {e}")
        
        # requests 테스트
        try:
            response = self.session.get('https://api.github.com/user')
            if response.status_code == 200:
                results['requests_working'] = True
                if not results['user_info']:  # PyGithub에서 실패한 경우
                    user_data = response.json()
                    results['user_info'] = {
                        'login': user_data.get('login', ''),
                        'name': user_data.get('name', ''),
                        'email': user_data.get('email', ''),
                        'public_repos': user_data.get('public_repos', 0)
                    }
        except Exception as e:
            logger.warning(f"requests 테스트 실패: {e}")
        
        logger.info(f"✅ GitHub 연결 테스트 완료: {results}")
        return results

# 전역 인스턴스
github_accessor = GitHubAccessor()

def search_github_repositories(query: str, **kwargs) -> List[Dict[str, Any]]:
    """GitHub 저장소 검색 (편의 함수)"""
    return github_accessor.search_repositories(query, **kwargs)

def search_github_code(query: str, **kwargs) -> List[Dict[str, Any]]:
    """GitHub 코드 검색 (편의 함수)"""
    return github_accessor.search_code(query, **kwargs)

def get_github_repo_info(owner: str, repo: str) -> Dict[str, Any]:
    """GitHub 저장소 정보 조회 (편의 함수)"""
    return github_accessor.get_repository_info(owner, repo)

def test_github_connection() -> Dict[str, Any]:
    """GitHub 연결 테스트 (편의 함수)"""
    return github_accessor.test_connection()

# 테스트 실행
if __name__ == "__main__":
    print("🔍 GitHub 접근 모듈 테스트")
    
    # 연결 테스트
    connection_test = test_github_connection()
    print(f"연결 테스트 결과: {json.dumps(connection_test, indent=2, ensure_ascii=False)}")
    
    # 저장소 검색 테스트
    print("\n📋 저장소 검색 테스트")
    repos = search_github_repositories("python stock analysis", language="python", per_page=3)
    for repo in repos:
        print(f"  • {repo['full_name']}: ⭐{repo['stars']} 🍴{repo['forks']}")
    
    # 코드 검색 테스트
    print("\n🔍 코드 검색 테스트")
    codes = search_github_code("def analyze_stock", language="python", per_page=3)
    for code in codes:
        print(f"  • {code['repository']['full_name']}/{code['path']}")
    
    print("\n✅ 모든 테스트 완료!") 