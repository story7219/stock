#!/usr/bin/env python3
"""
🤖 고급 AI 코드 리뷰어
Gemini AI를 활용한 투자 시스템 전용 고급 코드 리뷰 도구
"""

import os
import sys
import json
import ast
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
import asyncio
import aiohttp

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedAIReviewer:
    """고급 AI 코드 리뷰어"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.review_results = []
        self.analysis_stats = {
            'files_reviewed': 0,
            'issues_found': 0,
            'suggestions_made': 0,
            'security_issues': 0,
            'performance_issues': 0,
            'investment_logic_issues': 0
        }
        
        # 투자 시스템 특화 패턴
        self.investment_patterns = {
            'strategy_methods': [
                'analyze_market', 'calculate_score', 'filter_stocks',
                'evaluate_risk', 'generate_signals', 'backtest'
            ],
            'risk_keywords': [
                'stop_loss', 'risk_management', 'position_size',
                'volatility', 'drawdown', 'var', 'sharpe_ratio'
            ],
            'data_sources': [
                'yahoo_finance', 'alpha_vantage', 'quandl',
                'bloomberg', 'reuters', 'fred'
            ]
        }
        
        logger.info(f"🤖 고급 AI 리뷰어 초기화 (프로젝트: {self.project_root})")
    
    async def analyze_code_with_ai(self, file_path: Path, code_content: str) -> Dict[str, Any]:
        """AI를 활용한 코드 분석"""
        if not self.gemini_api_key:
            logger.warning("⚠️ GEMINI_API_KEY 환경 변수가 설정되지 않음")
            return self._fallback_analysis(file_path, code_content)
        
        try:
            # Gemini AI 분석 요청
            analysis_prompt = self._create_analysis_prompt(file_path, code_content)
            ai_response = await self._call_gemini_api(analysis_prompt)
            
            if ai_response:
                return self._parse_ai_response(file_path, ai_response)
            else:
                return self._fallback_analysis(file_path, code_content)
                
        except Exception as e:
            logger.error(f"❌ AI 분석 오류 {file_path}: {e}")
            return self._fallback_analysis(file_path, code_content)
    
    def _create_analysis_prompt(self, file_path: Path, code_content: str) -> str:
        """AI 분석용 프롬프트 생성"""
        file_type = "투자 전략" if "strategy" in str(file_path).lower() else "일반 코드"
        
        prompt = f"""
다음은 투자 시스템의 {file_type} 파일입니다. 전문적인 코드 리뷰를 수행해주세요.

파일: {file_path}

코드:
```python
{code_content}
```

다음 관점에서 분석해주세요:

1. **보안 검토**
   - API 키 하드코딩 여부
   - 인증 정보 노출 위험
   - 입력 검증 부족
   - SQL 인젝션 가능성

2. **성능 최적화**
   - 비효율적인 데이터 구조 사용
   - 불필요한 반복문
   - 메모리 누수 가능성
   - 비동기 처리 개선점

3. **투자 로직 검증**
   - 전략 로직의 타당성
   - 리스크 관리 부족
   - 백테스팅 고려사항
   - 데이터 품질 검증

4. **코드 품질**
   - 가독성 개선점
   - 함수 분할 필요성
   - 예외 처리 부족
   - 타입 힌트 누락

5. **투자 시스템 특화**
   - 시장 데이터 처리 방식
   - 포트폴리오 관리 로직
   - 실시간 처리 고려사항
   - 규제 준수 여부

JSON 형식으로 응답해주세요:
{{
    "overall_score": 85,
    "security_issues": [
        {{"type": "hardcoded_secret", "line": 15, "severity": "high", "description": "API 키가 하드코딩됨", "suggestion": "환경 변수 사용"}}
    ],
    "performance_issues": [
        {{"type": "inefficient_loop", "line": 25, "severity": "medium", "description": "비효율적인 반복문", "suggestion": "벡터화 연산 사용"}}
    ],
    "investment_logic_issues": [
        {{"type": "missing_risk_check", "line": 35, "severity": "high", "description": "리스크 검증 누락", "suggestion": "포지션 크기 제한 추가"}}
    ],
    "code_quality_issues": [
        {{"type": "missing_docstring", "line": 10, "severity": "low", "description": "함수 설명 누락", "suggestion": "docstring 추가"}}
    ],
    "suggestions": [
        "비동기 처리로 성능 개선",
        "에러 핸들링 강화",
        "로깅 추가"
    ]
}}
"""
        return prompt
    
    async def _call_gemini_api(self, prompt: str) -> Optional[str]:
        """Gemini API 호출"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent"
            headers = {
                'Content-Type': 'application/json',
                'x-goog-api-key': self.gemini_api_key
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['candidates'][0]['content']['parts'][0]['text']
                    else:
                        logger.error(f"❌ Gemini API 오류: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Gemini API 호출 실패: {e}")
            return None
    
    def _parse_ai_response(self, file_path: Path, ai_response: str) -> Dict[str, Any]:
        """AI 응답 파싱"""
        try:
            # JSON 부분 추출
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                ai_analysis = json.loads(json_str)
                
                # 파일 정보 추가
                ai_analysis['file'] = str(file_path.relative_to(self.project_root))
                ai_analysis['timestamp'] = datetime.now().isoformat()
                
                return ai_analysis
            else:
                logger.warning(f"⚠️ AI 응답에서 JSON을 찾을 수 없음: {file_path}")
                return self._fallback_analysis(file_path, "")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ AI 응답 JSON 파싱 오류 {file_path}: {e}")
            return self._fallback_analysis(file_path, "")
    
    def _fallback_analysis(self, file_path: Path, code_content: str) -> Dict[str, Any]:
        """AI 분석 실패 시 대체 분석"""
        logger.info(f"🔄 대체 분석 수행: {file_path}")
        
        analysis = {
            'file': str(file_path.relative_to(self.project_root)),
            'timestamp': datetime.now().isoformat(),
            'overall_score': 75,
            'security_issues': [],
            'performance_issues': [],
            'investment_logic_issues': [],
            'code_quality_issues': [],
            'suggestions': []
        }
        
        # 기본 패턴 분석
        if code_content:
            analysis.update(self._basic_pattern_analysis(code_content))
        
        return analysis
    
    def _basic_pattern_analysis(self, code_content: str) -> Dict[str, List[Dict[str, Any]]]:
        """기본 패턴 분석"""
        issues = {
            'security_issues': [],
            'performance_issues': [],
            'investment_logic_issues': [],
            'code_quality_issues': []
        }
        
        lines = code_content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # 보안 이슈
            if re.search(r'(api_key|secret|password)\s*=\s*["\'][^"\']{8,}["\']', line_lower):
                issues['security_issues'].append({
                    'type': 'hardcoded_secret',
                    'line': i,
                    'severity': 'high',
                    'description': '하드코딩된 비밀정보 발견',
                    'suggestion': '환경 변수 사용'
                })
            
            # 성능 이슈
            if 'for' in line_lower and 'range(len(' in line_lower:
                issues['performance_issues'].append({
                    'type': 'inefficient_loop',
                    'line': i,
                    'severity': 'medium',
                    'description': '비효율적인 반복문 패턴',
                    'suggestion': 'enumerate() 사용'
                })
            
            # 투자 로직 이슈
            if any(keyword in line_lower for keyword in ['buy', 'sell', 'trade']):
                if 'risk' not in line_lower and 'stop' not in line_lower:
                    issues['investment_logic_issues'].append({
                        'type': 'missing_risk_check',
                        'line': i,
                        'severity': 'medium',
                        'description': '거래 로직에 리스크 관리 부족',
                        'suggestion': '리스크 검증 로직 추가'
                    })
            
            # 코드 품질 이슈
            if line.strip().startswith('def ') and '"""' not in code_content[code_content.find(line):code_content.find(line) + 200]:
                issues['code_quality_issues'].append({
                    'type': 'missing_docstring',
                    'line': i,
                    'severity': 'low',
                    'description': '함수 docstring 누락',
                    'suggestion': 'docstring 추가'
                })
        
        return issues
    
    def analyze_investment_strategy_logic(self, file_path: Path, code_content: str) -> List[Dict[str, Any]]:
        """투자 전략 로직 특화 분석"""
        issues = []
        
        try:
            tree = ast.parse(code_content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # 전략 함수 분석
                    if any(pattern in node.name.lower() for pattern in self.investment_patterns['strategy_methods']):
                        strategy_issues = self._analyze_strategy_function(node, code_content)
                        issues.extend(strategy_issues)
                    
                    # 리스크 관리 함수 확인
                    if any(keyword in node.name.lower() for keyword in self.investment_patterns['risk_keywords']):
                        risk_issues = self._analyze_risk_function(node, code_content)
                        issues.extend(risk_issues)
        
        except Exception as e:
            logger.warning(f"⚠️ 투자 전략 분석 오류 {file_path}: {e}")
        
        return issues
    
    def _analyze_strategy_function(self, node: ast.FunctionDef, code_content: str) -> List[Dict[str, Any]]:
        """전략 함수 분석"""
        issues = []
        
        # 백테스팅 고려사항 확인
        func_source = ast.get_source_segment(code_content, node) or ""
        if 'backtest' not in func_source.lower() and 'historical' not in func_source.lower():
            issues.append({
                'type': 'missing_backtest',
                'line': node.lineno,
                'severity': 'medium',
                'description': f'전략 함수 {node.name}에 백테스팅 고려사항 부족',
                'suggestion': '백테스팅 로직 추가 고려'
            })
        
        # 데이터 검증 확인
        if 'validate' not in func_source.lower() and 'check' not in func_source.lower():
            issues.append({
                'type': 'missing_validation',
                'line': node.lineno,
                'severity': 'medium',
                'description': f'전략 함수 {node.name}에 데이터 검증 부족',
                'suggestion': '입력 데이터 검증 로직 추가'
            })
        
        return issues
    
    def _analyze_risk_function(self, node: ast.FunctionDef, code_content: str) -> List[Dict[str, Any]]:
        """리스크 관리 함수 분석"""
        issues = []
        
        func_source = ast.get_source_segment(code_content, node) or ""
        
        # 포지션 크기 제한 확인
        if 'position' in func_source.lower() and 'limit' not in func_source.lower():
            issues.append({
                'type': 'missing_position_limit',
                'line': node.lineno,
                'severity': 'high',
                'description': f'리스크 함수 {node.name}에 포지션 크기 제한 부족',
                'suggestion': '최대 포지션 크기 제한 추가'
            })
        
        # 손절 로직 확인
        if 'stop' not in func_source.lower() and 'loss' not in func_source.lower():
            issues.append({
                'type': 'missing_stop_loss',
                'line': node.lineno,
                'severity': 'high',
                'description': f'리스크 함수 {node.name}에 손절 로직 부족',
                'suggestion': '손절 로직 추가'
            })
        
        return issues
    
    async def review_file(self, file_path: Path) -> Dict[str, Any]:
        """개별 파일 리뷰"""
        logger.info(f"📝 파일 리뷰 중: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # AI 분석
            ai_analysis = await self.analyze_code_with_ai(file_path, code_content)
            
            # 투자 전략 특화 분석
            investment_issues = self.analyze_investment_strategy_logic(file_path, code_content)
            ai_analysis['investment_logic_issues'].extend(investment_issues)
            
            # 통계 업데이트
            self.analysis_stats['files_reviewed'] += 1
            self.analysis_stats['issues_found'] += len(ai_analysis.get('security_issues', []))
            self.analysis_stats['issues_found'] += len(ai_analysis.get('performance_issues', []))
            self.analysis_stats['issues_found'] += len(ai_analysis.get('investment_logic_issues', []))
            self.analysis_stats['issues_found'] += len(ai_analysis.get('code_quality_issues', []))
            
            self.analysis_stats['security_issues'] += len(ai_analysis.get('security_issues', []))
            self.analysis_stats['performance_issues'] += len(ai_analysis.get('performance_issues', []))
            self.analysis_stats['investment_logic_issues'] += len(ai_analysis.get('investment_logic_issues', []))
            self.analysis_stats['suggestions_made'] += len(ai_analysis.get('suggestions', []))
            
            return ai_analysis
            
        except Exception as e:
            logger.error(f"❌ 파일 리뷰 오류 {file_path}: {e}")
            return {
                'file': str(file_path.relative_to(self.project_root)),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_comprehensive_review(self) -> Dict[str, Any]:
        """종합적인 코드 리뷰 실행"""
        logger.info("🤖 고급 AI 코드 리뷰 시작")
        
        start_time = datetime.now()
        
        # Python 파일들 수집
        python_files = list(self.project_root.rglob('*.py'))
        logger.info(f"📁 {len(python_files)}개 Python 파일 발견")
        
        # 병렬 리뷰 실행
        tasks = []
        for py_file in python_files:
            if py_file.is_file():
                task = self.review_file(py_file)
                tasks.append(task)
        
        # 모든 리뷰 완료 대기
        review_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 정리
        successful_reviews = []
        failed_reviews = []
        
        for result in review_results:
            if isinstance(result, Exception):
                failed_reviews.append(str(result))
            elif 'error' in result:
                failed_reviews.append(result)
            else:
                successful_reviews.append(result)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 종합 결과
        comprehensive_result = {
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'stats': self.analysis_stats,
            'successful_reviews': successful_reviews,
            'failed_reviews': failed_reviews,
            'summary': {
                'total_files': len(python_files),
                'reviewed_files': len(successful_reviews),
                'failed_files': len(failed_reviews),
                'total_issues': self.analysis_stats['issues_found'],
                'average_score': self._calculate_average_score(successful_reviews)
            },
            'recommendations': self._generate_comprehensive_recommendations(successful_reviews)
        }
        
        logger.info(f"✅ 고급 AI 리뷰 완료: {len(successful_reviews)}개 파일 리뷰, {duration:.1f}초 소요")
        
        return comprehensive_result
    
    def _calculate_average_score(self, reviews: List[Dict[str, Any]]) -> float:
        """평균 점수 계산"""
        if not reviews:
            return 0.0
        
        total_score = sum(review.get('overall_score', 0) for review in reviews)
        return total_score / len(reviews)
    
    def _generate_comprehensive_recommendations(self, reviews: List[Dict[str, Any]]) -> List[str]:
        """종합 권장사항 생성"""
        recommendations = []
        
        # 보안 이슈 집계
        security_count = sum(len(review.get('security_issues', [])) for review in reviews)
        if security_count > 0:
            recommendations.append(f"🔒 {security_count}개 보안 이슈 해결 필요")
        
        # 성능 이슈 집계
        performance_count = sum(len(review.get('performance_issues', [])) for review in reviews)
        if performance_count > 0:
            recommendations.append(f"⚡ {performance_count}개 성능 최적화 필요")
        
        # 투자 로직 이슈 집계
        investment_count = sum(len(review.get('investment_logic_issues', [])) for review in reviews)
        if investment_count > 0:
            recommendations.append(f"💰 {investment_count}개 투자 로직 개선 필요")
        
        # 공통 제안사항 추출
        all_suggestions = []
        for review in reviews:
            all_suggestions.extend(review.get('suggestions', []))
        
        # 빈도 높은 제안사항 추가
        from collections import Counter
        common_suggestions = Counter(all_suggestions).most_common(5)
        for suggestion, count in common_suggestions:
            if count > 1:
                recommendations.append(f"💡 {suggestion} ({count}회 제안)")
        
        return recommendations
    
    def save_review_report(self, result: Dict[str, Any], output_file: str = "ai_review_report.json"):
        """리뷰 결과 저장"""
        output_path = self.project_root / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 AI 리뷰 보고서 저장: {output_path}")
        
        # 마크다운 보고서도 생성
        self.save_markdown_report(result, str(output_path).replace('.json', '.md'))
    
    def save_markdown_report(self, result: Dict[str, Any], output_file: str):
        """마크다운 형식 보고서 저장"""
        md_content = f"""# 🤖 고급 AI 코드 리뷰 보고서

**실행 시간**: {result['timestamp']}  
**소요 시간**: {result['duration_seconds']:.1f}초  
**평균 점수**: {result['summary']['average_score']:.1f}/100

## 📊 요약

- **총 파일**: {result['summary']['total_files']}개
- **리뷰 완료**: {result['summary']['reviewed_files']}개
- **리뷰 실패**: {result['summary']['failed_files']}개
- **총 이슈**: {result['summary']['total_issues']}개

### 이슈 분류
- **보안 이슈**: {result['stats']['security_issues']}개
- **성능 이슈**: {result['stats']['performance_issues']}개
- **투자 로직 이슈**: {result['stats']['investment_logic_issues']}개
- **제안사항**: {result['stats']['suggestions_made']}개

## 🎯 주요 권장사항

"""
        
        for rec in result['recommendations']:
            md_content += f"- {rec}\n"
        
        # 상세 리뷰 결과
        md_content += "\n## 📋 상세 리뷰 결과\n\n"
        
        for review in result['successful_reviews'][:10]:  # 상위 10개만 표시
            md_content += f"### {review['file']}\n\n"
            md_content += f"**점수**: {review.get('overall_score', 0)}/100\n\n"
            
            if review.get('security_issues'):
                md_content += "**보안 이슈**:\n"
                for issue in review['security_issues'][:3]:  # 상위 3개만
                    md_content += f"- 라인 {issue['line']}: {issue['description']}\n"
                md_content += "\n"
            
            if review.get('suggestions'):
                md_content += "**제안사항**:\n"
                for suggestion in review['suggestions'][:3]:  # 상위 3개만
                    md_content += f"- {suggestion}\n"
                md_content += "\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"📄 마크다운 보고서 저장: {output_file}")

async def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(description='고급 AI 코드 리뷰어')
    parser.add_argument('--project-root', default='.', help='프로젝트 루트 디렉토리')
    parser.add_argument('--output', default='ai_review_report.json', help='출력 파일명')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # AI 리뷰 실행
    reviewer = AdvancedAIReviewer(args.project_root)
    result = await reviewer.run_comprehensive_review()
    reviewer.save_review_report(result, args.output)
    
    # 결과 출력
    print(f"\n🤖 고급 AI 코드 리뷰 완료!")
    print(f"📊 평균 점수: {result['summary']['average_score']:.1f}/100")
    print(f"📁 리뷰된 파일: {result['summary']['reviewed_files']}개")
    print(f"🚨 총 이슈: {result['summary']['total_issues']}개")
    
    if result['summary']['average_score'] < 70:
        print("⚠️ 코드 품질 개선이 필요합니다")
        sys.exit(1)
    else:
        print("✅ 코드 품질이 양호합니다")

if __name__ == "__main__":
    asyncio.run(main()) 