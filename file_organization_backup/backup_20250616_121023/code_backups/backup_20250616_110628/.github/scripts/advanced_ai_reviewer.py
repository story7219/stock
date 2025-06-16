"""
고급 AI 코드 리뷰어
다중 AI 모델을 활용한 종합적 코드 분석
"""

import os
import sys
import json
import ast
import subprocess
from typing import List, Dict, Any
from pathlib import Path

# AI 모델 imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class AdvancedAIReviewer:
    """고급 AI 코드 리뷰어"""
    
    def __init__(self):
        self.setup_ai_models()
        self.review_results = []
        
    def setup_ai_models(self):
        """AI 모델 설정"""
        if GEMINI_AVAILABLE and os.getenv('GEMINI_API_KEY'):
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            print("✅ Gemini AI 모델 설정 완료")
        else:
            self.gemini_model = None
            print("⚠️ Gemini AI 사용 불가")
            
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.openai_available = True
            print("✅ OpenAI 모델 설정 완료")
        else:
            self.openai_available = False
            print("⚠️ OpenAI 사용 불가")
    
    def get_changed_files(self) -> List[str]:
        """변경된 파일 목록 가져오기"""
        try:
            # Git을 통해 변경된 파일 목록 가져오기
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD~1', 'HEAD'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                files = [f.strip() for f in result.stdout.split('\n') if f.strip().endswith('.py')]
                return files[:10]  # 최대 10개 파일만 분석
            else:
                print("⚠️ Git diff 실행 실패, 전체 Python 파일 분석")
                return list(Path('.').rglob('*.py'))[:10]
                
        except Exception as e:
            print(f"❌ 파일 목록 가져오기 실패: {e}")
            return []
    
    def analyze_file_structure(self, file_path: str) -> Dict[str, Any]:
        """파일 구조 상세 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                'file_path': file_path,
                'lines_of_code': len(content.split('\n')),
                'classes': [],
                'functions': [],
                'imports': [],
                'complexity_issues': [],
                'code_smells': [],
                'best_practices': []
            }
            
            # AST 분석
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    analysis['classes'].append({
                        'name': node.name,
                        'methods': methods,
                        'method_count': len(methods),
                        'line_start': node.lineno,
                        'docstring': ast.get_docstring(node)
                    })
                    
                    # 클래스 크기 체크
                    if len(methods) > 20:
                        analysis['code_smells'].append({
                            'type': 'large_class',
                            'location': f"Line {node.lineno}",
                            'message': f"클래스 '{node.name}'에 {len(methods)}개의 메서드가 있습니다. 단일 책임 원칙을 고려하세요."
                        })
                
                elif isinstance(node, ast.FunctionDef):
                    complexity = self.calculate_cyclomatic_complexity(node)
                    analysis['functions'].append({
                        'name': node.name,
                        'args_count': len(node.args.args),
                        'line_start': node.lineno,
                        'complexity': complexity,
                        'docstring': ast.get_docstring(node)
                    })
                    
                    # 복잡도 체크
                    if complexity > 15:
                        analysis['complexity_issues'].append({
                            'function': node.name,
                            'complexity': complexity,
                            'line': node.lineno,
                            'suggestion': '함수가 너무 복잡합니다. 더 작은 함수로 분리하세요.'
                        })
                    
                    # 문서화 체크
                    if not ast.get_docstring(node) and not node.name.startswith('_'):
                        analysis['best_practices'].append({
                            'type': 'missing_docstring',
                            'location': f"Line {node.lineno}",
                            'message': f"함수 '{node.name}'에 docstring이 없습니다."
                        })
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis['imports'].append(node.module)
            
            # 추가 코드 스멜 검사
            self.detect_code_smells(content, analysis)
            
            return analysis
            
        except Exception as e:
            return {'error': str(e), 'file_path': file_path}
    
    def calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """순환 복잡도 계산"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += 1
                complexity += len(child.handlers)
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def detect_code_smells(self, content: str, analysis: Dict):
        """코드 스멜 탐지"""
        lines = content.split('\n')
        
        # 긴 라인 체크
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                analysis['code_smells'].append({
                    'type': 'long_line',
                    'location': f"Line {i}",
                    'message': f"라인이 {len(line)}자로 너무 깁니다. (권장: 120자 이하)"
                })
        
        # TODO/FIXME 주석 체크
        for i, line in enumerate(lines, 1):
            if 'TODO' in line or 'FIXME' in line:
                analysis['code_smells'].append({
                    'type': 'todo_comment',
                    'location': f"Line {i}",
                    'message': "TODO/FIXME 주석이 발견되었습니다. 이슈로 등록하는 것을 고려하세요."
                })
        
        # 하드코딩된 값 체크 (간단한 버전)
        import re
        hardcoded_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']'
        ]
        
        for pattern in hardcoded_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                analysis['code_smells'].append({
                    'type': 'hardcoded_secret',
                    'location': f"Line {line_num}",
                    'message': "하드코딩된 비밀 정보가 발견되었습니다. 환경변수를 사용하세요."
                })
    
    async def ai_review_file(self, file_analysis: Dict) -> str:
        """AI를 통한 파일 리뷰"""
        if not self.gemini_model:
            return "AI 모델을 사용할 수 없습니다."
        
        prompt = f"""
다음은 Python 자동매매 시스템의 파일 분석 결과입니다:

파일: {file_analysis['file_path']}
코드 라인 수: {file_analysis['lines_of_code']}
클래스 수: {len(file_analysis.get('classes', []))}
함수 수: {len(file_analysis.get('functions', []))}

복잡도 이슈: {len(file_analysis.get('complexity_issues', []))}개
코드 스멜: {len(file_analysis.get('code_smells', []))}개

상세 분석:
{json.dumps(file_analysis, ensure_ascii=False, indent=2)}

다음 관점에서 코드를 리뷰해주세요:

1. **아키텍처 및 설계**:
   - 단일 책임 원칙 준수
   - 의존성 관리
   - 모듈화 수준

2. **코드 품질**:
   - 가독성 및 유지보수성
   - 네이밍 컨벤션
   - 코드 중복

3. **성능 및 최적화**:
   - 알고리즘 효율성
   - 메모리 사용
   - 비동기 처리

4. **보안 및 안정성**:
   - 에러 핸들링
   - 입력 검증
   - 보안 취약점

5. **자동매매 시스템 특화**:
   - 실시간 처리 적합성
   - 데이터 정확성
   - 리스크 관리

구체적인 개선 방안과 코드 예시를 포함하여 마크다운 형식으로 답변해주세요.
"""
        
        try:
            response = await self.gemini_model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"AI 리뷰 생성 실패: {str(e)}"
    
    async def generate_comprehensive_review(self, file_analyses: List[Dict]) -> str:
        """종합적인 리뷰 생성"""
        individual_reviews = []
        
        for analysis in file_analyses:
            if 'error' not in analysis:
                review = await self.ai_review_file(analysis)
                individual_reviews.append({
                    'file': analysis['file_path'],
                    'review': review
                })
        
        # 전체 프로젝트 종합 분석
        total_lines = sum(a.get('lines_of_code', 0) for a in file_analyses if 'error' not in a)
        total_complexity_issues = sum(len(a.get('complexity_issues', [])) for a in file_analyses if 'error' not in a)
        total_code_smells = sum(len(a.get('code_smells', [])) for a in file_analyses if 'error' not in a)
        
        comprehensive_review = f"""
# 🧠 AI 종합 코드 리뷰

## 📊 전체 분석 요약

- **분석된 파일**: {len(file_analyses)}개
- **총 코드 라인**: {total_lines:,}줄
- **복잡도 이슈**: {total_complexity_issues}개
- **코드 스멜**: {total_code_smells}개

## 📁 파일별 상세 리뷰

"""
        
        for review_data in individual_reviews:
            comprehensive_review += f"""
### 📄 {review_data['file']}

{review_data['review']}

---
"""
        
        # 전체 프로젝트 권장사항
        if self.gemini_model:
            try:
                project_summary_prompt = f"""
다음은 자동매매 시스템의 전체 분석 결과입니다:

- 총 {len(file_analyses)}개 파일 분석
- 총 {total_lines:,}줄의 코드
- {total_complexity_issues}개의 복잡도 이슈
- {total_code_smells}개의 코드 스멜

전체 프로젝트 관점에서 다음을 제안해주세요:

1. **우선순위 개선 사항** (상위 3개)
2. **아키텍처 개선 방향**
3. **성능 최적화 포인트**
4. **유지보수성 향상 방안**
5. **자동매매 시스템 안정성 강화**

마크다운 형식으로 답변해주세요.
"""
                
                project_recommendations = await self.gemini_model.generate_content_async(project_summary_prompt)
                comprehensive_review += f"""

## 🎯 전체 프로젝트 권장사항

{project_recommendations.text}
"""
            except Exception as e:
                comprehensive_review += f"\n⚠️ 전체 프로젝트 분석 실패: {str(e)}\n"
        
        return comprehensive_review

async def main():
    """메인 실행 함수"""
    print("🧠 고급 AI 코드 리뷰 시작...")
    
    reviewer = AdvancedAIReviewer()
    
    # 변경된 파일 분석
    changed_files = reviewer.get_changed_files()
    
    if not changed_files:
        print("❌ 분석할 파일이 없습니다.")
        return
    
    print(f"📁 분석할 파일: {len(changed_files)}개")
    
    # 각 파일 분석
    file_analyses = []
    for file_path in changed_files:
        print(f"🔍 분석 중: {file_path}")
        analysis = reviewer.analyze_file_structure(file_path)
        file_analyses.append(analysis)
    
    # 종합 리뷰 생성
    print("🤖 AI 리뷰 생성 중...")
    comprehensive_review = await reviewer.generate_comprehensive_review(file_analyses)
    
    # 결과 저장
    with open('ai_review_results.md', 'w', encoding='utf-8') as f:
        f.write(comprehensive_review)
    
    print("✅ AI 코드 리뷰 완료!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 