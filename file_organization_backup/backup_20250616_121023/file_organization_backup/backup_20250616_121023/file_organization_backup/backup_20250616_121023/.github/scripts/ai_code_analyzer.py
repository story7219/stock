"""
AI 기반 코드 분석기
코드의 모듈화, 함수 분리, 클래스 책임 분리를 분석
"""

import os
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any
import google.generativeai as genai

class CodeAnalyzer:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def analyze_file_structure(self, file_path: str) -> Dict[str, Any]:
        """파일 구조 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                'file_path': file_path,
                'classes': [],
                'functions': [],
                'imports': [],
                'lines_of_code': len(content.split('\n')),
                'complexity_issues': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    analysis['classes'].append({
                        'name': node.name,
                        'methods': methods,
                        'method_count': len(methods),
                        'line_start': node.lineno
                    })
                
                elif isinstance(node, ast.FunctionDef):
                    # 함수 복잡도 계산 (간단한 버전)
                    complexity = self.calculate_function_complexity(node)
                    analysis['functions'].append({
                        'name': node.name,
                        'args_count': len(node.args.args),
                        'line_start': node.lineno,
                        'complexity': complexity
                    })
                    
                    if complexity > 10:
                        analysis['complexity_issues'].append({
                            'function': node.name,
                            'complexity': complexity,
                            'suggestion': '함수가 너무 복잡합니다. 더 작은 함수로 분리를 고려하세요.'
                        })
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis['imports'].append(node.module)
            
            return analysis
            
        except Exception as e:
            return {'error': str(e), 'file_path': file_path}
    
    def calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """함수 복잡도 계산 (McCabe 복잡도 간단 버전)"""
        complexity = 1  # 기본 복잡도
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    async def ai_analyze_code_structure(self, file_analyses: List[Dict]) -> str:
        """AI를 통한 코드 구조 분석"""
        
        # 분석 데이터 요약
        summary = self.create_analysis_summary(file_analyses)
        
        prompt = f"""
다음은 Python 자동매매 시스템의 코드 분석 결과입니다:

{summary}

다음 관점에서 코드 구조를 분석하고 개선 방안을 제시해주세요:

1. **모듈화 분석**:
   - 각 파일의 역할이 명확한가?
   - 관련 기능들이 적절히 그룹화되어 있는가?
   - 파일 크기가 적절한가? (500줄 이하 권장)

2. **함수/클래스 책임 분리**:
   - 단일 책임 원칙(SRP)을 잘 지키고 있는가?
   - 함수가 너무 길거나 복잡하지 않은가?
   - 클래스의 메서드 수가 적절한가?

3. **의존성 관리**:
   - import 구조가 깔끔한가?
   - 순환 의존성은 없는가?

4. **리팩토링 우선순위**:
   - 가장 시급한 리팩토링 대상
   - 구체적인 개선 방안

마크다운 형식으로 답변해주세요.
"""
        
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"AI 분석 중 오류 발생: {str(e)}"
    
    def create_analysis_summary(self, file_analyses: List[Dict]) -> str:
        """분석 결과 요약 생성"""
        summary = "## 📁 파일별 분석 결과\n\n"
        
        for analysis in file_analyses:
            if 'error' in analysis:
                continue
                
            summary += f"### {analysis['file_path']}\n"
            summary += f"- 코드 라인 수: {analysis['lines_of_code']}\n"
            summary += f"- 클래스 수: {len(analysis['classes'])}\n"
            summary += f"- 함수 수: {len(analysis['functions'])}\n"
            summary += f"- Import 수: {len(analysis['imports'])}\n"
            
            if analysis['complexity_issues']:
                summary += f"- ⚠️ 복잡도 이슈: {len(analysis['complexity_issues'])}개\n"
                for issue in analysis['complexity_issues'][:3]:  # 상위 3개만
                    summary += f"  - {issue['function']}: 복잡도 {issue['complexity']}\n"
            
            summary += "\n"
        
        return summary

async def main():
    """메인 실행 함수"""
    analyzer = CodeAnalyzer()
    
    # Python 파일들 찾기
    python_files = []
    for root, dirs, files in os.walk('.'):
        # .git, __pycache__ 등 제외
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py') and not file.startswith('.'):
                python_files.append(os.path.join(root, file))
    
    # 각 파일 분석
    file_analyses = []
    for file_path in python_files[:10]:  # 최대 10개 파일만 분석
        analysis = analyzer.analyze_file_structure(file_path)
        file_analyses.append(analysis)
    
    # AI 분석 실행
    ai_analysis = await analyzer.ai_analyze_code_structure(file_analyses)
    
    # 결과 저장
    with open('ai_analysis.md', 'w', encoding='utf-8') as f:
        f.write(ai_analysis)
    
    print("✅ AI 코드 분석 완료!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 