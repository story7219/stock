"""
자동 리팩토링 제안 생성기
"""

import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple
import google.generativeai as genai

class AutoRefactorer:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def detect_refactoring_opportunities(self, file_path: str) -> List[Dict]:
        """리팩토링 기회 탐지"""
        opportunities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            tree = ast.parse(content)
            
            # 1. 긴 함수 탐지
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_lines = self.count_function_lines(node, lines)
                    if func_lines > 50:
                        opportunities.append({
                            'type': 'long_function',
                            'location': f"{file_path}:{node.lineno}",
                            'function_name': node.name,
                            'lines': func_lines,
                            'priority': 'high',
                            'suggestion': f"함수 '{node.name}'이 {func_lines}줄로 너무 깁니다. 더 작은 함수들로 분리하세요."
                        })
                
                # 2. 큰 클래스 탐지
                elif isinstance(node, ast.ClassDef):
                    method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    if method_count > 15:
                        opportunities.append({
                            'type': 'large_class',
                            'location': f"{file_path}:{node.lineno}",
                            'class_name': node.name,
                            'method_count': method_count,
                            'priority': 'medium',
                            'suggestion': f"클래스 '{node.name}'에 {method_count}개의 메서드가 있습니다. 책임을 분리하여 여러 클래스로 나누는 것을 고려하세요."
                        })
            
            # 3. 중복 코드 패턴 탐지 (간단한 버전)
            duplicate_patterns = self.find_duplicate_patterns(content)
            for pattern in duplicate_patterns:
                opportunities.append({
                    'type': 'duplicate_code',
                    'location': file_path,
                    'pattern': pattern['pattern'][:100] + '...',
                    'occurrences': pattern['count'],
                    'priority': 'medium',
                    'suggestion': f"중복된 코드 패턴이 {pattern['count']}번 발견되었습니다. 공통 함수로 추출하세요."
                })
            
            # 4. 파일 크기 체크
            if len(lines) > 500:
                opportunities.append({
                    'type': 'large_file',
                    'location': file_path,
                    'lines': len(lines),
                    'priority': 'high',
                    'suggestion': f"파일이 {len(lines)}줄로 너무 큽니다. 관련 기능별로 여러 파일로 분리하세요."
                })
            
        except Exception as e:
            opportunities.append({
                'type': 'analysis_error',
                'location': file_path,
                'error': str(e),
                'priority': 'low',
                'suggestion': f"파일 분석 중 오류 발생: {str(e)}"
            })
        
        return opportunities
    
    def count_function_lines(self, node: ast.FunctionDef, lines: List[str]) -> int:
        """함수의 실제 라인 수 계산"""
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
        
        # 빈 줄과 주석만 있는 줄 제외
        actual_lines = 0
        for i in range(start_line, min(end_line, len(lines))):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                actual_lines += 1
        
        return actual_lines
    
    def find_duplicate_patterns(self, content: str) -> List[Dict]:
        """중복 코드 패턴 찾기 (간단한 버전)"""
        lines = content.split('\n')
        patterns = {}
        
        # 3줄 이상의 연속된 패턴 찾기
        for i in range(len(lines) - 2):
            pattern = '\n'.join(lines[i:i+3]).strip()
            if len(pattern) > 50 and not pattern.startswith('#'):  # 주석 제외
                if pattern in patterns:
                    patterns[pattern] += 1
                else:
                    patterns[pattern] = 1
        
        # 2번 이상 나타나는 패턴만 반환
        duplicates = []
        for pattern, count in patterns.items():
            if count > 1:
                duplicates.append({'pattern': pattern, 'count': count})
        
        return duplicates
    
    async def generate_refactor_suggestions(self, opportunities: List[Dict]) -> str:
        """AI를 통한 리팩토링 제안 생성"""
        
        if not opportunities:
            return "## ✅ 리팩토링 제안 없음\n\n코드 구조가 양호합니다!"
        
        # 우선순위별로 정렬
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        opportunities.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        prompt = f"""
다음은 자동매매 시스템 코드에서 발견된 리팩토링 기회들입니다:

{self.format_opportunities(opportunities)}

각 이슈에 대해 다음을 제공해주세요:

1. **구체적인 리팩토링 방법**
2. **예상되는 개선 효과**
3. **구현 우선순위**
4. **주의사항**

특히 자동매매 시스템의 특성을 고려하여:
- 성능에 미치는 영향
- 안정성 및 신뢰성
- 유지보수성 향상

마크다운 형식으로 답변해주세요.
"""
        
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"## ❌ AI 제안 생성 실패\n\n오류: {str(e)}"
    
    def format_opportunities(self, opportunities: List[Dict]) -> str:
        """리팩토링 기회들을 포맷팅"""
        formatted = ""
        
        for i, opp in enumerate(opportunities[:10], 1):  # 최대 10개만
            formatted += f"\n### {i}. {opp['type'].replace('_', ' ').title()}\n"
            formatted += f"- **위치**: {opp['location']}\n"
            formatted += f"- **우선순위**: {opp['priority']}\n"
            formatted += f"- **제안**: {opp['suggestion']}\n"
            
            # 추가 정보
            if 'lines' in opp:
                formatted += f"- **라인 수**: {opp['lines']}\n"
            if 'method_count' in opp:
                formatted += f"- **메서드 수**: {opp['method_count']}\n"
            if 'occurrences' in opp:
                formatted += f"- **발생 횟수**: {opp['occurrences']}\n"
            
            formatted += "\n"
        
        return formatted

async def main():
    """메인 실행 함수"""
    refactorer = AutoRefactorer()
    
    # Python 파일들 찾기
    python_files = []
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py') and not file.startswith('.'):
                python_files.append(os.path.join(root, file))
    
    # 모든 리팩토링 기회 수집
    all_opportunities = []
    for file_path in python_files:
        opportunities = refactorer.detect_refactoring_opportunities(file_path)
        all_opportunities.extend(opportunities)
    
    # AI 제안 생성
    suggestions = await refactorer.generate_refactor_suggestions(all_opportunities)
    
    # 결과 저장
    with open('refactor_suggestions.md', 'w', encoding='utf-8') as f:
        f.write(suggestions)
    
    print(f"✅ 리팩토링 제안 완료! ({len(all_opportunities)}개 기회 발견)")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 