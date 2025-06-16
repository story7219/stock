"""
자동 리팩토링 제안 도구 (v2.0)
- 코드 품질 분석 및 개선 제안
- 간단하고 실용적인 분석 도구
"""
import ast
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SimpleCodeAnalyzer:
    """간단한 코드 분석 도구"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.python_files = [f for f in self.project_root.glob("**/*.py") 
                           if not any(skip in str(f) for skip in ['__pycache__', '.venv', '.git'])]
        self.issues = []
        
    def analyze_project(self):
        """프로젝트 분석 실행"""
        print("🔍 코드 품질 분석을 시작합니다...")
        
        for py_file in self.python_files:
            self._analyze_file(py_file)
        
        self._generate_report()
        
    def _analyze_file(self, file_path: Path):
        """개별 파일 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 함수 복잡도 검사
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_complexity(node)
                    if complexity > 10:
                        self.issues.append({
                            'file': file_path.name,
                            'line': node.lineno,
                            'type': 'complexity',
                            'severity': 'high' if complexity > 15 else 'medium',
                            'message': f"함수 '{node.name}'의 복잡도가 높습니다 (복잡도: {complexity})",
                            'suggestion': "함수를 더 작은 단위로 분리하세요."
                        })
                
                # bare except 검사
                elif isinstance(node, ast.Try):
                    for handler in node.handlers:
                        if handler.type is None:
                            self.issues.append({
                                'file': file_path.name,
                                'line': handler.lineno,
                                'type': 'error_handling',
                                'severity': 'medium',
                                'message': "bare except 사용을 피하세요",
                                'suggestion': "구체적인 예외 타입을 지정하세요."
                            })
                            
        except Exception as e:
            logger.warning(f"⚠️ {file_path} 분석 실패: {e}")
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """간단한 순환 복잡도 계산"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
        return complexity
    
    def _generate_report(self):
        """분석 결과 보고서 생성"""
        print("\n" + "="*60)
        print("🔍 코드 품질 분석 보고서")
        print("="*60)
        
        if not self.issues:
            print("\n✅ 발견된 이슈가 없습니다. 코드 품질이 양호합니다!")
            return
        
        # 심각도별 분류
        high_issues = [i for i in self.issues if i['severity'] == 'high']
        medium_issues = [i for i in self.issues if i['severity'] == 'medium']
        
        print(f"\n📊 이슈 요약:")
        print(f"  🔴 높음: {len(high_issues)}개")
        print(f"  🟡 보통: {len(medium_issues)}개")
        
        # 상세 이슈 출력
        if high_issues:
            print(f"\n🔴 높은 우선순위 이슈:")
            for issue in high_issues:
                print(f"  📁 {issue['file']}:{issue['line']}")
                print(f"     {issue['message']}")
                print(f"     💡 제안: {issue['suggestion']}")
                print()
        
        if medium_issues:
            print(f"\n🟡 보통 우선순위 이슈:")
            for issue in medium_issues[:5]:  # 상위 5개만
                print(f"  📁 {issue['file']}:{issue['line']}")
                print(f"     {issue['message']}")
                print(f"     💡 제안: {issue['suggestion']}")
                print()
        
        print("="*60)
        print(f"✅ 분석 완료: {len(self.python_files)}개 파일 분석됨")
        print("="*60)


def main():
    """메인 실행 함수"""
    analyzer = SimpleCodeAnalyzer()
    analyzer.analyze_project()


if __name__ == "__main__":
    main() 