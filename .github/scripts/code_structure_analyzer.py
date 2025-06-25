#!/usr/bin/env python3
"""
🏗️ 코드 구조 분석기
프로젝트의 코드 구조를 분석하고 아키텍처 품질을 평가하는 도구
"""

import os
import sys
import json
import ast
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeStructureAnalyzer:
    """코드 구조 분석기"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'modules': {},
            'dependencies': {},
            'metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        logger.info(f"🏗️ 코드 구조 분석기 초기화 (프로젝트: {self.project_root})")
    
    def analyze_file_structure(self) -> Dict[str, Any]:
        """파일 구조 분석"""
        logger.info("📁 파일 구조 분석 중...")
        
        structure = {
            'total_files': 0,
            'python_files': 0,
            'test_files': 0,
            'config_files': 0,
            'documentation_files': 0,
            'directories': [],
            'file_types': Counter(),
            'largest_files': [],
            'empty_files': []
        }
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                structure['total_files'] += 1
                
                # 파일 확장자별 분류
                suffix = file_path.suffix.lower()
                structure['file_types'][suffix] += 1
                
                # 파일 크기 확인
                file_size = file_path.stat().st_size
                
                if suffix == '.py':
                    structure['python_files'] += 1
                    
                    # 테스트 파일 확인
                    if 'test' in file_path.name.lower() or file_path.parent.name == 'tests':
                        structure['test_files'] += 1
                    
                    # 빈 파일 확인
                    if file_size == 0:
                        structure['empty_files'].append(str(file_path.relative_to(self.project_root)))
                    
                    # 큰 파일 추적 (상위 10개)
                    structure['largest_files'].append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'size': file_size,
                        'lines': self._count_lines(file_path)
                    })
                
                elif suffix in ['.ini', '.conf', '.cfg', '.json', '.yaml', '.yml', '.toml']:
                    structure['config_files'] += 1
                
                elif suffix in ['.md', '.rst', '.txt']:
                    structure['documentation_files'] += 1
            
            elif file_path.is_dir():
                structure['directories'].append(str(file_path.relative_to(self.project_root)))
        
        # 큰 파일 정렬 (상위 10개만)
        structure['largest_files'] = sorted(
            structure['largest_files'], 
            key=lambda x: x['size'], 
            reverse=True
        )[:10]
        
        return structure
    
    def _count_lines(self, file_path: Path) -> int:
        """파일의 라인 수 계산"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return len(f.readlines())
        except Exception:
            return 0
    
    def analyze_module_dependencies(self) -> Dict[str, Any]:
        """모듈 의존성 분석"""
        logger.info("🔗 모듈 의존성 분석 중...")
        
        dependencies = {
            'imports': defaultdict(set),
            'internal_dependencies': defaultdict(set),
            'external_dependencies': set(),
            'circular_dependencies': [],
            'unused_imports': [],
            'dependency_graph': {}
        }
        
        # 모든 Python 파일 분석
        for py_file in self.project_root.rglob('*.py'):
            if py_file.is_file():
                module_name = self._get_module_name(py_file)
                file_imports = self._analyze_imports(py_file)
                
                dependencies['imports'][module_name] = file_imports['all_imports']
                dependencies['external_dependencies'].update(file_imports['external'])
                dependencies['internal_dependencies'][module_name] = file_imports['internal']
                
                # 의존성 그래프 구축
                dependencies['dependency_graph'][module_name] = list(file_imports['internal'])
        
        # 순환 의존성 검사
        dependencies['circular_dependencies'] = self._find_circular_dependencies(
            dependencies['dependency_graph']
        )
        
        return dependencies
    
    def _get_module_name(self, file_path: Path) -> str:
        """파일 경로에서 모듈명 추출"""
        relative_path = file_path.relative_to(self.project_root)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        return '.'.join(module_parts).replace('__init__', '')
    
    def _analyze_imports(self, file_path: Path) -> Dict[str, Set[str]]:
        """파일의 import 분석"""
        imports = {
            'all_imports': set(),
            'internal': set(),
            'external': set(),
            'unused': set()
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.name
                        imports['all_imports'].add(import_name)
                        
                        if self._is_internal_module(import_name):
                            imports['internal'].add(import_name)
                        else:
                            imports['external'].add(import_name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module
                        imports['all_imports'].add(module_name)
                        
                        if self._is_internal_module(module_name):
                            imports['internal'].add(module_name)
                        else:
                            imports['external'].add(module_name)
                        
                        # from 절의 개별 import들도 추가
                        for alias in node.names:
                            full_name = f"{module_name}.{alias.name}"
                            imports['all_imports'].add(full_name)
        
        except Exception as e:
            logger.warning(f"⚠️ Import 분석 오류 {file_path}: {e}")
        
        return imports
    
    def _is_internal_module(self, module_name: str) -> bool:
        """내부 모듈인지 확인"""
        # 프로젝트 내부 모듈 패턴
        internal_patterns = ['src', 'modules', 'core', 'utils', 'config']
        
        for pattern in internal_patterns:
            if module_name.startswith(pattern):
                return True
        
        # 상대 경로 import
        if module_name.startswith('.'):
            return True
        
        # 프로젝트 루트에 해당 모듈 파일이 있는지 확인
        module_path = self.project_root / f"{module_name.replace('.', '/')}.py"
        if module_path.exists():
            return True
        
        return False
    
    def _find_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """순환 의존성 찾기"""
        circular_deps = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # 순환 발견
                cycle_start = path.index(node)
                circular_deps.append(path[cycle_start:] + [node])
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                if dfs(neighbor, path + [node]):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in dependency_graph:
            if node not in visited:
                dfs(node, [])
        
        return circular_deps
    
    def analyze_code_complexity(self) -> Dict[str, Any]:
        """코드 복잡도 분석"""
        logger.info("🧮 코드 복잡도 분석 중...")
        
        complexity = {
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'average_function_length': 0,
            'average_class_length': 0,
            'cyclomatic_complexity': {},
            'complex_functions': [],
            'large_classes': [],
            'long_functions': []
        }
        
        all_function_lengths = []
        all_class_lengths = []
        
        for py_file in self.project_root.rglob('*.py'):
            if py_file.is_file():
                file_analysis = self._analyze_file_complexity(py_file)
                
                complexity['total_lines'] += file_analysis['lines']
                complexity['total_functions'] += file_analysis['functions']
                complexity['total_classes'] += file_analysis['classes']
                
                all_function_lengths.extend(file_analysis['function_lengths'])
                all_class_lengths.extend(file_analysis['class_lengths'])
                
                # 복잡한 함수들 추가
                complexity['complex_functions'].extend(file_analysis['complex_functions'])
                complexity['large_classes'].extend(file_analysis['large_classes'])
                complexity['long_functions'].extend(file_analysis['long_functions'])
        
        # 평균 계산
        if all_function_lengths:
            complexity['average_function_length'] = sum(all_function_lengths) / len(all_function_lengths)
        
        if all_class_lengths:
            complexity['average_class_length'] = sum(all_class_lengths) / len(all_class_lengths)
        
        # 상위 복잡도 함수들만 유지
        complexity['complex_functions'] = sorted(
            complexity['complex_functions'], 
            key=lambda x: x['complexity'], 
            reverse=True
        )[:20]
        
        complexity['large_classes'] = sorted(
            complexity['large_classes'], 
            key=lambda x: x['lines'], 
            reverse=True
        )[:10]
        
        complexity['long_functions'] = sorted(
            complexity['long_functions'], 
            key=lambda x: x['lines'], 
            reverse=True
        )[:10]
        
        return complexity
    
    def _analyze_file_complexity(self, file_path: Path) -> Dict[str, Any]:
        """개별 파일 복잡도 분석"""
        analysis = {
            'lines': 0,
            'functions': 0,
            'classes': 0,
            'function_lengths': [],
            'class_lengths': [],
            'complex_functions': [],
            'large_classes': [],
            'long_functions': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                analysis['lines'] = len(content.split('\n'))
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'] += 1
                    func_lines = self._count_node_lines(node)
                    analysis['function_lengths'].append(func_lines)
                    
                    # 긴 함수 체크 (50줄 이상)
                    if func_lines > 50:
                        analysis['long_functions'].append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'name': node.name,
                            'lines': func_lines,
                            'start_line': node.lineno
                        })
                    
                    # 순환 복잡도 계산 (간단한 버전)
                    complexity = self._calculate_cyclomatic_complexity(node)
                    if complexity > 10:
                        analysis['complex_functions'].append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'name': node.name,
                            'complexity': complexity,
                            'lines': func_lines,
                            'start_line': node.lineno
                        })
                
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'] += 1
                    class_lines = self._count_node_lines(node)
                    analysis['class_lengths'].append(class_lines)
                    
                    # 큰 클래스 체크 (200줄 이상)
                    if class_lines > 200:
                        analysis['large_classes'].append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'name': node.name,
                            'lines': class_lines,
                            'start_line': node.lineno,
                            'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                        })
        
        except Exception as e:
            logger.warning(f"⚠️ 파일 복잡도 분석 오류 {file_path}: {e}")
        
        return analysis
    
    def _count_node_lines(self, node: ast.AST) -> int:
        """AST 노드의 라인 수 계산"""
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            return node.end_lineno - node.lineno + 1
        return 1
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """순환 복잡도 계산 (간단한 버전)"""
        complexity = 1  # 기본 복잡도
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                if isinstance(child.op, (ast.And, ast.Or)):
                    complexity += len(child.values) - 1
        
        return complexity
    
    def analyze_code_quality_patterns(self) -> Dict[str, Any]:
        """코드 품질 패턴 분석"""
        logger.info("✨ 코드 품질 패턴 분석 중...")
        
        patterns = {
            'docstring_coverage': 0,
            'type_hint_coverage': 0,
            'test_coverage_estimate': 0,
            'naming_issues': [],
            'code_smells': [],
            'best_practices': {
                'followed': [],
                'violated': []
            }
        }
        
        total_functions = 0
        documented_functions = 0
        type_hinted_functions = 0
        
        for py_file in self.project_root.rglob('*.py'):
            if py_file.is_file():
                file_patterns = self._analyze_file_patterns(py_file)
                
                total_functions += file_patterns['total_functions']
                documented_functions += file_patterns['documented_functions']
                type_hinted_functions += file_patterns['type_hinted_functions']
                
                patterns['naming_issues'].extend(file_patterns['naming_issues'])
                patterns['code_smells'].extend(file_patterns['code_smells'])
        
        # 커버리지 계산
        if total_functions > 0:
            patterns['docstring_coverage'] = (documented_functions / total_functions) * 100
            patterns['type_hint_coverage'] = (type_hinted_functions / total_functions) * 100
        
        # 테스트 커버리지 추정 (테스트 파일 수 기반)
        test_files = len(list(self.project_root.rglob('*test*.py')))
        source_files = len(list(self.project_root.rglob('*.py'))) - test_files
        if source_files > 0:
            patterns['test_coverage_estimate'] = min((test_files / source_files) * 100, 100)
        
        return patterns
    
    def _analyze_file_patterns(self, file_path: Path) -> Dict[str, Any]:
        """개별 파일 패턴 분석"""
        analysis = {
            'total_functions': 0,
            'documented_functions': 0,
            'type_hinted_functions': 0,
            'naming_issues': [],
            'code_smells': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['total_functions'] += 1
                    
                    # Docstring 체크
                    if ast.get_docstring(node):
                        analysis['documented_functions'] += 1
                    
                    # 타입 힌트 체크
                    if node.returns or any(arg.annotation for arg in node.args.args):
                        analysis['type_hinted_functions'] += 1
                    
                    # 네이밍 규칙 체크
                    if not self._is_valid_function_name(node.name):
                        analysis['naming_issues'].append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'type': 'function',
                            'name': node.name,
                            'line': node.lineno,
                            'issue': '함수명이 snake_case 규칙을 따르지 않음'
                        })
                    
                    # 함수 길이 체크 (코드 스멜)
                    func_lines = self._count_node_lines(node)
                    if func_lines > 100:
                        analysis['code_smells'].append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'type': 'long_function',
                            'name': node.name,
                            'line': node.lineno,
                            'lines': func_lines,
                            'description': f'함수가 너무 김 ({func_lines}줄)'
                        })
                
                elif isinstance(node, ast.ClassDef):
                    # 클래스 네이밍 규칙 체크
                    if not self._is_valid_class_name(node.name):
                        analysis['naming_issues'].append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'type': 'class',
                            'name': node.name,
                            'line': node.lineno,
                            'issue': '클래스명이 PascalCase 규칙을 따르지 않음'
                        })
        
        except Exception as e:
            logger.warning(f"⚠️ 파일 패턴 분석 오류 {file_path}: {e}")
        
        return analysis
    
    def _is_valid_function_name(self, name: str) -> bool:
        """함수명 유효성 검사 (snake_case)"""
        return re.match(r'^[a-z_][a-z0-9_]*$', name) is not None
    
    def _is_valid_class_name(self, name: str) -> bool:
        """클래스명 유효성 검사 (PascalCase)"""
        return re.match(r'^[A-Z][a-zA-Z0-9]*$', name) is not None
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """분석 결과 기반 권장사항 생성"""
        recommendations = []
        
        # 파일 구조 권장사항
        structure = analysis.get('file_structure', {})
        if structure.get('empty_files'):
            recommendations.append(f"📁 {len(structure['empty_files'])}개의 빈 파일을 정리하세요")
        
        if structure.get('python_files', 0) > 0 and structure.get('test_files', 0) == 0:
            recommendations.append("🧪 테스트 파일이 없습니다. 단위 테스트를 추가하세요")
        
        # 의존성 권장사항
        dependencies = analysis.get('dependencies', {})
        if dependencies.get('circular_dependencies'):
            recommendations.append(f"🔄 {len(dependencies['circular_dependencies'])}개의 순환 의존성을 해결하세요")
        
        # 복잡도 권장사항
        complexity = analysis.get('complexity', {})
        if complexity.get('complex_functions'):
            recommendations.append(f"🧮 {len(complexity['complex_functions'])}개의 복잡한 함수를 리팩토링하세요")
        
        if complexity.get('large_classes'):
            recommendations.append(f"📦 {len(complexity['large_classes'])}개의 큰 클래스를 분할하세요")
        
        # 품질 권장사항
        quality = analysis.get('quality_patterns', {})
        if quality.get('docstring_coverage', 0) < 70:
            recommendations.append(f"📝 Docstring 커버리지가 낮습니다 ({quality.get('docstring_coverage', 0):.1f}%)")
        
        if quality.get('type_hint_coverage', 0) < 50:
            recommendations.append(f"🏷️ 타입 힌트 커버리지가 낮습니다 ({quality.get('type_hint_coverage', 0):.1f}%)")
        
        return recommendations
    
    def calculate_overall_score(self, analysis: Dict[str, Any]) -> int:
        """전체 코드 품질 점수 계산"""
        score = 100
        
        # 구조 점수 (25%)
        structure = analysis.get('file_structure', {})
        if structure.get('empty_files'):
            score -= len(structure['empty_files']) * 2
        
        # 의존성 점수 (25%)
        dependencies = analysis.get('dependencies', {})
        score -= len(dependencies.get('circular_dependencies', [])) * 10
        
        # 복잡도 점수 (25%)
        complexity = analysis.get('complexity', {})
        score -= len(complexity.get('complex_functions', [])) * 2
        score -= len(complexity.get('large_classes', [])) * 5
        
        # 품질 점수 (25%)
        quality = analysis.get('quality_patterns', {})
        docstring_penalty = max(0, 70 - quality.get('docstring_coverage', 0)) * 0.3
        type_hint_penalty = max(0, 50 - quality.get('type_hint_coverage', 0)) * 0.2
        score -= docstring_penalty + type_hint_penalty
        
        return max(0, int(score))
    
    def run_analysis(self) -> Dict[str, Any]:
        """전체 구조 분석 실행"""
        logger.info("🏗️ 코드 구조 분석 시작")
        
        # 각 분석 실행
        self.analysis_result['file_structure'] = self.analyze_file_structure()
        self.analysis_result['dependencies'] = self.analyze_module_dependencies()
        self.analysis_result['complexity'] = self.analyze_code_complexity()
        self.analysis_result['quality_patterns'] = self.analyze_code_quality_patterns()
        
        # 권장사항 생성
        self.analysis_result['recommendations'] = self.generate_recommendations(self.analysis_result)
        
        # 전체 점수 계산
        self.analysis_result['overall_score'] = self.calculate_overall_score(self.analysis_result)
        
        logger.info(f"✅ 코드 구조 분석 완료: 점수 {self.analysis_result['overall_score']}/100")
        
        return self.analysis_result
    
    def save_report(self, output_file: str = "code_structure_report.json"):
        """분석 결과 저장"""
        output_path = self.project_root / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 구조 분석 보고서 저장: {output_path}")
        
        # 마크다운 보고서도 생성
        self.save_markdown_report(str(output_path).replace('.json', '.md'))
    
    def save_markdown_report(self, output_file: str):
        """마크다운 형식 보고서 저장"""
        result = self.analysis_result
        
        md_content = f"""# 🏗️ 코드 구조 분석 보고서

**생성 시간**: {result['timestamp']}  
**프로젝트**: {result['project_root']}  
**전체 점수**: {result['overall_score']}/100

## 📊 요약

### 📁 파일 구조
- **총 파일**: {result['file_structure']['total_files']}개
- **Python 파일**: {result['file_structure']['python_files']}개
- **테스트 파일**: {result['file_structure']['test_files']}개
- **설정 파일**: {result['file_structure']['config_files']}개

### 🔗 의존성
- **외부 의존성**: {len(result['dependencies']['external_dependencies'])}개
- **순환 의존성**: {len(result['dependencies']['circular_dependencies'])}개

### 🧮 복잡도
- **총 함수**: {result['complexity']['total_functions']}개
- **총 클래스**: {result['complexity']['total_classes']}개
- **평균 함수 길이**: {result['complexity']['average_function_length']:.1f}줄

### ✨ 품질
- **Docstring 커버리지**: {result['quality_patterns']['docstring_coverage']:.1f}%
- **타입 힌트 커버리지**: {result['quality_patterns']['type_hint_coverage']:.1f}%
- **테스트 커버리지 추정**: {result['quality_patterns']['test_coverage_estimate']:.1f}%

## 🎯 주요 권장사항

"""
        
        for rec in result['recommendations']:
            md_content += f"- {rec}\n"
        
        # 상세 이슈들
        if result['complexity']['complex_functions']:
            md_content += "\n## 🧮 복잡한 함수들\n\n"
            for func in result['complexity']['complex_functions'][:10]:
                md_content += f"- **{func['file']}:{func['start_line']}** `{func['name']}()` - 복잡도: {func['complexity']}, 길이: {func['lines']}줄\n"
        
        if result['dependencies']['circular_dependencies']:
            md_content += "\n## 🔄 순환 의존성\n\n"
            for cycle in result['dependencies']['circular_dependencies']:
                md_content += f"- {' → '.join(cycle)}\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"📄 마크다운 보고서 저장: {output_file}")

def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(description='코드 구조 분석 도구')
    parser.add_argument('--project-root', default='.', help='프로젝트 루트 디렉토리')
    parser.add_argument('--output', default='code_structure_report.json', help='출력 파일명')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 구조 분석 실행
    analyzer = CodeStructureAnalyzer(args.project_root)
    result = analyzer.run_analysis()
    analyzer.save_report(args.output)
    
    # 결과 출력
    print(f"\n🏗️ 코드 구조 분석 완료!")
    print(f"📊 전체 점수: {result['overall_score']}/100")
    print(f"📁 Python 파일: {result['file_structure']['python_files']}개")
    print(f"🧮 복잡한 함수: {len(result['complexity']['complex_functions'])}개")
    print(f"🔄 순환 의존성: {len(result['dependencies']['circular_dependencies'])}개")
    
    if result['overall_score'] < 70:
        print("⚠️ 코드 구조 개선이 필요합니다")
        sys.exit(1)
    else:
        print("✅ 코드 구조가 양호합니다")

if __name__ == "__main__":
    main() 