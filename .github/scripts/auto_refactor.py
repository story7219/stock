#!/usr/bin/env python3
"""
🔄 자동 리팩토링 도구
코드 품질을 자동으로 개선하는 리팩토링 시스템
"""

import os
import sys
import json
import ast
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime
import re
import tempfile
import shutil

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoRefactor:
    """자동 리팩토링 도구"""
    
    def __init__(self, project_root: str = ".", dry_run: bool = True):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.changes_made = []
        self.refactor_stats = {
            'files_processed': 0,
            'changes_applied': 0,
            'errors': 0,
            'improvements': []
        }
        
        logger.info(f"🔄 자동 리팩토링 도구 초기화 (DryRun: {dry_run})")
    
    def format_code_with_black(self) -> bool:
        """Black으로 코드 포맷팅"""
        logger.info("🎨 Black으로 코드 포맷팅 중...")
        
        try:
            cmd = ['black', '--check', '--diff', str(self.project_root)]
            if not self.dry_run:
                cmd.remove('--check')
                cmd.remove('--diff')
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ 코드가 이미 Black 포맷에 맞춰져 있습니다")
                return True
            elif result.returncode == 1 and self.dry_run:
                logger.info("📝 Black 포맷팅이 필요한 파일들이 있습니다")
                self.changes_made.append({
                    'type': 'formatting',
                    'tool': 'black',
                    'description': 'Code formatting improvements needed',
                    'diff': result.stdout
                })
                return True
            elif not self.dry_run:
                logger.info("✅ Black 포맷팅 적용 완료")
                self.refactor_stats['changes_applied'] += 1
                return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"⚠️ Black 실행 실패: {e}")
            self.refactor_stats['errors'] += 1
            return False
        
        return True
    
    def sort_imports_with_isort(self) -> bool:
        """isort로 import 정렬"""
        logger.info("📚 isort로 import 정렬 중...")
        
        try:
            cmd = ['isort', '--check-only', '--diff', str(self.project_root)]
            if not self.dry_run:
                cmd.remove('--check-only')
                cmd.remove('--diff')
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Import가 이미 정렬되어 있습니다")
                return True
            elif result.returncode == 1 and self.dry_run:
                logger.info("📝 Import 정렬이 필요한 파일들이 있습니다")
                self.changes_made.append({
                    'type': 'import_sorting',
                    'tool': 'isort',
                    'description': 'Import sorting improvements needed',
                    'diff': result.stdout
                })
                return True
            elif not self.dry_run:
                logger.info("✅ Import 정렬 적용 완료")
                self.refactor_stats['changes_applied'] += 1
                return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"⚠️ isort 실행 실패: {e}")
            self.refactor_stats['errors'] += 1
            return False
        
        return True
    
    def remove_unused_imports(self) -> bool:
        """사용하지 않는 import 제거"""
        logger.info("🧹 사용하지 않는 import 제거 중...")
        
        try:
            # autoflake 사용
            cmd = ['autoflake', '--check', '--recursive', str(self.project_root)]
            if not self.dry_run:
                cmd.remove('--check')
                cmd.extend(['--in-place', '--remove-all-unused-imports'])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                if self.dry_run:
                    logger.info("✅ 사용하지 않는 import가 없습니다")
                else:
                    logger.info("✅ 사용하지 않는 import 제거 완료")
                    self.refactor_stats['changes_applied'] += 1
                return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"⚠️ autoflake 실행 실패: {e}")
            self.refactor_stats['errors'] += 1
        
        # 수동으로 unused import 찾기
        return self._manual_remove_unused_imports()
    
    def _manual_remove_unused_imports(self) -> bool:
        """수동으로 사용하지 않는 import 찾기"""
        unused_imports = []
        
        for py_file in self.project_root.rglob('*.py'):
            if py_file.is_file():
                file_unused = self._find_unused_imports_in_file(py_file)
                unused_imports.extend(file_unused)
        
        if unused_imports:
            self.changes_made.append({
                'type': 'unused_imports',
                'tool': 'manual',
                'description': f'Found {len(unused_imports)} unused imports',
                'items': unused_imports
            })
            
            if not self.dry_run:
                self._remove_unused_imports(unused_imports)
                self.refactor_stats['changes_applied'] += len(unused_imports)
        
        return True
    
    def _find_unused_imports_in_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """파일에서 사용하지 않는 import 찾기"""
        unused = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Import 수집
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'name': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno,
                            'type': 'import'
                        })
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append({
                            'name': alias.name,
                            'alias': alias.asname,
                            'module': node.module,
                            'line': node.lineno,
                            'type': 'from_import'
                        })
            
            # 사용 여부 확인
            for imp in imports:
                name_to_check = imp.get('alias') or imp['name']
                if not self._is_name_used_in_content(content, name_to_check):
                    unused.append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': imp['line'],
                        'import': imp
                    })
        
        except Exception as e:
            logger.warning(f"⚠️ Import 분석 오류 {file_path}: {e}")
        
        return unused
    
    def _is_name_used_in_content(self, content: str, name: str) -> bool:
        """내용에서 이름이 사용되는지 확인"""
        # 간단한 정규식 기반 검사
        pattern = rf'\b{re.escape(name)}\b'
        matches = re.findall(pattern, content)
        
        # import 라인 제외하고 2번 이상 나타나면 사용됨
        return len(matches) > 1
    
    def refactor_long_functions(self) -> bool:
        """긴 함수 리팩토링 제안"""
        logger.info("📏 긴 함수 분석 중...")
        
        long_functions = []
        
        for py_file in self.project_root.rglob('*.py'):
            if py_file.is_file():
                functions = self._find_long_functions(py_file)
                long_functions.extend(functions)
        
        if long_functions:
            self.changes_made.append({
                'type': 'long_functions',
                'tool': 'analyzer',
                'description': f'Found {len(long_functions)} functions that could be refactored',
                'items': long_functions
            })
            
            self.refactor_stats['improvements'].append(f"긴 함수 {len(long_functions)}개 발견")
        
        return True
    
    def _find_long_functions(self, file_path: Path) -> List[Dict[str, Any]]:
        """파일에서 긴 함수 찾기"""
        long_functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_lines = self._count_function_lines(node, content)
                    
                    if func_lines > 50:  # 50줄 이상이면 긴 함수
                        long_functions.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'name': node.name,
                            'lines': func_lines,
                            'start_line': node.lineno,
                            'suggestion': self._suggest_function_refactoring(node, content)
                        })
        
        except Exception as e:
            logger.warning(f"⚠️ 긴 함수 분석 오류 {file_path}: {e}")
        
        return long_functions
    
    def _count_function_lines(self, node: ast.FunctionDef, content: str) -> int:
        """함수의 실제 라인 수 계산"""
        if hasattr(node, 'end_lineno'):
            return node.end_lineno - node.lineno + 1
        
        # end_lineno가 없으면 대략적으로 계산
        lines = content.split('\n')
        func_start = node.lineno - 1
        
        # 함수 끝 찾기 (간단한 버전)
        indent_level = len(lines[func_start]) - len(lines[func_start].lstrip())
        func_end = func_start + 1
        
        for i in range(func_start + 1, len(lines)):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= indent_level:
                if not line.strip().startswith(('"""', "'''", '#')):
                    func_end = i
                    break
        
        return func_end - func_start
    
    def _suggest_function_refactoring(self, node: ast.FunctionDef, content: str) -> str:
        """함수 리팩토링 제안"""
        suggestions = []
        
        # 복잡도 분석
        complexity = self._calculate_cyclomatic_complexity(node)
        if complexity > 10:
            suggestions.append(f"순환 복잡도가 높음 ({complexity})")
        
        # 중첩 레벨 확인
        max_depth = self._calculate_max_nesting_depth(node)
        if max_depth > 4:
            suggestions.append(f"중첩이 너무 깊음 (레벨 {max_depth})")
        
        # 매개변수 수 확인
        param_count = len(node.args.args)
        if param_count > 5:
            suggestions.append(f"매개변수가 너무 많음 ({param_count}개)")
        
        if not suggestions:
            suggestions.append("함수를 더 작은 단위로 분할 고려")
        
        return "; ".join(suggestions)
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """순환 복잡도 계산"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1
        
        return complexity
    
    def _calculate_max_nesting_depth(self, node: ast.FunctionDef) -> int:
        """최대 중첩 깊이 계산"""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return get_depth(node)
    
    def add_missing_docstrings(self) -> bool:
        """누락된 docstring 추가"""
        logger.info("📝 누락된 docstring 분석 중...")
        
        missing_docstrings = []
        
        for py_file in self.project_root.rglob('*.py'):
            if py_file.is_file():
                missing = self._find_missing_docstrings(py_file)
                missing_docstrings.extend(missing)
        
        if missing_docstrings:
            self.changes_made.append({
                'type': 'missing_docstrings',
                'tool': 'analyzer',
                'description': f'Found {len(missing_docstrings)} functions/classes without docstrings',
                'items': missing_docstrings
            })
            
            if not self.dry_run:
                self._add_docstrings(missing_docstrings)
                self.refactor_stats['changes_applied'] += len(missing_docstrings)
        
        return True
    
    def _find_missing_docstrings(self, file_path: Path) -> List[Dict[str, Any]]:
        """파일에서 docstring이 없는 함수/클래스 찾기"""
        missing = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        # private 함수/클래스는 제외
                        if not node.name.startswith('_'):
                            missing.append({
                                'file': str(file_path.relative_to(self.project_root)),
                                'name': node.name,
                                'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                                'line': node.lineno,
                                'suggested_docstring': self._generate_docstring(node)
                            })
        
        except Exception as e:
            logger.warning(f"⚠️ Docstring 분석 오류 {file_path}: {e}")
        
        return missing
    
    def _generate_docstring(self, node: ast.AST) -> str:
        """자동 docstring 생성"""
        if isinstance(node, ast.FunctionDef):
            return f'"""{node.name} 함수입니다.\n    \n    TODO: 함수 설명을 추가하세요.\n    """'
        elif isinstance(node, ast.ClassDef):
            return f'"""{node.name} 클래스입니다.\n    \n    TODO: 클래스 설명을 추가하세요.\n    """'
        return '"""TODO: 설명을 추가하세요."""'
    
    def optimize_imports(self) -> bool:
        """Import 최적화"""
        logger.info("🔧 Import 최적화 중...")
        
        optimizations = []
        
        for py_file in self.project_root.rglob('*.py'):
            if py_file.is_file():
                opts = self._optimize_file_imports(py_file)
                optimizations.extend(opts)
        
        if optimizations:
            self.changes_made.append({
                'type': 'import_optimization',
                'tool': 'optimizer',
                'description': f'Found {len(optimizations)} import optimizations',
                'items': optimizations
            })
            
            if not self.dry_run:
                self._apply_import_optimizations(optimizations)
                self.refactor_stats['changes_applied'] += len(optimizations)
        
        return True
    
    def _optimize_file_imports(self, file_path: Path) -> List[Dict[str, Any]]:
        """파일의 import 최적화"""
        optimizations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 중복 import 찾기
            imports_seen = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_str = alias.name
                        if import_str in imports_seen:
                            optimizations.append({
                                'file': str(file_path.relative_to(self.project_root)),
                                'line': node.lineno,
                                'type': 'duplicate_import',
                                'import': import_str,
                                'suggestion': f'중복된 import 제거: {import_str}'
                            })
                        imports_seen.add(import_str)
        
        except Exception as e:
            logger.warning(f"⚠️ Import 최적화 오류 {file_path}: {e}")
        
        return optimizations
    
    def run_refactoring(self) -> Dict[str, Any]:
        """전체 리팩토링 실행"""
        logger.info(f"🔄 자동 리팩토링 시작 (DryRun: {self.dry_run})")
        
        start_time = datetime.now()
        
        # 각 리팩토링 단계 실행
        steps = [
            ('코드 포맷팅', self.format_code_with_black),
            ('Import 정렬', self.sort_imports_with_isort),
            ('사용하지 않는 Import 제거', self.remove_unused_imports),
            ('긴 함수 분석', self.refactor_long_functions),
            ('Docstring 추가', self.add_missing_docstrings),
            ('Import 최적화', self.optimize_imports)
        ]
        
        for step_name, step_func in steps:
            try:
                logger.info(f"🔧 {step_name} 실행 중...")
                success = step_func()
                if success:
                    self.refactor_stats['files_processed'] += 1
                else:
                    self.refactor_stats['errors'] += 1
            except Exception as e:
                logger.error(f"❌ {step_name} 실행 오류: {e}")
                self.refactor_stats['errors'] += 1
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 결과 정리
        result = {
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'dry_run': self.dry_run,
            'stats': self.refactor_stats,
            'changes': self.changes_made,
            'summary': {
                'total_changes': len(self.changes_made),
                'files_processed': self.refactor_stats['files_processed'],
                'changes_applied': self.refactor_stats['changes_applied'],
                'errors': self.refactor_stats['errors']
            }
        }
        
        logger.info(f"✅ 리팩토링 완료: {len(self.changes_made)}개 변경사항, {duration:.1f}초 소요")
        
        return result
    
    def save_report(self, result: Dict[str, Any], output_file: str = "refactor_report.json"):
        """리팩토링 결과 저장"""
        output_path = self.project_root / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 리팩토링 보고서 저장: {output_path}")
        
        # 마크다운 보고서도 생성
        self.save_markdown_report(result, str(output_path).replace('.json', '.md'))
    
    def save_markdown_report(self, result: Dict[str, Any], output_file: str):
        """마크다운 형식 보고서 저장"""
        md_content = f"""# 🔄 자동 리팩토링 보고서

**실행 시간**: {result['timestamp']}  
**소요 시간**: {result['duration_seconds']:.1f}초  
**DryRun 모드**: {result['dry_run']}

## 📊 요약

- **총 변경사항**: {result['summary']['total_changes']}개
- **처리된 파일**: {result['summary']['files_processed']}개
- **적용된 변경**: {result['summary']['changes_applied']}개
- **오류**: {result['summary']['errors']}개

## 🔧 변경사항 상세

"""
        
        for change in result['changes']:
            md_content += f"### {change['type'].replace('_', ' ').title()}\n\n"
            md_content += f"- **도구**: {change['tool']}\n"
            md_content += f"- **설명**: {change['description']}\n"
            
            if 'items' in change:
                md_content += f"- **항목 수**: {len(change['items'])}개\n"
            
            md_content += "\n"
        
        if result['stats']['improvements']:
            md_content += "## 💡 개선사항\n\n"
            for improvement in result['stats']['improvements']:
                md_content += f"- {improvement}\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"📄 마크다운 보고서 저장: {output_file}")

def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(description='자동 리팩토링 도구')
    parser.add_argument('--project-root', default='.', help='프로젝트 루트 디렉토리')
    parser.add_argument('--output', default='refactor_report.json', help='출력 파일명')
    parser.add_argument('--dry-run', action='store_true', help='실제 변경 없이 분석만 수행')
    parser.add_argument('--apply', action='store_true', help='실제 변경사항 적용')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # dry_run 모드 결정
    dry_run = args.dry_run or not args.apply
    
    # 리팩토링 실행
    refactor = AutoRefactor(args.project_root, dry_run=dry_run)
    result = refactor.run_refactoring()
    refactor.save_report(result, args.output)
    
    # 결과 출력
    print(f"\n🔄 자동 리팩토링 완료!")
    print(f"📊 총 변경사항: {result['summary']['total_changes']}개")
    print(f"⏱️  소요 시간: {result['duration_seconds']:.1f}초")
    
    if dry_run:
        print("ℹ️  DryRun 모드: 실제 변경사항은 적용되지 않았습니다")
        print("💡 실제 적용하려면 --apply 옵션을 사용하세요")
    else:
        print(f"✅ {result['summary']['changes_applied']}개 변경사항이 적용되었습니다")
    
    if result['summary']['errors'] > 0:
        print(f"⚠️ {result['summary']['errors']}개 오류 발생")
        sys.exit(1)

if __name__ == "__main__":
    main() 