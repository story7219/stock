"""
프로젝트 파일 구조 분석 및 정리 도구
- 파일 역할 분석
- 중복 코드 탐지
- 통합 가능한 파일 식별
"""

import os
import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class FileInfo:
    """파일 정보 클래스"""
    path: str
    size: int
    lines: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    main_purpose: str
    similarity_hash: str

class ProjectStructureAnalyzer:
    """프로젝트 구조 분석기"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.files_info: Dict[str, FileInfo] = {}
        
    def analyze_project_structure(self) -> Dict[str, any]:
        """전체 프로젝트 구조 분석"""
        print("🔍 프로젝트 파일 구조 분석 중...")
        
        # Python 파일들만 분석
        python_files = list(self.project_path.glob("*.py"))
        
        analysis_result = {
            'total_files': len(python_files),
            'file_categories': defaultdict(list),
            'duplicate_groups': [],
            'integration_suggestions': [],
            'cleanup_recommendations': []
        }
        
        print(f"📁 발견된 Python 파일: {len(python_files)}개")
        
        # 각 파일 분석
        for i, file_path in enumerate(python_files, 1):
            try:
                print(f"  📄 분석 중 ({i}/{len(python_files)}): {file_path.name}")
                file_info = self.analyze_single_file(file_path)
                self.files_info[str(file_path)] = file_info
                
                # 파일 카테고리 분류
                category = self.categorize_file(file_info)
                analysis_result['file_categories'][category].append(str(file_path))
                
            except Exception as e:
                print(f"⚠️ 파일 분석 실패 {file_path}: {e}")
        
        # 중복 파일 그룹 찾기
        analysis_result['duplicate_groups'] = self.find_duplicate_groups()
        
        # 통합 제안 생성
        analysis_result['integration_suggestions'] = self.generate_integration_suggestions()
        
        # 정리 권장사항
        analysis_result['cleanup_recommendations'] = self.generate_cleanup_recommendations()
        
        return analysis_result
    
    def analyze_single_file(self, file_path: Path) -> FileInfo:
        """개별 파일 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # UTF-8로 읽기 실패 시 다른 인코딩 시도
            try:
                with open(file_path, 'r', encoding='cp949') as f:
                    content = f.read()
            except:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
        
        # 기본 정보
        size = file_path.stat().st_size
        lines = len(content.splitlines())
        
        # AST 파싱으로 구조 분석
        try:
            tree = ast.parse(content)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = self.extract_imports(tree)
        except:
            functions, classes, imports = [], [], []
        
        # 파일 목적 추론
        main_purpose = self.infer_file_purpose(file_path.name, content, functions, classes)
        
        # 유사성 해시 (중복 탐지용)
        similarity_hash = self.calculate_similarity_hash(content)
        
        return FileInfo(
            path=str(file_path),
            size=size,
            lines=lines,
            functions=functions,
            classes=classes,
            imports=imports,
            main_purpose=main_purpose,
            similarity_hash=similarity_hash
        )
    
    def extract_imports(self, tree: ast.AST) -> List[str]:
        """import 문 추출"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports
    
    def infer_file_purpose(self, filename: str, content: str, functions: List[str], classes: List[str]) -> str:
        """파일 목적 추론"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # 파일명 기반 분류
        if 'test' in filename_lower:
            return "테스트"
        elif 'config' in filename_lower or 'setting' in filename_lower:
            return "설정"
        elif 'util' in filename_lower or 'helper' in filename_lower:
            return "유틸리티"
        elif 'api' in filename_lower:
            return "API 연동"
        elif 'trading' in filename_lower or 'trade' in filename_lower:
            return "트레이딩 로직"
        elif 'analyzer' in filename_lower or 'analysis' in filename_lower:
            return "분석 도구"
        elif 'bot' in filename_lower or 'telegram' in filename_lower:
            return "봇/알림"
        elif 'quality' in filename_lower or 'refactor' in filename_lower:
            return "코드 품질"
        elif 'main' in filename_lower or filename_lower == 'app.py':
            return "메인 실행"
        elif 'debug' in filename_lower:
            return "디버깅 도구"
        elif 'chart' in filename_lower or 'graph' in filename_lower:
            return "차트/시각화"
        elif 'backtest' in filename_lower:
            return "백테스팅"
        elif 'logger' in filename_lower or 'log' in filename_lower:
            return "로깅"
        elif 'screener' in filename_lower or 'screen' in filename_lower:
            return "종목 스크리닝"
        elif 'cycle' in filename_lower:
            return "사이클 분석"
        elif 'fetch' in filename_lower:
            return "데이터 수집"
        elif 'spread' in filename_lower:
            return "스프레드 분석"
        elif 'throttle' in filename_lower:
            return "속도 제한"
        elif 'system' in filename_lower:
            return "시스템 관리"
        elif 'mixed' in filename_lower:
            return "혼합 전략"
        elif 'client' in filename_lower:
            return "클라이언트"
        elif 'sheets' in filename_lower:
            return "시트 연동"
        elif 'gemini' in filename_lower:
            return "AI 연동"
        
        # 내용 기반 분류
        if 'class' in content_lower and len(classes) > 0:
            if 'trading' in content_lower or 'stock' in content_lower:
                return "트레이딩 클래스"
            elif 'analyzer' in content_lower or 'analysis' in content_lower:
                return "분석 클래스"
            else:
                return "일반 클래스"
        elif len(functions) > 5:
            return "함수 모음"
        elif 'if __name__' in content:
            return "실행 스크립트"
        else:
            return "기타"
    
    def calculate_similarity_hash(self, content: str) -> str:
        """코드 유사성 해시 계산"""
        # 공백, 주석 제거 후 해시
        lines = []
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                lines.append(line)
        
        normalized_content = '\n'.join(lines)
        return hashlib.md5(normalized_content.encode()).hexdigest()[:8]
    
    def categorize_file(self, file_info: FileInfo) -> str:
        """파일 카테고리 분류"""
        return file_info.main_purpose
    
    def find_duplicate_groups(self) -> List[List[str]]:
        """중복 파일 그룹 찾기"""
        hash_groups = defaultdict(list)
        
        for file_path, file_info in self.files_info.items():
            if file_info.lines > 5:  # 너무 작은 파일은 제외
                hash_groups[file_info.similarity_hash].append(file_path)
        
        # 2개 이상인 그룹만 반환
        return [group for group in hash_groups.values() if len(group) > 1]
    
    def generate_integration_suggestions(self) -> List[Dict[str, any]]:
        """통합 제안 생성"""
        suggestions = []
        
        # 카테고리별 파일 그룹핑
        categories = defaultdict(list)
        for file_path, file_info in self.files_info.items():
            categories[file_info.main_purpose].append((file_path, file_info))
        
        for category, files in categories.items():
            if len(files) > 2 and category not in ['메인 실행', '설정']:  # 3개 이상인 카테고리
                suggestions.append({
                    'type': '카테고리 통합',
                    'category': category,
                    'files': [f[0] for f in files],
                    'reason': f'{category} 관련 파일들을 하나의 모듈로 통합',
                    'suggested_name': self.suggest_module_name(category)
                })
        
        # 작은 파일들 통합 제안
        small_files = [(path, info) for path, info in self.files_info.items() 
                      if info.lines < 50 and info.main_purpose not in ['메인 실행', '설정', '디버깅 도구']]
        
        if len(small_files) > 3:
            suggestions.append({
                'type': '소형 파일 통합',
                'files': [f[0] for f in small_files],
                'reason': '50줄 미만의 작은 파일들을 utils 모듈로 통합',
                'suggested_name': 'utils.py'
            })
        
        return suggestions
    
    def suggest_module_name(self, category: str) -> str:
        """모듈명 제안"""
        name_mapping = {
            '트레이딩 로직': 'trading_core.py',
            '분석 도구': 'analyzers.py',
            'API 연동': 'api_clients.py',
            '봇/알림': 'notification_bots.py',
            '코드 품질': 'quality_tools.py',
            '유틸리티': 'utils.py',
            '테스트': 'test_suite.py',
            '차트/시각화': 'visualization.py',
            '백테스팅': 'backtesting.py',
            '로깅': 'logging_utils.py',
            '종목 스크리닝': 'stock_screening.py',
            '사이클 분석': 'cycle_analysis.py',
            '데이터 수집': 'data_fetchers.py',
            '스프레드 분석': 'spread_analysis.py',
            '속도 제한': 'throttling.py',
            '시스템 관리': 'system_management.py',
            '혼합 전략': 'mixed_strategies.py',
            '클라이언트': 'clients.py',
            '시트 연동': 'sheet_integration.py',
            'AI 연동': 'ai_integration.py'
        }
        return name_mapping.get(category, f'{category.lower().replace(" ", "_").replace("/", "_")}.py')
    
    def generate_cleanup_recommendations(self) -> List[str]:
        """정리 권장사항 생성"""
        recommendations = []
        
        # 빈 파일이나 거의 빈 파일
        empty_files = [path for path, info in self.files_info.items() if info.lines < 5]
        if empty_files:
            recommendations.append(f"🗑️ 빈 파일 삭제 권장: {len(empty_files)}개")
        
        # 중복 파일
        duplicate_groups = self.find_duplicate_groups()
        if duplicate_groups:
            recommendations.append(f"🔄 중복 파일 정리 필요: {len(duplicate_groups)}개 그룹")
        
        # 너무 큰 파일
        large_files = [path for path, info in self.files_info.items() if info.lines > 500]
        if large_files:
            recommendations.append(f"📄 큰 파일 분할 고려: {len(large_files)}개 (500줄 이상)")
        
        # 함수가 너무 많은 파일
        function_heavy = [path for path, info in self.files_info.items() if len(info.functions) > 20]
        if function_heavy:
            recommendations.append(f"🔧 함수 과다 파일 모듈화: {len(function_heavy)}개 (20개 이상)")
        
        # 클래스가 없는 큰 파일
        no_class_large = [path for path, info in self.files_info.items() 
                         if info.lines > 200 and len(info.classes) == 0 and len(info.functions) > 10]
        if no_class_large:
            recommendations.append(f"🏗️ 클래스 구조화 권장: {len(no_class_large)}개 파일")
        
        return recommendations

def print_analysis_report(analysis: Dict[str, any]):
    """분석 결과 리포트 출력"""
    print("\n" + "="*80)
    print("📊 프로젝트 구조 분석 결과")
    print("="*80)
    
    print(f"\n📁 전체 파일 수: {analysis['total_files']}개")
    
    # 카테고리별 분류
    print(f"\n📋 파일 카테고리별 분류:")
    total_categorized = 0
    for category, files in sorted(analysis['file_categories'].items()):
        print(f"  📂 {category}: {len(files)}개")
        total_categorized += len(files)
        for file in files[:3]:  # 처음 3개만 표시
            print(f"    - {Path(file).name}")
        if len(files) > 3:
            print(f"    ... 외 {len(files)-3}개")
    
    # 중복 파일 그룹
    if analysis['duplicate_groups']:
        print(f"\n🔄 중복 파일 그룹: {len(analysis['duplicate_groups'])}개")
        for i, group in enumerate(analysis['duplicate_groups'], 1):
            print(f"  그룹 {i}: {[Path(f).name for f in group]}")
    else:
        print(f"\n✅ 중복 파일 없음")
    
    # 통합 제안
    if analysis['integration_suggestions']:
        print(f"\n💡 통합 제안: {len(analysis['integration_suggestions'])}개")
        for suggestion in analysis['integration_suggestions']:
            print(f"  📦 {suggestion['type']}: {suggestion['suggested_name']}")
            print(f"     대상: {len(suggestion['files'])}개 파일")
            print(f"     이유: {suggestion['reason']}")
    else:
        print(f"\n✅ 추가 통합 제안 없음")
    
    # 정리 권장사항
    if analysis['cleanup_recommendations']:
        print(f"\n🧹 정리 권장사항:")
        for rec in analysis['cleanup_recommendations']:
            print(f"  {rec}")
    else:
        print(f"\n✅ 정리할 항목 없음")
    
    # 정리 효과 예상
    print(f"\n📈 정리 효과 예상:")
    current_files = analysis['total_files']
    
    # 중복 제거 효과
    duplicate_reduction = sum(len(group) - 1 for group in analysis['duplicate_groups'])
    
    # 통합 효과
    integration_reduction = 0
    for suggestion in analysis['integration_suggestions']:
        if suggestion['type'] == '소형 파일 통합':
            integration_reduction += len(suggestion['files']) - 1
        elif suggestion['type'] == '카테고리 통합':
            integration_reduction += len(suggestion['files']) - 1
    
    estimated_final = current_files - duplicate_reduction - integration_reduction
    reduction_percent = ((current_files - estimated_final) / current_files * 100) if current_files > 0 else 0
    
    print(f"  📉 파일 수 감소: {current_files}개 → {estimated_final}개 ({reduction_percent:.1f}% 감소)")
    print(f"  🗑️ 중복 제거: {duplicate_reduction}개")
    print(f"  📦 통합 효과: {integration_reduction}개")

def main():
    """메인 실행 함수"""
    try:
        print("🚀 프로젝트 파일 구조 분석 시작")
        print("="*50)
        
        analyzer = ProjectStructureAnalyzer()
        analysis = analyzer.analyze_project_structure()
        
        # 결과 출력
        print_analysis_report(analysis)
        
        # JSON 파일로 저장
        with open('project_structure_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 상세 분석 결과가 'project_structure_analysis.json'에 저장되었습니다.")
        
        # 사용자 선택
        print(f"\n🎯 다음 단계를 선택하세요:")
        print(f"1. 자동 파일 정리 실행")
        print(f"2. 수동 검토 후 정리")
        print(f"3. 분석만 하고 종료")
        
        choice = input("선택 (1-3): ").strip()
        
        if choice == '1':
            print("🚀 자동 파일 정리를 시작합니다...")
            # 자동 정리 실행
            import subprocess
            try:
                result = subprocess.run(['python', 'auto_file_organizer.py'], 
                                      capture_output=True, text=True, encoding='utf-8')
                if result.returncode == 0:
                    print("✅ 자동 정리 완료!")
                    print(result.stdout)
                else:
                    print("❌ 자동 정리 실패:")
                    print(result.stderr)
            except Exception as e:
                print(f"❌ 자동 정리 실행 실패: {e}")
                print("💡 수동으로 'python auto_file_organizer.py'를 실행해주세요.")
                
        elif choice == '2':
            print("📋 수동 검토 모드입니다.")
            print("💡 'project_structure_analysis.json' 파일을 확인하고")
            print("💡 'python auto_file_organizer.py'를 실행하여 정리하세요.")
        else:
            print("📊 분석 완료. 결과를 검토하세요.")
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 