"""
자동 파일 정리 도구
- 중복 파일 제거
- 카테고리별 통합
- 백업 생성
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import ast

class AutoFileOrganizer:
    """자동 파일 정리기"""
    
    def __init__(self):
        self.backup_dir = Path("file_organization_backup")
        self.backup_dir.mkdir(exist_ok=True)
        
    def create_backup(self) -> str:
        """전체 백업 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        
        # 현재 디렉토리 백업
        shutil.copytree(".", backup_path, 
                       ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
        
        print(f"💾 백업 생성 완료: {backup_path}")
        return str(backup_path)
    
    def organize_files_by_category(self, analysis: Dict[str, any]):
        """카테고리별 파일 정리"""
        print("📁 카테고리별 파일 정리 중...")
        
        # 디렉토리 구조 생성
        directories = {
            '트레이딩 로직': 'trading',
            '분석 도구': 'analyzers', 
            'API 연동': 'api',
            '봇/알림': 'bots',
            '코드 품질': 'quality',
            '유틸리티': 'utils',
            '테스트': 'tests',
            '설정': 'config'
        }
        
        for category, files in analysis['file_categories'].items():
            if category in directories and len(files) > 1:
                dir_name = directories[category]
                dir_path = Path(dir_name)
                dir_path.mkdir(exist_ok=True)
                
                print(f"📂 {category} 파일들을 {dir_name}/ 디렉토리로 이동...")
                
                for file_path in files:
                    file_name = Path(file_path).name
                    if file_name != 'main.py':  # 메인 파일은 루트에 유지
                        try:
                            shutil.move(file_path, dir_path / file_name)
                            print(f"  ✅ {file_name} → {dir_name}/")
                        except Exception as e:
                            print(f"  ❌ {file_name} 이동 실패: {e}")
    
    def merge_similar_files(self, analysis: Dict[str, any]):
        """유사한 파일들 병합"""
        print("🔄 유사한 파일들 병합 중...")
        
        for suggestion in analysis['integration_suggestions']:
            if suggestion['type'] == '소형 파일 통합':
                self.merge_small_files(suggestion['files'], suggestion['suggested_name'])
            elif suggestion['type'] == '카테고리 통합':
                self.merge_category_files(suggestion['files'], suggestion['suggested_name'])
    
    def merge_small_files(self, files: List[str], target_name: str):
        """작은 파일들을 하나로 병합"""
        print(f"📦 작은 파일들을 {target_name}으로 병합...")
        
        merged_content = []
        merged_content.append(f'"""\n{target_name} - 통합된 유틸리티 함수들\n생성일: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n"""')
        merged_content.append("")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_name = Path(file_path).name
                merged_content.append(f"# ===== {file_name}에서 가져온 코드 =====")
                merged_content.append(content)
                merged_content.append("")
                
                # 원본 파일 삭제
                os.remove(file_path)
                print(f"  ✅ {file_name} 병합 완료")
                
            except Exception as e:
                print(f"  ❌ {Path(file_path).name} 병합 실패: {e}")
        
        # 병합된 파일 저장
        with open(target_name, 'w', encoding='utf-8') as f:
            f.write('\n'.join(merged_content))
        
        print(f"💾 {target_name} 생성 완료")
    
    def remove_duplicate_files(self, duplicate_groups: List[List[str]]):
        """중복 파일 제거"""
        print("🗑️ 중복 파일 제거 중...")
        
        for group in duplicate_groups:
            if len(group) > 1:
                # 가장 최근 파일 유지, 나머지 삭제
                files_with_time = [(f, os.path.getmtime(f)) for f in group]
                files_with_time.sort(key=lambda x: x[1], reverse=True)
                
                keep_file = files_with_time[0][0]
                remove_files = [f[0] for f in files_with_time[1:]]
                
                print(f"📁 중복 그룹: {[Path(f).name for f in group]}")
                print(f"  ✅ 유지: {Path(keep_file).name}")
                
                for remove_file in remove_files:
                    try:
                        os.remove(remove_file)
                        print(f"  🗑️ 삭제: {Path(remove_file).name}")
                    except Exception as e:
                        print(f"  ❌ 삭제 실패 {Path(remove_file).name}: {e}")
    
    def create_init_files(self):
        """__init__.py 파일 생성"""
        print("📝 __init__.py 파일 생성 중...")
        
        directories = ['trading', 'analyzers', 'api', 'bots', 'quality', 'utils', 'tests', 'config']
        
        for dir_name in directories:
            dir_path = Path(dir_name)
            if dir_path.exists() and dir_path.is_dir():
                init_file = dir_path / '__init__.py'
                if not init_file.exists():
                    with open(init_file, 'w', encoding='utf-8') as f:
                        f.write(f'"""{dir_name} 모듈"""\n')
                    print(f"  ✅ {dir_name}/__init__.py 생성")
    
    def update_imports(self):
        """import 문 업데이트"""
        print("🔄 import 문 업데이트 중...")
        
        # 모든 Python 파일에서 import 문 수정
        for py_file in Path('.').rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 간단한 import 경로 수정 (실제로는 더 복잡한 로직 필요)
                updated_content = self.fix_imports(content, py_file)
                
                if updated_content != content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    print(f"  ✅ {py_file} import 업데이트")
                    
            except Exception as e:
                print(f"  ❌ {py_file} import 업데이트 실패: {e}")
    
    def fix_imports(self, content: str, file_path: Path) -> str:
        """import 문 수정"""
        # 기본적인 import 경로 수정
        # 실제 프로젝트에서는 더 정교한 로직이 필요
        
        lines = content.splitlines()
        updated_lines = []
        
        for line in lines:
            # 상대 import를 절대 import로 변경하는 등의 작업
            if line.strip().startswith('from ') or line.strip().startswith('import '):
                # 여기서 실제 import 경로 수정 로직 구현
                updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        return '\n'.join(updated_lines)

def main():
    """메인 실행 함수"""
    try:
        # 분석 결과 로드
        if not Path('project_structure_analysis.json').exists():
            print("❌ 먼저 file_structure_analyzer.py를 실행하여 분석을 수행하세요.")
            return
        
        with open('project_structure_analysis.json', 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        organizer = AutoFileOrganizer()
        
        print("🚀 자동 파일 정리 시작")
        print("="*50)
        
        # 1. 백업 생성
        backup_path = organizer.create_backup()
        
        # 2. 중복 파일 제거
        if analysis['duplicate_groups']:
            organizer.remove_duplicate_files(analysis['duplicate_groups'])
        
        # 3. 유사한 파일 병합
        if analysis['integration_suggestions']:
            organizer.merge_similar_files(analysis)
        
        # 4. 카테고리별 정리
        organizer.organize_files_by_category(analysis)
        
        # 5. __init__.py 파일 생성
        organizer.create_init_files()
        
        # 6. import 문 업데이트
        organizer.update_imports()
        
        print("\n🎉 파일 정리 완료!")
        print(f"💾 백업 위치: {backup_path}")
        print("📋 정리 결과를 확인하고 테스트해보세요.")
        
    except Exception as e:
        print(f"❌ 파일 정리 실패: {e}")

if __name__ == "__main__":
    main() 