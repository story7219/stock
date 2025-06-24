#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📁 파일 관리 마스터 시스템 v1.0
체계적이고 효율적인 프로젝트 파일 관리 도구

Features:
- 🔍 전체 파일 스캔 및 분석
- 🗂️ 자동 파일 분류 및 정리
- 📊 프로젝트 구조 시각화
- 🧹 중복 파일 제거
- 📈 파일 사용량 통계
- 🔧 자동 백업 시스템
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass, asdict
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/file_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

@dataclass
class FileInfo:
    """파일 정보 데이터 클래스"""
    path: str
    name: str
    size: int
    created: str
    modified: str
    extension: str
    category: str
    hash_md5: Optional[str] = None
    is_duplicate: bool = False
    usage_score: int = 0

class FileManager:
    """🗂️ 파일 관리 마스터 클래스"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.files_data: Dict[str, FileInfo] = {}
        self.categories = {
            'core': ['app.py', 'launcher.py', 'run_analysis.py'],
            'modules': ['.py'],
            'config': ['.env', '.json', '.yaml', '.yml', '.ini', '.conf'],
            'docs': ['.md', '.txt', '.rst', '.pdf'],
            'scripts': ['.bat', '.sh', '.ps1', 'Makefile'],
            'data': ['.csv', '.xlsx', '.json', '.pickle', '.pkl'],
            'logs': ['.log'],
            'reports': ['.html', '.pdf', '.json'],
            'tests': ['test_', '_test.py', '.pytest'],
            'backup': ['backup_', 'old_', '.bak'],
            'temp': ['.tmp', '.temp', '__pycache__'],
            'media': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
            'archive': ['.zip', '.tar', '.gz', '.rar']
        }
        
        # 프로젝트 구조 정의
        self.target_structure = {
            'core/': '핵심 실행 파일들',
            'modules/': '분석 모듈들',
            'config/': '설정 파일들',
            'docs/': '문서 파일들',
            'scripts/': '스크립트 파일들',
            'data/': '데이터 파일들',
            'logs/': '로그 파일들',
            'reports/': '리포트 파일들',
            'tests/': '테스트 파일들',
            'backup_old_files/': '백업 파일들',
            'src/': '기존 소스 파일들'
        }
        
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """필요한 디렉토리 생성"""
        for directory in self.target_structure.keys():
            dir_path = self.project_root / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"✅ 디렉토리 생성: {directory}")
    
    def scan_all_files(self, exclude_dirs: List[str] = None) -> None:
        """📊 전체 파일 스캔 및 분석"""
        if exclude_dirs is None:
            exclude_dirs = ['.git', '.venv', '__pycache__', 'node_modules', '.vscode']
        
        print("🔍 파일 스캔 시작...")
        start_time = time.time()
        
        for root, dirs, files in os.walk(self.project_root):
            # 제외할 디렉토리 건너뛰기
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                file_path = Path(root) / file
                try:
                    self._analyze_file(file_path)
                except Exception as e:
                    logging.warning(f"파일 분석 실패: {file_path} - {e}")
        
        scan_time = time.time() - start_time
        print(f"✅ 스캔 완료! ({len(self.files_data)}개 파일, {scan_time:.2f}초)")
    
    def _analyze_file(self, file_path: Path) -> None:
        """개별 파일 분석"""
        try:
            stat = file_path.stat()
            
            file_info = FileInfo(
                path=str(file_path.relative_to(self.project_root)),
                name=file_path.name,
                size=stat.st_size,
                created=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                extension=file_path.suffix.lower(),
                category=self._categorize_file(file_path),
                usage_score=self._calculate_usage_score(file_path)
            )
            
            # MD5 해시 계산 (중복 파일 검출용)
            if stat.st_size < 50 * 1024 * 1024:  # 50MB 미만만
                file_info.hash_md5 = self._calculate_md5(file_path)
            
            self.files_data[str(file_path)] = file_info
            
        except Exception as e:
            logging.error(f"파일 분석 오류: {file_path} - {e}")
    
    def _categorize_file(self, file_path: Path) -> str:
        """파일 카테고리 분류"""
        file_name = file_path.name.lower()
        extension = file_path.suffix.lower()
        
        # 특별한 파일들 우선 체크
        if file_name in ['requirements.txt', 'setup.py', 'pyproject.toml', '.gitignore']:
            return 'config'
        
        # 카테고리별 패턴 매칭
        for category, patterns in self.categories.items():
            for pattern in patterns:
                if pattern.startswith('.') and extension == pattern:
                    return category
                elif file_name.startswith(pattern) or pattern in file_name:
                    return category
        
        # 디렉토리 기반 분류
        parts = file_path.parts
        for part in parts:
            if part in self.categories:
                return part
        
        return 'misc'
    
    def _calculate_usage_score(self, file_path: Path) -> int:
        """파일 중요도 점수 계산"""
        score = 0
        file_name = file_path.name.lower()
        
        # 핵심 파일들
        if file_name in ['app.py', 'launcher.py', 'main.py']:
            score += 10
        elif file_name == 'requirements.txt':
            score += 9
        elif file_name.endswith('_analyzer.py'):
            score += 8
        elif file_name.endswith('.py'):
            score += 5
        
        # 최근 수정된 파일
        try:
            mtime = file_path.stat().st_mtime
            days_old = (time.time() - mtime) / (24 * 3600)
            if days_old < 1:
                score += 5
            elif days_old < 7:
                score += 3
            elif days_old < 30:
                score += 1
        except:
            pass
        
        return score
    
    def _calculate_md5(self, file_path: Path) -> str:
        """MD5 해시 계산"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return ""
    
    def find_duplicates(self) -> Dict[str, List[str]]:
        """🔍 중복 파일 찾기"""
        print("🔍 중복 파일 검색 중...")
        
        hash_groups = defaultdict(list)
        for file_path, file_info in self.files_data.items():
            if file_info.hash_md5:
                hash_groups[file_info.hash_md5].append(file_path)
        
        duplicates = {h: files for h, files in hash_groups.items() if len(files) > 1}
        
        # 중복 플래그 설정
        for file_list in duplicates.values():
            for file_path in file_list[1:]:  # 첫 번째 제외하고 중복 표시
                if file_path in self.files_data:
                    self.files_data[file_path].is_duplicate = True
        
        print(f"✅ 중복 파일 {len(duplicates)}개 그룹 발견")
        return duplicates
    
    def auto_organize(self, dry_run: bool = True) -> Dict[str, int]:
        """🗂️ 자동 파일 정리"""
        print(f"🗂️ 자동 파일 정리 {'(시뮬레이션)' if dry_run else '(실행)'}")
        
        moves = defaultdict(int)
        
        for file_path, file_info in self.files_data.items():
            current_path = Path(file_path)
            
            # 이미 올바른 위치에 있는지 체크
            if self._is_in_correct_location(current_path, file_info.category):
                continue
            
            # 새 위치 결정
            new_location = self._get_target_location(file_info)
            if new_location:
                new_path = self.project_root / new_location / current_path.name
                
                if not dry_run:
                    try:
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(current_path), str(new_path))
                        logging.info(f"이동: {current_path} → {new_path}")
                    except Exception as e:
                        logging.error(f"이동 실패: {current_path} - {e}")
                
                moves[file_info.category] += 1
        
        return dict(moves)
    
    def _is_in_correct_location(self, file_path: Path, category: str) -> bool:
        """파일이 올바른 위치에 있는지 확인"""
        parts = file_path.parts
        if len(parts) > 1 and parts[0] == category:
            return True
        return False
    
    def _get_target_location(self, file_info: FileInfo) -> Optional[str]:
        """파일의 목표 위치 결정"""
        category_mapping = {
            'core': 'core',
            'modules': 'modules',
            'config': 'config',
            'docs': 'docs',
            'scripts': 'scripts',
            'data': 'data',
            'logs': 'logs',
            'reports': 'reports',
            'tests': 'tests',
            'backup': 'backup_old_files',
            'temp': 'backup_old_files',
            'media': 'docs',
            'archive': 'backup_old_files'
        }
        return category_mapping.get(file_info.category)
    
    def generate_report(self) -> str:
        """📊 상세 분석 리포트 생성"""
        report = []
        report.append("📁 파일 관리 시스템 분석 리포트")
        report.append("=" * 50)
        report.append(f"📅 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📂 프로젝트 경로: {self.project_root}")
        report.append(f"📊 총 파일 수: {len(self.files_data)}")
        report.append("")
        
        # 카테고리별 통계
        category_stats = Counter(f.category for f in self.files_data.values())
        report.append("📂 카테고리별 파일 분포:")
        for category, count in category_stats.most_common():
            percentage = (count / len(self.files_data)) * 100
            report.append(f"  {category:12} : {count:4}개 ({percentage:5.1f}%)")
        report.append("")
        
        # 크기별 통계
        total_size = sum(f.size for f in self.files_data.values())
        report.append(f"💾 총 용량: {self._format_size(total_size)}")
        
        # 상위 10개 큰 파일
        large_files = sorted(self.files_data.values(), key=lambda x: x.size, reverse=True)[:10]
        report.append("\n📏 크기가 큰 파일 Top 10:")
        for i, file_info in enumerate(large_files, 1):
            report.append(f"  {i:2}. {file_info.name:30} ({self._format_size(file_info.size)})")
        
        # 중요도 높은 파일
        important_files = sorted(self.files_data.values(), key=lambda x: x.usage_score, reverse=True)[:10]
        report.append("\n⭐ 중요도 높은 파일 Top 10:")
        for i, file_info in enumerate(important_files, 1):
            report.append(f"  {i:2}. {file_info.name:30} (점수: {file_info.usage_score})")
        
        # 확장자별 통계
        ext_stats = Counter(f.extension for f in self.files_data.values() if f.extension)
        report.append("\n📄 확장자별 파일 수:")
        for ext, count in ext_stats.most_common(10):
            report.append(f"  {ext:8} : {count:4}개")
        
        return "\n".join(report)
    
    def _format_size(self, size: int) -> str:
        """파일 크기 포맷팅"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:3.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def create_backup(self) -> str:
        """💾 전체 프로젝트 백업"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"project_backup_{timestamp}"
        backup_path = self.project_root.parent / backup_name
        
        print(f"💾 백업 생성 중: {backup_path}")
        
        try:
            shutil.copytree(
                self.project_root,
                backup_path,
                ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__', '*.pyc')
            )
            print(f"✅ 백업 완료: {backup_path}")
            return str(backup_path)
        except Exception as e:
            print(f"❌ 백업 실패: {e}")
            return ""
    
    def cleanup_temp_files(self) -> int:
        """🧹 임시 파일 정리"""
        print("🧹 임시 파일 정리 중...")
        
        temp_patterns = ['.tmp', '.temp', '.log', '__pycache__', '.pyc', '.pyo']
        cleaned = 0
        
        for file_path, file_info in self.files_data.items():
            if any(pattern in file_info.name.lower() for pattern in temp_patterns):
                try:
                    full_path = self.project_root / file_path
                    if full_path.exists():
                        if full_path.is_file():
                            full_path.unlink()
                        else:
                            shutil.rmtree(full_path)
                        cleaned += 1
                        logging.info(f"삭제: {file_path}")
                except Exception as e:
                    logging.warning(f"삭제 실패: {file_path} - {e}")
        
        print(f"✅ {cleaned}개 임시 파일 정리 완료")
        return cleaned
    
    def save_analysis(self, filename: str = None) -> str:
        """💾 분석 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"file_analysis_{timestamp}.json"
        
        output_path = self.project_root / "reports" / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # 분석 데이터 준비
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'total_files': len(self.files_data),
            'files': [asdict(file_info) for file_info in self.files_data.values()],
            'statistics': {
                'categories': dict(Counter(f.category for f in self.files_data.values())),
                'extensions': dict(Counter(f.extension for f in self.files_data.values())),
                'total_size': sum(f.size for f in self.files_data.values())
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 분석 결과 저장: {output_path}")
        return str(output_path)
    
    def interactive_menu(self) -> None:
        """🎯 대화형 메뉴"""
        while True:
            print("\n" + "="*60)
            print("📁 파일 관리 마스터 시스템")
            print("="*60)
            print("1. 📊 전체 파일 스캔")
            print("2. 🗂️  자동 파일 정리 (시뮬레이션)")
            print("3. 🗂️  자동 파일 정리 (실제 실행)")
            print("4. 🔍 중복 파일 검색")
            print("5. 📋 분석 리포트 생성")
            print("6. 💾 프로젝트 백업")
            print("7. 🧹 임시 파일 정리")
            print("8. 💾 분석 결과 저장")
            print("9. 📊 현재 구조 보기")
            print("0. 🚪 종료")
            print("-"*60)
            
            choice = input("선택하세요 (0-9): ").strip()
            
            try:
                if choice == '1':
                    self.scan_all_files()
                elif choice == '2':
                    moves = self.auto_organize(dry_run=True)
                    print("시뮬레이션 결과:", moves)
                elif choice == '3':
                    confirm = input("❗ 실제로 파일을 이동하시겠습니까? (y/N): ")
                    if confirm.lower() == 'y':
                        moves = self.auto_organize(dry_run=False)
                        print("파일 이동 완료:", moves)
                elif choice == '4':
                    duplicates = self.find_duplicates()
                    if duplicates:
                        print(f"\n🔍 중복 파일 {len(duplicates)}개 그룹:")
                        for i, (hash_val, files) in enumerate(duplicates.items(), 1):
                            print(f"  그룹 {i}: {len(files)}개 파일")
                            for file_path in files:
                                print(f"    - {file_path}")
                elif choice == '5':
                    if not self.files_data:
                        print("❗ 먼저 파일 스캔을 실행하세요.")
                    else:
                        report = self.generate_report()
                        print("\n" + report)
                elif choice == '6':
                    backup_path = self.create_backup()
                    if backup_path:
                        print(f"✅ 백업 완료: {backup_path}")
                elif choice == '7':
                    cleaned = self.cleanup_temp_files()
                    print(f"✅ {cleaned}개 파일 정리 완료")
                elif choice == '8':
                    if not self.files_data:
                        print("❗ 먼저 파일 스캔을 실행하세요.")
                    else:
                        save_path = self.save_analysis()
                        print(f"✅ 저장 완료: {save_path}")
                elif choice == '9':
                    self.show_current_structure()
                elif choice == '0':
                    print("👋 파일 관리 시스템을 종료합니다.")
                    break
                else:
                    print("❌ 잘못된 선택입니다.")
            
            except KeyboardInterrupt:
                print("\n👋 사용자에 의해 중단되었습니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                logging.error(f"메뉴 실행 오류: {e}")
    
    def show_current_structure(self) -> None:
        """📊 현재 프로젝트 구조 보기"""
        print("\n📊 현재 프로젝트 구조:")
        print("-" * 40)
        
        for root, dirs, files in os.walk(self.project_root):
            level = root.replace(str(self.project_root), '').count(os.sep)
            indent = ' ' * 2 * level
            folder_name = os.path.basename(root) if level > 0 else "📁 프로젝트 루트"
            print(f"{indent}📁 {folder_name}/")
            
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # 처음 5개만 표시
                print(f"{subindent}📄 {file}")
            
            if len(files) > 5:
                print(f"{subindent}   ... 외 {len(files) - 5}개 파일")
            
            if level >= 2:  # 너무 깊이 들어가지 않기
                dirs.clear()

def main():
    """메인 실행 함수"""
    print("🚀 파일 관리 마스터 시스템 v1.0 시작!")
    
    # 현재 디렉토리에서 시작
    file_manager = FileManager()
    
    # 대화형 메뉴 시작
    file_manager.interactive_menu()

if __name__ == "__main__":
    main()