"""
적극적인 파일 통합 도구
- 실제 파일 수를 대폭 줄이는 통합
- 기능별 대형 모듈로 재구성
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AggressiveFileMerger:
    """적극적인 파일 통합기"""
    
    def __init__(self):
        self.merge_plan = {
            # 트레이딩 관련 모든 파일을 하나로
            'trading_system.py': [
                'trading/kis_trading_system.py',
                'trading/trading_bot.py', 
                'trading/trading_simulator.py',
                'trading/trading_dashboard.py',
                'trading/backtest.py'
            ],
            
            # 분석 도구들을 하나로
            'analysis_tools.py': [
                'analyzers/quality_analyzer.py',
                'analyzers/oneil_canslim_analyzer.py',
                'analyzers/auto_refactoring_system.py',
                'analyzers/visualization.py'
            ],
            
            # API 관련 모든 파일을 하나로
            'api_integrations.py': [
                'api/kis_api.py',
                'api/telegram_bot.py',
                'api/google_sheets_integration.py'
            ],
            
            # 유틸리티와 설정을 하나로
            'utilities.py': [
                'utils/utils.py',
                'utils/config.py'
            ],
            
            # 테스트 파일들을 하나로
            'test_suite.py': [
                'tests/test_kis_api.py',
                'tests/test_quality.py',
                'tests/test_telegram.py'
            ],
            
            # 품질 관리 도구들을 하나로
            'quality_tools.py': [
                'quality/auto_quality_service.py',
                'quality/auto_file_organizer.py',
                'quality/file_structure_analyzer.py'
            ]
        }

    def create_mega_backup(self):
        """메가 백업 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"mega_backup_{timestamp}")
        backup_path.mkdir(exist_ok=True)
        
        # 현재 전체 구조 백업
        for item in Path('.').iterdir():
            if item.is_file() and item.suffix == '.py':
                shutil.copy2(item, backup_path)
            elif item.is_dir() and item.name not in ['backup_safe', 'mega_backup_*']:
                shutil.copytree(item, backup_path / item.name, ignore_errors=True)
        
        logger.info(f"🔄 메가 백업 생성: {backup_path}")
        return backup_path

    def merge_files_aggressively(self):
        """적극적인 파일 통합"""
        logger.info("🔥 적극적인 파일 통합 시작")
        
        # 백업 생성
        backup_path = self.create_mega_backup()
        
        merged_count = 0
        
        for target_file, source_files in self.merge_plan.items():
            logger.info(f"📦 {target_file} 생성 중...")
            
            merged_content = []
            merged_content.append(f'"""')
            merged_content.append(f'{target_file} - 통합 모듈')
            merged_content.append(f'생성일: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            merged_content.append(f'통합된 파일: {len(source_files)}개')
            merged_content.append('"""')
            merged_content.append('')
            
            # 모든 import 수집
            all_imports = set()
            file_contents = {}
            
            for source_file in source_files:
                source_path = Path(source_file)
                if source_path.exists():
                    try:
                        with open(source_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        file_contents[source_file] = content
                        
                        # import 문 추출
                        lines = content.splitlines()
                        for line in lines:
                            line = line.strip()
                            if line.startswith('import ') or line.startswith('from '):
                                all_imports.add(line)
                                
                    except Exception as e:
                        logger.error(f"파일 읽기 실패 {source_file}: {e}")
            
            # import 문 정리
            sorted_imports = sorted(all_imports)
            for imp in sorted_imports:
                merged_content.append(imp)
            
            merged_content.append('')
            merged_content.append('')
            
            # 파일 내용 병합
            for source_file, content in file_contents.items():
                file_name = Path(source_file).name
                
                # import 문 제거한 실제 코드만 추출
                clean_content = self.extract_clean_code(content)
                
                if clean_content.strip():
                    merged_content.append(f"# " + "="*60)
                    merged_content.append(f"# {file_name}에서 가져온 코드")
                    merged_content.append(f"# " + "="*60)
                    merged_content.append(clean_content)
                    merged_content.append('')
            
            # 통합 파일 저장
            try:
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(merged_content))
                
                logger.info(f"✅ {target_file} 생성 완료 ({len(source_files)}개 파일 통합)")
                merged_count += 1
                
            except Exception as e:
                logger.error(f"통합 파일 저장 실패 {target_file}: {e}")
        
        # 원본 디렉토리들 제거
        directories_to_remove = ['trading', 'analyzers', 'api', 'utils', 'tests', 'quality']
        for dir_name in directories_to_remove:
            dir_path = Path(dir_name)
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"🗑️ {dir_name}/ 디렉토리 제거")
        
        logger.info(f"🎉 적극적인 통합 완료: {merged_count}개 대형 모듈 생성")
        logger.info(f"💾 백업 위치: {backup_path}")
        
        return merged_count

    def extract_clean_code(self, content: str) -> str:
        """import문과 docstring 제거한 실제 코드만 추출"""
        lines = content.splitlines()
        clean_lines = []
        
        skip_imports = True
        in_docstring = False
        docstring_char = None
        
        for line in lines:
            stripped = line.strip()
            
            # docstring 처리
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    in_docstring = True
                    docstring_char = stripped[:3]
                    if stripped.endswith(docstring_char) and len(stripped) > 3:
                        in_docstring = False
                    continue
                elif stripped.endswith(docstring_char):
                    in_docstring = False
                    continue
            
            if in_docstring:
                continue
            
            # import 구간 건너뛰기
            if skip_imports:
                if (stripped.startswith('import ') or 
                    stripped.startswith('from ') or 
                    stripped.startswith('#') or 
                    not stripped):
                    continue
                else:
                    skip_imports = False
            
            # 실제 코드 라인 추가
            clean_lines.append(line)
        
        return '\n'.join(clean_lines)

def main():
    """메인 실행 함수"""
    print("⚠️  경고: 이 작업은 모든 파일을 6개의 대형 모듈로 통합합니다.")
    print("📁 현재 31개 파일 → 6개 파일로 대폭 축소")
    print("💾 자동으로 백업이 생성되지만, 신중하게 결정하세요.")
    print()
    
    response = input("정말로 적극적인 파일 통합을 진행하시겠습니까? (yes/no): ")
    
    if response.lower() == 'yes':
        merger = AggressiveFileMerger()
        merged_count = merger.merge_files_aggressively()
        
        print(f"\n🎉 통합 완료!")
        print(f"📊 31개 파일 → {merged_count + 4}개 파일 (main.py, debug_tools.py, fix_imports.py, run_tests.py 포함)")
        print(f"📉 파일 수 감소: {31 - (merged_count + 4)}개")
        
    else:
        print("❌ 작업이 취소되었습니다.")

if __name__ == "__main__":
    main() 