"""
파일 자동 정리 모듈
- 파일의 역할에 따라 카테고리별 하위 폴더로 이동하여 프로젝트 구조 정리
"""
import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FileOrganizer:
    CATEGORIES = {
        'core': ['main.py', 'config.py'],
        'trading': ['core_trader.py'],
        'analysis': ['analysis_engine.py'],
        'maintenance': ['auto_file_organizer.py', 'auto_quality_analyzer.py', 'auto_refactor_proposer.py', 'run_maintenance.py']
    }

    def organize_project_files(self, project_path="."):
        logger.info("프로젝트 파일 자동 정리를 시작합니다...")
        base_path = Path(project_path)
        
        flat_map = {file: cat for cat, files in self.CATEGORIES.items() for file in files}
        
        all_files_to_move = [f for files in self.CATEGORIES.values() for f in files]

        for file_name in all_files_to_move:
            source_path = base_path / file_name
            if not source_path.exists():
                logger.warning(f"'{file_name}' 파일이 없어 건너뜁니다.")
                continue

            category = flat_map.get(file_name)
            if not category:
                continue

            target_dir = base_path / category
            target_dir.mkdir(exist_ok=True)
            target_path = target_dir / file_name

            try:
                shutil.move(str(source_path), str(target_path))
                logger.info(f"✅ '{source_path}' -> '{target_path}' 이동 완료")
            except Exception as e:
                logger.error(f"❌ '{file_name}' 이동 실패: {e}")
        
        logger.info("파일 정리가 완료되었습니다.") 