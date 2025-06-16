"""
통합된 파일 구조에 맞는 Import 경로 수정
"""

import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MergedImportFixer:
    """통합된 구조용 Import 수정기"""
    
    def __init__(self):
        # 새로운 import 매핑
        self.new_import_mapping = {
            # 트레이딩 관련
            'from trading.kis_trading_system import': 'from trading_system import',
            'from trading.trading_bot import': 'from trading_system import',
            'from trading.backtest import': 'from trading_system import',
            'from trading.trading_simulator import': 'from trading_system import',
            'from trading.trading_dashboard import': 'from trading_system import',
            
            # 분석 도구
            'from analyzers.quality_analyzer import': 'from analysis_tools import',
            'from analyzers.oneil_canslim_analyzer import': 'from analysis_tools import',
            'from analyzers.auto_refactoring_system import': 'from analysis_tools import',
            'from analyzers.visualization import': 'from analysis_tools import',
            
            # API 연동
            'from api.kis_api import': 'from api_integrations import',
            'from api.telegram_bot import': 'from api_integrations import',
            'from api.google_sheets_integration import': 'from api_integrations import',
            
            # 유틸리티
            'from utils.utils import': 'from utilities import',
            'from utils.config import': 'from utilities import',
            
            # 테스트
            'from tests.test_kis_api import': 'from test_suite import',
            'from tests.test_quality import': 'from test_suite import',
            'from tests.test_telegram import': 'from test_suite import',
            
            # 품질 도구
            'from quality.auto_quality_service import': 'from quality_tools import',
            'from quality.auto_file_organizer import': 'from quality_tools import',
            'from quality.file_structure_analyzer import': 'from quality_tools import'
        }

    def fix_merged_imports(self):
        """통합된 파일들의 import 수정"""
        logger.info("🔧 통합된 구조에 맞는 Import 수정 시작")
        
        # 수정할 파일들
        files_to_fix = [
            'main.py',
            'trading_system.py',
            'analysis_tools.py', 
            'api_integrations.py',
            'utilities.py',
            'test_suite.py',
            'quality_tools.py',
            'debug_tools.py'
        ]
        
        total_fixed = 0
        
        for file_name in files_to_fix:
            file_path = Path(file_name)
            if file_path.exists():
                fixed_count = self.fix_file_imports(file_path)
                total_fixed += fixed_count
        
        logger.info(f"✅ 통합 구조 Import 수정 완료: {total_fixed}개 수정")

    def fix_file_imports(self, file_path: Path) -> int:
        """개별 파일의 import 수정"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixed_count = 0
            
            # 매핑된 import 수정
            for old_import, new_import in self.new_import_mapping.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    fixed_count += 1
                    logger.info(f"  수정: {old_import} → {new_import}")
            
            # 변경사항이 있으면 파일 저장
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"✅ {file_path}: {fixed_count}개 import 수정")
            
            return fixed_count
            
        except Exception as e:
            logger.error(f"Import 수정 실패 {file_path}: {e}")
            return 0

def main():
    """메인 실행 함수"""
    fixer = MergedImportFixer()
    fixer.fix_merged_imports()

if __name__ == "__main__":
    main() 