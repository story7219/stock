"""
Import 경로 자동 수정 도구
- 파일 이동 후 깨진 import 경로 자동 수정
- 새로운 디렉토리 구조에 맞게 경로 업데이트
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImportPathFixer:
    """Import 경로 자동 수정기"""
    
    def __init__(self):
        # 파일 이동 매핑 (이전 경로 → 새 경로)
        self.file_mapping = {
            'kis_trading_system.py': 'trading.kis_trading_system',
            'trading_bot.py': 'trading.trading_bot',
            'trading_simulator.py': 'trading.trading_simulator',
            'trading_dashboard.py': 'trading.trading_dashboard',
            'backtest.py': 'trading.backtest',
            
            'quality_analyzer.py': 'analyzers.quality_analyzer',
            'oneil_canslim_analyzer.py': 'analyzers.oneil_canslim_analyzer',
            'auto_refactoring_system.py': 'analyzers.auto_refactoring_system',
            'visualization.py': 'analyzers.visualization',
            
            'kis_api.py': 'api.kis_api',
            'telegram_bot.py': 'api.telegram_bot',
            'google_sheets_integration.py': 'api.google_sheets_integration',
            
            'utils.py': 'utils.utils',
            'config.py': 'utils.config',
            
            'test_kis_api.py': 'tests.test_kis_api',
            'test_quality.py': 'tests.test_quality',
            'test_telegram.py': 'tests.test_telegram',
            
            'auto_quality_service.py': 'quality.auto_quality_service',
            'auto_file_organizer.py': 'quality.auto_file_organizer',
            'file_structure_analyzer.py': 'quality.file_structure_analyzer',
            
            'debug_gemini.py': 'debug_tools'  # 통합된 파일
        }
        
        # 클래스/함수 매핑 (필요시)
        self.class_mapping = {
            'CodeQualityAnalyzer': 'analyzers.quality_analyzer',
            'OneilAnalyzer': 'analyzers.oneil_canslim_analyzer',
            'KisAPI': 'api.kis_api',
            'TelegramBot': 'api.telegram_bot',
            'TradingSystem': 'trading.kis_trading_system'
        }

    def find_all_python_files(self) -> List[Path]:
        """모든 Python 파일 찾기"""
        python_files = []
        
        # 루트 디렉토리 파일들
        for file in Path('.').glob('*.py'):
            python_files.append(file)
        
        # 서브 디렉토리 파일들
        for directory in ['trading', 'analyzers', 'api', 'utils', 'tests', 'quality']:
            dir_path = Path(directory)
            if dir_path.exists():
                for file in dir_path.glob('*.py'):
                    if file.name != '__init__.py':
                        python_files.append(file)
        
        return python_files

    def analyze_imports(self, file_path: Path) -> List[Dict]:
        """파일의 import 문 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            lines = content.splitlines()
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # import 문 패턴 매칭
                if line.startswith('import ') or line.startswith('from '):
                    imports.append({
                        'line_number': i,
                        'original': line,
                        'type': 'import' if line.startswith('import ') else 'from_import'
                    })
            
            return imports
            
        except Exception as e:
            logger.error(f"Import 분석 실패 {file_path}: {e}")
            return []

    def fix_import_line(self, import_line: str) -> str:
        """개별 import 라인 수정"""
        original_line = import_line
        
        # from X import Y 패턴
        from_match = re.match(r'from\s+([^\s]+)\s+import\s+(.+)', import_line)
        if from_match:
            module_name = from_match.group(1)
            imports = from_match.group(2)
            
            # 파일명에서 모듈명으로 변환
            for old_file, new_module in self.file_mapping.items():
                old_module = old_file.replace('.py', '')
                if module_name == old_module:
                    return f"from {new_module} import {imports}"
                elif module_name.endswith(f'.{old_module}'):
                    base_module = module_name.replace(f'.{old_module}', '')
                    return f"from {base_module}.{new_module} import {imports}"
        
        # import X 패턴
        import_match = re.match(r'import\s+(.+)', import_line)
        if import_match:
            modules = import_match.group(1).split(',')
            fixed_modules = []
            
            for module in modules:
                module = module.strip()
                
                # 파일명에서 모듈명으로 변환
                for old_file, new_module in self.file_mapping.items():
                    old_module = old_file.replace('.py', '')
                    if module == old_module:
                        fixed_modules.append(new_module)
                        break
                else:
                    fixed_modules.append(module)
            
            if fixed_modules != [m.strip() for m in modules]:
                return f"import {', '.join(fixed_modules)}"
        
        return original_line

    def fix_file_imports(self, file_path: Path) -> int:
        """파일의 모든 import 수정"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            fixed_count = 0
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    fixed_line = self.fix_import_line(stripped)
                    if fixed_line != stripped:
                        # 원래 들여쓰기 유지
                        indent = line[:len(line) - len(line.lstrip())]
                        lines[i] = indent + fixed_line + '\n'
                        fixed_count += 1
                        logger.info(f"  수정: {stripped} → {fixed_line}")
            
            if fixed_count > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                logger.info(f"✅ {file_path}: {fixed_count}개 import 수정")
            
            return fixed_count
            
        except Exception as e:
            logger.error(f"Import 수정 실패 {file_path}: {e}")
            return 0

    def run_import_fix(self):
        """전체 import 수정 실행"""
        logger.info("🔧 Import 경로 자동 수정 시작")
        logger.info("="*50)
        
        python_files = self.find_all_python_files()
        total_fixed = 0
        
        for file_path in python_files:
            logger.info(f"📄 수정 중: {file_path}")
            fixed_count = self.fix_file_imports(file_path)
            total_fixed += fixed_count
        
        logger.info("="*50)
        logger.info(f"✅ Import 수정 완료: {len(python_files)}개 파일, {total_fixed}개 import 수정")

def main():
    """메인 실행 함수"""
    fixer = ImportPathFixer()
    fixer.run_import_fix()

if __name__ == "__main__":
    main() 