"""
자동 테스트 실행 도구
- 모든 모듈 import 테스트
- 기본 기능 동작 확인
- 오류 발생 시 상세 리포트
"""

import sys
import importlib
import traceback
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoTester:
    """자동 테스트 실행기"""
    
    def __init__(self):
        self.test_results = {
            'passed': [],
            'failed': [],
            'errors': []
        }
        
        # 테스트할 모듈들
        self.modules_to_test = [
            'trading.kis_trading_system',
            'trading.trading_bot',
            'trading.backtest',
            'analyzers.quality_analyzer',
            'analyzers.oneil_canslim_analyzer',
            'api.kis_api',
            'api.telegram_bot',
            'utils.utils',
            'utils.config'
        ]

    def test_module_import(self, module_name: str) -> bool:
        """모듈 import 테스트"""
        try:
            logger.info(f"  📦 {module_name} import 테스트...")
            
            # 모듈 import 시도
            module = importlib.import_module(module_name)
            
            # 기본 속성 확인
            if hasattr(module, '__file__'):
                logger.info(f"    ✅ 파일 위치: {module.__file__}")
            
            # 주요 클래스/함수 확인
            classes = [name for name in dir(module) if not name.startswith('_') and callable(getattr(module, name))]
            if classes:
                logger.info(f"    📋 발견된 클래스/함수: {len(classes)}개")
            
            self.test_results['passed'].append(module_name)
            return True
            
        except ImportError as e:
            logger.error(f"    ❌ Import 오류: {e}")
            self.test_results['failed'].append((module_name, f"Import 오류: {e}"))
            return False
            
        except Exception as e:
            logger.error(f"    ❌ 예상치 못한 오류: {e}")
            self.test_results['errors'].append((module_name, f"예상치 못한 오류: {e}"))
            return False

    def test_main_functionality(self):
        """주요 기능 테스트"""
        logger.info("🔍 주요 기능 테스트...")
        
        # main.py 실행 테스트
        try:
            logger.info("  📄 main.py 구문 검사...")
            with open('main.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 구문 오류 검사
            compile(content, 'main.py', 'exec')
            logger.info("    ✅ main.py 구문 검사 통과")
            self.test_results['passed'].append('main.py 구문 검사')
            
        except SyntaxError as e:
            logger.error(f"    ❌ main.py 구문 오류: {e}")
            self.test_results['failed'].append(('main.py', f"구문 오류: {e}"))
        except FileNotFoundError:
            logger.warning("    ⚠️ main.py 파일을 찾을 수 없습니다")
        except Exception as e:
            logger.error(f"    ❌ main.py 테스트 오류: {e}")
            self.test_results['errors'].append(('main.py', f"테스트 오류: {e}"))

    def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("🧪 자동 테스트 시작")
        logger.info("="*50)
        
        # 모듈 import 테스트
        logger.info("📦 모듈 Import 테스트...")
        for module_name in self.modules_to_test:
            self.test_module_import(module_name)
        
        # 주요 기능 테스트
        self.test_main_functionality()
        
        # 결과 리포트
        self.generate_test_report()

    def generate_test_report(self):
        """테스트 결과 리포트 생성"""
        logger.info("="*50)
        logger.info("📊 테스트 결과 리포트")
        logger.info("="*50)
        
        total_tests = len(self.test_results['passed']) + len(self.test_results['failed']) + len(self.test_results['errors'])
        
        logger.info(f"✅ 성공: {len(self.test_results['passed'])}개")
        logger.info(f"❌ 실패: {len(self.test_results['failed'])}개")
        logger.info(f"⚠️ 오류: {len(self.test_results['errors'])}개")
        logger.info(f"📊 전체: {total_tests}개")
        
        if self.test_results['passed']:
            logger.info("\n✅ 성공한 테스트:")
            for test in self.test_results['passed']:
                logger.info(f"  - {test}")
        
        if self.test_results['failed']:
            logger.info("\n❌ 실패한 테스트:")
            for test, error in self.test_results['failed']:
                logger.info(f"  - {test}: {error}")
        
        if self.test_results['errors']:
            logger.info("\n⚠️ 오류가 발생한 테스트:")
            for test, error in self.test_results['errors']:
                logger.info(f"  - {test}: {error}")
        
        # 성공률 계산
        if total_tests > 0:
            success_rate = (len(self.test_results['passed']) / total_tests) * 100
            logger.info(f"\n📈 성공률: {success_rate:.1f}%")
            
            if success_rate >= 90:
                logger.info("🎉 테스트 결과: 우수!")
            elif success_rate >= 70:
                logger.info("👍 테스트 결과: 양호")
            else:
                logger.info("⚠️ 테스트 결과: 개선 필요")

def main():
    """메인 실행 함수"""
    tester = AutoTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 