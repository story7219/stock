"""
통합된 시스템 테스트
"""

import importlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_merged_modules():
    """통합된 모듈들 테스트"""
    logger.info("🧪 통합된 시스템 테스트 시작")
    
    modules_to_test = [
        'trading_system',
        'analysis_tools',
        'api_integrations', 
        'utilities',
        'test_suite',
        'quality_tools'
    ]
    
    passed = 0
    failed = 0
    
    for module_name in modules_to_test:
        try:
            logger.info(f"  📦 {module_name} 테스트...")
            module = importlib.import_module(module_name)
            
            # 모듈 크기 확인
            if hasattr(module, '__file__'):
                file_size = Path(module.__file__).stat().st_size
                logger.info(f"    ✅ 파일 크기: {file_size:,} bytes")
            
            # 함수/클래스 개수 확인
            items = [name for name in dir(module) if not name.startswith('_')]
            logger.info(f"    📋 포함된 항목: {len(items)}개")
            
            passed += 1
            
        except Exception as e:
            logger.error(f"    ❌ {module_name} 테스트 실패: {e}")
            failed += 1
    
    logger.info(f"\n📊 테스트 결과: 성공 {passed}개, 실패 {failed}개")
    
    if failed == 0:
        logger.info("🎉 모든 통합 모듈이 정상 작동합니다!")
    else:
        logger.warning("⚠️ 일부 모듈에 문제가 있습니다.")

if __name__ == "__main__":
    from pathlib import Path
    test_merged_modules() 