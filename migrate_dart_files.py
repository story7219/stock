#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: migrate_dart_files.py
모듈: 기존 DART 파일 마이그레이션 스크립트
목적: 기존 DART 관련 파일들을 백업하고 새로운 통합 시스템으로 마이그레이션

Author: Trading AI System
Created: 2025-01-07
Version: 1.0.0
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backup_existing_files():
    """기존 DART 파일들 백업"""
    backup_dir = Path('backup_dart_files') / datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    dart_files = [
        'data/dart.py',
        'data/dart_all_collector.py', 
        'data/dart_realtime_api.py'
    ]
    
    backed_up_files = []
    
    for file_path in dart_files:
        if Path(file_path).exists():
            try:
                # 백업 디렉토리 생성
                file_backup_dir = backup_dir / Path(file_path).parent.name
                file_backup_dir.mkdir(parents=True, exist_ok=True)
                
                # 파일 복사
                backup_path = file_backup_dir / Path(file_path).name
                shutil.copy2(file_path, backup_path)
                
                backed_up_files.append(file_path)
                logger.info(f"✅ 백업 완료: {file_path} -> {backup_path}")
                
            except Exception as e:
                logger.error(f"❌ 백업 실패 {file_path}: {e}")
    
    logger.info(f"📦 총 {len(backed_up_files)}개 파일 백업 완료")
    return backup_dir


def create_migration_guide():
    """마이그레이션 가이드 생성"""
    guide_content = """# DART 파일 마이그레이션 가이드

## 📋 변경 사항

### 기존 파일들
- `data/dart.py` - 실시간 공시 모니터링
- `data/dart_all_collector.py` - 대용량 데이터 수집  
- `data/dart_realtime_api.py` - 실시간 API 호출

### 새로운 통합 파일
- `dart_unified_collector.py` - 모든 기능 통합

## 🔄 마이그레이션 방법

### 1. 기존 코드 사용법 (레거시)

```python
# 기존 실시간 모니터링
from data.dart import DARTMonitor
monitor = DARTMonitor()
await monitor.start_monitoring()

# 기존 대용량 수집
from data.dart_all_collector import DARTDataCollector
collector = DARTDataCollector()
await collector.run_full_collection()

# 기존 실시간 API
from data.dart_realtime_api import DartRealtimeAPI
api = DartRealtimeAPI()
await api.fetch_all(corp_code, year)
```

### 2. 새로운 통합 코드 사용법

```python
from dart_unified_collector import DARTUnifiedCollector, UnifiedConfig

# 설정
config = UnifiedConfig(
    api_key=os.environ.get('DART_API_KEY'),
    include_disclosures=True,
    include_financials=True,
    enable_monitoring=True,
    enable_realtime=True
)

# 통합 실행
async with DARTUnifiedCollector(config) as collector:
    await collector.run_unified_system()
```

## 🆕 새로운 기능

### 1. 통합 설정
- 모든 설정을 하나의 `UnifiedConfig`로 관리
- JSON 설정 파일 지원

### 2. 향상된 에러 처리
- 더 안정적인 예외 처리
- 자동 재시도 메커니즘

### 3. 성능 최적화
- 비동기 처리 개선
- 메모리 사용량 최적화

### 4. 모니터링 강화
- 실시간 진행률 표시
- 상세한 로깅

## ⚠️ 주의사항

1. **기존 파일들은 백업됨**: `backup_dart_files/` 디렉토리에서 확인 가능
2. **API 호환성**: 기존 API와 호환되도록 설계됨
3. **점진적 마이그레이션**: 기존 코드를 단계적으로 교체 가능

## 🔧 문제 해결

### 기존 코드가 작동하지 않는 경우
1. 백업된 파일에서 원본 코드 복원
2. 새로운 통합 시스템으로 점진적 마이그레이션
3. 필요시 기존 코드 유지

### 성능 문제
1. 설정 조정 (`request_delay`, `batch_size` 등)
2. 메모리 사용량 모니터링
3. 로그 레벨 조정

## 📞 지원

문제가 발생하거나 개선 사항이 있으시면 이슈를 등록해 주세요.

---

**© 2025 Trading AI System. All rights reserved.**
"""
    
    with open('DART_MIGRATION_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    logger.info("📖 마이그레이션 가이드 생성 완료: DART_MIGRATION_GUIDE.md")


def create_compatibility_wrapper():
    """호환성 래퍼 생성"""
    wrapper_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: dart_compatibility.py
모듈: 기존 DART 코드 호환성 래퍼
목적: 기존 코드와의 호환성을 위한 래퍼 클래스들

Author: Trading AI System
Created: 2025-01-07
Version: 1.0.0
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

# 새로운 통합 시스템 import
from dart_unified_collector import DARTUnifiedCollector, UnifiedConfig, DisclosureData, DisclosureAlert


class DARTMonitor:
    """기존 DARTMonitor 호환성 래퍼"""
    
    def __init__(self):
        self.config = UnifiedConfig(
            api_key=os.environ.get('DART_API_KEY', ''),
            enable_monitoring=True
        )
        self.collector = None
        self.monitored_corps = []
        self.keywords = []
        self.callbacks = []
        self.running = False
        
    async def start_monitoring(self, corps: List[str] = None, keywords: List[str] = None):
        """공시 모니터링 시작 (기존 호환성)"""
        if corps:
            self.monitored_corps = corps
        if keywords:
            self.keywords = keywords
            
        self.collector = DARTUnifiedCollector(self.config)
        await self.collector.start_monitoring(self.monitored_corps, self.keywords)
        
    async def stop_monitoring(self):
        """공시 모니터링 중지 (기존 호환성)"""
        if self.collector:
            await self.collector.stop_monitoring()
            
    def add_callback(self, callback: Callable):
        """콜백 함수 추가 (기존 호환성)"""
        if self.collector:
            self.collector.add_callback(callback)
            
    def get_recent_disclosures(self, hours: int = 24) -> List[DisclosureData]:
        """최근 공시 조회 (기존 호환성)"""
        if self.collector:
            return self.collector.get_recent_disclosures(hours)
        return []
        
    def get_disclosures_by_corp(self, corp_code: str) -> List[DisclosureData]:
        """기업별 공시 조회 (기존 호환성)"""
        if self.collector:
            return self.collector.get_disclosures_by_corp(corp_code)
        return []
        
    def get_disclosures_by_keyword(self, keyword: str) -> List[DisclosureData]:
        """키워드별 공시 조회 (기존 호환성)"""
        if self.collector:
            return self.collector.get_disclosures_by_keyword(keyword)
        return []


class DARTDataCollector:
    """기존 DARTDataCollector 호환성 래퍼"""
    
    def __init__(self):
        self.config = UnifiedConfig(
            api_key=os.environ.get('DART_API_KEY', ''),
            include_disclosures=True,
            include_financials=True,
            include_executives=True,
            include_dividends=True,
            include_auditors=True,
            include_corp_info=True
        )
        self.collector = None
        
    async def __aenter__(self):
        self.collector = DARTUnifiedCollector(self.config)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def run_full_collection(self):
        """전체 수집 실행 (기존 호환성)"""
        if self.collector:
            await self.collector.collect_all_historical_data()


class DartRealtimeAPI:
    """기존 DartRealtimeAPI 호환성 래퍼"""
    
    def __init__(self, api_key: str = None):
        self.config = UnifiedConfig(
            api_key=api_key or os.environ.get('DART_API_KEY', ''),
            enable_realtime=True
        )
        self.collector = None
        
    async def initialize(self):
        """초기화 (기존 호환성)"""
        self.collector = DARTUnifiedCollector(self.config)
        
    async def close(self):
        """종료 (기존 호환성)"""
        pass
        
    async def fetch_all(self, corp_code: str, year: int) -> Dict[str, Any]:
        """통합 데이터 수집 (기존 호환성)"""
        if self.collector:
            return await self.collector.fetch_all_realtime_data(corp_code, year)
        return {}
        
    async def periodic_task(self, corp_code: str, year: int, interval_min: int = 60):
        """주기적 작업 (기존 호환성)"""
        if self.collector:
            await self.collector.periodic_realtime_task(corp_code, year, interval_min)


# 기존 import 호환성을 위한 별칭
__all__ = ['DARTMonitor', 'DARTDataCollector', 'DartRealtimeAPI']
'''
    
    with open('dart_compatibility.py', 'w', encoding='utf-8') as f:
        f.write(wrapper_content)
    
    logger.info("🔧 호환성 래퍼 생성 완료: dart_compatibility.py")


def main():
    """메인 실행 함수"""
    logger.info("🔄 DART 파일 마이그레이션 시작")
    
    try:
        # 1. 기존 파일 백업
        backup_dir = backup_existing_files()
        
        # 2. 마이그레이션 가이드 생성
        create_migration_guide()
        
        # 3. 호환성 래퍼 생성
        create_compatibility_wrapper()
        
        logger.info("✅ 마이그레이션 완료!")
        logger.info(f"📁 백업 위치: {backup_dir}")
        logger.info("📖 가이드: DART_MIGRATION_GUIDE.md")
        logger.info("🔧 호환성: dart_compatibility.py")
        
    except Exception as e:
        logger.error(f"❌ 마이그레이션 실패: {e}")


if __name__ == "__main__":
    main() 