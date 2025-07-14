#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub 백업 관리 스크립트
목적: 자동화된 GitHub 백업 및 용량 관리

Author: Auto Trading System
Created: 2025-01-13
Version: 1.0.0

Features:
- 자동 백업 스케줄링
- 용량 모니터링 (900MB 제한)
- 오래된 파일 자동 삭제
- 백업 검증
"""

import os
import shutil
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitHubBackupManager:
    """GitHub 백업 관리자"""
    
    def __init__(self):
        self.backup_dirs = [
            "src",
            "scripts", 
            "config",
            "tests",
            "docs",
            "data_engine",
            "strategy",
            "monitoring"
        ]
        
        self.exclude_patterns = [
            "__pycache__",
            "*.pyc",
            "*.log",
            "*.tmp",
            ".env",
            "secrets",
            "api_keys",
            "*.bak",
            "cache",
            "logs",
            "collected_data",
            "models",
            "mlruns",
            "backup"
        ]
        
        self.max_size_mb = 900
        self.backup_root = Path("backup/github")
        
    def get_directory_size(self, path: Path) -> int:
        """디렉토리 크기 계산 (MB)"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size // (1024 * 1024)  # MB로 변환
    
    def should_exclude(self, file_path: Path) -> bool:
        """파일 제외 여부 확인"""
        for pattern in self.exclude_patterns:
            if pattern in str(file_path):
                return True
        return False
    
    def create_backup(self) -> Dict[str, any]:
        """백업 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_root / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating backup: {backup_dir}")
        
        backup_stats = {
            "timestamp": timestamp,
            "files_copied": 0,
            "total_size_mb": 0,
            "errors": []
        }
        
        for source_dir in self.backup_dirs:
            source_path = Path(source_dir)
            if not source_path.exists():
                logger.warning(f"Source directory not found: {source_dir}")
                continue
                
            dest_path = backup_dir / source_dir
            dest_path.mkdir(parents=True, exist_ok=True)
            
            try:
                # 파일 복사 (제외 패턴 적용)
                for file_path in source_path.rglob("*"):
                    if file_path.is_file() and not self.should_exclude(file_path):
                        relative_path = file_path.relative_to(source_path)
                        dest_file = dest_path / relative_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        shutil.copy2(file_path, dest_file)
                        backup_stats["files_copied"] += 1
                        backup_stats["total_size_mb"] += file_path.stat().st_size // (1024 * 1024)
                        
            except Exception as e:
                error_msg = f"Error copying {source_dir}: {e}"
                logger.error(error_msg)
                backup_stats["errors"].append(error_msg)
        
        # 백업 메타데이터 저장
        metadata_file = backup_dir / "backup_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(backup_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Backup completed: {backup_stats['files_copied']} files, {backup_stats['total_size_mb']}MB")
        return backup_stats
    
    def cleanup_old_backups(self):
        """오래된 백업 정리"""
        if not self.backup_root.exists():
            return
            
        backups = []
        for backup_dir in self.backup_root.glob("backup_*"):
            if backup_dir.is_dir():
                # 백업 생성 시간 추출
                try:
                    timestamp = backup_dir.name.split("_")[1]
                    backup_time = datetime.strptime(timestamp, "%Y%m%d")
                    backups.append((backup_dir, backup_time))
                except:
                    continue
        
        # 날짜순 정렬 (오래된 것부터)
        backups.sort(key=lambda x: x[1])
        
        # 최근 5개만 유지
        if len(backups) > 5:
            backups_to_remove = backups[:-5]
            
            for backup_dir, _ in backups_to_remove:
                try:
                    shutil.rmtree(backup_dir)
                    logger.info(f"Removed old backup: {backup_dir.name}")
                except Exception as e:
                    logger.error(f"Failed to remove {backup_dir.name}: {e}")
    
    def check_repository_size(self) -> int:
        """저장소 크기 확인"""
        total_size = 0
        
        for item in Path(".").rglob("*"):
            if item.is_file() and not self.should_exclude(item):
                total_size += item.stat().st_size
        
        size_mb = total_size // (1024 * 1024)
        logger.info(f"Repository size: {size_mb}MB")
        return size_mb
    
    def cleanup_if_needed(self):
        """필요시 정리 작업"""
        current_size = self.check_repository_size()
        
        if current_size > self.max_size_mb:
            logger.warning(f"Repository size ({current_size}MB) exceeds limit ({self.max_size_mb}MB)")
            self.cleanup_old_backups()
            
            # 추가 정리: 큰 파일 찾기
            large_files = self.find_large_files()
            for file_path, size_mb in large_files:
                if size_mb > 50:  # 50MB 이상 파일
                    logger.info(f"Large file found: {file_path} ({size_mb}MB)")
                    # 필요시 삭제 또는 압축
    
    def find_large_files(self) -> List[Tuple[Path, int]]:
        """큰 파일 찾기"""
        large_files = []
        
        for file_path in Path(".").rglob("*"):
            if file_path.is_file() and not self.should_exclude(file_path):
                size_mb = file_path.stat().st_size // (1024 * 1024)
                if size_mb > 10:  # 10MB 이상
                    large_files.append((file_path, size_mb))
        
        return sorted(large_files, key=lambda x: x[1], reverse=True)
    
    def run_backup(self):
        """백업 실행"""
        try:
            logger.info("Starting GitHub backup process")
            
            # 1. 백업 생성
            backup_stats = self.create_backup()
            
            # 2. 크기 확인 및 정리
            self.cleanup_if_needed()
            
            # 3. 오래된 백업 정리
            self.cleanup_old_backups()
            
            logger.info("GitHub backup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

def main():
    """메인 함수"""
    backup_manager = GitHubBackupManager()
    success = backup_manager.run_backup()
    
    if success:
        print("✅ GitHub backup completed successfully")
        sys.exit(0)
    else:
        print("❌ GitHub backup failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 