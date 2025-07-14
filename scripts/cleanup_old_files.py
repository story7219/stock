#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
오래된 파일 자동 삭제 스크립트
목적: GitHub 저장소 크기 관리 (900MB 제한)

Author: Auto Trading System
Created: 2025-01-13
Version: 1.0.0
"""

import os
import shutil
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict
import subprocess

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileCleanupManager:
    """파일 정리 관리자"""
    
    def __init__(self):
        self.max_size_mb = 900
        self.backup_dirs = [
            "backup/github",
            "backup/auto_fix",
            "backup/krx",
            "backup/ichimoku"
        ]
        
        # 삭제 우선순위 (높은 우선순위부터)
        self.cleanup_priorities = [
            # 1순위: 임시 파일들
            {
                "patterns": ["*.tmp", "*.bak", "*.log", "*.cache"],
                "max_age_days": 7,
                "description": "임시 파일"
            },
            # 2순위: 오래된 백업
            {
                "patterns": ["backup_*"],
                "max_age_days": 30,
                "description": "오래된 백업"
            },
            # 3순위: 큰 파일들
            {
                "patterns": ["*.parquet", "*.h5", "*.hdf5"],
                "max_age_days": 60,
                "description": "큰 데이터 파일"
            }
        ]
    
    def get_file_size_mb(self, file_path: Path) -> int:
        """파일 크기 계산 (MB)"""
        try:
            return file_path.stat().st_size // (1024 * 1024)
        except:
            return 0
    
    def get_file_age_days(self, file_path: Path) -> int:
        """파일 나이 계산 (일)"""
        try:
            mtime = file_path.stat().st_mtime
            file_date = datetime.fromtimestamp(mtime)
            age = datetime.now() - file_date
            return age.days
        except:
            return 999  # 오류 시 매우 오래된 것으로 처리
    
    def find_files_to_cleanup(self) -> List[Tuple[Path, Dict]]:
        """정리할 파일 찾기"""
        files_to_cleanup = []
        
        for priority in self.cleanup_priorities:
            max_age = priority["max_age_days"]
            patterns = priority["patterns"]
            
            for pattern in patterns:
                for file_path in Path(".").rglob(pattern):
                    if file_path.is_file():
                        age_days = self.get_file_age_days(file_path)
                        
                        if age_days > max_age:
                            file_info = {
                                "size_mb": self.get_file_size_mb(file_path),
                                "age_days": age_days,
                                "priority": priority["description"],
                                "pattern": pattern
                            }
                            files_to_cleanup.append((file_path, file_info))
        
        # 크기순 정렬 (큰 파일부터)
        files_to_cleanup.sort(key=lambda x: x[1]["size_mb"], reverse=True)
        return files_to_cleanup
    
    def calculate_total_size(self) -> int:
        """전체 저장소 크기 계산"""
        total_size = 0
        
        for file_path in Path(".").rglob("*"):
            if file_path.is_file():
                total_size += self.get_file_size_mb(file_path)
        
        return total_size
    
    def cleanup_files(self, files_to_cleanup: List[Tuple[Path, Dict]]) -> Dict:
        """파일 정리 실행"""
        cleanup_stats = {
            "files_removed": 0,
            "size_freed_mb": 0,
            "errors": []
        }
        
        current_size = self.calculate_total_size()
        target_size = self.max_size_mb - 100  # 100MB 여유 공간
        
        logger.info(f"Current size: {current_size}MB, Target: {target_size}MB")
        
        for file_path, file_info in files_to_cleanup:
            if current_size <= target_size:
                break
                
            try:
                size_mb = file_info["size_mb"]
                
                # 파일 삭제
                file_path.unlink()
                
                cleanup_stats["files_removed"] += 1
                cleanup_stats["size_freed_mb"] += size_mb
                current_size -= size_mb
                
                logger.info(f"Removed: {file_path.name} ({size_mb}MB, {file_info['age_days']} days old)")
                
            except Exception as e:
                error_msg = f"Failed to remove {file_path}: {e}"
                logger.error(error_msg)
                cleanup_stats["errors"].append(error_msg)
        
        return cleanup_stats
    
    def cleanup_old_backups(self):
        """오래된 백업 디렉토리 정리"""
        for backup_dir in self.backup_dirs:
            backup_path = Path(backup_dir)
            if not backup_path.exists():
                continue
                
            # 백업 디렉토리 내의 하위 디렉토리들
            subdirs = [d for d in backup_path.iterdir() if d.is_dir()]
            
            # 날짜순 정렬
            dated_dirs = []
            for subdir in subdirs:
                try:
                    # 디렉토리명에서 날짜 추출 시도
                    dir_name = subdir.name
                    if "backup_" in dir_name:
                        date_str = dir_name.split("_")[1]
                        date_obj = datetime.strptime(date_str, "%Y%m%d")
                        dated_dirs.append((subdir, date_obj))
                except:
                    continue
            
            # 오래된 디렉토리 삭제 (최근 5개만 유지)
            if len(dated_dirs) > 5:
                dated_dirs.sort(key=lambda x: x[1])
                dirs_to_remove = dated_dirs[:-5]
                
                for dir_path, _ in dirs_to_remove:
                    try:
                        shutil.rmtree(dir_path)
                        logger.info(f"Removed old backup directory: {dir_path}")
                    except Exception as e:
                        logger.error(f"Failed to remove {dir_path}: {e}")
    
    def run_cleanup(self) -> bool:
        """정리 작업 실행"""
        try:
            logger.info("Starting file cleanup process")
            
            # 1. 현재 크기 확인
            current_size = self.calculate_total_size()
            logger.info(f"Current repository size: {current_size}MB")
            
            if current_size <= self.max_size_mb:
                logger.info("Repository size is within limits")
                return True
            
            # 2. 정리할 파일 찾기
            files_to_cleanup = self.find_files_to_cleanup()
            logger.info(f"Found {len(files_to_cleanup)} files to cleanup")
            
            # 3. 파일 정리
            cleanup_stats = self.cleanup_files(files_to_cleanup)
            
            # 4. 백업 디렉토리 정리
            self.cleanup_old_backups()
            
            # 5. 결과 보고
            final_size = self.calculate_total_size()
            logger.info(f"Cleanup completed:")
            logger.info(f"  - Files removed: {cleanup_stats['files_removed']}")
            logger.info(f"  - Size freed: {cleanup_stats['size_freed_mb']}MB")
            logger.info(f"  - Final size: {final_size}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

def main():
    """메인 함수"""
    cleanup_manager = FileCleanupManager()
    success = cleanup_manager.run_cleanup()
    
    if success:
        print("✅ File cleanup completed successfully")
    else:
        print("❌ File cleanup failed")
        exit(1)

if __name__ == "__main__":
    main() 