#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
네이버 클라우드 업로드 스크립트
목적: 자동화된 네이버 클라우드 백업 (30GB 무료)

Author: Auto Trading System
Created: 2025-01-13
Version: 1.0.0

Features:
- 네이버 클라우드 API 연동
- 자동 백업 업로드
- 증분 백업 지원
- 업로드 상태 모니터링
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile
import tempfile
import hashlib

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NaverCloudUploader:
    """네이버 클라우드 업로더"""
    
    def __init__(self):
        self.api_url = "https://api.cloud.naver.com"
        self.access_key = os.getenv("NAVER_CLOUD_ACCESS_KEY")
        self.secret_key = os.getenv("NAVER_CLOUD_SECRET_KEY")
        self.bucket_name = os.getenv("NAVER_CLOUD_BUCKET", "auto-trading-backup")
        
        # 업로드할 디렉토리들
        self.upload_dirs = [
            "src",
            "scripts",
            "config",
            "tests",
            "docs",
            "data_engine",
            "strategy",
            "monitoring"
        ]
        
        # 제외할 패턴들
        self.exclude_patterns = [
            "__pycache__",
            "*.pyc",
            "*.log",
            "*.tmp",
            ".env",
            "secrets",
            "api_keys",
            "*.bak"
        ]
        
        if not self.access_key or not self.secret_key:
            logger.warning("Naver Cloud credentials not found. Skipping upload.")
            self.enabled = False
        else:
            self.enabled = True
    
    def create_headers(self) -> Dict[str, str]:
        """API 헤더 생성"""
        return {
            "Authorization": f"Bearer {self.access_key}",
            "Content-Type": "application/json",
            "X-NCP-APIGW-API-KEY": self.secret_key
        }
    
    def should_exclude(self, file_path: Path) -> bool:
        """파일 제외 여부 확인"""
        for pattern in self.exclude_patterns:
            if pattern in str(file_path):
                return True
        return False
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def create_backup_archive(self) -> Tuple[Path, Dict]:
        """백업 아카이브 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"auto_trading_backup_{timestamp}.zip"
        archive_path = Path("backup") / archive_name
        
        # 백업 디렉토리 생성
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        backup_stats = {
            "timestamp": timestamp,
            "files_included": 0,
            "total_size_mb": 0,
            "file_hashes": {}
        }
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for source_dir in self.upload_dirs:
                source_path = Path(source_dir)
                if not source_path.exists():
                    continue
                
                for file_path in source_path.rglob("*"):
                    if file_path.is_file() and not self.should_exclude(file_path):
                        try:
                            # 상대 경로로 아카이브에 추가
                            arcname = file_path.relative_to(Path.cwd())
                            zipf.write(file_path, arcname)
                            
                            # 통계 업데이트
                            backup_stats["files_included"] += 1
                            backup_stats["total_size_mb"] += file_path.stat().st_size // (1024 * 1024)
                            backup_stats["file_hashes"][str(arcname)] = self.calculate_file_hash(file_path)
                            
                        except Exception as e:
                            logger.error(f"Failed to add {file_path} to archive: {e}")
        
        # 메타데이터 저장
        metadata_path = archive_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(backup_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created backup archive: {archive_path} ({backup_stats['files_included']} files, {backup_stats['total_size_mb']}MB)")
        
        return archive_path, backup_stats
    
    def upload_to_naver_cloud(self, file_path: Path) -> bool:
        """네이버 클라우드에 업로드"""
        if not self.enabled:
            logger.warning("Naver Cloud upload disabled (credentials not found)")
            return False
        
        try:
            # 파일 업로드 API 호출
            upload_url = f"{self.api_url}/object-storage/v2/upload"
            
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/zip')}
                data = {
                    'bucketName': self.bucket_name,
                    'objectName': f"backups/{file_path.name}"
                }
                
                response = requests.post(
                    upload_url,
                    headers=self.create_headers(),
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully uploaded {file_path.name} to Naver Cloud")
                    return True
                else:
                    logger.error(f"Upload failed: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False
    
    def cleanup_local_archive(self, archive_path: Path):
        """로컬 아카이브 정리"""
        try:
            # 업로드 완료 후 로컬 파일 삭제
            archive_path.unlink()
            metadata_path = archive_path.with_suffix('.json')
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Cleaned up local archive: {archive_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup local archive: {e}")
    
    def check_naver_cloud_space(self) -> Optional[int]:
        """네이버 클라우드 사용량 확인"""
        if not self.enabled:
            return None
        
        try:
            # 사용량 확인 API 호출
            usage_url = f"{self.api_url}/object-storage/v2/usage"
            response = requests.get(usage_url, headers=self.create_headers())
            
            if response.status_code == 200:
                usage_data = response.json()
                used_gb = usage_data.get('usedSize', 0) / (1024 * 1024 * 1024)
                logger.info(f"Naver Cloud usage: {used_gb:.2f}GB")
                return int(used_gb)
            else:
                logger.warning(f"Failed to get usage info: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Usage check error: {e}")
            return None
    
    def run_upload(self) -> bool:
        """업로드 실행"""
        try:
            logger.info("Starting Naver Cloud upload process")
            
            # 1. 사용량 확인
            current_usage = self.check_naver_cloud_space()
            if current_usage and current_usage > 25:  # 25GB 이상 사용 시 경고
                logger.warning(f"Naver Cloud usage is high: {current_usage}GB")
            
            # 2. 백업 아카이브 생성
            archive_path, backup_stats = self.create_backup_archive()
            
            # 3. 네이버 클라우드 업로드
            upload_success = self.upload_to_naver_cloud(archive_path)
            
            if upload_success:
                # 4. 로컬 아카이브 정리
                self.cleanup_local_archive(archive_path)
                logger.info("Naver Cloud upload completed successfully")
                return True
            else:
                logger.error("Naver Cloud upload failed")
                return False
                
        except Exception as e:
            logger.error(f"Upload process failed: {e}")
            return False

def main():
    """메인 함수"""
    uploader = NaverCloudUploader()
    success = uploader.run_upload()
    
    if success:
        print("✅ Naver Cloud upload completed successfully")
    else:
        print("❌ Naver Cloud upload failed")
        exit(1)

if __name__ == "__main__":
    main() 