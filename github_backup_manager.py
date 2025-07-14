#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: github_backup_manager.py
목적: 중복 파일을 GitHub에 백업하고 구조 정리 시 복원
Author: GitHub Backup Manager
Created: 2025-07-13
Version: 1.0.0

Features:
    - 중복 파일 자동 감지
    - GitHub 백업 및 복원
    - 파일 버전 관리
    - 안전한 구조 정리
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
import subprocess
import logging

class GitHubBackupManager:
    """GitHub 백업 관리자"""
    
    def __init__(self, repo_name: str = "auto-trading-backup"):
        self.repo_name = repo_name
        self.backup_dir = Path("backup/github")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/github_backup.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 백업 메타데이터
        self.backup_metadata_file = self.backup_dir / "backup_metadata.json"
        self.backup_metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """백업 메타데이터 로드"""
        if self.backup_metadata_file.exists():
            with open(self.backup_metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "backups": {},
            "duplicates": {},
            "last_backup": None,
            "total_files": 0
        }
    
    def _save_metadata(self) -> None:
        """백업 메타데이터 저장"""
        with open(self.backup_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.backup_metadata, f, indent=2, ensure_ascii=False)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _find_duplicate_files(self) -> Dict[str, List[str]]:
        """중복 파일 찾기"""
        file_hashes = {}
        duplicates = {}
        
        # 모든 파일 스캔
        for file_path in Path(".").rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    file_hash = self._calculate_file_hash(file_path)
                    
                    if file_hash in file_hashes:
                        if file_hash not in duplicates:
                            duplicates[file_hash] = [file_hashes[file_hash]]
                        duplicates[file_hash].append(str(file_path))
                    else:
                        file_hashes[file_hash] = str(file_path)
                        
                except Exception as e:
                    self.logger.warning(f"파일 해시 계산 실패: {file_path} - {e}")
        
        return duplicates
    
    def backup_duplicates_to_github(self) -> Dict[str, Any]:
        """중복 파일을 GitHub에 백업"""
        try:
            self.logger.info("중복 파일 백업 시작")
            
            # 중복 파일 찾기
            duplicates = self._find_duplicate_files()
            
            if not duplicates:
                self.logger.info("중복 파일이 없습니다")
                return {"success": True, "backed_up": 0}
            
            # 백업 디렉토리 생성
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{backup_timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            backed_up_files = []
            
            # 중복 파일 백업
            for file_hash, file_paths in duplicates.items():
                if len(file_paths) > 1:
                    # 첫 번째 파일을 원본으로 사용
                    original_path = Path(file_paths[0])
                    backup_file_path = backup_path / f"{file_hash}_{original_path.name}"
                    
                    try:
                        shutil.copy2(original_path, backup_file_path)
                        backed_up_files.append({
                            "hash": file_hash,
                            "original": str(original_path),
                            "backup": str(backup_file_path),
                            "duplicates": file_paths[1:]
                        })
                        
                        self.logger.info(f"백업 완료: {original_path.name}")
                        
                    except Exception as e:
                        self.logger.error(f"백업 실패: {original_path} - {e}")
            
            # 메타데이터 업데이트
            self.backup_metadata["backups"][backup_timestamp] = {
                "timestamp": datetime.now().isoformat(),
                "files": backed_up_files,
                "total_files": len(backed_up_files)
            }
            self.backup_metadata["duplicates"] = duplicates
            self.backup_metadata["last_backup"] = datetime.now().isoformat()
            self.backup_metadata["total_files"] += len(backed_up_files)
            
            self._save_metadata()
            
            # GitHub에 푸시
            self._push_to_github(backup_timestamp)
            
            self.logger.info(f"백업 완료: {len(backed_up_files)}개 파일")
            
            return {
                "success": True,
                "backed_up": len(backed_up_files),
                "backup_path": str(backup_path),
                "timestamp": backup_timestamp
            }
            
        except Exception as e:
            self.logger.error(f"백업 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def _push_to_github(self, backup_timestamp: str) -> bool:
        """GitHub에 푸시"""
        try:
            # Git 초기화 (이미 있으면 스킵)
            if not (self.backup_dir / ".git").exists():
                subprocess.run(["git", "init"], cwd=self.backup_dir, check=True)
            
            # 원격 저장소 추가 (이미 있으면 스킵)
            try:
                subprocess.run(["git", "remote", "add", "origin", f"https://github.com/your-username/{self.repo_name}.git"], 
                             cwd=self.backup_dir, check=True)
            except subprocess.CalledProcessError:
                # 이미 존재하는 경우 무시
                pass
            
            # 파일 추가
            subprocess.run(["git", "add", "."], cwd=self.backup_dir, check=True)
            
            # 커밋
            commit_message = f"Backup duplicate files - {backup_timestamp}"
            subprocess.run(["git", "commit", "-m", commit_message], cwd=self.backup_dir, check=True)
            
            # 푸시
            subprocess.run(["git", "push", "origin", "main"], cwd=self.backup_dir, check=True)
            
            self.logger.info("GitHub 푸시 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"GitHub 푸시 실패: {e}")
            return False
    
    def restore_from_github(self, backup_timestamp: Optional[str] = None) -> Dict[str, Any]:
        """GitHub에서 복원"""
        try:
            self.logger.info("GitHub에서 복원 시작")
            
            # 최신 백업 가져오기
            subprocess.run(["git", "pull", "origin", "main"], cwd=self.backup_dir, check=True)
            
            # 메타데이터 다시 로드
            self.backup_metadata = self._load_metadata()
            
            if not backup_timestamp:
                # 가장 최근 백업 사용
                backup_timestamp = self.backup_metadata.get("last_backup", "").split("T")[0].replace("-", "")
            
            if backup_timestamp not in self.backup_metadata["backups"]:
                return {"success": False, "error": f"백업을 찾을 수 없음: {backup_timestamp}"}
            
            backup_info = self.backup_metadata["backups"][backup_timestamp]
            restored_files = []
            
            # 파일 복원
            for file_info in backup_info["files"]:
                backup_file_path = Path(file_info["backup"])
                if backup_file_path.exists():
                    # 원본 위치에 복원
                    original_path = Path(file_info["original"])
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(backup_file_path, original_path)
                    restored_files.append(str(original_path))
                    
                    self.logger.info(f"복원 완료: {original_path.name}")
            
            self.logger.info(f"복원 완료: {len(restored_files)}개 파일")
            
            return {
                "success": True,
                "restored": len(restored_files),
                "files": restored_files,
                "timestamp": backup_timestamp
            }
            
        except Exception as e:
            self.logger.error(f"복원 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """백업 목록 조회"""
        backups = []
        for timestamp, info in self.backup_metadata["backups"].items():
            backups.append({
                "timestamp": timestamp,
                "date": info["timestamp"],
                "files_count": info["total_files"],
                "files": [f["original"] for f in info["files"]]
            })
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
    
    def get_duplicate_report(self) -> Dict[str, Any]:
        """중복 파일 리포트"""
        duplicates = self._find_duplicate_files()
        
        report = {
            "total_duplicates": len(duplicates),
            "total_files": sum(len(paths) for paths in duplicates.values()),
            "duplicates": {}
        }
        
        for file_hash, file_paths in duplicates.items():
            if len(file_paths) > 1:
                report["duplicates"][file_hash] = {
                    "count": len(file_paths),
                    "files": file_paths
                }
        
        return report

def main():
    """메인 실행 함수"""
    backup_manager = GitHubBackupManager()
    
    print("🔍 중복 파일 분석 중...")
    duplicate_report = backup_manager.get_duplicate_report()
    
    print(f"📊 중복 파일 리포트:")
    print(f"  - 중복 그룹: {duplicate_report['total_duplicates']}개")
    print(f"  - 총 파일: {duplicate_report['total_files']}개")
    
    if duplicate_report['total_duplicates'] > 0:
        print("\n📦 GitHub 백업 시작...")
        result = backup_manager.backup_duplicates_to_github()
        
        if result["success"]:
            print(f"✅ 백업 완료: {result['backed_up']}개 파일")
            print(f"📁 백업 위치: {result['backup_path']}")
        else:
            print(f"❌ 백업 실패: {result['error']}")
    
    print("\n📋 백업 목록:")
    backups = backup_manager.list_backups()
    for backup in backups[:5]:  # 최근 5개만 표시
        print(f"  📅 {backup['date']}: {backup['files_count']}개 파일")

if __name__ == "__main__":
    main() 