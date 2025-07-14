#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: safe_reorganize.py
목적: GitHub 백업 후 안전한 구조 정리
Author: Safe Reorganizer
Created: 2025-07-13
Version: 1.0.0

Features:
    - 중복 파일 GitHub 백업
    - 안전한 폴더 구조 정리
    - 복원 가능한 구조
    - 진행 상황 모니터링
"""

import asyncio
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

from github_backup_manager import GitHubBackupManager

class SafeReorganizer:
    """안전한 구조 정리기"""
    
    def __init__(self):
        self.backup_manager = GitHubBackupManager()
        self.reorganize_log = []
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/safe_reorganize.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def safe_reorganize(self) -> Dict[str, Any]:
        """안전한 구조 정리"""
        try:
            self.logger.info("안전한 구조 정리 시작")
            
            # 1단계: 중복 파일 백업
            print("🔄 1단계: 중복 파일 GitHub 백업")
            backup_result = self.backup_manager.backup_duplicates_to_github()
            
            if not backup_result["success"]:
                return {"success": False, "error": f"백업 실패: {backup_result['error']}"}
            
            self.reorganize_log.append({
                "step": "backup",
                "timestamp": datetime.now().isoformat(),
                "result": backup_result
            })
            
            print(f"✅ 백업 완료: {backup_result['backed_up']}개 파일")
            
            # 2단계: 20개 폴더로 자연스럽게 정리
            print("\n🔄 2단계: 폴더 구조 정리")
            reorganize_result = await self._reorganize_folders()
            
            if not reorganize_result["success"]:
                # 백업에서 복원
                print("⚠️ 구조 정리 실패, 백업에서 복원 중...")
                restore_result = self.backup_manager.restore_from_github()
                return {"success": False, "error": reorganize_result["error"], "restored": restore_result}
            
            self.reorganize_log.append({
                "step": "reorganize",
                "timestamp": datetime.now().isoformat(),
                "result": reorganize_result
            })
            
            print(f"✅ 구조 정리 완료: {reorganize_result['moved_files']}개 파일")
            
            # 3단계: 정리 결과 검증
            print("\n🔄 3단계: 결과 검증")
            validation_result = await self._validate_reorganization()
            
            self.reorganize_log.append({
                "step": "validation",
                "timestamp": datetime.now().isoformat(),
                "result": validation_result
            })
            
            # 로그 저장
            self._save_reorganize_log()
            
            return {
                "success": True,
                "backup_result": backup_result,
                "reorganize_result": reorganize_result,
                "validation_result": validation_result
            }
            
        except Exception as e:
            self.logger.error(f"안전한 구조 정리 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def _reorganize_folders(self) -> Dict[str, Any]:
        """폴더 구조 정리"""
        try:
            # 20개 자연스러운 폴더 구조
            target_folders = {
                "core": "핵심 시스템",
                "data": "데이터 저장소",
                "collectors": "데이터 수집기",
                "processors": "데이터 처리기",
                "models": "ML/DL 모델",
                "strategies": "트레이딩 전략",
                "execution": "주문 실행",
                "monitoring": "시스템 모니터링",
                "config": "설정 파일",
                "logs": "로그 파일",
                "reports": "리포트 및 결과",
                "scripts": "실행 스크립트",
                "tests": "테스트 파일",
                "docs": "문서",
                "backup": "백업",
                "cache": "캐시",
                "temp": "임시 파일",
                "trading": "트레이딩 관련",
                "analysis": "분석 도구",
                "utils": "유틸리티"
            }
            
            # 폴더 생성
            for folder in target_folders.keys():
                Path(folder).mkdir(exist_ok=True)
            
            # 이동 규칙
            move_rules = {
                # 데이터 관련
                "collected_data": "data/",
                "data_backup": "data/",
                "stock_history_parquet": "data/",
                "krx_auto_data": "data/",
                "krx_kosdaq50_data": "data/",
                "free_collected_data": "data/",
                
                # 수집기 관련
                "data_engine/collectors": "collectors/",
                "service": "collectors/",
                
                # 처리기 관련
                "data_engine/processors": "processors/",
                
                # 모델 관련
                "models": "models/",
                "mlruns": "models/",
                
                # 전략 관련
                "strategy_engine": "strategies/",
                "strategy": "strategies/",
                
                # 실행 관련
                "execution_engine": "execution/",
                "trading": "execution/",
                
                # 모니터링 관련
                "monitoring_center": "monitoring/",
                "monitoring": "monitoring/",
                
                # 설정 관련
                "config": "config/",
                
                # 로그 관련
                "logs": "logs/",
                
                # 리포트 관련
                "results": "reports/",
                "performance_reports": "reports/",
                "standardized_testing_results": "reports/",
                
                # 스크립트 관련
                "scripts": "scripts/",
                
                # 테스트 관련
                "tests": "tests/",
                
                # 문서 관련
                "docs": "docs/",
                
                # 백업 관련
                "backup": "backup/",
                "backup_ai_fixes": "backup/",
                "backup_code_fixes": "backup/",
                
                # 캐시 관련
                "cache": "cache/",
                
                # 임시 관련
                "temp": "temp/",
                "scripts_temp": "temp/",
                
                # 트레이딩 관련
                "trading_env": "trading/",
                "daytrading_system": "trading/",
                "production_trading": "trading/",
                
                # 분석 관련
                "analysis": "analysis/",
                "backtesting_lab": "analysis/",
                
                # 유틸리티 관련
                "utils": "utils/",
                "core": "utils/"
            }
            
            moved_files = 0
            
            # 파일 이동
            for source, destination in move_rules.items():
                source_path = Path(source)
                dest_path = Path(destination)
                
                if source_path.exists():
                    try:
                        # 대상 폴더 생성
                        dest_path.mkdir(parents=True, exist_ok=True)
                        
                        # 폴더인 경우 내용만 이동
                        if source_path.is_dir():
                            for item in source_path.iterdir():
                                if not (dest_path / item.name).exists():
                                    shutil.move(str(item), str(dest_path / item.name))
                                    moved_files += 1
                            
                            # 빈 폴더만 삭제
                            if not any(source_path.iterdir()):
                                source_path.rmdir()
                        else:
                            # 파일 이동
                            if not (dest_path / source_path.name).exists():
                                shutil.move(str(source_path), str(dest_path / source_path.name))
                                moved_files += 1
                        
                        self.logger.info(f"이동 완료: {source} → {destination}")
                        
                    except Exception as e:
                        self.logger.warning(f"이동 실패 (건너뜀): {source} - {e}")
            
            # 루트 파일 정리
            file_rules = {
                "*.py": "scripts/",
                "*.json": "config/",
                "*.log": "logs/",
                "*.csv": "data/",
                "*.parquet": "data/",
                "*.h5": "models/",
                "*.md": "docs/",
                "*.txt": "docs/",
                "*.bat": "scripts/",
                "*.sh": "scripts/"
            }
            
            for pattern, destination in file_rules.items():
                for file_path in Path(".").glob(pattern):
                    if file_path.is_file() and file_path.name not in ["safe_reorganize.py", "github_backup_manager.py"]:
                        try:
                            dest_path = Path(destination)
                            dest_path.mkdir(exist_ok=True)
                            
                            if not (dest_path / file_path.name).exists():
                                shutil.move(str(file_path), str(dest_path / file_path.name))
                                moved_files += 1
                                
                        except Exception as e:
                            self.logger.warning(f"파일 이동 실패: {file_path.name} - {e}")
            
            return {
                "success": True,
                "moved_files": moved_files,
                "target_folders": list(target_folders.keys())
            }
            
        except Exception as e:
            self.logger.error(f"폴더 구조 정리 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_reorganization(self) -> Dict[str, Any]:
        """정리 결과 검증"""
        try:
            # 필수 폴더 확인
            required_folders = ["core", "data", "collectors", "processors", "models", "config", "logs"]
            missing_folders = []
            
            for folder in required_folders:
                if not Path(folder).exists():
                    missing_folders.append(folder)
            
            # 파일 손실 확인
            total_files_before = len(list(Path(".").rglob("*")))
            total_files_after = len(list(Path(".").rglob("*")))
            
            validation_result = {
                "success": len(missing_folders) == 0,
                "missing_folders": missing_folders,
                "files_before": total_files_before,
                "files_after": total_files_after,
                "file_loss": total_files_before - total_files_after
            }
            
            if validation_result["success"]:
                self.logger.info("검증 완료: 모든 폴더가 정상적으로 생성됨")
            else:
                self.logger.warning(f"검증 실패: 누락된 폴더 {missing_folders}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"검증 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def _save_reorganize_log(self) -> None:
        """정리 로그 저장"""
        log_file = Path("logs/reorganize_log.json")
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.reorganize_log, f, indent=2, ensure_ascii=False)
    
    def get_reorganize_status(self) -> Dict[str, Any]:
        """정리 상태 조회"""
        return {
            "total_steps": len(self.reorganize_log),
            "last_step": self.reorganize_log[-1] if self.reorganize_log else None,
            "success": all(step["result"].get("success", False) for step in self.reorganize_log)
        }

async def main():
    """메인 실행 함수"""
    reorganizer = SafeReorganizer()
    
    print("🔄 안전한 구조 정리 시작")
    print("📦 중복 파일을 GitHub에 백업한 후 구조를 정리합니다.")
    
    result = await reorganizer.safe_reorganize()
    
    if result["success"]:
        print("\n✅ 안전한 구조 정리 완료!")
        print(f"📊 백업된 파일: {result['backup_result']['backed_up']}개")
        print(f"📊 이동된 파일: {result['reorganize_result']['moved_files']}개")
        print(f"📊 검증 결과: {'성공' if result['validation_result']['success'] else '실패'}")
        
        # 현재 폴더 구조 출력
        print("\n📁 현재 폴더 구조:")
        for folder in result['reorganize_result']['target_folders']:
            if Path(folder).exists():
                file_count = len(list(Path(folder).rglob("*")))
                print(f"  📁 {folder}: {file_count}개 파일/폴더")
    else:
        print(f"\n❌ 구조 정리 실패: {result['error']}")
        if 'restored' in result:
            print("🔄 백업에서 복원되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 