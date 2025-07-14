#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: smart_duplicate_cleaner.py
목적: 중복 파일을 똑똑한 순서로 2등까지 남기고 나머지 삭제 (모듈화 파일 예외)
Author: Smart Duplicate Cleaner
Created: 2025-07-13
Version: 1.0.0

Features:
    - 중복 파일 지능 순서 평가
    - 모듈화 파일 예외 처리
    - 상위 2개 파일 보존
    - 안전한 삭제 로직
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple
import logging
import re

class SmartDuplicateCleaner:
    """스마트 중복 파일 정리기"""
    
    def __init__(self):
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/smart_cleaner.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 모듈화 파일 예외 패턴
        self.modular_exceptions = {
            # 핵심 모듈 파일들
            "core/", "domain/", "infrastructure/", "application/",
            "data_engine/", "strategy_engine/", "execution_engine/",
            "monitoring/", "service/", "utils/",
            
            # 설정 파일들
            "config/", "settings.py", "requirements.txt", "setup.py",
            
            # 테스트 파일들
            "tests/", "test_", "_test.py", "test_",
            
            # 문서 파일들
            "docs/", "README", "LICENSE", "CHANGELOG",
            
            # 스크립트 파일들
            "scripts/", "run_", "main.py", "app.py",
            
            # 특정 파일 패턴
            "__init__.py", "models.py", "views.py", "controllers.py",
            "services.py", "repositories.py", "entities.py",
            
            # 데이터 관련
            "collectors/", "processors/", "models/", "strategies/",
            
            # 실행 관련
            "execution/", "trading/", "monitoring/",
        }
        
        # 지능 순서 평가 기준
        self.intelligence_criteria = {
            "file_size": 0.1,      # 파일 크기 (작을수록 좋음)
            "line_count": 0.15,     # 코드 라인 수 (적당할수록 좋음)
            "comment_ratio": 0.2,   # 주석 비율 (높을수록 좋음)
            "function_count": 0.15, # 함수 수 (적당할수록 좋음)
            "class_count": 0.1,     # 클래스 수 (적당할수록 좋음)
            "import_count": 0.1,    # import 수 (적당할수록 좋음)
            "error_handling": 0.2,  # 에러 처리 (있을수록 좋음)
        }
    
    def is_modular_file(self, file_path: Path) -> bool:
        """모듈화 파일인지 확인"""
        file_str = str(file_path)
        
        for pattern in self.modular_exceptions:
            if pattern in file_str:
                return True
        
        # 특정 파일 확장자 예외
        if file_path.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.h']:
            return True
        
        return False
    
    def calculate_intelligence_score(self, file_path: Path) -> float:
        """파일의 지능 점수 계산"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            line_count = len(lines)
            
            # 기본 점수
            score = 0.0
            
            # 1. 파일 크기 점수 (작을수록 좋음)
            file_size = len(content)
            if file_size < 1000:
                score += self.intelligence_criteria["file_size"] * 1.0
            elif file_size < 5000:
                score += self.intelligence_criteria["file_size"] * 0.8
            elif file_size < 10000:
                score += self.intelligence_criteria["file_size"] * 0.6
            else:
                score += self.intelligence_criteria["file_size"] * 0.3
            
            # 2. 라인 수 점수 (적당할수록 좋음)
            if 10 <= line_count <= 500:
                score += self.intelligence_criteria["line_count"] * 1.0
            elif 5 <= line_count < 10 or 500 < line_count <= 1000:
                score += self.intelligence_criteria["line_count"] * 0.7
            else:
                score += self.intelligence_criteria["line_count"] * 0.3
            
            # 3. 주석 비율 점수 (높을수록 좋음)
            comment_lines = sum(1 for line in lines if line.strip().startswith('#') or '"""' in line or "'''" in line)
            comment_ratio = comment_lines / max(line_count, 1)
            score += self.intelligence_criteria["comment_ratio"] * min(comment_ratio * 10, 1.0)
            
            # 4. 함수 수 점수 (적당할수록 좋음)
            function_count = len(re.findall(r'def\s+\w+', content))
            if 1 <= function_count <= 20:
                score += self.intelligence_criteria["function_count"] * 1.0
            elif function_count > 20:
                score += self.intelligence_criteria["function_count"] * 0.5
            else:
                score += self.intelligence_criteria["function_count"] * 0.3
            
            # 5. 클래스 수 점수 (적당할수록 좋음)
            class_count = len(re.findall(r'class\s+\w+', content))
            if 0 <= class_count <= 5:
                score += self.intelligence_criteria["class_count"] * 1.0
            elif class_count > 5:
                score += self.intelligence_criteria["class_count"] * 0.5
            
            # 6. import 수 점수 (적당할수록 좋음)
            import_count = len(re.findall(r'^import\s+|^from\s+', content, re.MULTILINE))
            if 0 <= import_count <= 10:
                score += self.intelligence_criteria["import_count"] * 1.0
            elif import_count > 10:
                score += self.intelligence_criteria["import_count"] * 0.5
            
            # 7. 에러 처리 점수 (있을수록 좋음)
            error_patterns = ['try:', 'except:', 'finally:', 'raise', 'assert']
            error_count = sum(content.count(pattern) for pattern in error_patterns)
            if error_count > 0:
                score += self.intelligence_criteria["error_handling"] * min(error_count / 10, 1.0)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"지능 점수 계산 실패: {file_path} - {e}")
            return 0.0
    
    def find_duplicates_with_intelligence(self) -> Dict[str, List[Tuple[str, float]]]:
        """지능 점수를 포함한 중복 파일 찾기"""
        file_hashes = {}
        duplicates = {}
        
        # 모든 파일 스캔
        for file_path in Path(".").rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    file_hash = self._calculate_file_hash(file_path)
                    
                    if file_hash in file_hashes:
                        if file_hash not in duplicates:
                            # 첫 번째 파일도 추가
                            first_file = file_hashes[file_hash]
                            first_score = self.calculate_intelligence_score(Path(first_file))
                            duplicates[file_hash] = [(first_file, first_score)]
                        
                        # 현재 파일 추가
                        current_score = self.calculate_intelligence_score(file_path)
                        duplicates[file_hash].append((str(file_path), current_score))
                    else:
                        file_hashes[file_hash] = str(file_path)
                        
                except Exception as e:
                    self.logger.warning(f"중복 파일 검사 실패: {file_path} - {e}")
        
        return duplicates
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def smart_clean_duplicates(self) -> Dict[str, Any]:
        """스마트 중복 파일 정리"""
        try:
            self.logger.info("스마트 중복 파일 정리 시작")
            
            # 중복 파일 찾기
            duplicates = self.find_duplicates_with_intelligence()
            
            if not duplicates:
                self.logger.info("중복 파일이 없습니다")
                return {"success": True, "cleaned": 0, "preserved": 0}
            
            cleaned_files = []
            preserved_files = []
            modular_exceptions = []
            
            # 각 중복 그룹 처리
            for file_hash, file_list in duplicates.items():
                if len(file_list) > 1:
                    # 지능 점수로 정렬 (높은 순서)
                    sorted_files = sorted(file_list, key=lambda x: x[1], reverse=True)
                    
                    # 모듈화 파일 확인
                    modular_files = []
                    regular_files = []
                    
                    for file_path, score in sorted_files:
                        if self.is_modular_file(Path(file_path)):
                            modular_files.append((file_path, score))
                        else:
                            regular_files.append((file_path, score))
                    
                    # 모듈화 파일은 모두 보존
                    for file_path, score in modular_files:
                        preserved_files.append({
                            "file": file_path,
                            "score": score,
                            "reason": "modular_file"
                        })
                        modular_exceptions.append(file_path)
                    
                    # 일반 파일은 상위 2개만 보존
                    for i, (file_path, score) in enumerate(regular_files):
                        if i < 2:  # 상위 2개 보존
                            preserved_files.append({
                                "file": file_path,
                                "score": score,
                                "reason": f"top_{i+1}_intelligent"
                            })
                        else:  # 나머지 삭제
                            try:
                                os.remove(file_path)
                                cleaned_files.append({
                                    "file": file_path,
                                    "score": score,
                                    "reason": "duplicate_removed"
                                })
                                self.logger.info(f"삭제 완료: {file_path} (점수: {score:.3f})")
                            except Exception as e:
                                self.logger.error(f"삭제 실패: {file_path} - {e}")
            
            # 결과 요약
            result = {
                "success": True,
                "cleaned": len(cleaned_files),
                "preserved": len(preserved_files),
                "modular_exceptions": len(modular_exceptions),
                "cleaned_files": cleaned_files,
                "preserved_files": preserved_files,
                "modular_files": modular_exceptions
            }
            
            self.logger.info(f"정리 완료: {len(cleaned_files)}개 삭제, {len(preserved_files)}개 보존")
            
            return result
            
        except Exception as e:
            self.logger.error(f"스마트 정리 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """정리 리포트 생성"""
        duplicates = self.find_duplicates_with_intelligence()
        
        report = {
            "total_duplicate_groups": len(duplicates),
            "total_duplicate_files": sum(len(files) for files in duplicates.values()),
            "modular_files_count": 0,
            "regular_files_count": 0,
            "duplicate_groups": {}
        }
        
        for file_hash, file_list in duplicates.items():
            if len(file_list) > 1:
                modular_count = sum(1 for file_path, _ in file_list if self.is_modular_file(Path(file_path)))
                regular_count = len(file_list) - modular_count
                
                report["modular_files_count"] += modular_count
                report["regular_files_count"] += regular_count
                
                report["duplicate_groups"][file_hash] = {
                    "total_files": len(file_list),
                    "modular_files": modular_count,
                    "regular_files": regular_count,
                    "files": [(path, score) for path, score in file_list]
                }
        
        return report

def main():
    """메인 실행 함수"""
    cleaner = SmartDuplicateCleaner()
    
    print("🔍 중복 파일 분석 중...")
    report = cleaner.get_cleaning_report()
    
    print(f"📊 중복 파일 리포트:")
    print(f"  - 중복 그룹: {report['total_duplicate_groups']}개")
    print(f"  - 총 중복 파일: {report['total_duplicate_files']}개")
    print(f"  - 모듈화 파일: {report['modular_files_count']}개")
    print(f"  - 일반 파일: {report['regular_files_count']}개")
    
    if report['total_duplicate_groups'] > 0:
        print("\n🧠 스마트 정리 시작...")
        print("📋 정리 규칙:")
        print("  ✅ 모듈화 파일: 모두 보존")
        print("  ✅ 일반 파일: 지능 점수 상위 2개만 보존")
        print("  ❌ 나머지: 삭제")
        
        result = cleaner.smart_clean_duplicates()
        
        if result["success"]:
            print(f"\n✅ 스마트 정리 완료!")
            print(f"  🗑️ 삭제된 파일: {result['cleaned']}개")
            print(f"  💾 보존된 파일: {result['preserved']}개")
            print(f"  🛡️ 모듈화 예외: {result['modular_exceptions']}개")
            
            # 상위 보존 파일들 표시
            if result['preserved_files']:
                print(f"\n🏆 보존된 파일들:")
                for i, file_info in enumerate(result['preserved_files'][:10]):  # 상위 10개만
                    print(f"  {i+1}. {file_info['file']} (점수: {file_info['score']:.3f})")
        else:
            print(f"❌ 정리 실패: {result['error']}")
    else:
        print("🎉 중복 파일이 없습니다!")

if __name__ == "__main__":
    main() 