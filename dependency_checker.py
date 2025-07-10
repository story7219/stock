#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
의존성 모니터링 스크립트
한투 API 연동 시스템의 필수 라이브러리 상태를 확인합니다.
"""

import importlib
import sys
import subprocess
import pkg_resources
from typing import Dict, List, Tuple
from datetime import datetime
import json

class DependencyChecker:
    """의존성 상태 확인 클래스"""
    
    def __init__(self):
        # 핵심 의존성 패키지들
        self.core_packages = [
            'tensorflow', 'torch', 'pandas', 'numpy',
            'aiohttp', 'pykis', 'ta', 'fastapi'
        ]
        
        # 선택적 의존성 패키지들
        self.optional_packages = [
            'konlpy', 'ta-lib', 'sklearn', 'scipy',
            'matplotlib', 'seaborn', 'plotly'
        ]
        
        # 문제가 있는 패키지들
        self.problematic_packages = [
            'konlpy', 'ta-lib'
        ]
    
    def check_package_installation(self, package_name: str) -> Tuple[bool, str]:
        """패키지 설치 상태 확인"""
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            return True, version
        except ImportError:
            return False, 'not installed'
        except Exception as e:
            return False, f'error: {str(e)}'
    
    def check_package_versions(self) -> Dict[str, str]:
        """설치된 패키지들의 버전 정보 수집"""
        versions = {}
        for package in pkg_resources.working_set:
            versions[package.project_name] = package.version
        return versions
    
    def check_system_info(self) -> Dict[str, str]:
        """시스템 정보 수집"""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'architecture': sys.maxsize > 2**32 and '64bit' or '32bit'
        }
    
    def check_gpu_support(self) -> Dict[str, bool]:
        """GPU 지원 상태 확인"""
        gpu_info = {
            'tensorflow_gpu': False,
            'pytorch_gpu': False,
            'cuda_available': False
        }
        
        # TensorFlow GPU 확인
        try:
            import tensorflow as tf
            gpu_devices = tf.config.list_physical_devices('GPU')
            gpu_info['tensorflow_gpu'] = len(gpu_devices) > 0
        except:
            pass
        
        # PyTorch GPU 확인
        try:
            import torch
            gpu_info['pytorch_gpu'] = torch.cuda.is_available()
        except:
            pass
        
        # CUDA 확인
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            gpu_info['cuda_available'] = result.returncode == 0
        except:
            pass
        
        return gpu_info
    
    def check_package_compatibility(self) -> Dict[str, List[str]]:
        """패키지 호환성 확인"""
        compatibility_issues = {
            'version_conflicts': [],
            'missing_dependencies': [],
            'deprecated_packages': []
        }
        
        # 버전 충돌 확인
        versions = self.check_package_versions()
        
        # TensorFlow와 PyTorch 버전 확인
        if 'tensorflow' in versions and 'torch' in versions:
            tf_version = versions['tensorflow']
            torch_version = versions['torch']
            
            # TensorFlow 2.x와 PyTorch 호환성
            if tf_version.startswith('2.') and torch_version.startswith('2.'):
                pass  # 호환됨
            else:
                compatibility_issues['version_conflicts'].append(
                    f"TensorFlow {tf_version} and PyTorch {torch_version} may have compatibility issues"
                )
        
        # 누락된 의존성 확인
        for package in self.core_packages:
            installed, version = self.check_package_installation(package)
            if not installed:
                compatibility_issues['missing_dependencies'].append(package)
        
        return compatibility_issues
    
    def generate_detailed_report(self) -> Dict:
        """상세한 의존성 보고서 생성"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.check_system_info(),
            'gpu_support': self.check_gpu_support(),
            'package_status': {},
            'compatibility_issues': self.check_package_compatibility(),
            'recommendations': []
        }
        
        # 핵심 패키지 상태 확인
        for package in self.core_packages:
            installed, version = self.check_package_installation(package)
            report['package_status'][package] = {
                'installed': installed,
                'version': version,
                'critical': True
            }
        
        # 선택적 패키지 상태 확인
        for package in self.optional_packages:
            installed, version = self.check_package_installation(package)
            report['package_status'][package] = {
                'installed': installed,
                'version': version,
                'critical': False
            }
        
        # 권장사항 생성
        self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict):
        """권장사항 생성"""
        recommendations = []
        
        # GPU 지원 권장사항
        gpu_support = report['gpu_support']
        if not gpu_support['tensorflow_gpu'] and not gpu_support['pytorch_gpu']:
            recommendations.append("GPU 지원을 활성화하여 성능을 향상시킬 수 있습니다.")
        
        # 누락된 핵심 패키지
        missing_critical = [
            package for package, status in report['package_status'].items()
            if status['critical'] and not status['installed']
        ]
        if missing_critical:
            recommendations.append(f"핵심 패키지 설치 필요: {', '.join(missing_critical)}")
        
        # 문제가 있는 패키지 대체
        for package in self.problematic_packages:
            if package in report['package_status'] and not report['package_status'][package]['installed']:
                if package == 'konlpy':
                    recommendations.append("KoNLPy 대신 textblob 또는 vaderSentiment 사용 권장")
                elif package == 'ta-lib':
                    recommendations.append("TA-Lib 대신 pandas_ta 사용 권장")
        
        report['recommendations'] = recommendations
    
    def print_summary_report(self):
        """요약 보고서 출력"""
        report = self.generate_detailed_report()
        
        print("=" * 60)
        print("의존성 상태 보고서")
        print("=" * 60)
        print(f"생성 시간: {report['timestamp']}")
        print(f"Python 버전: {report['system_info']['python_version'].split()[0]}")
        print(f"플랫폼: {report['system_info']['platform']}")
        print()
        
        # GPU 지원 상태
        gpu_support = report['gpu_support']
        print("GPU 지원 상태:")
        print(f"  TensorFlow GPU: {'✅' if gpu_support['tensorflow_gpu'] else '❌'}")
        print(f"  PyTorch GPU: {'✅' if gpu_support['pytorch_gpu'] else '❌'}")
        print(f"  CUDA: {'✅' if gpu_support['cuda_available'] else '❌'}")
        print()
        
        # 패키지 상태
        print("패키지 상태:")
        for package, status in report['package_status'].items():
            icon = "✅" if status['installed'] else "❌"
            critical = " (핵심)" if status['critical'] else ""
            print(f"  {package}: {icon} {status['version']}{critical}")
        print()
        
        # 호환성 문제
        issues = report['compatibility_issues']
        if issues['version_conflicts'] or issues['missing_dependencies']:
            print("발견된 문제:")
            for issue in issues['version_conflicts']:
                print(f"  ⚠️  {issue}")
            for package in issues['missing_dependencies']:
                print(f"  ❌ 누락된 패키지: {package}")
            print()
        
        # 권장사항
        if report['recommendations']:
            print("권장사항:")
            for rec in report['recommendations']:
                print(f"  💡 {rec}")
            print()
        
        print("=" * 60)
    
    def save_report(self, filename: str = "dependency_report.json"):
        """보고서를 JSON 파일로 저장"""
        report = self.generate_detailed_report()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"보고서가 {filename}에 저장되었습니다.")

def main():
    """메인 함수"""
    checker = DependencyChecker()
    checker.print_summary_report()
    checker.save_report()

if __name__ == "__main__":
    main() 