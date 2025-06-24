#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 마스터 런처 v2.0 - 통합 프로젝트 관리 시스템
모든 기능을 하나의 인터페이스에서 관리할 수 있는 마스터 런처

Features:
- 🎯 투자 분석 시스템 실행
- 📁 파일 관리 시스템
- 🔧 시스템 점검 및 최적화
- 📊 프로젝트 상태 모니터링
- 🗂️ 백업 및 복원
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
import time

class MasterLauncher:
    """🚀 마스터 런처 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.logo = """
        ╔══════════════════════════════════════════╗
        ║     🚀 마스터 런처 v2.0                    ║
        ║     투자 분석 & 파일 관리 통합 시스템         ║
        ╚══════════════════════════════════════════╝
        """
    
    def show_status(self):
        """📊 현재 프로젝트 상태 표시"""
        print("\n📊 프로젝트 상태 정보:")
        print("-" * 50)
        
        # 핵심 파일들 체크
        core_files = [
            "core/app.py",
            "core/launcher.py", 
            "core/run_analysis.py",
            "file_manager.py",
            "requirements.txt"
        ]
        
        print("📁 핵심 파일 상태:")
        for file_path in core_files:
            full_path = self.project_root / file_path
            status = "✅" if full_path.exists() else "❌"
            print(f"  {status} {file_path}")
        
        # 디렉토리 구조 체크
        directories = [
            "core", "modules", "config", "docs", 
            "scripts", "logs", "reports", "tests"
        ]
        
        print("\n📂 디렉토리 구조:")
        for directory in directories:
            full_path = self.project_root / directory
            status = "✅" if full_path.exists() else "❌"
            file_count = len(list(full_path.glob("*"))) if full_path.exists() else 0
            print(f"  {status} {directory}/ ({file_count}개 파일)")
    
    def run_investment_analysis(self):
        """🎯 투자 분석 시스템 실행"""
        print("\n🎯 투자 분석 시스템을 시작합니다...")
        
        # 런처 실행
        launcher_path = self.project_root / "core" / "launcher.py"
        if launcher_path.exists():
            try:
                subprocess.run([sys.executable, str(launcher_path)], cwd=self.project_root)
            except Exception as e:
                print(f"❌ 런처 실행 실패: {e}")
        else:
            print("❌ 런처 파일을 찾을 수 없습니다.")
    
    def run_file_manager(self):
        """📁 파일 관리 시스템 실행"""
        print("\n📁 파일 관리 시스템을 시작합니다...")
        
        file_manager_path = self.project_root / "file_manager.py"
        if file_manager_path.exists():
            try:
                subprocess.run([sys.executable, str(file_manager_path)], cwd=self.project_root)
            except Exception as e:
                print(f"❌ 파일 관리자 실행 실패: {e}")
        else:
            print("❌ 파일 관리자를 찾을 수 없습니다.")
    
    def check_system(self):
        """🔧 시스템 점검 실행"""
        print("\n🔧 시스템 점검을 시작합니다...")
        
        check_script = self.project_root / "scripts" / "check_system_specs.py"
        if check_script.exists():
            try:
                subprocess.run([sys.executable, str(check_script)], cwd=self.project_root)
            except Exception as e:
                print(f"❌ 시스템 점검 실패: {e}")
        else:
            print("❌ 시스템 점검 스크립트를 찾을 수 없습니다.")
    
    def quick_analysis(self):
        """⚡ 빠른 분석 실행"""
        print("\n⚡ 빠른 투자 분석을 시작합니다...")
        
        run_analysis_path = self.project_root / "core" / "run_analysis.py"
        if run_analysis_path.exists():
            try:
                subprocess.run([sys.executable, str(run_analysis_path)], cwd=self.project_root)
            except Exception as e:
                print(f"❌ 빠른 분석 실행 실패: {e}")
        else:
            print("❌ 분석 스크립트를 찾을 수 없습니다.")
    
    def backup_project(self):
        """💾 프로젝트 백업"""
        print("\n💾 프로젝트 백업을 시작합니다...")
        
        try:
            # 파일 관리자 모듈을 임포트해서 백업 기능 사용
            spec = importlib.util.spec_from_file_location("file_manager", self.project_root / "file_manager.py")
            file_manager_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(file_manager_module)
            
            file_manager = file_manager_module.FileManager(str(self.project_root))
            backup_path = file_manager.create_backup()
            
            if backup_path:
                print(f"✅ 백업 완료: {backup_path}")
            else:
                print("❌ 백업 실패")
                
        except Exception as e:
            print(f"❌ 백업 실행 실패: {e}")
    
    def show_project_info(self):
        """ℹ️ 프로젝트 정보 표시"""
        print("\nℹ️ 프로젝트 정보:")
        print("-" * 50)
        print(f"📂 프로젝트 경로: {self.project_root}")
        print(f"📅 현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🐍 Python 버전: {sys.version}")
        print(f"💻 운영체제: {os.name}")
        
        # requirements.txt 체크
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                print(f"📦 의존성 패키지: {len(lines)}개")
            except:
                print("📦 의존성 패키지: 확인 불가")
    
    def main_menu(self):
        """📋 메인 메뉴 표시 및 실행"""
        while True:
            print(self.logo)
            print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\n🎯 메인 메뉴:")
            print("=" * 60)
            print("1. 🎯 투자 분석 시스템 실행")
            print("2. 📁 파일 관리 시스템 실행") 
            print("3. ⚡ 빠른 투자 분석")
            print("4. 🔧 시스템 점검")
            print("5. 💾 프로젝트 백업")
            print("6. 📊 프로젝트 상태 확인")
            print("7. ℹ️  프로젝트 정보")
            print("8. 📂 탐색기에서 열기")
            print("0. 🚪 종료")
            print("=" * 60)
            
            try:
                choice = input("\n선택하세요 (0-8): ").strip()
                
                if choice == '1':
                    self.run_investment_analysis()
                elif choice == '2':
                    self.run_file_manager()
                elif choice == '3':
                    self.quick_analysis()
                elif choice == '4':
                    self.check_system()
                elif choice == '5':
                    self.backup_project()
                elif choice == '6':
                    self.show_status()
                elif choice == '7':
                    self.show_project_info()
                elif choice == '8':
                    if os.name == 'nt':  # Windows
                        os.startfile(str(self.project_root))
                    else:  # macOS/Linux
                        subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', str(self.project_root)])
                elif choice == '0':
                    print("\n👋 마스터 런처를 종료합니다.")
                    print("🙏 이용해 주셔서 감사합니다!")
                    break
                else:
                    print("❌ 잘못된 선택입니다. 다시 선택해주세요.")
                
                if choice != '0':
                    input("\n⏎ 계속하려면 Enter를 누르세요...")
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
            except KeyboardInterrupt:
                print("\n\n👋 사용자에 의해 중단되었습니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                input("⏎ 계속하려면 Enter를 누르세요...")

def main():
    """메인 실행 함수"""
    launcher = MasterLauncher()
    launcher.main_menu()

if __name__ == "__main__":
    main()