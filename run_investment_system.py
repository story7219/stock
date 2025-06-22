#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 종합 투자 분석 시스템 실행기
HTS 스타일 GUI + 고해상도 차트 + AI 분석 통합 시스템
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox
import webbrowser

def check_dependencies():
    """필수 패키지 확인 및 설치"""
    required_packages = [
        'yfinance',
        'matplotlib',
        'pandas',
        'numpy',
        'requests',
        'google-generativeai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("📦 필수 패키지 설치 중...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} 설치 완료")
            except subprocess.CalledProcessError:
                print(f"❌ {package} 설치 실패")
                return False
    
    return True

def show_welcome_screen():
    """환영 화면 표시"""
    welcome = tk.Tk()
    welcome.title("🚀 종합 투자 분석 시스템")
    welcome.geometry("600x400")
    welcome.configure(bg='#0a0a0a')
    welcome.resizable(False, False)
    
    # 중앙 정렬
    welcome.eval('tk::PlaceWindow . center')
    
    # 메인 프레임
    main_frame = tk.Frame(welcome, bg='#0a0a0a')
    main_frame.pack(expand=True, fill='both', padx=20, pady=20)
    
    # 제목
    title_label = tk.Label(main_frame, 
                          text="🚀 종합 투자 분석 시스템", 
                          font=('맑은 고딕', 24, 'bold'),
                          fg='#00ff88', bg='#0a0a0a')
    title_label.pack(pady=(20, 10))
    
    # 부제목
    subtitle_label = tk.Label(main_frame, 
                             text="Professional HTS Style Investment Analysis Platform", 
                             font=('맑은 고딕', 12),
                             fg='#ffffff', bg='#0a0a0a')
    subtitle_label.pack(pady=(0, 20))
    
    # 기능 소개
    features_frame = tk.Frame(main_frame, bg='#1a1a1a', relief='raised', bd=2)
    features_frame.pack(fill='x', pady=10)
    
    features_text = """
    ✨ 주요 기능:
    
    📊 HTS 스타일 실시간 차트 (일목균형표, 이동평균선)
    📈 고해상도 캔들스틱 차트 저장 (PNG/SVG)
    🤖 AI 기반 투자 분석 (Gemini 1.5 Pro)
    📋 40개 종목 퀀트 분석 (한국/미국 주식)
    💎 전문적인 투자 점수 시스템
    🔄 실시간 데이터 업데이트
    """
    
    features_label = tk.Label(features_frame, text=features_text,
                             font=('맑은 고딕', 11), fg='#ffffff', bg='#1a1a1a',
                             justify='left', anchor='w')
    features_label.pack(padx=20, pady=15)
    
    # 버튼 프레임
    button_frame = tk.Frame(main_frame, bg='#0a0a0a')
    button_frame.pack(pady=20)
    
    # 시작 버튼
    start_button = tk.Button(button_frame, text="🚀 시스템 시작", 
                            command=lambda: start_system(welcome),
                            font=('맑은 고딕', 14, 'bold'),
                            bg='#00ff88', fg='#000000',
                            width=15, height=2,
                            relief='raised', bd=3)
    start_button.pack(side='left', padx=10)
    
    # 고해상도 차트 테스트 버튼
    chart_button = tk.Button(button_frame, text="📊 차트 테스트", 
                            command=lambda: test_chart(welcome),
                            font=('맑은 고딕', 12, 'bold'),
                            bg='#4488ff', fg='#ffffff',
                            width=12, height=2,
                            relief='raised', bd=3)
    chart_button.pack(side='left', padx=10)
    
    # 종료 버튼
    exit_button = tk.Button(button_frame, text="❌ 종료", 
                           command=welcome.quit,
                           font=('맑은 고딕', 12, 'bold'),
                           bg='#ff4444', fg='#ffffff',
                           width=8, height=2,
                           relief='raised', bd=3)
    exit_button.pack(side='left', padx=10)
    
    # 하단 정보
    info_label = tk.Label(main_frame, 
                         text="💡 Tip: 종목을 선택한 후 '고품질 차트' 버튼으로 고해상도 차트를 저장할 수 있습니다.",
                         font=('맑은 고딕', 9), fg='#999999', bg='#0a0a0a')
    info_label.pack(side='bottom', pady=10)
    
    welcome.mainloop()

def start_system(welcome_window):
    """메인 시스템 시작"""
    try:
        welcome_window.destroy()
        print("🚀 종합 투자 분석 시스템 시작...")
        
        # 메인 시스템 실행
        from comprehensive_hts_gui import ComprehensiveHTS
        app = ComprehensiveHTS()
        app.run()
        
    except Exception as e:
        messagebox.showerror("오류", f"시스템 시작 오류: {e}")
        print(f"❌ 시스템 시작 오류: {e}")

def test_chart(welcome_window):
    """고해상도 차트 테스트"""
    try:
        print("📊 고해상도 차트 테스트 시작...")
        
        from high_resolution_chart import HighResolutionCandlestickChart
        chart_generator = HighResolutionCandlestickChart()
        
        # 삼성전자 차트 생성
        fig = chart_generator.create_comprehensive_chart(
            symbol="005930.KS",
            korean_name="삼성전자",
            start_date="2024-01-01",
            end_date="2025-06-21",
            save_format="png"
        )
        
        if fig:
            messagebox.showinfo("성공", "고해상도 차트가 성공적으로 생성되었습니다!\n파일을 확인해보세요.")
        else:
            messagebox.showwarning("실패", "차트 생성에 실패했습니다.")
            
    except Exception as e:
        messagebox.showerror("오류", f"차트 테스트 오류: {e}")
        print(f"❌ 차트 테스트 오류: {e}")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🚀 종합 투자 분석 시스템 초기화")
    print("=" * 60)
    
    # 의존성 확인
    if not check_dependencies():
        print("❌ 필수 패키지 설치 실패")
        input("Press Enter to exit...")
        return
    
    print("✅ 모든 의존성 확인 완료")
    
    # 환영 화면 표시
    try:
        show_welcome_screen()
    except KeyboardInterrupt:
        print("\n👋 시스템 종료")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 