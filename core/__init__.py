#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 핵심 실행 모듈 패키지
메인 애플리케이션 로직과 런처가 포함되어 있습니다.
"""

__version__ = "1.0.0"

# 핵심 모듈들
try:
    from .launcher import main as launch_app
    from .app import main as run_app
    from .run_analysis import main as run_analysis
    
    __all__ = ['launch_app', 'run_app', 'run_analysis']
    
except ImportError as e:
    print(f"⚠️ 핵심 모듈 임포트 실패: {e}")
    __all__ = []