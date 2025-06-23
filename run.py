#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultra Stock Analysis System 실행 스크립트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from main import main
    main() 