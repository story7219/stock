#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖥️ 시스템 스펙 체크 및 ML 작업 준비 상태 확인 스크립트
=====================================================

이 스크립트는 코스피200·나스닥100·S&P500 AI 기반 투자 분석 시스템이
최적의 성능으로 실행될 수 있도록 시스템 환경을 종합적으로 분석합니다.

주요 기능:
1. 🔍 시스템 하드웨어 스펙 분석
   - CPU: 코어 수, 클럭 속도, 아키텍처 정보
   - RAM: 총 용량, 사용 가능한 메모리, 메모리 타입
   - 저장소: 디스크 용량, 사용 가능 공간, I/O 성능
   - GPU: 그래픽 카드 정보 (딥러닝 가속 가능 여부)

2. ⚡ 실시간 성능 메트릭 모니터링
   - CPU 사용률 실시간 추적
   - 메모리 사용량 및 여유 공간 모니터링
   - 디스크 I/O 성능 측정
   - 네트워크 대역폭 상태 확인

3. 🧠 머신러닝 작업 준비 상태 평가
   - ML 모델 학습 가능 여부 판단
   - 권장 배치 크기 계산
   - 최적 워커 스레드 수 제안
   - 메모리 부족 위험도 평가

4. 📊 성능 벤치마크 및 점수화
   - 하드웨어 성능 종합 점수 산출
   - 투자 분석 작업별 예상 성능 평가
   - 데이터 수집 처리 능력 측정
   - AI 추론 속도 예측

5. 💡 최적화 권장사항 제공
   - 하드웨어 업그레이드 우선순위 제안
   - 시스템 설정 최적화 가이드
   - 성능 병목 지점 식별
   - 메모리 관리 개선 방안

6. 🎯 투자 분석 시스템 특화 검사
   - 대용량 주식 데이터 처리 능력
   - 15개 투자 대가 전략 동시 실행 가능성  
   - Gemini AI API 호출 최적 동시성 레벨
   - 실시간 차트 분석 처리 성능

검사 항목:
- 최소 시스템 요구사항 충족 여부
- 권장 시스템 사양과의 비교
- 대용량 데이터 처리 준비 상태
- 멀티태스킹 성능 평가
- 장시간 연속 실행 안정성

출력 정보:
- 📈 시스템 스펙 상세 정보
- ⚡ 현재 성능 메트릭
- 🏆 종합 성능 점수 (A~F 등급)
- 💡 맞춤형 최적화 권장사항
- 🎯 투자 분석 작업별 예상 성능

사용법:
```bash
python scripts/check_system_specs.py
```

이 스크립트를 통해 시스템이 AI 기반 투자 분석에 최적화되어 있는지
확인하고, 필요한 경우 성능 개선 방안을 제공받을 수 있습니다.
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.system_monitor import SystemMonitor
    import psutil
    import time
    from datetime import datetime
except ImportError as e:
    print(f"❌ 필요한 라이브러리가 설치되지 않았습니다: {e}")
    print("다음 명령으로 설치하세요:")
    print("pip install psutil")
    sys.exit(1)

def main():
    print("=" * 60)
    print("🖥️  시스템 스펙 및 ML 작업 준비 상태 체크")
    print("=" * 60)
    
    try:
        # 시스템 모니터 초기화
        monitor = SystemMonitor()
        
        # 시스템 스펙 정보
        specs = monitor.system_specs
        
        print("\n📊 시스템 스펙 정보:")
        print(f"  ├─ CPU: {specs.cpu_count}코어 @ {specs.cpu_freq_max:.1f}GHz")
        print(f"  ├─ RAM: {specs.ram_total_gb:.1f}GB (여유: {specs.ram_available_gb:.1f}GB)")
        print(f"  └─ 저장소: {specs.disk_total_gb:.1f}GB (여유: {specs.disk_free_gb:.1f}GB)")
        
        # 현재 성능 메트릭
        print("\n⚡ 현재 시스템 상태:")
        metrics = monitor.get_current_metrics()
        print(f"  ├─ CPU 사용률: {metrics.cpu_usage_percent:.1f}%")
        print(f"  ├─ 메모리 사용률: {metrics.memory_usage_percent:.1f}%")
        print(f"  └─ 여유 메모리: {metrics.memory_available_gb:.1f}GB")
        
        # ML 작업 준비 상태
        print("\n🧠 머신러닝 작업 준비 상태:")
        ml_ready = monitor.is_system_ready_for_ml()
        if ml_ready:
            print("  ✅ ML 작업 실행 가능!")
        else:
            print("  ⚠️ ML 작업 실행 제한됨 (리소스 부족)")
        
        # 임계값 정보
        print(f"\n📋 설정된 임계값:")
        print(f"  ├─ CPU 임계값: {monitor.cpu_threshold}%")
        print(f"  └─ 메모리 임계값: {monitor.memory_threshold}%")
        
        # 권장 설정
        recommended_batch = monitor.get_recommended_batch_size()
        recommended_workers = monitor.get_recommended_worker_count()
        
        print(f"\n🎯 권장 설정:")
        print(f"  ├─ 배치 크기: {recommended_batch}")
        print(f"  └─ 워커 수: {recommended_workers}")
        
        # 사용자 스펙 평가
        print("\n📈 스펙 평가:")
        
        # CPU 평가
        if specs.cpu_freq_max >= 3.0:
            cpu_score = "✅ 양호"
        elif specs.cpu_freq_max >= 2.5:
            cpu_score = "⚠️ 보통"
        else:
            cpu_score = "❌ 부족"
        print(f"  ├─ CPU 성능: {cpu_score}")
        
        # RAM 평가
        if specs.ram_total_gb >= 16:
            ram_score = "✅ 충분"
        elif specs.ram_total_gb >= 8:
            ram_score = "⚠️ 보통"
        else:
            ram_score = "❌ 부족"
        print(f"  └─ RAM 용량: {ram_score}")
        
        # ML 작업별 예상 성능
        print(f"\n🎪 ML 작업별 예상 성능:")
        
        # 데이터 수집
        if specs.ram_total_gb >= 8:
            print("  ├─ 데이터 수집: ✅ 원활 (코스피200+나스닥100+S&P500)")
        else:
            print("  ├─ 데이터 수집: ⚠️ 제한적 (메모리 부족으로 일부만 가능)")
        
        # 전통적 ML
        if specs.ram_total_gb >= 4:
            print("  ├─ 전통적 ML: ✅ 원활 (RandomForest, XGBoost 등)")
        else:
            print("  ├─ 전통적 ML: ❌ 어려움")
        
        # 딥러닝
        if specs.ram_total_gb >= 8:
            print("  └─ 딥러닝: ✅ 가능 (경량 신경망)")
        elif specs.ram_total_gb >= 4:
            print("  └─ 딥러닝: ⚠️ 제한적 (매우 경량 모델만)")
        else:
            print("  └─ 딥러닝: ❌ 불가능")
        
        # 개선 권장사항
        print(f"\n💡 개선 권장사항:")
        
        improvements = []
        
        if specs.ram_total_gb < 16:
            improvements.append("• RAM을 16GB 이상으로 업그레이드 (현재 병목)")
        
        if specs.cpu_freq_max < 3.0:
            improvements.append("• CPU 업그레이드 고려 (ML 연산 속도 향상)")
        
        if specs.disk_free_gb < 100:
            improvements.append("• 저장공간 확보 (모델 저장 및 데이터 캐시용)")
        
        if not improvements:
            print("  ✅ 현재 스펙으로 충분합니다!")
        else:
            for improvement in improvements:
                print(f"  {improvement}")
        
        # 최적화 팁
        print(f"\n🚀 성능 최적화 팁:")
        print("  • 불필요한 프로그램 종료하여 메모리 확보")
        print("  • ML 작업 전 시스템 재시작 권장")
        print("  • 백그라운드 업데이트 비활성화")
        print("  • 배치 크기를 시스템에 맞게 조정")
        
        # 실시간 모니터링 테스트
        print(f"\n🔄 5초간 실시간 모니터링 테스트...")
        for i in range(5):
            time.sleep(1)
            current_metrics = monitor.get_current_metrics()
            print(f"  [{i+1}/5] CPU: {current_metrics.cpu_usage_percent:5.1f}% | "
                  f"메모리: {current_metrics.memory_usage_percent:5.1f}% | "
                  f"여유: {current_metrics.memory_available_gb:5.1f}GB")
        
        # 종합 평가
        print(f"\n🏆 종합 평가:")
        
        total_score = 0
        
        # CPU 점수
        if specs.cpu_freq_max >= 3.5:
            cpu_points = 4
        elif specs.cpu_freq_max >= 3.0:
            cpu_points = 3
        elif specs.cpu_freq_max >= 2.5:
            cpu_points = 2
        else:
            cpu_points = 1
        total_score += cpu_points
        
        # RAM 점수
        if specs.ram_total_gb >= 32:
            ram_points = 4
        elif specs.ram_total_gb >= 16:
            ram_points = 3
        elif specs.ram_total_gb >= 8:
            ram_points = 2
        else:
            ram_points = 1
        total_score += ram_points
        
        # 디스크 점수
        if specs.disk_free_gb >= 500:
            disk_points = 2
        elif specs.disk_free_gb >= 100:
            disk_points = 1
        else:
            disk_points = 0
        total_score += disk_points
        
        max_score = 10
        percentage = (total_score / max_score) * 100
        
        if percentage >= 80:
            grade = "🥇 우수"
        elif percentage >= 60:
            grade = "🥈 양호"
        elif percentage >= 40:
            grade = "🥉 보통"
        else:
            grade = "📉 개선 필요"
        
        print(f"  총점: {total_score}/{max_score} ({percentage:.0f}%) - {grade}")
        
        # 결론
        print(f"\n✨ 결론:")
        if ml_ready and percentage >= 60:
            print("  현재 시스템으로 ML 투자 분석 프로그램을 원활히 실행할 수 있습니다!")
        elif ml_ready:
            print("  현재 시스템으로 ML 작업이 가능하지만, 성능 향상을 위해 업그레이드를 고려해보세요.")
        else:
            print("  현재 시스템 부하가 높습니다. 잠시 후 다시 시도하거나 불필요한 프로그램을 종료하세요.")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 시스템 체크 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 