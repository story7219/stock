from __future__ import annotations
from core.config import config
from core.logger import setup_logging, get_logger
from core.models import News, NewsCategory, SentimentType, Theme, Signal, StrategyType, TradeType
from datetime import datetime, timedelta, timezone
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from strategy.short_term_optimized import ShortTermOptimizedStrategy
from typing import Any, Dict, List
import asyncio
import logging
import numpy as np
import pandas as pd
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: test_optimized_strategy.py
모듈: 단기매매 최적화 전략 테스트
목적: 새로운 최적화 전략의 성능 및 기능 테스트

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 1.0.0

Dependencies:
    - Python 3.11+
    - pandas==2.1.4
    - numpy==1.24.0
    - rich==13.7.0

Performance:
    - 테스트 실행: < 30초
    - 메모리사용량: < 200MB

Security:
    - 데이터 검증
    - 에러 처리
    - 로깅

License: MIT
"""





# Rich 콘솔 설정
console = Console()

# 로깅 설정
setup_logging(logging.INFO)
logger = get_logger(__name__)


def create_sample_news_data() -> List[News]:
    """샘플 뉴스 데이터 생성"""
    news_list = []

    # 긍정적 뉴스
    positive_news = [
        {
            'title': '삼성전자, AI 반도체 시장 진출 확대',
            'content': '삼성전자가 AI 반도체 시장에서 혁신적인 기술 개발을 통해 성장세를 보이고 있습니다.',
            'sentiment': SentimentType.POSITIVE,
            'sentiment_score': 0.8,
            'related_stocks': ['005930']
        },
        {
            'title': 'SK하이닉스, HBM 기술 혁신 성과',
            'content': 'SK하이닉스가 고대역폭 메모리(HBM) 기술에서 세계 최고 수준의 성과를 달성했습니다.',
            'sentiment': SentimentType.POSITIVE,
            'sentiment_score': 0.7,
            'related_stocks': ['000660']
        },
        {
            'title': 'LG에너지솔루션, 전기차 배터리 시장 점유율 확대',
            'content': 'LG에너지솔루션이 글로벌 전기차 배터리 시장에서 점유율을 확대하고 있습니다.',
            'sentiment': SentimentType.POSITIVE,
            'sentiment_score': 0.6,
            'related_stocks': ['373220']
        }
    ]

    # 부정적 뉴스
    negative_news = [
        {
            'title': '일부 반도체 업체, 공급 과잉 우려',
            'content': '반도체 업계에서 공급 과잉으로 인한 가격 하락 우려가 제기되고 있습니다.',
            'sentiment': SentimentType.NEGATIVE,
            'sentiment_score': -0.5,
            'related_stocks': ['005930', '000660']
        }
    ]

    # 뉴스 생성
    for i, news_data in enumerate(positive_news + negative_news):
        news = News(
            id=f"news_{i+1}",
            title=news_data['title'],
            content=news_data['content'],
            url=f"https://example.com/news/{i+1}",
            source="테스트 뉴스",
            published_at=datetime.now(timezone.utc) - timedelta(hours=i),
            category=NewsCategory.TECHNOLOGICAL,
            sentiment=news_data['sentiment'],
            sentiment_score=news_data['sentiment_score'],
            related_stocks=news_data['related_stocks']
        )
        news_list.append(news)

    return news_list


def create_sample_theme_data() -> List[Theme]:
    """샘플 테마 데이터 생성"""
    themes = [
        Theme(
            id="theme_1",
            name="AI 반도체",
            description="인공지능 반도체 관련 테마",
            category="technology",
            related_stocks=['005930', '000660', '051910'],
            momentum_score=0.8
        ),
        Theme(
            id="theme_2",
            name="전기차 배터리",
            description="전기차 배터리 관련 테마",
            category="energy",
            related_stocks=['373220', '006400'],
            momentum_score=0.7
        ),
        Theme(
            id="theme_3",
            name="바이오 기술",
            description="바이오 기술 관련 테마",
            category="healthcare",
            related_stocks=['068270', '091990'],
            momentum_score=0.6
        )
    ]

    return themes


def create_sample_stock_data() -> Dict[str, pd.DataFrame]:
    """샘플 주식 데이터 생성"""
    stock_data = {}

    # 종목별 샘플 데이터 생성
    stock_codes = ['005930', '000660', '373220', '051910', '006400']

    for stock_code in stock_codes:
        # 100일치 데이터 생성
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

        # 기본 가격 (종목별로 다르게)
        base_price = 50000 + hash(stock_code) % 50000

        # 가격 데이터 생성 (랜덤 워크)
        np.random.seed(hash(stock_code) % 1000)
        price_changes = np.random.normal(0, 0.02, 100)  # 2% 일일 변동성
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # 최소 1000원

        # 거래량 데이터 생성
        volumes = np.random.randint(1000000, 10000000, 100)

        # 데이터프레임 생성
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': volumes
        })

        stock_data[stock_code] = df

    return stock_data


def create_sample_market_data() -> Dict[str, Dict[str, Any]]:
    """샘플 시장 데이터 생성"""
    market_data = {
        '005930': {
            'market_cap': 3_500_000_000_000,  # 3.5조원 (중형주)
            'pe_ratio': 15.5,
            'pb_ratio': 1.2,
            'atr': 2500,
            'close_price': 75000
        },
        '000660': {
            'market_cap': 2_800_000_000_000,  # 2.8조원 (중형주)
            'pe_ratio': 18.2,
            'pb_ratio': 1.5,
            'atr': 3200,
            'close_price': 120000
        },
        '373220': {
            'market_cap': 800_000_000_000,   # 8000억원 (소형주)
            'pe_ratio': 25.1,
            'pb_ratio': 2.1,
            'atr': 1800,
            'close_price': 450000
        },
        '051910': {
            'market_cap': 1_200_000_000_000, # 1.2조원 (중형주)
            'pe_ratio': 22.3,
            'pb_ratio': 1.8,
            'atr': 2100,
            'close_price': 85000
        },
        '006400': {
            'market_cap': 600_000_000_000,   # 6000억원 (소형주)
            'pe_ratio': 30.5,
            'pb_ratio': 2.5,
            'atr': 1500,
            'close_price': 35000
        }
    }

    return market_data


def display_strategy_metrics(metrics: Dict[str, Any]) -> None:
    """전략 성과 지표 표시"""
    table = Table(title="📊 단기매매 최적화 전략 성과 지표")

    table.add_column("지표", style="cyan")
    table.add_column("값", style="green")
    table.add_column("설명", style="yellow")

    # 기본 지표
    table.add_row("전략명", metrics['strategy_name'], "전략 이름")
    table.add_row("일일 거래", f"{metrics['daily_trades']}/{metrics['max_daily_trades']}", "현재/최대 거래 수")
    table.add_row("일일 신호", str(metrics['daily_signals_count']), "생성된 신호 수")

    # 성과 지표
    perf_metrics = metrics['performance_metrics']
    table.add_row("총 신호", str(perf_metrics['total_signals']), "총 생성된 신호 수")
    table.add_row("성공 신호", str(perf_metrics['successful_signals']), "성공한 신호 수")
    table.add_row("승률", f"{perf_metrics['win_rate']:.2%}", "성공률")
    table.add_row("평균 수익률", f"{perf_metrics['avg_return']:.2%}", "평균 수익률")
    table.add_row("평균 보유기간", f"{perf_metrics['avg_holding_days']:.1f}일", "평균 보유 기간")

    # 전략 파라미터
    params = metrics['strategy_parameters']
    table.add_row("최소 신뢰도", f"{params['min_confidence_score']:.2f}", "최소 신뢰도 점수")
    table.add_row("최소 거래량 비율", f"{params['min_volume_ratio']:.1f}", "평균 대비 최소 거래량")
    table.add_row("최소 가격 변화", f"{params['min_price_change']:.1%}", "최소 가격 변화율")
    table.add_row("최대 보유기간", f"{params['max_holding_days']}일", "최대 보유 기간")
    table.add_row("최소 보유기간", f"{params['min_holding_days']}일", "최소 보유 기간")

    console.print(table)


def display_signals(signals: List[Signal]) -> None:
    """신호 목록 표시"""
    if not signals:
        console.print("📭 생성된 신호가 없습니다.", style="yellow")
        return

    table = Table(title="🎯 생성된 매매 신호")

    table.add_column("종목코드", style="cyan")
    table.add_column("신호타입", style="green")
    table.add_column("신뢰도", style="yellow")
    table.add_column("근거", style="white")
    table.add_column("생성시간", style="blue")

    for signal in signals:
        signal_type_emoji = "📈" if signal.signal_type == TradeType.BUY else "📉"
        confidence_color = "green" if signal.confidence_score >= 0.7 else "yellow" if signal.confidence_score >= 0.5 else "red"

        table.add_row(
            signal.stock_code,
            f"{signal_type_emoji} {signal.signal_type.value.upper()}",
            f"[{confidence_color}]{signal.confidence_score:.2f}[/{confidence_color}]",
            signal.reasoning[:50] + "..." if len(signal.reasoning) > 50 else signal.reasoning,
            signal.created_at.strftime("%H:%M:%S")
        )

    console.print(table)


async def test_optimized_strategy() -> None:
    """단기매매 최적화 전략 테스트"""
    console.print(Panel.fit("🚀 단기매매 최적화 전략 테스트 시작", style="bold blue"))

    try:
        # 1. 전략 초기화
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("전략 초기화 중...", total=None)
            strategy = ShortTermOptimizedStrategy()
            progress.update(task, description="✅ 전략 초기화 완료")

        # 2. 샘플 데이터 생성
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("샘플 데이터 생성 중...", total=None)

            news_list = create_sample_news_data()
            themes = create_sample_theme_data()
            stock_data = create_sample_stock_data()
            market_data = create_sample_market_data()

            progress.update(task, description="✅ 샘플 데이터 생성 완료")

        # 3. 신호 생성
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("신호 생성 중...", total=None)

            signals = await strategy.generate_signals(
                news_list=news_list,
                themes=themes,
                stock_data=stock_data,
                market_data=market_data,
                target_stocks=['005930', '000660', '373220', '051910', '006400']
            )

            progress.update(task, description="✅ 신호 생성 완료")

        # 4. 결과 표시
        console.print(f"\n📊 데이터 요약:")
        console.print(f"   뉴스: {len(news_list)}개")
        console.print(f"   테마: {len(themes)}개")
        console.print(f"   종목: {len(stock_data)}개")
        console.print(f"   신호: {len(signals)}개")

        # 5. 신호 표시
        display_signals(signals)

        # 6. 전략 성과 지표 표시
        metrics = strategy.get_strategy_metrics()
        display_strategy_metrics(metrics)

        # 7. 성과 업데이트 시뮬레이션
        console.print("\n🔄 성과 업데이트 시뮬레이션:")

        # 가상 거래 결과로 성과 업데이트
        sample_trade_results = [
            {'profit': 50000, 'return_pct': 0.05, 'holding_days': 3},
            {'profit': -20000, 'return_pct': -0.02, 'holding_days': 2},
            {'profit': 80000, 'return_pct': 0.08, 'holding_days': 5}
        ]

        for i, result in enumerate(sample_trade_results, 1):
            strategy.update_performance(result)
            console.print(f"   거래 {i}: 수익 {result['profit']:,}원 ({result['return_pct']:+.1%}), 보유 {result['holding_days']}일")

        # 업데이트된 성과 지표 표시
        console.print("\n📈 업데이트된 성과 지표:")
        updated_metrics = strategy.get_strategy_metrics()
        perf = updated_metrics['performance_metrics']
        console.print(f"   승률: {perf['win_rate']:.2%}")
        console.print(f"   평균 수익률: {perf['avg_return']:.2%}")
        console.print(f"   평균 보유기간: {perf['avg_holding_days']:.1f}일")

        console.print(Panel.fit("✅ 단기매매 최적화 전략 테스트 완료!", style="bold green"))

    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        console.print(f"❌ 오류: {e}", style="bold red")


async def test_strategy_limits() -> None:
    """전략 제한사항 테스트"""
    console.print(Panel.fit("🔒 전략 제한사항 테스트", style="bold blue"))

    try:
        strategy = ShortTermOptimizedStrategy()

        # 일일 거래 한도 테스트
        console.print("📊 일일 거래 한도 테스트:")
        console.print(f"   최대 거래 수: {strategy.max_daily_trades}")
        console.print(f"   현재 거래 수: {strategy.daily_trades}")

        # 연속 신호 생성 시뮬레이션
        news_list = create_sample_news_data()
        themes = create_sample_theme_data()
        stock_data = create_sample_stock_data()

        total_signals = 0
        for i in range(5):  # 5번 연속 실행
            signals = await strategy.generate_signals(
                news_list=news_list,
                themes=themes,
                stock_data=stock_data
            )
            total_signals += len(signals)
            console.print(f"   실행 {i+1}: {len(signals)}개 신호 생성")

        console.print(f"   총 생성된 신호: {total_signals}개")
        console.print(f"   최종 일일 거래 수: {strategy.daily_trades}")

        # 보유기간 제한 확인
        console.print(f"\n📅 보유기간 제한:")
        console.print(f"   최소 보유기간: {strategy.min_holding_days}일")
        console.print(f"   최대 보유기간: {strategy.max_holding_days}일")

        # 시가총액 제한 확인
        console.print(f"\n💰 시가총액 제한:")
        console.print(f"   최소 시가총액: {strategy.min_market_cap:,}원")
        console.print(f"   최대 시가총액: {strategy.max_market_cap:,}원")

        console.print(Panel.fit("✅ 전략 제한사항 테스트 완료!", style="bold green"))

    except Exception as e:
        logger.error(f"제한사항 테스트 중 오류 발생: {e}")
        console.print(f"❌ 오류: {e}", style="bold red")


async def main() -> None:
    """메인 함수"""
    console.print("🚀 단기매매 최적화 전략 종합 테스트", style="bold blue")

    # 1. 기본 전략 테스트
    await test_optimized_strategy()

    console.print("\n" + "="*80 + "\n")

    # 2. 제한사항 테스트
    await test_strategy_limits()

    console.print("\n🎉 모든 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())

