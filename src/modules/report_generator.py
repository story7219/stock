#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 AI 기반 투자 분석 시스템 - 리포트 생성기
분석 결과를 다양한 형태로 시각화하고 리포트를 생성하는 모듈
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO
import logging
from dataclasses import dataclass
from jinja2 import Template
import pdfkit
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReportData:
    """리포트 데이터 구조"""
    top5_selections: List[Dict[str, Any]]
    market_analysis: Dict[str, Any]
    strategy_weights: Dict[str, float]
    timestamp: datetime
    total_analyzed: int
    analysis_duration: float
    confidence_scores: Dict[str, float]

class ReportGenerator:
    """리포트 생성 클래스"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        리포트 생성기 초기화
        
        Args:
            output_dir: 리포트 출력 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
        plt.rcParams['axes.unicode_minus'] = False
        
    def generate_comprehensive_report(
        self, 
        report_data: ReportData,
        formats: List[str] = ["html", "json", "pdf"]
    ) -> Dict[str, str]:
        """
        종합 리포트 생성
        
        Args:
            report_data: 리포트 데이터
            formats: 생성할 리포트 형식 리스트
            
        Returns:
            생성된 리포트 파일 경로들
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}
        
        try:
            # HTML 리포트 생성
            if "html" in formats:
                html_path = self._generate_html_report(report_data, timestamp)
                report_files["html"] = html_path
                logger.info(f"HTML 리포트 생성 완료: {html_path}")
            
            # JSON 리포트 생성
            if "json" in formats:
                json_path = self._generate_json_report(report_data, timestamp)
                report_files["json"] = json_path
                logger.info(f"JSON 리포트 생성 완료: {json_path}")
            
            # PDF 리포트 생성
            if "pdf" in formats:
                pdf_path = self._generate_pdf_report(report_data, timestamp)
                report_files["pdf"] = pdf_path
                logger.info(f"PDF 리포트 생성 완료: {pdf_path}")
                
        except Exception as e:
            logger.error(f"리포트 생성 중 오류 발생: {e}")
            
        return report_files
    
    def _generate_html_report(self, report_data: ReportData, timestamp: str) -> str:
        """HTML 리포트 생성"""
        
        # 차트 생성
        charts = self._generate_charts(report_data)
        
        # HTML 템플릿
        html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 AI 투자 분석 리포트</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .meta-info {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
            font-size: 0.9em;
            opacity: 0.9;
        }
        .content {
            padding: 30px;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .top5-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stock-card {
            background: #f8f9ff;
            border: 1px solid #e1e5f7;
            border-radius: 8px;
            padding: 20px;
            transition: transform 0.2s;
        }
        .stock-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .stock-rank {
            background: #667eea;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .stock-symbol {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .stock-score {
            font-size: 1.1em;
            color: #667eea;
            font-weight: bold;
        }
        .reasoning {
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
            border-left: 4px solid #667eea;
            font-size: 0.9em;
            line-height: 1.4;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9ff;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e1e5f7;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .footer {
            background: #f8f9ff;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e1e5f7;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 AI 투자 분석 리포트</h1>
            <div class="meta-info">
                <div>생성일시: {{ report_data.timestamp.strftime('%Y년 %m월 %d일 %H:%M:%S') }}</div>
                <div>분석 종목 수: {{ report_data.total_analyzed }}개</div>
                <div>분석 소요시간: {{ "%.2f"|format(report_data.analysis_duration) }}초</div>
            </div>
        </div>
        
        <div class="content">
            <!-- 통계 요약 -->
            <div class="section">
                <h2>📊 분석 요약</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{{ report_data.top5_selections|length }}</div>
                        <div class="stat-label">선정 종목</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ report_data.total_analyzed }}</div>
                        <div class="stat-label">분석 종목</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ report_data.strategy_weights|length }}</div>
                        <div class="stat-label">적용 전략</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.1f"|format(report_data.analysis_duration) }}s</div>
                        <div class="stat-label">분석 시간</div>
                    </div>
                </div>
            </div>
            
            <!-- Top 5 종목 -->
            <div class="section">
                <h2>🏆 Top 5 추천 종목</h2>
                <div class="top5-grid">
                    {% for stock in report_data.top5_selections %}
                    <div class="stock-card">
                        <div class="stock-rank">{{ loop.index }}</div>
                        <div class="stock-symbol">{{ stock.symbol }}</div>
                        <div class="stock-score">종합점수: {{ "%.2f"|format(stock.total_score) }}</div>
                        <div class="reasoning">
                            <strong>선정 이유:</strong><br>
                            {{ stock.reasoning[:200] }}...
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <!-- 차트 섹션 -->
            <div class="section">
                <h2>📈 분석 차트</h2>
                {% for chart_name, chart_data in charts.items() %}
                <div class="chart-container">
                    <h3>{{ chart_name }}</h3>
                    <img src="data:image/png;base64,{{ chart_data }}" alt="{{ chart_name }}">
                </div>
                {% endfor %}
            </div>
            
            <!-- 전략별 가중치 -->
            <div class="section">
                <h2>⚖️ 투자 전략별 가중치</h2>
                <div class="stats-grid">
                    {% for strategy, weight in report_data.strategy_weights.items() %}
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.1f"|format(weight * 100) }}%</div>
                        <div class="stat-label">{{ strategy.replace('_', ' ').title() }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🤖 AI 기반 투자 분석 시스템 | 생성시간: {{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}</p>
            <p>⚠️ 이 리포트는 투자 참고용이며, 실제 투자 결정은 신중히 하시기 바랍니다.</p>
        </div>
    </div>
</body>
</html>
        """
        
        # 템플릿 렌더링
        template = Template(html_template)
        html_content = template.render(
            report_data=report_data,
            charts=charts,
            datetime=datetime
        )
        
        # 파일 저장
        filename = f"investment_report_{timestamp}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return str(filepath)
    
    def _generate_json_report(self, report_data: ReportData, timestamp: str) -> str:
        """JSON 리포트 생성"""
        
        json_data = {
            "metadata": {
                "generated_at": report_data.timestamp.isoformat(),
                "total_analyzed": report_data.total_analyzed,
                "analysis_duration_seconds": report_data.analysis_duration,
                "report_version": "1.0"
            },
            "top5_selections": [
                {
                    "rank": i + 1,
                    "symbol": stock["symbol"],
                    "total_score": stock["total_score"],
                    "confidence": stock.get("confidence", 0.0),
                    "risk_level": stock.get("risk_level", "medium"),
                    "reasoning": stock["reasoning"],
                    "strategy_scores": stock.get("strategy_scores", {}),
                    "technical_indicators": stock.get("technical_indicators", {})
                }
                for i, stock in enumerate(report_data.top5_selections)
            ],
            "strategy_weights": report_data.strategy_weights,
            "market_analysis": report_data.market_analysis,
            "confidence_scores": report_data.confidence_scores
        }
        
        filename = f"investment_report_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            
        return str(filepath)
    
    def _generate_pdf_report(self, report_data: ReportData, timestamp: str) -> str:
        """PDF 리포트 생성"""
        
        # HTML 리포트를 먼저 생성
        html_path = self._generate_html_report(report_data, timestamp)
        
        # PDF 변환 옵션
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        filename = f"investment_report_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        try:
            # HTML을 PDF로 변환
            pdfkit.from_file(html_path, str(filepath), options=options)
        except Exception as e:
            logger.warning(f"PDF 생성 실패: {e}")
            # PDF 생성 실패시 HTML 경로 반환
            return html_path
            
        return str(filepath)
    
    def _generate_charts(self, report_data: ReportData) -> Dict[str, str]:
        """차트 생성 및 base64 인코딩"""
        charts = {}
        
        try:
            # 1. Top 5 종목 점수 차트
            charts["top5_scores"] = self._create_top5_scores_chart(report_data)
            
            # 2. 전략별 가중치 파이 차트
            charts["strategy_weights"] = self._create_strategy_weights_chart(report_data)
            
            # 3. 신뢰도 분포 차트
            charts["confidence_distribution"] = self._create_confidence_chart(report_data)
            
        except Exception as e:
            logger.error(f"차트 생성 중 오류: {e}")
            
        return charts
    
    def _create_top5_scores_chart(self, report_data: ReportData) -> str:
        """Top 5 종목 점수 막대 차트"""
        plt.figure(figsize=(12, 6))
        
        symbols = [stock["symbol"] for stock in report_data.top5_selections]
        scores = [stock["total_score"] for stock in report_data.top5_selections]
        
        bars = plt.bar(symbols, scores, color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'])
        
        plt.title('🏆 Top 5 추천 종목 점수', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('종목 심볼', fontsize=12)
        plt.ylabel('종합 점수', fontsize=12)
        plt.xticks(rotation=45)
        
        # 막대 위에 점수 표시
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return self._fig_to_base64()
    
    def _create_strategy_weights_chart(self, report_data: ReportData) -> str:
        """전략별 가중치 파이 차트"""
        plt.figure(figsize=(10, 8))
        
        strategies = list(report_data.strategy_weights.keys())
        weights = list(report_data.strategy_weights.values())
        
        # 상위 8개 전략만 표시 (나머지는 기타로 합침)
        if len(strategies) > 8:
            top_strategies = sorted(zip(strategies, weights), key=lambda x: x[1], reverse=True)[:7]
            other_weight = sum(weight for _, weight in sorted(zip(strategies, weights), key=lambda x: x[1], reverse=True)[7:])
            
            strategies = [s for s, _ in top_strategies] + ['기타']
            weights = [w for _, w in top_strategies] + [other_weight]
        
        colors = plt.cm.Set3(range(len(strategies)))
        
        wedges, texts, autotexts = plt.pie(weights, labels=strategies, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        
        plt.title('⚖️ 투자 전략별 가중치 분배', fontsize=16, fontweight='bold', pad=20)
        
        # 범례 추가
        plt.legend(wedges, strategies, title="투자 전략", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        return self._fig_to_base64()
    
    def _create_confidence_chart(self, report_data: ReportData) -> str:
        """신뢰도 분포 차트"""
        plt.figure(figsize=(10, 6))
        
        if report_data.confidence_scores:
            symbols = list(report_data.confidence_scores.keys())
            confidences = list(report_data.confidence_scores.values())
            
            plt.bar(symbols, confidences, color='#4facfe', alpha=0.7)
            plt.title('📊 종목별 신뢰도 분포', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('종목 심볼', fontsize=12)
            plt.ylabel('신뢰도 (%)', fontsize=12)
            plt.xticks(rotation=45)
            plt.ylim(0, 100)
            
            # 평균선 추가
            avg_confidence = sum(confidences) / len(confidences)
            plt.axhline(y=avg_confidence, color='red', linestyle='--', alpha=0.7,
                       label=f'평균: {avg_confidence:.1f}%')
            plt.legend()
            
        else:
            plt.text(0.5, 0.5, '신뢰도 데이터 없음', ha='center', va='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title('📊 종목별 신뢰도 분포', fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return self._fig_to_base64()
    
    def _fig_to_base64(self) -> str:
        """matplotlib 그래프를 base64 문자열로 변환"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64
    
    def create_summary_dashboard(self, report_data: ReportData) -> str:
        """요약 대시보드 생성"""
        
        # Plotly를 사용한 인터랙티브 대시보드
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top 5 종목 점수', '전략별 가중치', '신뢰도 분포', '위험도 분석'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Top 5 점수
        symbols = [stock["symbol"] for stock in report_data.top5_selections]
        scores = [stock["total_score"] for stock in report_data.top5_selections]
        
        fig.add_trace(
            go.Bar(x=symbols, y=scores, name="종합점수", marker_color='#667eea'),
            row=1, col=1
        )
        
        # 전략 가중치
        strategies = list(report_data.strategy_weights.keys())[:8]  # 상위 8개만
        weights = list(report_data.strategy_weights.values())[:8]
        
        fig.add_trace(
            go.Pie(labels=strategies, values=weights, name="전략 가중치"),
            row=1, col=2
        )
        
        # 신뢰도 분포
        if report_data.confidence_scores:
            conf_symbols = list(report_data.confidence_scores.keys())
            confidences = list(report_data.confidence_scores.values())
            
            fig.add_trace(
                go.Bar(x=conf_symbols, y=confidences, name="신뢰도", marker_color='#4facfe'),
                row=2, col=1
            )
        
        # 위험도 vs 수익률 (예시)
        risk_levels = [stock.get("risk_level", "medium") for stock in report_data.top5_selections]
        risk_numeric = [{"low": 1, "medium": 2, "high": 3}.get(risk, 2) for risk in risk_levels]
        
        fig.add_trace(
            go.Scatter(
                x=risk_numeric, y=scores, mode='markers+text',
                text=symbols, textposition="top center",
                marker=dict(size=10, color=scores, colorscale='Viridis'),
                name="위험도-수익률"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="🎯 AI 투자 분석 대시보드",
            showlegend=False,
            height=800
        )
        
        # HTML로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dashboard_{timestamp}.html"
        filepath = self.output_dir / filename
        
        fig.write_html(str(filepath))
        
        return str(filepath)

    async def generate_analysis_report(self, analysis_data: Dict[str, Any]) -> str:
        """분석 리포트 생성"""
        try:
            logger.info("분석 리포트 생성 시작")
            
            # 리포트 데이터 준비
            report_data = {
                'title': '투자 분석 리포트',
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_data': analysis_data,
                'summary': self._generate_summary(analysis_data),
                'recommendations': self._extract_recommendations(analysis_data)
            }
            
            # HTML 리포트 생성
            html_report = self._generate_html_report(report_data)
            
            # 파일 저장
            report_path = f"reports/analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Path("reports").mkdir(exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            logger.info(f"분석 리포트 생성 완료: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"분석 리포트 생성 중 오류: {e}")
            raise
    
    def _generate_summary(self, analysis_data: Dict[str, Any]) -> str:
        """분석 요약 생성"""
        # Mock 요약 생성
        return "투자 분석 결과 요약: 전체적으로 긍정적인 투자 기회가 확인되었습니다."
    
    def _extract_recommendations(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """추천 종목 추출"""
        # Mock 추천 종목 생성
        return [
            {'symbol': 'AAPL', 'action': 'BUY', 'confidence': 0.85},
            {'symbol': 'GOOGL', 'action': 'BUY', 'confidence': 0.78}
        ]
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """HTML 리포트 생성"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data['title']}</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>{report_data['title']}</h1>
            <p>생성일시: {report_data['generated_at']}</p>
            <h2>분석 요약</h2>
            <p>{report_data['summary']}</p>
            <h2>추천 종목</h2>
            <ul>
            {''.join([f"<li>{rec['symbol']}: {rec['action']} (신뢰도: {rec['confidence']:.1%})</li>" for rec in report_data['recommendations']])}
            </ul>
        </body>
        </html>
        """
        return html_template

# 사용 예시
if __name__ == "__main__":
    # 테스트 데이터
    test_data = ReportData(
        top5_selections=[
            {"symbol": "AAPL", "total_score": 0.85, "reasoning": "강력한 기술적 지표와 안정적인 성장세"},
            {"symbol": "MSFT", "total_score": 0.82, "reasoning": "클라우드 사업 성장과 안정적인 수익성"},
            {"symbol": "GOOGL", "total_score": 0.79, "reasoning": "AI 기술 리더십과 광고 수익 안정성"},
            {"symbol": "TSLA", "total_score": 0.76, "reasoning": "전기차 시장 선도와 혁신적 기술"},
            {"symbol": "NVDA", "total_score": 0.74, "reasoning": "AI 칩 시장 독점과 높은 성장 잠재력"}
        ],
        market_analysis={"trend": "bullish", "volatility": "medium"},
        strategy_weights={"warren_buffett": 0.15, "peter_lynch": 0.10, "benjamin_graham": 0.12},
        timestamp=datetime.now(),
        total_analyzed=100,
        analysis_duration=45.2,
        confidence_scores={"AAPL": 85, "MSFT": 82, "GOOGL": 79, "TSLA": 76, "NVDA": 74}
    )
    
    generator = ReportGenerator()
    files = generator.generate_comprehensive_report(test_data)
    print("생성된 리포트:", files) 