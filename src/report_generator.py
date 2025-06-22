"""
리포트 생성 모듈
Top5 종목 선정 결과를 다양한 형식으로 리포트 생성
"""

import logging
import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
from jinja2 import Template, Environment, FileSystemLoader
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns
from .gemini_analyzer import GeminiAnalysisResult, Top5Selection
from .technical_analyzer import TechnicalSignal, ChartPattern
from .data_collector import StockData

logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    """리포트 설정"""
    title: str = "Ultra HTS v5.0 - Top5 종목 선정 리포트"
    subtitle: str = "Gemini AI 기반 투자 분석 시스템"
    author: str = "Ultra HTS Team"
    company: str = "Ultra HTS"
    logo_path: Optional[str] = None
    theme_color: str = "#1f77b4"
    output_directory: str = "reports"
    include_charts: bool = True
    include_technical_analysis: bool = True
    include_risk_warnings: bool = True

class ReportGenerator:
    """종합 리포트 생성기"""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self._ensure_output_directory()
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # HTML 템플릿 환경 설정
        self.jinja_env = Environment(
            loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates'))
        )
        
    def _ensure_output_directory(self):
        """출력 디렉토리 생성"""
        os.makedirs(self.config.output_directory, exist_ok=True)
        
    def generate_comprehensive_report(self, 
                                    gemini_result: GeminiAnalysisResult,
                                    market_data: Dict[str, List[StockData]],
                                    strategy_results: Dict[str, Any],
                                    technical_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        종합 리포트 생성 (모든 형식)
        
        Args:
            gemini_result: Gemini AI 분석 결과
            market_data: 시장 데이터
            strategy_results: 전략별 결과
            technical_analysis: 기술적 분석 결과
            
        Returns:
            Dict[str, str]: 생성된 파일 경로들
        """
        logger.info("종합 리포트 생성 시작")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_files = {}
            
            # 1. HTML 리포트 생성
            html_path = self._generate_html_report(
                gemini_result, market_data, strategy_results, technical_analysis, timestamp
            )
            report_files['html'] = html_path
            
            # 2. PDF 리포트 생성
            pdf_path = self._generate_pdf_report(
                gemini_result, market_data, strategy_results, timestamp
            )
            report_files['pdf'] = pdf_path
            
            # 3. CSV 데이터 내보내기
            csv_path = self._generate_csv_report(gemini_result, timestamp)
            report_files['csv'] = csv_path
            
            # 4. JSON 데이터 내보내기
            json_path = self._generate_json_report(
                gemini_result, market_data, strategy_results, timestamp
            )
            report_files['json'] = json_path
            
            # 5. 차트 이미지 생성
            charts_dir = self._generate_chart_images(
                gemini_result, market_data, strategy_results, timestamp
            )
            report_files['charts'] = charts_dir
            
            logger.info(f"종합 리포트 생성 완료: {len(report_files)}개 파일")
            return report_files
            
        except Exception as e:
            logger.error(f"종합 리포트 생성 실패: {e}")
            raise
    
    def _generate_html_report(self, 
                            gemini_result: GeminiAnalysisResult,
                            market_data: Dict[str, List[StockData]],
                            strategy_results: Dict[str, Any],
                            technical_analysis: Optional[Dict[str, Any]],
                            timestamp: str) -> str:
        """HTML 리포트 생성"""
        try:
            # HTML 템플릿
            html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid {{ config.theme_color }};
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: {{ config.theme_color }};
            margin: 0;
            font-size: 2.5em;
        }
        .header h2 {
            color: #666;
            margin: 10px 0;
            font-weight: normal;
        }
        .summary-box {
            background: linear-gradient(135deg, {{ config.theme_color }}, #4a90e2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .top5-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stock-card {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            background: #fafafa;
            transition: transform 0.3s;
        }
        .stock-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .rank-badge {
            display: inline-block;
            background: {{ config.theme_color }};
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .score {
            font-size: 2em;
            font-weight: bold;
            color: {{ config.theme_color }};
        }
        .strategy-scores {
            margin: 15px 0;
        }
        .strategy-score {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            padding: 5px 10px;
            background: white;
            border-radius: 5px;
        }
        .analysis-section {
            margin: 30px 0;
            padding: 20px;
            border-left: 4px solid {{ config.theme_color }};
            background: #f9f9f9;
        }
        .risk-warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .risk-warning h4 {
            color: #856404;
            margin-top: 0;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: {{ config.theme_color }};
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <!-- 헤더 -->
        <div class="header">
            <h1>{{ config.title }}</h1>
            <h2>{{ config.subtitle }}</h2>
            <p><strong>분석 시간:</strong> {{ analysis_timestamp }}</p>
            <p><strong>신뢰도:</strong> {{ confidence_score }}%</p>
        </div>

        <!-- 요약 -->
        <div class="summary-box">
            <h3>📊 분석 요약</h3>
            <p>{{ analysis_summary }}</p>
        </div>

        <!-- Top 5 종목 -->
        <div class="analysis-section">
            <h2>🏆 Top 5 선정 종목</h2>
            <div class="top5-grid">
                {% for stock in top5_selections %}
                <div class="stock-card">
                    <div class="rank-badge">{{ stock.rank }}위</div>
                    <h3>{{ stock.symbol }} - {{ stock.name }}</h3>
                    <div class="score">{{ "%.1f"|format(stock.final_score) }}점</div>
                    
                    <div class="strategy-scores">
                        <h4>전략별 점수</h4>
                        {% for strategy, score in stock.strategy_scores.items() %}
                        <div class="strategy-score">
                            <span>{{ strategy }}</span>
                            <span><strong>{{ "%.1f"|format(score) }}점</strong></span>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div>
                        <h4>선정 이유</h4>
                        <p>{{ stock.selection_reason }}</p>
                        
                        <h4>기술적 분석</h4>
                        <p>{{ stock.technical_analysis }}</p>
                        
                        <h4>리스크 평가</h4>
                        <p>{{ stock.risk_assessment }}</p>
                        
                        <h4>Gemini AI 분석</h4>
                        <p>{{ stock.gemini_reasoning }}</p>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- 시장 전망 -->
        <div class="analysis-section">
            <h2>🔮 시장 전망</h2>
            <p>{{ market_outlook }}</p>
        </div>

        <!-- 위험 요소 -->
        {% if risk_warnings %}
        <div class="analysis-section">
            <h2>⚠️ 위험 요소</h2>
            {% for warning in risk_warnings %}
            <div class="risk-warning">
                <h4>주의사항</h4>
                <p>{{ warning }}</p>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <!-- 대안 후보 -->
        {% if alternative_candidates %}
        <div class="analysis-section">
            <h2>💡 대안 후보</h2>
            <p>다음 종목들도 고려해볼 만합니다:</p>
            <ul>
                {% for candidate in alternative_candidates %}
                <li>{{ candidate }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <!-- 시장별 데이터 요약 -->
        <div class="analysis-section">
            <h2>📈 시장별 데이터 요약</h2>
            <table>
                <thead>
                    <tr>
                        <th>시장</th>
                        <th>분석 종목 수</th>
                        <th>평균 점수</th>
                        <th>Top 종목</th>
                    </tr>
                </thead>
                <tbody>
                    {% for market, data in market_summary.items() %}
                    <tr>
                        <td>{{ market }}</td>
                        <td>{{ data.total_stocks }}</td>
                        <td>{{ "%.1f"|format(data.avg_score) }}</td>
                        <td>{{ data.top_stock }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- 푸터 -->
        <div class="footer">
            <p>Generated by {{ config.company }} - {{ config.author }}</p>
            <p>이 리포트는 투자 참고용이며, 투자 결정은 본인의 판단과 책임하에 이루어져야 합니다.</p>
        </div>
    </div>
</body>
</html>
            """
            
            # 템플릿 데이터 준비
            template_data = {
                'config': self.config,
                'analysis_timestamp': gemini_result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'confidence_score': gemini_result.confidence_score,
                'analysis_summary': gemini_result.analysis_summary,
                'market_outlook': gemini_result.market_outlook,
                'risk_warnings': gemini_result.risk_warnings,
                'alternative_candidates': gemini_result.alternative_candidates,
                'top5_selections': gemini_result.top5_selections,
                'market_summary': self._calculate_market_summary(market_data, strategy_results)
            }
            
            # HTML 생성
            template = Template(html_template)
            html_content = template.render(**template_data)
            
            # 파일 저장
            filename = f"top5_report_{timestamp}.html"
            filepath = os.path.join(self.config.output_directory, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML 리포트 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"HTML 리포트 생성 실패: {e}")
            raise
    
    def _generate_pdf_report(self, 
                           gemini_result: GeminiAnalysisResult,
                           market_data: Dict[str, List[StockData]],
                           strategy_results: Dict[str, Any],
                           timestamp: str) -> str:
        """PDF 리포트 생성"""
        try:
            filename = f"top5_report_{timestamp}.pdf"
            filepath = os.path.join(self.config.output_directory, filename)
            
            # PDF 문서 생성
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # 제목 스타일
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1,  # 중앙 정렬
                textColor=colors.HexColor(self.config.theme_color)
            )
            
            # 제목
            story.append(Paragraph(self.config.title, title_style))
            story.append(Paragraph(self.config.subtitle, styles['Heading2']))
            story.append(Spacer(1, 20))
            
            # 분석 정보
            analysis_info = [
                ['분석 시간', gemini_result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')],
                ['신뢰도', f"{gemini_result.confidence_score}%"],
                ['분석 종목 수', str(sum(len(stocks) for stocks in market_data.values()))]
            ]
            
            info_table = Table(analysis_info, colWidths=[2*inch, 3*inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(info_table)
            story.append(Spacer(1, 20))
            
            # 분석 요약
            story.append(Paragraph("분석 요약", styles['Heading2']))
            story.append(Paragraph(gemini_result.analysis_summary, styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Top 5 종목
            story.append(Paragraph("Top 5 선정 종목", styles['Heading2']))
            
            for stock in gemini_result.top5_selections:
                # 종목 정보 테이블
                stock_data = [
                    ['순위', str(stock.rank)],
                    ['종목', f"{stock.symbol} - {stock.name}"],
                    ['최종 점수', f"{stock.final_score:.1f}점"],
                    ['선정 이유', stock.selection_reason[:100] + "..." if len(stock.selection_reason) > 100 else stock.selection_reason]
                ]
                
                stock_table = Table(stock_data, colWidths=[1.5*inch, 4*inch])
                stock_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP')
                ]))
                
                story.append(stock_table)
                story.append(Spacer(1, 15))
            
            # 시장 전망
            story.append(Paragraph("시장 전망", styles['Heading2']))
            story.append(Paragraph(gemini_result.market_outlook, styles['Normal']))
            story.append(Spacer(1, 15))
            
            # 위험 요소
            if gemini_result.risk_warnings:
                story.append(Paragraph("위험 요소", styles['Heading2']))
                for warning in gemini_result.risk_warnings:
                    story.append(Paragraph(f"• {warning}", styles['Normal']))
                story.append(Spacer(1, 15))
            
            # PDF 생성
            doc.build(story)
            
            logger.info(f"PDF 리포트 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"PDF 리포트 생성 실패: {e}")
            raise
    
    def _generate_csv_report(self, gemini_result: GeminiAnalysisResult, timestamp: str) -> str:
        """CSV 리포트 생성"""
        try:
            filename = f"top5_report_{timestamp}.csv"
            filepath = os.path.join(self.config.output_directory, filename)
            
            # Top 5 데이터를 DataFrame으로 변환
            data = []
            for stock in gemini_result.top5_selections:
                row = {
                    'Rank': stock.rank,
                    'Symbol': stock.symbol,
                    'Name': stock.name,
                    'Final_Score': stock.final_score,
                    'Selection_Reason': stock.selection_reason,
                    'Technical_Analysis': stock.technical_analysis,
                    'Risk_Assessment': stock.risk_assessment,
                    'Gemini_Reasoning': stock.gemini_reasoning
                }
                
                # 전략별 점수 추가
                for strategy, score in stock.strategy_scores.items():
                    row[f'Strategy_{strategy}'] = score
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            logger.info(f"CSV 리포트 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"CSV 리포트 생성 실패: {e}")
            raise
    
    def _generate_json_report(self, 
                            gemini_result: GeminiAnalysisResult,
                            market_data: Dict[str, List[StockData]],
                            strategy_results: Dict[str, Any],
                            timestamp: str) -> str:
        """JSON 리포트 생성"""
        try:
            filename = f"top5_report_{timestamp}.json"
            filepath = os.path.join(self.config.output_directory, filename)
            
            # 데이터 직렬화 준비
            report_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'analysis_timestamp': gemini_result.analysis_timestamp.isoformat(),
                    'confidence_score': gemini_result.confidence_score,
                    'total_analyzed_stocks': sum(len(stocks) for stocks in market_data.values())
                },
                'gemini_analysis': {
                    'analysis_summary': gemini_result.analysis_summary,
                    'market_outlook': gemini_result.market_outlook,
                    'risk_warnings': gemini_result.risk_warnings,
                    'alternative_candidates': gemini_result.alternative_candidates
                },
                'top5_selections': [
                    {
                        'rank': stock.rank,
                        'symbol': stock.symbol,
                        'name': stock.name,
                        'final_score': stock.final_score,
                        'selection_reason': stock.selection_reason,
                        'strategy_scores': stock.strategy_scores,
                        'technical_analysis': stock.technical_analysis,
                        'risk_assessment': stock.risk_assessment,
                        'gemini_reasoning': stock.gemini_reasoning
                    }
                    for stock in gemini_result.top5_selections
                ],
                'market_summary': self._calculate_market_summary(market_data, strategy_results)
            }
            
            # JSON 파일 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON 리포트 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"JSON 리포트 생성 실패: {e}")
            raise
    
    def _generate_chart_images(self, 
                             gemini_result: GeminiAnalysisResult,
                             market_data: Dict[str, List[StockData]],
                             strategy_results: Dict[str, Any],
                             timestamp: str) -> str:
        """차트 이미지 생성"""
        try:
            charts_dir = os.path.join(self.config.output_directory, f"charts_{timestamp}")
            os.makedirs(charts_dir, exist_ok=True)
            
            # 1. Top 5 점수 비교 차트
            self._create_top5_comparison_chart(gemini_result, charts_dir)
            
            # 2. 전략별 점수 분포 차트
            self._create_strategy_distribution_chart(gemini_result, charts_dir)
            
            # 3. 시장별 분석 차트
            self._create_market_analysis_chart(market_data, charts_dir)
            
            # 4. 신뢰도 게이지 차트
            self._create_confidence_gauge_chart(gemini_result, charts_dir)
            
            logger.info(f"차트 이미지 생성 완료: {charts_dir}")
            return charts_dir
            
        except Exception as e:
            logger.error(f"차트 이미지 생성 실패: {e}")
            raise
    
    def _create_top5_comparison_chart(self, gemini_result: GeminiAnalysisResult, output_dir: str):
        """Top 5 점수 비교 차트"""
        try:
            symbols = [stock.symbol for stock in gemini_result.top5_selections]
            scores = [stock.final_score for stock in gemini_result.top5_selections]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(symbols, scores, color=plt.cm.viridis(np.linspace(0, 1, len(symbols))))
            
            plt.title('Top 5 종목 점수 비교', fontsize=16, fontweight='bold')
            plt.xlabel('종목', fontsize=12)
            plt.ylabel('점수', fontsize=12)
            plt.ylim(0, 100)
            
            # 점수 라벨 추가
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top5_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Top 5 비교 차트 생성 실패: {e}")
    
    def _create_strategy_distribution_chart(self, gemini_result: GeminiAnalysisResult, output_dir: str):
        """전략별 점수 분포 차트"""
        try:
            # 전략별 점수 수집
            strategy_data = {}
            for stock in gemini_result.top5_selections:
                for strategy, score in stock.strategy_scores.items():
                    if strategy not in strategy_data:
                        strategy_data[strategy] = []
                    strategy_data[strategy].append(score)
            
            if strategy_data:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 박스플롯 생성
                data_to_plot = [scores for scores in strategy_data.values()]
                labels = list(strategy_data.keys())
                
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # 색상 설정
                colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                plt.title('전략별 점수 분포', fontsize=16, fontweight='bold')
                plt.xlabel('투자 전략', fontsize=12)
                plt.ylabel('점수', fontsize=12)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'strategy_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            logger.error(f"전략별 분포 차트 생성 실패: {e}")
    
    def _create_market_analysis_chart(self, market_data: Dict[str, List[StockData]], output_dir: str):
        """시장별 분석 차트"""
        try:
            markets = list(market_data.keys())
            stock_counts = [len(stocks) for stocks in market_data.values()]
            
            plt.figure(figsize=(10, 6))
            plt.pie(stock_counts, labels=markets, autopct='%1.1f%%', startangle=90)
            plt.title('시장별 분석 종목 수', fontsize=16, fontweight='bold')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'market_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"시장별 분석 차트 생성 실패: {e}")
    
    def _create_confidence_gauge_chart(self, gemini_result: GeminiAnalysisResult, output_dir: str):
        """신뢰도 게이지 차트"""
        try:
            confidence = gemini_result.confidence_score
            
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
            
            # 게이지 생성
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            # 배경
            ax.fill_between(theta, 0, r, alpha=0.3, color='lightgray')
            
            # 신뢰도에 따른 색상
            if confidence >= 80:
                color = 'green'
            elif confidence >= 60:
                color = 'orange'
            else:
                color = 'red'
            
            # 신뢰도 표시
            confidence_theta = np.linspace(0, np.pi * confidence / 100, 50)
            confidence_r = np.ones_like(confidence_theta)
            ax.fill_between(confidence_theta, 0, confidence_r, alpha=0.7, color=color)
            
            # 설정
            ax.set_ylim(0, 1)
            ax.set_theta_zero_location('W')
            ax.set_theta_direction(1)
            ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180], 
                            ['100%', '83%', '67%', '50%', '33%', '17%', '0%'])
            ax.set_rgrids([])
            ax.set_title(f'분석 신뢰도: {confidence:.1f}%', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confidence_gauge.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"신뢰도 게이지 차트 생성 실패: {e}")
    
    def _calculate_market_summary(self, 
                                market_data: Dict[str, List[StockData]], 
                                strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """시장별 요약 계산"""
        summary = {}
        
        try:
            for market, stocks in market_data.items():
                if stocks:
                    # 기본 통계
                    total_stocks = len(stocks)
                    avg_price = np.mean([stock.price for stock in stocks])
                    
                    # 임시로 점수 계산 (실제로는 전략 결과에서 가져와야 함)
                    avg_score = 75.0  # 기본값
                    top_stock = stocks[0].symbol if stocks else "N/A"
                    
                    summary[market] = {
                        'total_stocks': total_stocks,
                        'avg_price': avg_price,
                        'avg_score': avg_score,
                        'top_stock': top_stock
                    }
        
        except Exception as e:
            logger.error(f"시장별 요약 계산 실패: {e}")
        
        return summary 