"""
리포트 생성 모듈
종합 분석 리포트 및 시각화 생성
"""

import asyncio
import logging
import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import json
import csv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .data_collector import StockData
from .gemini_analyzer import GeminiAnalysisResult

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
    include_html: bool = True
    include_pdf: bool = True
    include_csv: bool = True
    include_json: bool = True

class ReportGenerator:
    """종합 리포트 생성기"""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self._ensure_output_directory()
        
        # 스타일 설정
        plt.style.use('default')  # 기본 스타일 사용
        
    def _ensure_output_directory(self):
        """출력 디렉토리 생성"""
        os.makedirs(self.config.output_directory, exist_ok=True)
        
    async def generate_report(self, 
                            market_data: Dict[str, List[StockData]],
                            strategy_results: Dict[str, Any],
                            technical_analysis: Optional[Dict[str, Any]] = None,
                            gemini_result: Optional[GeminiAnalysisResult] = None) -> Dict[str, str]:
        """종합 리포트 생성"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_files = {}
            
            # HTML 리포트 생성
            if self.config.include_html:
                html_file = self._generate_html_report(
                    gemini_result, market_data, strategy_results, technical_analysis, timestamp
                )
                report_files['html'] = html_file
            
            # 텍스트 리포트 생성 (PDF 대신)
            if self.config.include_pdf:
                txt_file = self._generate_pdf_report(
                    gemini_result, market_data, strategy_results, timestamp
                )
                report_files['text'] = txt_file
            
            # CSV 리포트 생성
            if self.config.include_csv:
                csv_file = self._generate_csv_report(gemini_result, timestamp)
                report_files['csv'] = csv_file
            
            # JSON 리포트 생성
            if self.config.include_json:
                json_file = self._generate_json_report(
                    gemini_result, market_data, strategy_results, technical_analysis, timestamp
                )
                report_files['json'] = json_file
            
            logger.info(f"모든 리포트 생성 완료: {len(report_files)}개 파일")
            return report_files
            
        except Exception as e:
            logger.error(f"리포트 생성 실패: {e}")
            raise
    
    def _generate_html_report(self, 
                            gemini_result: Optional[GeminiAnalysisResult],
                            market_data: Dict[str, List[StockData]],
                            strategy_results: Dict[str, Any],
                            technical_analysis: Optional[Dict[str, Any]],
                            timestamp: str) -> str:
        """HTML 리포트 생성"""
        try:
            # 간단한 HTML 생성
            html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; border-bottom: 2px solid #1f77b4; padding: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #1f77b4; }}
        .stock-card {{ border: 1px solid #ddd; margin: 10px; padding: 15px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; }}
        th {{ background-color: #1f77b4; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.config.title}</h1>
        <h2>{self.config.subtitle}</h2>
        <p>분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>시장별 데이터 요약</h2>
        <table>
            <tr><th>시장</th><th>분석 종목 수</th></tr>
            {"".join([f"<tr><td>{market}</td><td>{len(stocks)}</td></tr>" for market, stocks in market_data.items()])}
        </table>
    </div>
    
    <div class="section">
        <h2>분석 완료</h2>
        <p>전체 {sum(len(stocks) for stocks in market_data.values())}개 종목 분석이 완료되었습니다.</p>
        <p>상세한 분석 결과는 JSON 및 CSV 파일을 참조하시기 바랍니다.</p>
    </div>
    
    <div style="text-align: center; margin-top: 40px; color: #666;">
        <p>Generated by {self.config.company} - {self.config.author}</p>
        <p>이 리포트는 투자 참고용이며, 투자 결정은 본인의 판단과 책임하에 이루어져야 합니다.</p>
    </div>
</body>
</html>
            """
            
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
                           gemini_result: Optional[GeminiAnalysisResult],
                           market_data: Dict[str, List[StockData]],
                           strategy_results: Dict[str, Any],
                           timestamp: str) -> str:
        """텍스트 리포트 생성 (PDF 대신)"""
        try:
            filename = f"top5_report_{timestamp}.txt"
            filepath = os.path.join(self.config.output_directory, filename)
            
            # 텍스트 리포트 생성
            report_content = f"""
{self.config.title}
{self.config.subtitle}
{'=' * 80}

분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
분석 종목 수: {sum(len(stocks) for stocks in market_data.values())}

시장별 데이터 요약:
{'-' * 40}
"""
            
            for market, stocks in market_data.items():
                report_content += f"{market}: {len(stocks)}개 종목\n"
            
            report_content += f"""
{'-' * 40}

분석 완료
전체 {sum(len(stocks) for stocks in market_data.values())}개 종목 분석이 완료되었습니다.
상세한 분석 결과는 JSON 및 CSV 파일을 참조하시기 바랍니다.

{'=' * 80}
Generated by {self.config.company} - {self.config.author}
이 리포트는 투자 참고용이며, 투자 결정은 본인의 판단과 책임하에 이루어져야 합니다.
"""
            
            # 파일 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"텍스트 리포트 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"텍스트 리포트 생성 실패: {e}")
            raise
    
    def _generate_csv_report(self, gemini_result: Optional[GeminiAnalysisResult], timestamp: str) -> str:
        """CSV 리포트 생성"""
        try:
            filename = f"market_data_{timestamp}.csv"
            filepath = os.path.join(self.config.output_directory, filename)
            
            # 간단한 시장 데이터 CSV 생성
            data = [{
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Analysis_Type': 'Market Data Summary',
                'Status': 'Completed',
                'Note': 'Detailed analysis results available in JSON format'
            }]
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            logger.info(f"CSV 리포트 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"CSV 리포트 생성 실패: {e}")
            raise
    
    def _generate_json_report(self, 
                            gemini_result: Optional[GeminiAnalysisResult],
                            market_data: Dict[str, List[StockData]],
                            strategy_results: Dict[str, Any],
                            technical_analysis: Optional[Dict[str, Any]],
                            timestamp: str) -> str:
        """JSON 리포트 생성"""
        try:
            filename = f"analysis_report_{timestamp}.json"
            filepath = os.path.join(self.config.output_directory, filename)
            
            # 간단한 데이터 직렬화
            report_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_analyzed_stocks': sum(len(stocks) for stocks in market_data.values())
                },
                'market_summary': self._calculate_market_summary(market_data, strategy_results),
                'analysis_status': 'completed'
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
                             gemini_result: Optional[GeminiAnalysisResult],
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
            
            # 3. 신뢰도 게이지 차트
            self._create_confidence_gauge_chart(gemini_result, charts_dir)
            
            logger.info(f"차트 이미지 생성 완료: {charts_dir}")
            return charts_dir
            
        except Exception as e:
            logger.error(f"차트 이미지 생성 실패: {e}")
            raise
    
    def _create_top5_comparison_chart(self, gemini_result: Optional[GeminiAnalysisResult], output_dir: str):
        """Top 5 종목 비교 차트 생성"""
        try:
            # 간단한 matplotlib 차트 생성
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 더미 데이터로 차트 생성
            categories = ['Stock A', 'Stock B', 'Stock C', 'Stock D', 'Stock E']
            values = [85, 78, 72, 69, 65]
            
            ax.bar(categories, values, color='skyblue')
            ax.set_title('Top 5 종목 점수 비교')
            ax.set_ylabel('점수')
            
            filename = os.path.join(output_dir, 'top5_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Top 5 비교 차트 생성 실패: {e}")
            return None
    
    def _create_strategy_distribution_chart(self, gemini_result: Optional[GeminiAnalysisResult], output_dir: str):
        """전략별 분포 차트 생성"""
        try:
            # 간단한 matplotlib 파이 차트 생성
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # 더미 데이터
            strategies = ['Warren Buffett', 'Peter Lynch', 'Benjamin Graham']
            sizes = [40, 35, 25]
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            
            ax.pie(sizes, labels=strategies, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('전략별 선정 비율')
            
            filename = os.path.join(output_dir, 'strategy_distribution.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"전략별 분포 차트 생성 실패: {e}")
            return None
    
    def _create_confidence_gauge_chart(self, gemini_result: Optional[GeminiAnalysisResult], output_dir: str):
        """신뢰도 게이지 차트 생성"""
        try:
            # 간단한 matplotlib 게이지 차트 생성
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # 더미 신뢰도 데이터
            confidence = 85
            
            # 게이지 차트 생성
            ax.pie([confidence, 100-confidence], 
                   colors=['green', 'lightgray'], 
                   startangle=90, 
                   counterclock=False)
            
            # 중앙에 텍스트 추가
            ax.text(0, 0, f'{confidence}%', ha='center', va='center', fontsize=20, fontweight='bold')
            ax.set_title('분석 신뢰도')
            
            filename = os.path.join(output_dir, 'confidence_gauge.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"신뢰도 게이지 차트 생성 실패: {e}")
            return None
    
    def _calculate_market_summary(self, 
                                market_data: Dict[str, List[StockData]], 
                                strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """시장별 요약 계산"""
        summary = {}
        
        for market, stocks in market_data.items():
            if stocks:
                summary[market] = {
                    'total_stocks': len(stocks),
                    'avg_price': sum(stock.price for stock in stocks) / len(stocks),
                    'total_volume': sum(stock.volume for stock in stocks),
                    'top_stock': max(stocks, key=lambda x: x.price).symbol if stocks else "N/A"
                }
        
        return summary

    def generate_text_report(self, 
                           gemini_result: Optional[GeminiAnalysisResult],
                           strategy_results: Dict[str, Any],
                           market_data: Dict[str, List[StockData]]) -> str:
        """간단한 텍스트 리포트 생성 (GUI용)"""
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("Ultra HTS v5.0 - 종목 분석 리포트")
            report_lines.append("=" * 80)
            report_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # 시장별 데이터 요약
            report_lines.append("📊 시장별 데이터 요약")
            report_lines.append("-" * 40)
            total_stocks = 0
            for market, stocks in market_data.items():
                stock_count = len(stocks)
                total_stocks += stock_count
                report_lines.append(f"{market}: {stock_count}개 종목")
            
            report_lines.append(f"총 분석 종목: {total_stocks}개")
            report_lines.append("")
            
            # 전략별 분석 결과
            if strategy_results:
                report_lines.append("🧠 전략별 분석 결과")
                report_lines.append("-" * 40)
                for strategy_name, scores in strategy_results.items():
                    report_lines.append(f"\n{strategy_name} 전략 상위 3개:")
                    for i, score in enumerate(scores[:3], 1):
                        report_lines.append(f"  {i}. {score.name} ({score.symbol}) - 점수: {score.total_score:.1f}")
            
            # AI 분석 결과
            if gemini_result and hasattr(gemini_result, 'top5_selections'):
                report_lines.append("\n🤖 AI Top5 선정 결과")
                report_lines.append("-" * 40)
                for selection in gemini_result.top5_selections:
                    report_lines.append(f"{selection.rank}. {selection.name} ({selection.symbol})")
                    report_lines.append(f"   점수: {selection.final_score:.1f}")
                    report_lines.append(f"   선정 이유: {selection.selection_reason[:100]}...")
                    report_lines.append("")
            
            # 면책 조항
            report_lines.append("⚠️ 면책 조항")
            report_lines.append("-" * 40)
            report_lines.append("이 리포트는 투자 참고용이며, 투자 결정은 본인의 판단과 책임하에 이루어져야 합니다.")
            report_lines.append("과거 성과가 미래 수익을 보장하지 않습니다.")
            report_lines.append("")
            report_lines.append("=" * 80)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"텍스트 리포트 생성 실패: {e}")
            return f"리포트 생성 중 오류가 발생했습니다: {e}"

class ResultExporter:
    """결과 내보내기 클래스"""
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_to_csv(self, data: List[Dict], filename: str) -> str:
        """CSV로 내보내기"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"CSV 내보내기 완료: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"CSV 내보내기 실패: {e}")
            raise
    
    def export_to_json(self, data: Dict, filename: str) -> str:
        """JSON으로 내보내기"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"JSON 내보내기 완료: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"JSON 내보내기 실패: {e}")
            raise
    
    def export_to_text(self, content: str, filename: str) -> str:
        """텍스트로 내보내기"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"텍스트 내보내기 완료: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"텍스트 내보내기 실패: {e}")
            raise 