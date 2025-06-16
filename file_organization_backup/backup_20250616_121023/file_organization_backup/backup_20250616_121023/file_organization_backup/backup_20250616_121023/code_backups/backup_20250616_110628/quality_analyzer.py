import os
import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import logging
from dataclasses import dataclass, asdict
import google.generativeai as genai
from pathlib import Path
import subprocess
import ast

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quality_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CodeMetrics:
    """코드 품질 메트릭 데이터 클래스"""
    file_path: str
    lines_of_code: int
    complexity: int
    maintainability_index: float
    test_coverage: float
    code_smells: List[str]
    security_issues: List[str]
    performance_issues: List[str]

@dataclass
class QualityReport:
    """품질 분석 보고서 데이터 클래스"""
    timestamp: str
    overall_score: float
    file_metrics: List[CodeMetrics]
    gemini_analysis: str
    recommendations: List[str]
    trend_analysis: str

class CodeQualityAnalyzer:
    """코드 품질 분석기 클래스"""
    
    def __init__(self):
        self.setup_gemini()
        self.project_root = Path.cwd()
        self.reports_dir = self.project_root / "quality_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
    def setup_gemini(self):
        """Gemini API 설정 - 1.5 Flash 모델 고정"""
        try:
            # .env 파일에서 API 키 로드
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
            
            genai.configure(api_key=api_key)
            
            # Gemini 1.5 Flash 모델로 고정
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("✅ Gemini 1.5 Flash 모델 설정 완료 (가성비 최적화)")
            
            # 테스트 요청으로 연결 확인
            test_response = self.model.generate_content("테스트")
            if test_response and test_response.text:
                logger.info("✅ Gemini 1.5 Flash API 연결 확인 완료")
            else:
                raise Exception("Gemini API 응답 없음")
            
        except Exception as e:
            logger.error(f"Gemini API 설정 실패: {e}")
            self.model = None
            raise

    def analyze_file_complexity(self, file_path: str) -> int:
        """파일의 순환 복잡도 계산"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            complexity = 1  # 기본 복잡도
            
            for node in ast.walk(tree):
                # 조건문, 반복문, 예외처리 등으로 복잡도 증가
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                    
            return complexity
            
        except Exception as e:
            logger.warning(f"복잡도 분석 실패 {file_path}: {e}")
            return 0

    def count_lines_of_code(self, file_path: str) -> int:
        """실제 코드 라인 수 계산 (주석, 빈 줄 제외)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            code_lines = 0
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    code_lines += 1
                    
            return code_lines
            
        except Exception as e:
            logger.warning(f"코드 라인 계산 실패 {file_path}: {e}")
            return 0

    def detect_code_smells(self, file_path: str) -> List[str]:
        """코드 스멜 탐지"""
        smells = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # 긴 함수 탐지
            in_function = False
            function_lines = 0
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('def '):
                    in_function = True
                    function_lines = 0
                elif in_function:
                    if stripped and not stripped.startswith('#'):
                        function_lines += 1
                    if stripped.startswith('def ') or stripped.startswith('class '):
                        if function_lines > 50:
                            smells.append("긴 함수 발견 (50줄 초과)")
                        in_function = False
            
            # 중복 코드 패턴 탐지
            if content.count('if __name__ == "__main__"') > 1:
                smells.append("중복된 메인 블록")
            
            # 매직 넘버 탐지
            import re
            magic_numbers = re.findall(r'\b\d{3,}\b', content)
            if len(magic_numbers) > 5:
                smells.append(f"매직 넘버 과다 사용 ({len(magic_numbers)}개)")
                
        except Exception as e:
            logger.warning(f"코드 스멜 탐지 실패 {file_path}: {e}")
            
        return smells

    def detect_security_issues(self, file_path: str) -> List[str]:
        """보안 이슈 탐지"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 하드코딩된 비밀번호/키 탐지
            import re
            
            # API 키 패턴
            if re.search(r'["\'](?:api_key|password|secret)["\']?\s*[:=]\s*["\'][^"\']{10,}["\']', content, re.IGNORECASE):
                issues.append("하드코딩된 API 키 또는 비밀번호 의심")
            
            # SQL 인젝션 위험
            if 'execute(' in content and '%' in content:
                issues.append("SQL 인젝션 위험 가능성")
            
            # eval() 사용
            if 'eval(' in content:
                issues.append("eval() 함수 사용으로 인한 보안 위험")
                
        except Exception as e:
            logger.warning(f"보안 이슈 탐지 실패 {file_path}: {e}")
            
        return issues

    def analyze_python_files(self) -> List[CodeMetrics]:
        """Python 파일들 분석"""
        metrics = []
        
        for py_file in self.project_root.glob("**/*.py"):
            if py_file.name.startswith('.') or 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                file_metrics = CodeMetrics(
                    file_path=str(py_file.relative_to(self.project_root)),
                    lines_of_code=self.count_lines_of_code(str(py_file)),
                    complexity=self.analyze_file_complexity(str(py_file)),
                    maintainability_index=self.calculate_maintainability_index(str(py_file)),
                    test_coverage=0.0,  # 실제 테스트 커버리지는 별도 도구 필요
                    code_smells=self.detect_code_smells(str(py_file)),
                    security_issues=self.detect_security_issues(str(py_file)),
                    performance_issues=[]  # 성능 이슈는 프로파일링 도구 필요
                )
                metrics.append(file_metrics)
                
            except Exception as e:
                logger.error(f"파일 분석 실패 {py_file}: {e}")
                
        return metrics

    def calculate_maintainability_index(self, file_path: str) -> float:
        """유지보수성 지수 계산"""
        try:
            loc = self.count_lines_of_code(file_path)
            complexity = self.analyze_file_complexity(file_path)
            
            if loc == 0:
                return 0.0
            
            # 간단한 유지보수성 지수 계산
            # 실제로는 더 복잡한 공식 사용
            maintainability = max(0, 100 - (complexity * 2) - (loc / 10))
            return round(maintainability, 2)
            
        except Exception:
            return 0.0

    async def get_gemini_analysis(self, metrics: List[CodeMetrics]) -> str:
        """Gemini API를 통한 고급 분석 (에러 처리 강화)"""
        try:
            # 메트릭 데이터를 텍스트로 변환
            analysis_data = {
                "총_파일_수": len(metrics),
                "총_코드_라인": sum(m.lines_of_code for m in metrics),
                "평균_복잡도": round(sum(m.complexity for m in metrics) / len(metrics), 2) if metrics else 0,
                "평균_유지보수성": round(sum(m.maintainability_index for m in metrics) / len(metrics), 2) if metrics else 0,
                "총_코드_스멜": sum(len(m.code_smells) for m in metrics),
                "총_보안_이슈": sum(len(m.security_issues) for m in metrics),
                "문제가_많은_파일들": [
                    {
                        "파일": m.file_path,
                        "복잡도": m.complexity,
                        "코드스멜": len(m.code_smells),
                        "보안이슈": len(m.security_issues)
                    }
                    for m in sorted(metrics, key=lambda x: x.complexity + len(x.code_smells), reverse=True)[:5]
                ]
            }
            
            prompt = f"""
다음은 Python 프로젝트의 코드 품질 분석 결과입니다:

{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

이 데이터를 바탕으로 다음 사항들을 분석해주세요:

1. 전체적인 코드 품질 평가 (1-10점)
2. 주요 문제점과 개선 방향
3. 우선순위별 리팩토링 권장사항
4. 코드 아키텍처 개선 제안
5. 성능 최적화 방안
6. 보안 강화 방안

한국어로 상세하고 실용적인 분석을 제공해주세요.
"""

            # 여러 방법으로 API 호출 시도
            response = None
            
            # 방법 1: 일반적인 방법
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content, prompt
                )
            except Exception as e1:
                logger.warning(f"첫 번째 시도 실패: {e1}")
                
                # 방법 2: 동기 방식으로 시도
                try:
                    response = self.model.generate_content(prompt)
                except Exception as e2:
                    logger.warning(f"두 번째 시도 실패: {e2}")
                    
                    # 방법 3: 짧은 프롬프트로 시도
                    try:
                        short_prompt = f"Python 코드 {len(metrics)}개 파일 분석 결과를 요약해주세요. 평균 복잡도: {analysis_data['평균_복잡도']}, 코드 스멜: {analysis_data['총_코드_스멜']}개"
                        response = self.model.generate_content(short_prompt)
                    except Exception as e3:
                        logger.error(f"모든 시도 실패: {e3}")
                        raise e3
            
            if response and response.text:
                return response.text
            else:
                raise ValueError("Gemini API에서 응답을 받지 못했습니다.")
            
        except Exception as e:
            logger.error(f"Gemini 분석 실패: {e}")
            # 상세한 오류 정보 로깅
            logger.error(f"오류 타입: {type(e).__name__}")
            logger.error(f"오류 메시지: {str(e)}")
            raise e

    def generate_recommendations(self, metrics: List[CodeMetrics]) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 복잡도 기반 권장사항
        high_complexity_files = [m for m in metrics if m.complexity > 10]
        if high_complexity_files:
            recommendations.append(
                f"높은 복잡도 파일 {len(high_complexity_files)}개 리팩토링 필요"
            )
        
        # 코드 스멜 기반 권장사항
        total_smells = sum(len(m.code_smells) for m in metrics)
        if total_smells > 0:
            recommendations.append(f"총 {total_smells}개의 코드 스멜 해결 필요")
        
        # 보안 이슈 기반 권장사항
        total_security_issues = sum(len(m.security_issues) for m in metrics)
        if total_security_issues > 0:
            recommendations.append(f"총 {total_security_issues}개의 보안 이슈 해결 필요")
        
        # 유지보수성 기반 권장사항
        low_maintainability = [m for m in metrics if m.maintainability_index < 50]
        if low_maintainability:
            recommendations.append(
                f"낮은 유지보수성 파일 {len(low_maintainability)}개 개선 필요"
            )
        
        return recommendations

    def analyze_trends(self) -> str:
        """품질 트렌드 분석"""
        try:
            # 최근 7일간의 보고서 로드
            recent_reports = []
            for report_file in sorted(self.reports_dir.glob("quality_report_*.json"))[-7:]:
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                        recent_reports.append(report_data)
                except Exception as e:
                    logger.warning(f"보고서 로드 실패 {report_file}: {e}")
            
            if len(recent_reports) < 2:
                return "트렌드 분석을 위한 충분한 데이터가 없습니다."
            
            # 점수 변화 분석
            scores = [report['overall_score'] for report in recent_reports]
            if scores[-1] > scores[0]:
                trend = f"품질 점수가 {scores[0]:.1f}에서 {scores[-1]:.1f}로 개선되었습니다."
            elif scores[-1] < scores[0]:
                trend = f"품질 점수가 {scores[0]:.1f}에서 {scores[-1]:.1f}로 하락했습니다."
            else:
                trend = "품질 점수가 안정적으로 유지되고 있습니다."
            
            return trend
            
        except Exception as e:
            logger.error(f"트렌드 분석 실패: {e}")
            return "트렌드 분석 중 오류가 발생했습니다."

    async def run_quality_analysis(self) -> QualityReport:
        """전체 품질 분석 실행"""
        logger.info("코드 품질 분석 시작")
        
        try:
            # 1. 파일 메트릭 분석
            logger.info("파일 메트릭 분석 중...")
            file_metrics = self.analyze_python_files()
            
            # 2. 전체 점수 계산 (Gemini 분석과 독립적으로)
            if file_metrics:
                avg_maintainability = sum(m.maintainability_index for m in file_metrics) / len(file_metrics)
                total_issues = sum(len(m.code_smells) + len(m.security_issues) for m in file_metrics)
                # 점수 계산 로직 개선
                base_score = avg_maintainability
                penalty = min(total_issues * 2, 50)  # 최대 50점 감점
                overall_score = max(0, base_score - penalty)
            else:
                overall_score = 0
            
            # 3. Gemini 고급 분석 (실패해도 전체 분석은 계속)
            logger.info("Gemini AI 분석 중...")
            try:
                gemini_analysis = await self.get_gemini_analysis(file_metrics)
            except Exception as e:
                logger.warning(f"Gemini 분석 실패, 기본 분석으로 대체: {e}")
                gemini_analysis = self.generate_fallback_analysis(file_metrics)
            
            # 4. 권장사항 생성
            recommendations = self.generate_recommendations(file_metrics)
            
            # 5. 트렌드 분석
            trend_analysis = self.analyze_trends()
            
            # 6. 보고서 생성
            report = QualityReport(
                timestamp=datetime.now().isoformat(),
                overall_score=round(overall_score, 2),
                file_metrics=file_metrics,
                gemini_analysis=gemini_analysis,
                recommendations=recommendations,
                trend_analysis=trend_analysis
            )
            
            # 7. 보고서 저장
            self.save_report(report)
            
            logger.info(f"품질 분석 완료 - 전체 점수: {overall_score:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"품질 분석 실패: {e}")
            raise

    def save_report(self, report: QualityReport):
        """보고서 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"quality_report_{timestamp}.json"
            
            # JSON 직렬화 가능한 형태로 변환
            report_dict = asdict(report)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"보고서 저장 완료: {report_file}")
            
            # HTML 보고서도 생성
            self.generate_html_report(report, timestamp)
            
        except Exception as e:
            logger.error(f"보고서 저장 실패: {e}")

    def generate_html_report(self, report: QualityReport, timestamp: str):
        """HTML 보고서 생성"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>코드 품질 분석 보고서 - {timestamp}</title>
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }}
        .score {{ font-size: 2em; font-weight: bold; color: #e74c3c; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
        .file-metric {{ background: #f8f9fa; margin: 10px 0; padding: 10px; border-radius: 4px; }}
        .issue {{ color: #e74c3c; font-weight: bold; }}
        .recommendation {{ background: #d4edda; padding: 10px; margin: 5px 0; border-radius: 4px; }}
        .gemini-analysis {{ background: #e3f2fd; padding: 15px; border-radius: 8px; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>코드 품질 분석 보고서</h1>
        <p>분석 시간: {report.timestamp}</p>
        <div class="score">전체 점수: {report.overall_score}/100</div>
    </div>
    
    <div class="section">
        <h2>📊 파일별 메트릭</h2>
        {''.join([f'''
        <div class="file-metric">
            <h3>{metric.file_path}</h3>
            <p>코드 라인: {metric.lines_of_code} | 복잡도: {metric.complexity} | 유지보수성: {metric.maintainability_index}</p>
            {f'<p class="issue">코드 스멜: {", ".join(metric.code_smells)}</p>' if metric.code_smells else ''}
            {f'<p class="issue">보안 이슈: {", ".join(metric.security_issues)}</p>' if metric.security_issues else ''}
        </div>
        ''' for metric in report.file_metrics])}
    </div>
    
    <div class="section">
        <h2>🤖 Gemini AI 분석</h2>
        <div class="gemini-analysis">{report.gemini_analysis}</div>
    </div>
    
    <div class="section">
        <h2>💡 개선 권장사항</h2>
        {''.join([f'<div class="recommendation">{rec}</div>' for rec in report.recommendations])}
    </div>
    
    <div class="section">
        <h2>📈 트렌드 분석</h2>
        <p>{report.trend_analysis}</p>
    </div>
</body>
</html>
"""
            
            html_file = self.reports_dir / f"quality_report_{timestamp}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML 보고서 생성 완료: {html_file}")
            
        except Exception as e:
            logger.error(f"HTML 보고서 생성 실패: {e}")

    def generate_fallback_analysis(self, metrics: List[CodeMetrics]) -> str:
        """Gemini API 실패 시 대체 분석"""
        if not metrics:
            return "분석할 파일이 없습니다."
        
        total_files = len(metrics)
        total_loc = sum(m.lines_of_code for m in metrics)
        avg_complexity = sum(m.complexity for m in metrics) / total_files
        avg_maintainability = sum(m.maintainability_index for m in metrics) / total_files
        total_smells = sum(len(m.code_smells) for m in metrics)
        total_security = sum(len(m.security_issues) for m in metrics)
        
        # 품질 등급 결정
        if avg_maintainability >= 80:
            grade = "우수"
        elif avg_maintainability >= 60:
            grade = "양호"
        elif avg_maintainability >= 40:
            grade = "보통"
        else:
            grade = "개선 필요"
        
        analysis = f"""
📊 코드 품질 분석 결과 (기본 분석)

🎯 전체 평가: {grade}
📈 평균 유지보수성: {avg_maintainability:.1f}/100
🔢 평균 복잡도: {avg_complexity:.1f}
📁 총 파일 수: {total_files}개
📝 총 코드 라인: {total_loc:,}줄

⚠️ 발견된 이슈:
- 코드 스멜: {total_smells}개
- 보안 이슈: {total_security}개

💡 주요 권장사항:
1. 복잡도가 높은 함수들을 작은 단위로 분리하세요
2. 코드 스멜을 해결하여 가독성을 향상시키세요
3. 보안 이슈를 우선적으로 해결하세요
4. 정기적인 리팩토링을 통해 코드 품질을 유지하세요

📈 개선 방향:
- 함수당 최대 50줄 이하로 유지
- 복잡도 10 이하로 관리
- 주석과 문서화 강화
- 테스트 코드 작성
"""
        
        return analysis

class QualityScheduler:
    """품질 검사 스케줄러"""
    
    def __init__(self):
        self.analyzer = CodeQualityAnalyzer()
        
    async def run_scheduled_analysis(self):
        """스케줄된 분석 실행"""
        try:
            logger.info("=== 자동 품질 검사 시작 ===")
            report = await self.analyzer.run_quality_analysis()
            
            # 중요한 이슈가 있으면 알림
            if report.overall_score < 50:
                logger.warning(f"⚠️ 품질 점수가 낮습니다: {report.overall_score}")
            
            total_issues = sum(len(m.code_smells) + len(m.security_issues) for m in report.file_metrics)
            if total_issues > 10:
                logger.warning(f"⚠️ 총 {total_issues}개의 이슈가 발견되었습니다")
            
            logger.info("=== 자동 품질 검사 완료 ===")
            
        except Exception as e:
            logger.error(f"스케줄된 분석 실패: {e}")

    def start_scheduler(self):
        """스케줄러 시작"""
        # 매일 오전 7시에 실행
        schedule.every().day.at("07:00").do(
            lambda: asyncio.run(self.run_scheduled_analysis())
        )
        
        logger.info("품질 검사 스케줄러 시작 - 매일 오전 7시 실행")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 체크

# 즉시 실행 함수
async def run_immediate_analysis():
    """즉시 분석 실행"""
    analyzer = CodeQualityAnalyzer()
    report = await analyzer.run_quality_analysis()
    print(f"\n✅ 분석 완료!")
    print(f"📊 전체 점수: {report.overall_score}/100")
    print(f"📁 분석된 파일 수: {len(report.file_metrics)}")
    print(f"💡 권장사항 수: {len(report.recommendations)}")
    return report

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "now":
        # 즉시 실행
        asyncio.run(run_immediate_analysis())
    else:
        # 스케줄러 시작
        scheduler = QualityScheduler()
        scheduler.start_scheduler() 