"""
프로젝트의 코드 품질을 분석하고 리포트를 생성하는 모듈
"""
import subprocess
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class CodeQualityReport:
    """코드 품질 분석 결과를 담는 데이터 클래스"""
    total_files: int = 0
    performance_score: float = 100.0
    syntax_errors: List[Dict[str, Any]] = field(default_factory=list)
    complexity_issues: List[Dict[str, Any]] = field(default_factory=list)
    code_smells: List[Dict[str, Any]] = field(default_factory=list)
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class CodeAnalyzer:
    """
    pylint를 사용하여 코드 품질을 분석하고 개선점을 제안합니다.
    """
    def __init__(self, target_directory: str = '.'):
        self.target_directory = target_directory

    def _run_pylint(self) -> List[Dict[str, Any]]:
        """pylint를 실행하고 결과를 JSON 형식으로 반환합니다."""
        output_file = "pylint_report.json"
        command = [
            "pylint",
            self.target_directory,
            "--output-format=json",
            f"--rcfile=.pylintrc", # 프로젝트 루트의 pylint 설정 파일 사용
            "--disable=C0114,C0115,C0116", # 문서화 관련 경고 비활성화
        ]
        
        try:
            # JSON 파일로 직접 출력
            with open(output_file, 'w', encoding='utf-8') as f:
                subprocess.run(command, stdout=f, check=True, text=True, encoding='utf-8')

            # 파일에서 결과 읽기
            with open(output_file, 'r', encoding='utf-8') as f:
                # 파일이 비어있을 경우 빈 리스트 반환
                content = f.read()
                if not content:
                    return []
                return json.loads(content)

        except FileNotFoundError:
            print("❌ pylint가 설치되지 않았습니다. `pip install pylint`로 설치해주세요.")
            return []
        except subprocess.CalledProcessError as e:
            print(f"pylint 실행 중 오류 발생: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"pylint 결과 파싱 실패: {e}")
            return []
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def analyze(self) -> CodeQualityReport:
        """코드 품질을 분석하고 리포트를 생성합니다."""
        report = CodeQualityReport()
        pylint_results = self._run_pylint()

        if not pylint_results:
            report.recommendations.append("pylint 분석을 실행할 수 없습니다. 설치를 확인하세요.")
            return report

        # 파일 수 계산 (중복 제거)
        report.total_files = len(set(item['path'] for item in pylint_results))
        
        # 점수 계산 로직 (pylint의 global score 대신 자체 로직으로 간소화)
        total_issues = len(pylint_results)
        # 기본 점수 100점에서 이슈 1개당 0.5점씩 감점 (최소 0점)
        report.performance_score = max(0, 100 - (total_issues * 0.5))

        for item in pylint_results:
            message = {
                "file": item['path'],
                "line": item['line'],
                "message": item['message'],
                "symbol": item['symbol']
            }
            
            if item['type'] == 'error' or item['type'] == 'fatal':
                report.syntax_errors.append(message)
            elif item['symbol'] == 'too-many-branches' or item['symbol'] == 'too-many-statements':
                report.complexity_issues.append(message)
            elif item['type'] == 'convention' or item['type'] == 'refactor':
                report.code_smells.append(message)
            elif 'security' in item['symbol']: # 예시
                report.security_issues.append(message)
        
        self._generate_recommendations(report)
        return report

    def _generate_recommendations(self, report: CodeQualityReport):
        """분석 리포트를 기반으로 개선 권장사항을 생성합니다."""
        if report.syntax_errors:
            report.recommendations.append(f"🚨 {len(report.syntax_errors)}개의 심각한 구문 오류를 해결해야 합니다.")
        
        if len(report.complexity_issues) > 3:
            report.recommendations.append(f"🤔 {len(report.complexity_issues)}개의 복잡도 높은 코드가 있습니다. 리팩토링을 고려하세요.")
        
        if report.performance_score < 80:
            report.recommendations.append(f"📉 전체 성능 점수({report.performance_score:.1f})가 낮습니다. 코드 스멜을 줄여보세요.")
        
        if not report.recommendations:
            report.recommendations.append("🎉 코드가 매우 깔끔합니다! 좋은 상태를 유지하세요.")

async def main():
    """테스트용 메인 함수"""
    print("코드 품질 분석기 테스트 시작...")
    analyzer = CodeAnalyzer(target_directory="personal_blackrock") # 특정 디렉토리 분석
    report = analyzer.analyze()
    
    print("\n--- 코드 품질 분석 리포트 ---")
    print(f"분석 파일 수: {report.total_files}개")
    print(f"성능 점수: {report.performance_score:.1f}/100")
    print(f"구문 오류: {len(report.syntax_errors)}개")
    print(f"복잡도 이슈: {len(report.complexity_issues)}개")
    print("\n권장사항:")
    for rec in report.recommendations:
        print(f"- {rec}")

if __name__ == "__main__":
    asyncio.run(main()) 