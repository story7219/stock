import os
import asyncio
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
import ast
import re
from dataclasses import dataclass, asdict
import google.generativeai as genai

# 필수 import 추가
try:
    from quality_analyzer import CodeQualityAnalyzer, QualityReport, CodeMetrics
except ImportError as e:
    print(f"❌ 필수 모듈 import 실패: {e}")
    print("💡 quality_analyzer.py 파일이 같은 디렉토리에 있는지 확인하세요")
    exit(1)

logger = logging.getLogger(__name__)

@dataclass
class RefactoringProposal:
    """리팩토링 제안 데이터 클래스"""
    file_path: str
    issue_type: str
    description: str
    original_code: str
    proposed_code: str
    confidence: float  # 0.0 ~ 1.0
    risk_level: str    # "LOW", "MEDIUM", "HIGH"
    explanation: str

@dataclass
class RefactoringSession:
    """리팩토링 세션 데이터 클래스"""
    session_id: str
    timestamp: str
    proposals: List[RefactoringProposal]
    approved_count: int = 0
    rejected_count: int = 0
    applied_count: int = 0

class AutoRefactoringSystem:
    """반자동 리팩토링 시스템 (에러 처리 강화)"""
    
    def __init__(self):
        try:
            self.analyzer = CodeQualityAnalyzer()
            self.backup_dir = Path("code_backups")
            self.proposals_dir = Path("refactoring_proposals")
            self.backup_dir.mkdir(exist_ok=True)
            self.proposals_dir.mkdir(exist_ok=True)
            
            # Gemini 모델 설정
            self.setup_gemini()
            
        except Exception as e:
            logger.error(f"AutoRefactoringSystem 초기화 실패: {e}")
            raise
    
    def setup_gemini(self):
        """Gemini API 설정 - 에러 처리 강화"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
            
            genai.configure(api_key=api_key)
            
            # Gemini 2.0 Flash 모델 사용
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("🚀 리팩토링용 Gemini 2.0 Flash 모델 설정 완료")
            
            # 연결 테스트
            test_response = self.model.generate_content("테스트")
            if test_response and test_response.text:
                logger.info("✅ Gemini 2.0 Flash API 연결 확인 완료")
            else:
                raise Exception("Gemini API 응답 없음")
            
        except Exception as e:
            logger.error(f"Gemini API 설정 실패: {e}")
            self.model = None
            raise

    async def safe_gemini_call(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """안전한 Gemini API 호출"""
        if not self.model:
            logger.error("Gemini 모델이 설정되지 않았습니다")
            return None
            
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content, prompt
                )
                
                if response and response.text:
                    return response.text
                else:
                    logger.warning(f"Gemini API 응답 없음 (시도 {attempt + 1}/{max_retries})")
                    
            except Exception as e:
                logger.error(f"Gemini API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 지수 백오프
                
        logger.error("모든 Gemini API 호출 시도 실패")
        return None

    def create_backup(self) -> str:
        """전체 프로젝트 백업 생성"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{timestamp}"
            
            project_root = Path.cwd()
            exclude_dirs = {
                '__pycache__', '.git', 'venv', 'env', 
                'node_modules', '.pytest_cache', 'code_backups'
            }
            
            backup_path.mkdir(parents=True, exist_ok=True)
            
            for item in project_root.iterdir():
                if item.name not in exclude_dirs:
                    try:
                        if item.is_file():
                            shutil.copy2(item, backup_path / item.name)
                        elif item.is_dir() and not any(exclude_dir in item.parts for exclude_dir in exclude_dirs):
                            shutil.copytree(item, backup_path / item.name, 
                                          ignore=shutil.ignore_patterns(*exclude_dirs))
                    except Exception as e:
                        logger.warning(f"백업 중 파일 건너뜀 {item}: {e}")
            
            logger.info(f"백업 생성 완료: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"백업 생성 실패: {e}")
            raise

    async def generate_refactoring_proposals(self, report: QualityReport) -> List[RefactoringProposal]:
        """리팩토링 제안 생성"""
        proposals = []
        
        logger.info("리팩토링 제안 생성 중...")
        
        total_files = len(report.file_metrics)
        processed_files = 0
        
        for metric in report.file_metrics:
            try:
                file_proposals = await self.analyze_file_for_refactoring(metric)
                proposals.extend(file_proposals)
                
                processed_files += 1
                if processed_files % 10 == 0 or processed_files == total_files:
                    logger.info(f"진행률: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"파일 처리 실패 {metric.file_path}: {e}")
                processed_files += 1
        
        proposals.sort(key=lambda p: (p.risk_level, -p.confidence))
        logger.info(f"총 {len(proposals)}개의 리팩토링 제안 생성 완료")
        return proposals

    async def analyze_file_for_refactoring(self, metric: CodeMetrics) -> List[RefactoringProposal]:
        """개별 파일 리팩토링 분석"""
        proposals = []
        
        try:
            if not os.path.exists(metric.file_path):
                return proposals
            
            with open(metric.file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            if not file_content.strip():
                return proposals
            
            # 각 분석 유형별로 안전하게 처리
            analysis_tasks = [
                ("매직 넘버", lambda: any("매직 넘버" in smell for smell in metric.code_smells), 
                 self.propose_magic_number_refactoring),
                ("긴 함수", lambda: any("긴 함수" in smell for smell in metric.code_smells), 
                 self.propose_function_splitting),
                ("높은 복잡도", lambda: metric.complexity > 15, 
                 self.propose_complexity_reduction),
                ("보안 이슈", lambda: bool(metric.security_issues), 
                 lambda fp, fc: self.propose_security_fixes(fp, fc, metric.security_issues))
            ]
            
            for task_name, condition, task_func in analysis_tasks:
                try:
                    if condition():
                        task_proposals = await task_func(metric.file_path, file_content)
                        proposals.extend(task_proposals)
                        logger.debug(f"{task_name} 분석 완료: {len(task_proposals)}개 제안")
                except Exception as e:
                    logger.error(f"{task_name} 분석 실패 {metric.file_path}: {e}")
            
        except Exception as e:
            logger.error(f"파일 분석 전체 실패 {metric.file_path}: {e}")
        
        return proposals

    async def propose_magic_number_refactoring(self, file_path: str, content: str) -> List[RefactoringProposal]:
        """매직 넘버 리팩토링 제안"""
        proposals = []
        
        try:
            magic_numbers = re.findall(r'\b(\d{3,})\b', content)
            
            if len(magic_numbers) > 3:
                prompt = f"""
Python 코드에서 매직 넘버를 상수로 추출하는 리팩토링을 제안해주세요:

발견된 매직 넘버: {list(set(magic_numbers))}

간단한 예시로 답변해주세요:
1. 문제점 설명
2. 상수 정의 예시
3. 개선된 코드 예시

한국어로 답변해주세요.
"""
                
                response_text = await self.safe_gemini_call(prompt)
                
                if response_text:
                    proposal = RefactoringProposal(
                        file_path=file_path,
                        issue_type="매직 넘버",
                        description=f"{len(set(magic_numbers))}개의 매직 넘버를 상수로 추출",
                        original_code=content[:300] + "...",
                        proposed_code=response_text,
                        confidence=0.8,
                        risk_level="LOW",
                        explanation="매직 넘버를 의미있는 상수로 변경하여 가독성 향상"
                    )
                    proposals.append(proposal)
                    
        except Exception as e:
            logger.error(f"매직 넘버 분석 실패: {e}")
        
        return proposals

    async def propose_function_splitting(self, file_path: str, content: str) -> List[RefactoringProposal]:
        """긴 함수 분할 제안"""
        proposals = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_lines = getattr(node, 'end_lineno', node.lineno + 10) - node.lineno
                    
                    if func_lines > 50:
                        prompt = f"""
함수 '{node.name}'이 {func_lines}줄로 너무 깁니다.
이 함수를 작은 함수들로 분할하는 방법을 간단히 제안해주세요.

한국어로 답변해주세요.
"""
                        
                        response = await self.safe_gemini_call(prompt)
                        
                        if response:
                            proposal = RefactoringProposal(
                                file_path=file_path,
                                issue_type="긴 함수",
                                description=f"함수 '{node.name}' ({func_lines}줄) 분할 제안",
                                original_code=f"def {node.name}(...): # {func_lines}줄",
                                proposed_code=response,
                                confidence=0.7,
                                risk_level="MEDIUM",
                                explanation="긴 함수를 작은 단위로 분할하여 가독성 향상"
                            )
                            proposals.append(proposal)
                            
        except Exception as e:
            logger.error(f"함수 분할 분석 실패: {e}")
        
        return proposals

    async def propose_complexity_reduction(self, file_path: str, content: str) -> List[RefactoringProposal]:
        """복잡도 감소 제안"""
        proposals = []
        
        try:
            prompt = f"""
Python 코드의 복잡도가 높습니다. 
복잡도를 줄이는 간단한 방법을 제안해주세요.

한국어로 답변해주세요.
"""
            
            response = await self.safe_gemini_call(prompt)
            
            if response:
                proposal = RefactoringProposal(
                    file_path=file_path,
                    issue_type="높은 복잡도",
                    description="복잡도 감소를 위한 구조 개선",
                    original_code=content[:200] + "...",
                    proposed_code=response,
                    confidence=0.6,
                    risk_level="HIGH",
                    explanation="복잡한 로직을 단순화하여 이해하기 쉬운 코드로 개선"
                )
                proposals.append(proposal)
                
        except Exception as e:
            logger.error(f"복잡도 분석 실패: {e}")
        
        return proposals

    async def propose_security_fixes(self, file_path: str, content: str, security_issues: List[str]) -> List[RefactoringProposal]:
        """보안 이슈 해결 제안"""
        proposals = []
        
        try:
            for issue in security_issues:
                try:
                    prompt = f"""
보안 이슈가 발견되었습니다: {issue}

이 보안 이슈를 해결하는 간단한 방법을 제안해주세요.

한국어로 답변해주세요.
"""
                    
                    response = await self.safe_gemini_call(prompt)
                    
                    if response:
                        proposal = RefactoringProposal(
                            file_path=file_path,
                            issue_type="보안 이슈",
                            description=f"보안 이슈 해결: {issue}",
                            original_code=content[:200] + "...",
                            proposed_code=response,
                            confidence=0.9,
                            risk_level="HIGH",
                            explanation=f"보안 취약점 '{issue}' 해결로 시스템 보안 강화"
                        )
                        proposals.append(proposal)
                        
                except Exception as e:
                    logger.error(f"개별 보안 이슈 분석 실패 ({issue}): {e}")
                
        except Exception as e:
            logger.error(f"보안 분석 전체 실패: {e}")
        
        return proposals

    def save_refactoring_session(self, session: RefactoringSession):
        """리팩토링 세션 저장"""
        try:
            session_file = self.proposals_dir / f"session_{session.session_id}.json"
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, ensure_ascii=False, indent=2)
            
            logger.info(f"리팩토링 세션 저장: {session_file}")
            
        except Exception as e:
            logger.error(f"세션 저장 실패: {e}")

    def generate_interactive_html(self, session: RefactoringSession) -> str:
        """간단한 HTML 리포트 생성"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>리팩토링 제안 - {session.session_id}</title>
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; margin: 20px; }}
        .proposal {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .risk-low {{ border-left: 5px solid green; }}
        .risk-medium {{ border-left: 5px solid orange; }}
        .risk-high {{ border-left: 5px solid red; }}
        .code {{ background: #f5f5f5; padding: 10px; border-radius: 3px; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <h1>🔧 리팩토링 제안</h1>
    <p>세션 ID: {session.session_id}</p>
    <p>총 제안 수: {len(session.proposals)}개</p>
    
    {''.join([f'''
    <div class="proposal risk-{proposal.risk_level.lower()}">
        <h3>📁 {proposal.file_path}</h3>
        <p><strong>유형:</strong> {proposal.issue_type} ({proposal.risk_level} 위험도)</p>
        <p><strong>설명:</strong> {proposal.description}</p>
        <p><strong>신뢰도:</strong> {proposal.confidence:.0%}</p>
        <details>
            <summary>제안 내용 보기</summary>
            <div class="code">{proposal.proposed_code}</div>
        </details>
    </div>
    ''' for proposal in session.proposals])}
</body>
</html>
"""
            
            html_file = self.proposals_dir / f"review_{session.session_id}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML 리포트 생성: {html_file}")
            return str(html_file)
            
        except Exception as e:
            logger.error(f"HTML 생성 실패: {e}")
            return ""

    async def run_semi_automatic_refactoring(self, report: QualityReport) -> RefactoringSession:
        """반자동 리팩토링 실행"""
        try:
            logger.info("=== 반자동 리팩토링 시작 ===")
            
            # 1. 백업 생성
            backup_path = self.create_backup()
            
            # 2. 리팩토링 제안 생성
            proposals = await self.generate_refactoring_proposals(report)
            
            # 3. 세션 생성
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            session = RefactoringSession(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                proposals=proposals
            )
            
            # 4. 세션 저장
            self.save_refactoring_session(session)
            
            # 5. HTML 생성
            html_file = self.generate_interactive_html(session)
            
            logger.info(f"=== 리팩토링 제안 준비 완료 ===")
            logger.info(f"📊 총 {len(proposals)}개 제안 생성")
            logger.info(f"🌐 검토 페이지: {html_file}")
            logger.info(f"💾 백업 위치: {backup_path}")
            
            # 브라우저에서 열기 시도
            if html_file:
                try:
                    import webbrowser
                    webbrowser.open(f"file://{Path(html_file).absolute()}")
                    logger.info("🌐 브라우저에서 검토 페이지를 열었습니다")
                except Exception as e:
                    logger.info(f"브라우저 자동 열기 실패: {e}")
            
            return session
            
        except Exception as e:
            logger.error(f"반자동 리팩토링 실패: {e}")
            raise

# 즉시 실행 함수
async def run_semi_auto_refactoring():
    """반자동 리팩토링 즉시 실행"""
    try:
        print("🚀 반자동 리팩토링 시스템 시작")
        
        # 1. 품질 분석 실행
        print("📊 코드 품질 분석 중...")
        analyzer = CodeQualityAnalyzer()
        report = await analyzer.run_quality_analysis()
        
        # 2. 리팩토링 시스템 실행
        print("🤖 리팩토링 제안 생성 중...")
        refactoring_system = AutoRefactoringSystem()
        session = await refactoring_system.run_semi_automatic_refactoring(report)
        
        print(f"\n🎯 반자동 리팩토링 준비 완료!")
        print(f"📊 총 제안 수: {len(session.proposals)}")
        print(f"🌐 검토 페이지가 생성되었습니다")
        print(f"💡 제안을 검토하고 승인/거부를 선택하세요")
        
        return session
        
    except Exception as e:
        logger.error(f"반자동 리팩토링 실행 실패: {e}")
        print(f"❌ 실행 실패: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "refactor":
        asyncio.run(run_semi_auto_refactoring())
    else:
        print("사용법: python auto_refactoring_system.py refactor") 