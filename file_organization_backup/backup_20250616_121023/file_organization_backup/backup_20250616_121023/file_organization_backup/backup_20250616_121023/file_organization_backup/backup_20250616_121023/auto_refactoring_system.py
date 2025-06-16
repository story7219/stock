import os
import asyncio
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
import ast
import re
from dataclasses import dataclass, asdict
import google.generativeai as genai

from quality_analyzer import CodeQualityAnalyzer, QualityReport, CodeMetrics

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
    """반자동 리팩토링 시스템"""
    
    def __init__(self):
        self.analyzer = CodeQualityAnalyzer()
        self.backup_dir = Path("code_backups")
        self.proposals_dir = Path("refactoring_proposals")
        self.backup_dir.mkdir(exist_ok=True)
        self.proposals_dir.mkdir(exist_ok=True)
        
        # Gemini 모델 설정
        self.setup_gemini()
    
    def setup_gemini(self):
        """Gemini API 설정 - 2.0 Flash 모델 사용 (최신 버전)"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")
            
            genai.configure(api_key=api_key)
            
            # Gemini 2.0 Flash 모델 사용 (최신 성능)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("🚀 리팩토링용 Gemini 2.0 Flash 모델 설정 완료 (최신 성능)")
            
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

    def create_backup(self) -> str:
        """전체 프로젝트 백업 생성"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{timestamp}"
            
            # 현재 프로젝트 디렉토리 백업
            project_root = Path.cwd()
            
            # 백업에서 제외할 디렉토리들
            exclude_dirs = {
                '__pycache__', '.git', 'venv', 'env', 
                'node_modules', '.pytest_cache', 'code_backups'
            }
            
            def should_exclude(path: Path) -> bool:
                return any(exclude_dir in path.parts for exclude_dir in exclude_dirs)
            
            backup_path.mkdir(parents=True, exist_ok=True)
            
            for item in project_root.iterdir():
                if not should_exclude(item) and item.name not in exclude_dirs:
                    if item.is_file():
                        shutil.copy2(item, backup_path / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, backup_path / item.name, 
                                      ignore=shutil.ignore_patterns(*exclude_dirs))
            
            logger.info(f"백업 생성 완료: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"백업 생성 실패: {e}")
            raise

    async def generate_refactoring_proposals(self, report: QualityReport) -> List[RefactoringProposal]:
        """리팩토링 제안 생성 (에러 처리 강화)"""
        proposals = []
        
        logger.info("리팩토링 제안 생성 중...")
        
        # 파일별 처리 진행률 표시
        total_files = len(report.file_metrics)
        processed_files = 0
        
        for metric in report.file_metrics:
            try:
                # 각 파일별로 제안 생성
                file_proposals = await self.analyze_file_for_refactoring(metric)
                proposals.extend(file_proposals)
                
                processed_files += 1
                logger.info(f"진행률: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"파일 처리 실패 {metric.file_path}: {e}")
                processed_files += 1
                # 다음 파일 계속 처리
        
        # 제안들을 위험도와 신뢰도로 정렬
        proposals.sort(key=lambda p: (p.risk_level, -p.confidence))
        
        logger.info(f"총 {len(proposals)}개의 리팩토링 제안 생성 완료")
        return proposals

    async def analyze_file_for_refactoring(self, metric: CodeMetrics) -> List[RefactoringProposal]:
        """개별 파일 리팩토링 분석 (에러 처리 강화)"""
        proposals = []
        
        try:
            # 파일 존재 여부 확인
            if not os.path.exists(metric.file_path):
                logger.warning(f"파일을 찾을 수 없습니다: {metric.file_path}")
                return proposals
            
            with open(metric.file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # 파일이 비어있는 경우 처리
            if not file_content.strip():
                logger.info(f"빈 파일 건너뜀: {metric.file_path}")
                return proposals
            
            # 1. 매직 넘버 리팩토링 (에러 처리 추가)
            try:
                if any("매직 넘버" in smell for smell in metric.code_smells):
                    magic_proposals = await self.propose_magic_number_refactoring(
                        metric.file_path, file_content
                    )
                    proposals.extend(magic_proposals)
            except Exception as e:
                logger.error(f"매직 넘버 분석 실패 {metric.file_path}: {e}")
            
            # 2. 긴 함수 분할 (에러 처리 추가)
            try:
                if any("긴 함수" in smell for smell in metric.code_smells):
                    function_proposals = await self.propose_function_splitting(
                        metric.file_path, file_content
                    )
                    proposals.extend(function_proposals)
            except Exception as e:
                logger.error(f"함수 분할 분석 실패 {metric.file_path}: {e}")
            
            # 3. 복잡도 감소 (에러 처리 추가)
            try:
                if metric.complexity > 15:
                    complexity_proposals = await self.propose_complexity_reduction(
                        metric.file_path, file_content
                    )
                    proposals.extend(complexity_proposals)
            except Exception as e:
                logger.error(f"복잡도 분석 실패 {metric.file_path}: {e}")
            
            # 4. 보안 이슈 해결 (에러 처리 강화)
            try:
                if metric.security_issues:
                    security_proposals = await self.propose_security_fixes(
                        metric.file_path, file_content, metric.security_issues
                    )
                    proposals.extend(security_proposals)
            except Exception as e:
                logger.error(f"보안 분석 실패 {metric.file_path}: {e}")
            
        except Exception as e:
            logger.error(f"파일 분석 전체 실패 {metric.file_path}: {e}")
        
        return proposals

    async def propose_magic_number_refactoring(self, file_path: str, content: str) -> List[RefactoringProposal]:
        """매직 넘버 리팩토링 제안 (개선된 버전)"""
        proposals = []
        
        try:
            # 매직 넘버 패턴 찾기 (더 정확한 패턴)
            magic_numbers = re.findall(r'\b(\d{3,})\b', content)
            
            if len(magic_numbers) > 3:  # 3개 이상일 때만 제안
                prompt = f"""
다음 Python 코드에서 매직 넘버들을 상수로 추출하는 리팩토링을 제안해주세요:

```python
{content[:2000]}  # 처음 2000자만
```

발견된 매직 넘버들: {list(set(magic_numbers))}

다음 형식으로 구체적으로 응답해주세요:

## 🎯 발견된 문제점
- 어떤 매직 넘버들이 문제인지 구체적으로 설명
- 각 숫자가 코드에서 어떤 의미인지 분석

## 💡 제안하는 상수들
```python
# 추출할 상수들과 의미있는 이름
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 5000
BUFFER_SIZE = 1024
```

## 🔧 리팩토링된 코드 예시
```python
# Before (원본 코드)
if retry_count > 3:
    timeout = 5000
    buffer = 1024

# After (개선된 코드)  
if retry_count > MAX_RETRY_COUNT:
    timeout = DEFAULT_TIMEOUT
    buffer = BUFFER_SIZE
```

## ✅ 개선 효과
1. 가독성 향상: 숫자의 의미가 명확해짐
2. 유지보수성 향상: 값 변경 시 한 곳에서만 수정
3. 실수 방지: 같은 값을 여러 곳에서 사용할 때 일관성 보장

## ⚠️ 주의사항
- 상수명은 의미를 명확히 표현해야 함
- 파일 상단에 상수 정의 섹션 추가 권장

한국어로 상세하게 답변해주세요.
"""
            
            # 안전한 API 호출 사용
            response_text = await self.safe_gemini_call(prompt)
            
            if response_text:
                proposal = RefactoringProposal(
                    file_path=file_path,
                    issue_type="매직 넘버",
                    description=f"{len(set(magic_numbers))}개의 매직 넘버를 상수로 추출",
                    original_code=content[:500] + "...",
                    proposed_code=response_text,
                    confidence=0.8,
                    risk_level="LOW",
                    explanation="매직 넘버를 의미있는 상수로 변경하여 가독성과 유지보수성 향상"
                )
                proposals.append(proposal)
                logger.info(f"매직 넘버 제안 생성 완료: {file_path}")
                
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
                    # 함수 라인 수 계산
                    func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    
                    if func_lines > 50:  # 50줄 이상인 함수
                        func_code = ast.get_source_segment(content, node)
                        
                        prompt = f"""
다음 Python 함수가 너무 길어서 분할이 필요합니다 ({func_lines}줄):

```python
{func_code}
```

이 함수를 더 작은 함수들로 분할하는 방법을 구체적으로 제안해주세요:

## 🔍 현재 함수 분석
- 함수명: {node.name}
- 총 라인 수: {func_lines}줄
- 주요 기능들을 단계별로 분석

## 💡 분할 제안
### 1단계: 책임 분리
- 어떤 부분들로 나눌 수 있는지 구체적으로 설명
- 각 부분의 역할과 책임 명시

### 2단계: 새로운 함수 설계
```python
def validate_input_data(data):
    '''입력 데이터 검증 전용 함수'''
    # 구체적인 코드 예시
    pass

def process_business_logic(validated_data):
    '''비즈니스 로직 처리 전용 함수'''
    # 구체적인 코드 예시
    pass

def format_output_result(processed_data):
    '''결과 포맷팅 전용 함수'''
    # 구체적인 코드 예시
    pass
```

### 3단계: 리팩토링된 메인 함수
```python
def {node.name}(original_params):
    '''리팩토링된 메인 함수'''
    validated_data = validate_input_data(original_params)
    processed_data = process_business_logic(validated_data)
    result = format_output_result(processed_data)
    return result
```

## ✅ 개선 효과
1. **가독성**: 각 함수가 하나의 책임만 가짐
2. **테스트 용이성**: 작은 단위로 개별 테스트 가능
3. **재사용성**: 분리된 함수들을 다른 곳에서도 활용
4. **디버깅**: 문제 발생 시 원인 파악이 쉬움

## ⚠️ 주의사항
- 함수 간 데이터 전달 방식 고려
- 성능에 미치는 영향 검토
- 기존 호출부 코드 수정 필요

한국어로 상세하게 답변해주세요.
"""
                        
                        response = await self.safe_gemini_call(prompt)
                        
                        if response:
                            proposal = RefactoringProposal(
                                file_path=file_path,
                                issue_type="긴 함수",
                                description=f"함수 '{node.name}' ({func_lines}줄) 분할 제안",
                                original_code=func_code,
                                proposed_code=response,
                                confidence=0.7,
                                risk_level="MEDIUM",
                                explanation="긴 함수를 작은 단위로 분할하여 가독성과 테스트 용이성 향상"
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
다음 Python 코드의 복잡도가 높습니다. 복잡도를 줄이는 리팩토링을 제안해주세요:

```python
{content[:1500]}  # 처음 1500자만
```

다음 관점에서 제안해주세요:
1. 중첩된 if문 개선
2. 반복되는 패턴 추출
3. 조건문 단순화
4. 디자인 패턴 적용

구체적인 코드 예시와 함께 한국어로 답변해주세요.
"""
            
            response = await self.safe_gemini_call(prompt)
            
            if response:
                proposal = RefactoringProposal(
                    file_path=file_path,
                    issue_type="높은 복잡도",
                    description="복잡도 감소를 위한 구조 개선",
                    original_code=content[:500] + "...",
                    proposed_code=response,
                    confidence=0.6,
                    risk_level="HIGH",
                    explanation="복잡한 로직을 단순화하여 이해하기 쉽고 버그가 적은 코드로 개선"
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
다음 Python 코드에서 보안 이슈가 발견되었습니다:

**🚨 발견된 보안 이슈:** {issue}

```python
{content[:1000]}  # 처음 1000자만
```

이 보안 이슈를 해결하는 안전한 코드로 수정해주세요:

## 🔍 보안 위험 분석
- 현재 코드의 어떤 부분이 위험한지 구체적으로 설명
- 공격자가 어떻게 악용할 수 있는지 시나리오 제시
- 발생 가능한 피해 규모 평가

## 🛡️ 보안 강화 방안
### 방법 1: 환경변수 활용
```python
# Before (위험한 코드)
api_key = "sk-1234567890abcdef"  # 하드코딩된 API 키

# After (안전한 코드)
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API_KEY 환경변수가 설정되지 않았습니다")
```

### 방법 2: 입력값 검증 강화
```python
# Before (위험한 코드)
query = f"SELECT * FROM users WHERE id = {{user_input}}"

# After (안전한 코드)
import sqlite3
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_input,))
```

## ✅ 보안 강화 효과
1. **데이터 보호**: 민감한 정보 노출 방지
2. **인젝션 공격 방지**: SQL/코드 인젝션 차단
3. **접근 제어**: 권한 없는 접근 차단
4. **감사 추적**: 보안 이벤트 로깅 가능

## 📋 추가 권장사항
- 정기적인 보안 스캔 실시
- 의존성 라이브러리 보안 업데이트
- 로깅 및 모니터링 강화
- 보안 정책 문서화

## ⚠️ 구현 시 주의사항
- 기존 기능 동작 확인
- 성능 영향 최소화
- 팀원들에게 변경사항 공유

한국어로 상세하게 답변해주세요.
"""
                
                    response = await self.safe_gemini_call(prompt)
                    
                    if response:
                        proposal = RefactoringProposal(
                            file_path=file_path,
                            issue_type="보안 이슈",
                            description=f"보안 이슈 해결: {issue}",
                            original_code=content[:300] + "...",
                            proposed_code=response,
                            confidence=0.9,
                            risk_level="HIGH",
                            explanation=f"보안 취약점 '{issue}' 해결로 시스템 보안 강화"
                        )
                        proposals.append(proposal)
                        logger.info(f"보안 제안 생성 완료: {issue}")
                        
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
        """인터랙티브 HTML 리포트 생성"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>리팩토링 제안 검토 - {session.session_id}</title>
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .proposal {{ background: white; margin: 15px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .risk-low {{ border-left: 5px solid #27ae60; }}
        .risk-medium {{ border-left: 5px solid #f39c12; }}
        .risk-high {{ border-left: 5px solid #e74c3c; }}
        .confidence {{ background: #3498db; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8em; }}
        .code-block {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 4px; overflow-x: auto; margin: 10px 0; white-space: pre-wrap; }}
        .buttons {{ margin-top: 15px; }}
        .btn {{ padding: 10px 20px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }}
        .btn-approve {{ background: #27ae60; color: white; }}
        .btn-reject {{ background: #e74c3c; color: white; }}
        .btn-modify {{ background: #f39c12; color: white; }}
        .stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .stat-card {{ background: white; padding: 15px; border-radius: 8px; text-align: center; flex: 1; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔧 리팩토링 제안 검토</h1>
        <p>세션 ID: {session.session_id}</p>
        <p>생성 시간: {session.timestamp}</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>{len(session.proposals)}</h3>
            <p>총 제안 수</p>
        </div>
        <div class="stat-card">
            <h3>{len([p for p in session.proposals if p.risk_level == 'LOW'])}</h3>
            <p>낮은 위험도</p>
        </div>
        <div class="stat-card">
            <h3>{len([p for p in session.proposals if p.risk_level == 'MEDIUM'])}</h3>
            <p>중간 위험도</p>
        </div>
        <div class="stat-card">
            <h3>{len([p for p in session.proposals if p.risk_level == 'HIGH'])}</h3>
            <p>높은 위험도</p>
        </div>
    </div>
    
    <div id="proposals">
        {''.join([f'''
        <div class="proposal risk-{proposal.risk_level.lower()}" id="proposal-{i}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3>📁 {proposal.file_path}</h3>
                <span class="confidence">신뢰도: {proposal.confidence:.0%}</span>
            </div>
            
            <div style="margin-bottom: 10px;">
                <strong>🎯 이슈 유형:</strong> {proposal.issue_type} 
                <span style="color: #e74c3c; font-weight: bold;">({proposal.risk_level} 위험도)</span>
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>📝 설명:</strong> {proposal.description}
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>💡 개선 효과:</strong> {proposal.explanation}
            </div>
            
            <details style="margin-bottom: 15px;">
                <summary style="cursor: pointer; font-weight: bold;">🔍 원본 코드 보기</summary>
                <div class="code-block">{proposal.original_code}</div>
            </details>
            
            <details style="margin-bottom: 15px;">
                <summary style="cursor: pointer; font-weight: bold;">✨ 제안된 코드 보기</summary>
                <div class="code-block">{proposal.proposed_code}</div>
            </details>
            
            <div class="buttons">
                <button class="btn btn-approve" onclick="approveProposal({i})">✅ 승인</button>
                <button class="btn btn-reject" onclick="rejectProposal({i})">❌ 거부</button>
                <button class="btn btn-modify" onclick="modifyProposal({i})">✏️ 수정 요청</button>
            </div>
            
            <div id="status-{i}" style="margin-top: 10px; font-weight: bold;"></div>
        </div>
        ''' for i, proposal in enumerate(session.proposals)])}
    </div>
    
    <script>
        let approvedProposals = [];
        let rejectedProposals = [];
        
        function approveProposal(index) {{
            approvedProposals.push(index);
            document.getElementById('status-' + index).innerHTML = '✅ 승인됨';
            document.getElementById('status-' + index).style.color = '#27ae60';
        }}
        
        function rejectProposal(index) {{
            rejectedProposals.push(index);
            document.getElementById('status-' + index).innerHTML = '❌ 거부됨';
            document.getElementById('status-' + index).style.color = '#e74c3c';
        }}
        
        function modifyProposal(index) {{
            const comment = prompt('수정 요청 사항을 입력해주세요:');
            if (comment) {{
                document.getElementById('status-' + index).innerHTML = '✏️ 수정 요청: ' + comment;
                document.getElementById('status-' + index).style.color = '#f39c12';
            }}
        }}
        
        function applyApprovedChanges() {{
            if (approvedProposals.length === 0) {{
                alert('승인된 제안이 없습니다.');
                return;
            }}
            
            alert(`${{approvedProposals.length}}개의 승인된 변경사항이 기록되었습니다.`);
        }}
    </script>
</body>
</html>
"""
        
            html_file = self.proposals_dir / f"review_{session.session_id}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"인터랙티브 HTML 생성: {html_file}")
            return str(html_file)
        
        except Exception as e:
            logger.error(f"HTML 생성 실패: {e}")
            # 기본 HTML 파일 생성
            simple_html = f"""
<!DOCTYPE html>
<html lang="ko">
<head><title>리팩토링 제안</title></head>
<body>
    <h1>리팩토링 제안 - {session.session_id}</h1>
    <p>총 {len(session.proposals)}개의 제안이 생성되었습니다.</p>
    <p>상세 내용은 JSON 파일을 확인하세요.</p>
</body>
</html>
"""
            fallback_file = self.proposals_dir / f"simple_{session.session_id}.html"
            with open(fallback_file, 'w', encoding='utf-8') as f:
                f.write(simple_html)
            return str(fallback_file)

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
            
            # 5. 인터랙티브 HTML 생성
            html_file = self.generate_interactive_html(session)
            
            logger.info(f"=== 리팩토링 제안 준비 완료 ===")
            logger.info(f"📊 총 {len(proposals)}개 제안 생성")
            logger.info(f"🌐 검토 페이지: {html_file}")
            logger.info(f"💾 백업 위치: {backup_path}")
            
            # 브라우저에서 자동으로 열기
            try:
                import webbrowser
                webbrowser.open(f"file://{Path(html_file).absolute()}")
                logger.info("🌐 브라우저에서 검토 페이지를 열었습니다")
            except:
                logger.info("브라우저 자동 열기 실패 - 수동으로 HTML 파일을 열어주세요")
            
            return session
            
        except Exception as e:
            logger.error(f"반자동 리팩토링 실패: {e}")
            raise

    def create_safe_prompt(self, template: str, **kwargs) -> str:
        """안전한 프롬프트 생성 (변수 치환 에러 방지)"""
        try:
            # 기본값 설정
            safe_kwargs = {
                'user_input': 'example_input',
                'user_id': 'example_user_id',
                'file_path': kwargs.get('file_path', 'example.py'),
                'content': kwargs.get('content', '# 예시 코드'),
                'issue': kwargs.get('issue', '일반적인 보안 이슈'),
                **kwargs  # 실제 전달된 값들로 덮어쓰기
            }
            
            return template.format(**safe_kwargs)
            
        except KeyError as e:
            logger.error(f"프롬프트 변수 누락: {e}")
            # 기본 프롬프트 반환
            return "코드 개선 방안을 제안해주세요."
        except Exception as e:
            logger.error(f"프롬프트 생성 실패: {e}")
            return "코드 분석을 도와주세요."

    async def safe_gemini_call(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """안전한 Gemini API 호출 (재시도 로직 포함)"""
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
                    # 재시도 전 잠시 대기
                    await asyncio.sleep(2 ** attempt)  # 지수 백오프
                
        logger.error("모든 Gemini API 호출 시도 실패")
        return None

    async def apply_approved_proposals(self, session_id: str, approved_indices: List[int]) -> Dict[str, Any]:
        """승인된 리팩토링 제안들을 실제 코드에 적용 (async 수정)"""
        try:
            # 세션 파일 로드
            session_file = self.proposals_dir / f"session_{session_id}.json"
            if not session_file.exists():
                raise FileNotFoundError(f"세션 파일을 찾을 수 없습니다: {session_file}")
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session = RefactoringSession(**session_data)
            
            # 적용 결과 추적
            results = {
                'applied_count': 0,
                'failed_count': 0,
                'applied_files': [],
                'failed_files': [],
                'backup_created': None
            }
            
            # 적용 전 추가 백업 생성
            backup_path = self.create_backup()
            results['backup_created'] = backup_path
            logger.info(f"적용 전 백업 생성: {backup_path}")
            
            # 승인된 제안들만 필터링
            approved_proposals = [session.proposals[i] for i in approved_indices 
                                if i < len(session.proposals)]
            
            logger.info(f"=== 승인된 제안 적용 시작 ===")
            logger.info(f"적용할 제안 수: {len(approved_proposals)}")
            
            for i, proposal in enumerate(approved_proposals):
                try:
                    success = await self.apply_single_proposal(proposal)
                    
                    if success:
                        results['applied_count'] += 1
                        results['applied_files'].append(proposal.file_path)
                        logger.info(f"✅ 적용 완료 ({i+1}/{len(approved_proposals)}): {proposal.file_path}")
                    else:
                        results['failed_count'] += 1
                        results['failed_files'].append(proposal.file_path)
                        logger.error(f"❌ 적용 실패 ({i+1}/{len(approved_proposals)}): {proposal.file_path}")
                        
                except Exception as e:
                    results['failed_count'] += 1
                    results['failed_files'].append(proposal.file_path)
                    logger.error(f"❌ 적용 중 오류 ({i+1}/{len(approved_proposals)}): {proposal.file_path} - {e}")
            
            # 적용 결과 저장
            self.save_application_results(session_id, results, approved_indices)
            
            logger.info(f"=== 적용 완료 ===")
            logger.info(f"✅ 성공: {results['applied_count']}개")
            logger.info(f"❌ 실패: {results['failed_count']}개")
            
            return results
            
        except Exception as e:
            logger.error(f"제안 적용 실패: {e}")
            raise

    async def apply_single_proposal(self, proposal: RefactoringProposal) -> bool:
        """개별 제안을 실제 파일에 적용"""
        try:
            file_path = Path(proposal.file_path)
            
            # 파일 존재 확인
            if not file_path.exists():
                logger.warning(f"파일이 존재하지 않습니다: {file_path}")
                return False
            
            # 현재 파일 내용 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
            
            # 제안 유형별 적용 방식
            if proposal.issue_type == "매직 넘버":
                new_content = await self.apply_magic_number_fix(current_content, proposal)
            elif proposal.issue_type == "보안 이슈":
                new_content = await self.apply_security_fix(current_content, proposal)
            elif proposal.issue_type == "긴 함수":
                new_content = await self.apply_function_splitting(current_content, proposal)
            elif proposal.issue_type == "높은 복잡도":
                new_content = await self.apply_complexity_reduction(current_content, proposal)
            else:
                # 일반적인 적용 방식
                new_content = await self.apply_generic_fix(current_content, proposal)
            
            if new_content and new_content != current_content:
                # 파일에 적용
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                logger.info(f"파일 수정 완료: {file_path}")
                return True
            else:
                logger.warning(f"적용할 변경사항이 없습니다: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"개별 제안 적용 실패 {proposal.file_path}: {e}")
            return False

    async def apply_magic_number_fix(self, content: str, proposal: RefactoringProposal) -> str:
        """매직 넘버 수정 적용"""
        try:
            # Gemini에게 실제 적용 가능한 코드 요청
            prompt = f"""
다음 Python 코드에서 매직 넘버를 상수로 추출해주세요.
원본 코드를 수정하여 완전한 파일 내용을 반환해주세요.

원본 코드:
```python
{content}
```

요구사항:
1. 매직 넘버들을 의미있는 상수명으로 추출
2. 파일 상단에 상수 정의 섹션 추가
3. 기존 기능은 그대로 유지
4. 완전한 파일 내용을 반환

수정된 전체 코드만 반환해주세요 (설명 없이):
"""
            
            response = await self.safe_gemini_call(prompt)
            
            if response:
                # 코드 블록에서 실제 코드만 추출
                code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
                if code_match:
                    return code_match.group(1)
                else:
                    # 코드 블록이 없으면 전체 응답을 코드로 간주
                    return response.strip()
            
            return content  # 실패 시 원본 반환
            
        except Exception as e:
            logger.error(f"매직 넘버 수정 실패: {e}")
            return content

    async def apply_security_fix(self, content: str, proposal: RefactoringProposal) -> str:
        """보안 이슈 수정 적용"""
        try:
            prompt = f"""
다음 Python 코드의 보안 이슈를 수정해주세요.
원본 코드를 수정하여 완전한 파일 내용을 반환해주세요.

원본 코드:
```python
{content}
```

보안 이슈: {proposal.description}

요구사항:
1. 보안 취약점을 안전한 코드로 수정
2. 기존 기능은 그대로 유지
3. 필요한 import문 추가
4. 완전한 파일 내용을 반환

수정된 전체 코드만 반환해주세요 (설명 없이):
"""
            
            response = await self.safe_gemini_call(prompt)
            
            if response:
                code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
                if code_match:
                    return code_match.group(1)
                else:
                    return response.strip()
            
            return content
            
        except Exception as e:
            logger.error(f"보안 수정 실패: {e}")
            return content

    async def apply_generic_fix(self, content: str, proposal: RefactoringProposal) -> str:
        """일반적인 수정 적용"""
        try:
            prompt = f"""
다음 Python 코드를 개선해주세요.

원본 코드:
```python
{content}
```

개선 사항: {proposal.description}
설명: {proposal.explanation}

요구사항:
1. 제안된 개선사항을 적용
2. 기존 기능은 그대로 유지
3. 완전한 파일 내용을 반환

수정된 전체 코드만 반환해주세요 (설명 없이):
"""
            
            response = await self.safe_gemini_call(prompt)
            
            if response:
                code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
                if code_match:
                    return code_match.group(1)
                else:
                    return response.strip()
            
            return content
            
        except Exception as e:
            logger.error(f"일반 수정 실패: {e}")
            return content

    def save_application_results(self, session_id: str, results: Dict[str, Any], approved_indices: List[int]):
        """적용 결과 저장"""
        try:
            results_file = self.proposals_dir / f"applied_{session_id}.json"
            
            application_data = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'approved_indices': approved_indices,
                'results': results
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(application_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"적용 결과 저장: {results_file}")
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")

    def interactive_proposal_selection(self, session_id: str) -> List[int]:
        """인터랙티브 제안 선택 시스템"""
        try:
            # 세션 파일 로드
            session_file = self.proposals_dir / f"session_{session_id}.json"
            if not session_file.exists():
                raise FileNotFoundError(f"세션 파일을 찾을 수 없습니다: {session_file}")
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session = RefactoringSession(**session_data)
            
            print(f"\n🔍 리팩토링 제안 검토 - 세션 {session_id}")
            print(f"📊 총 {len(session.proposals)}개의 제안이 있습니다.")
            print("\n" + "="*80)
            
            approved_indices = []
            
            for i, proposal in enumerate(session.proposals):
                print(f"\n📋 제안 #{i}")
                print(f"📁 파일: {proposal.file_path}")
                print(f"🎯 이슈: {proposal.issue_type}")
                print(f"📝 설명: {proposal.description}")
                print(f"💡 개선효과: {proposal.explanation}")
                print(f"🔒 위험도: {proposal.risk_level}")
                print(f"📊 신뢰도: {proposal.confidence:.0%}")
                
                # 원본 코드 미리보기 (처음 200자)
                preview = proposal.original_code[:200].replace('\n', ' ')
                print(f"👀 코드 미리보기: {preview}...")
                
                print("\n" + "-"*50)
                
                while True:
                    choice = input(f"제안 #{i} 선택 (1:승인, 2:거부, s:건너뛰기, q:종료): ").strip().lower()
                    
                    if choice == '1':
                        approved_indices.append(i)
                        print(f"✅ 제안 #{i} 승인됨")
                        break
                    elif choice == '2':
                        print(f"❌ 제안 #{i} 거부됨")
                        break
                    elif choice == 's':
                        print(f"⏭️ 제안 #{i} 건너뜀")
                        break
                    elif choice == 'q':
                        print("🛑 선택 중단")
                        return approved_indices
                    else:
                        print("❗ 잘못된 입력입니다. 1(승인), 2(거부), s(건너뛰기), q(종료) 중 선택하세요.")
                
                print("="*80)
            
            print(f"\n🎯 선택 완료!")
            print(f"✅ 승인된 제안: {len(approved_indices)}개")
            print(f"📋 승인된 제안 번호: {approved_indices}")
            
            return approved_indices
            
        except Exception as e:
            logger.error(f"인터랙티브 선택 실패: {e}")
            return []

    def quick_selection_menu(self, session_id: str) -> List[int]:
        """빠른 선택 메뉴"""
        try:
            # 세션 파일 로드
            session_file = self.proposals_dir / f"session_{session_id}.json"
            if not session_file.exists():
                raise FileNotFoundError(f"세션 파일을 찾을 수 없습니다: {session_file}")
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session = RefactoringSession(**session_data)
            
            print(f"\n🚀 빠른 선택 메뉴 - 세션 {session_id}")
            print(f"📊 총 {len(session.proposals)}개의 제안")
            
            # 위험도별 분류
            low_risk = [i for i, p in enumerate(session.proposals) if p.risk_level == 'LOW']
            medium_risk = [i for i, p in enumerate(session.proposals) if p.risk_level == 'MEDIUM']
            high_risk = [i for i, p in enumerate(session.proposals) if p.risk_level == 'HIGH']
            
            # 이슈 유형별 분류
            magic_numbers = [i for i, p in enumerate(session.proposals) if p.issue_type == '매직 넘버']
            security_issues = [i for i, p in enumerate(session.proposals) if p.issue_type == '보안 이슈']
            
            print(f"\n📊 제안 분류:")
            print(f"🟢 낮은 위험도: {len(low_risk)}개")
            print(f"🟡 중간 위험도: {len(medium_risk)}개")
            print(f"🔴 높은 위험도: {len(high_risk)}개")
            print(f"🔢 매직 넘버: {len(magic_numbers)}개")
            print(f"🛡️ 보안 이슈: {len(security_issues)}개")
            
            print(f"\n🎯 빠른 선택 옵션:")
            print(f"1. 낮은 위험도만 모두 승인 ({len(low_risk)}개)")
            print(f"2. 매직 넘버만 모두 승인 ({len(magic_numbers)}개)")
            print(f"3. 낮은+중간 위험도 승인 ({len(low_risk + medium_risk)}개)")
            print(f"4. 처음 5개만 승인")
            print(f"5. 처음 10개만 승인")
            print(f"6. 개별 선택 모드")
            print(f"7. 모두 승인 (⚠️ 주의)")
            print(f"0. 취소")
            
            while True:
                choice = input("\n선택하세요 (0-7): ").strip()
                
                if choice == '1':
                    return low_risk
                elif choice == '2':
                    return magic_numbers
                elif choice == '3':
                    return low_risk + medium_risk
                elif choice == '4':
                    return list(range(min(5, len(session.proposals))))
                elif choice == '5':
                    return list(range(min(10, len(session.proposals))))
                elif choice == '6':
                    return self.interactive_proposal_selection(session_id)
                elif choice == '7':
                    confirm = input("⚠️ 모든 제안을 승인하시겠습니까? (y/N): ").strip().lower()
                    if confirm == 'y':
                        return list(range(len(session.proposals)))
                    else:
                        continue
                elif choice == '0':
                    print("🛑 취소됨")
                    return []
                else:
                    print("❗ 잘못된 선택입니다. 0-7 중에서 선택하세요.")
            
        except Exception as e:
            logger.error(f"빠른 선택 실패: {e}")
            return []

async def auto_apply_safe_refactoring_from_session(session_id: str):
    """기존 세션에서 안전한 제안들만 자동으로 적용"""
    try:
        system = AutoRefactoringSystem()
        
        # 세션 파일 로드
        session_file = system.proposals_dir / f"session_{session_id}.json"
        if not session_file.exists():
            raise FileNotFoundError(f"세션 파일을 찾을 수 없습니다: {session_id}")
        
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        session = RefactoringSession(**session_data)
        
        # 안전한 제안들만 자동 선택 (더 엄격한 기준)
        safe_indices = []
        
        for i, proposal in enumerate(session.proposals):
            # 매우 안전한 제안만 자동 승인
            is_ultra_safe = (
                proposal.risk_level == 'LOW' and 
                proposal.confidence >= 0.85 and  # 더 높은 신뢰도 요구
                proposal.issue_type in {'매직 넘버', '코드 스타일'} and  # 가장 안전한 유형만
                proposal.priority >= 100  # 높은 우선순위만
            )
            
            if is_ultra_safe:
                safe_indices.append(i)
        
        print(f"\n🤖 AI 초안전 자동 리팩토링 시작")
        print(f"📊 총 제안: {len(session.proposals)}개")
        print(f"✅ 자동 승인: {len(safe_indices)}개 (초안전 기준)")
        print(f"📋 승인 기준: 낮은 위험도 + 85% 이상 신뢰도 + 매직넘버/스타일만 + 높은 우선순위")
        
        if not safe_indices:
            print("🛑 자동 승인할 초안전 제안이 없습니다.")
            print("💡 수동 검토를 권장합니다: python auto_refactoring_system.py apply", session_id)
            return
        
        # 자동 적용
        result = await system.apply_approved_proposals(session_id, safe_indices)
        
        print(f"\n🎯 AI 초안전 자동 리팩토링 완료!")
        print(f"✅ 성공: {result.applied_count}개")
        print(f"❌ 실패: {result.failed_count}개")
        print(f"💾 백업: {result.backup_created}")
        print(f"⏱️ 처리 시간: {result.processing_time:.2f}초")
        
        if result.applied_files:
            print(f"\n📝 자동 수정된 파일들:")
            for file_path in result.applied_files:
                print(f"  - {file_path}")
        
        # 남은 제안들 안내
        remaining_count = len(session.proposals) - len(safe_indices)
        if remaining_count > 0:
            print(f"\n💡 추가 검토 필요한 제안: {remaining_count}개")
            print(f"   수동 검토: python auto_refactoring_system.py apply {session_id}")
            print(f"   세션 요약: python auto_refactoring_system.py summary {session_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"AI 자동 리팩토링 실패: {e}")
        raise

def list_sessions():
    """세션 목록 표시"""
    try:
        proposals_dir = Path("refactoring_proposals")
        if not proposals_dir.exists():
            print("❌ 리팩토링 세션이 없습니다.")
            return
        
        session_files = list(proposals_dir.glob("session_*.json"))
        
        if not session_files:
            print("❌ 리팩토링 세션이 없습니다.")
            return
        
        print(f"\n📋 리팩토링 세션 목록 ({len(session_files)}개)")
        print("="*80)
        
        sessions_info = []
        
        for session_file in session_files:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                session = RefactoringSession(**session_data)
                sessions_info.append({
                    'id': session.session_id,
                    'timestamp': session.timestamp,
                    'proposals': len(session.proposals),
                    'auto_applied': session.auto_applied_count,
                    'processing_time': session.processing_time,
                    'files_analyzed': session.total_files_analyzed
                })
                
            except Exception as e:
                logger.error(f"세션 파일 읽기 실패 {session_file}: {e}")
        
        # 최신 순으로 정렬
        sessions_info.sort(key=lambda x: x['timestamp'], reverse=True)
        
        for i, info in enumerate(sessions_info, 1):
            timestamp = datetime.fromisoformat(info['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i:2d}. 세션 ID: {info['id']}")
            print(f"    📅 생성시간: {timestamp}")
            print(f"    📊 제안수: {info['proposals']}개 (자동적용: {info['auto_applied']}개)")
            print(f"    📁 분석파일: {info['files_analyzed']}개")
            print(f"    ⏱️ 처리시간: {info['processing_time']:.1f}초")
            print(f"    🔧 명령어: python auto_refactoring_system.py apply {info['id']}")
            print()
        
        print("💡 사용법:")
        print("  - 세션 적용: python auto_refactoring_system.py apply <session_id>")
        print("  - 세션 요약: python auto_refactoring_system.py summary <session_id>")
        print("  - 자동 적용: python auto_refactoring_system.py auto <session_id>")
        
    except Exception as e:
        logger.error(f"세션 목록 표시 실패: {e}")

def clean_old_sessions(days: int = 30):
    """오래된 세션 파일 정리"""
    try:
        proposals_dir = Path("refactoring_proposals")
        if not proposals_dir.exists():
            print("❌ 리팩토링 디렉토리가 없습니다.")
            return
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        session_files = list(proposals_dir.glob("session_*.json"))
        html_files = list(proposals_dir.glob("*.html"))
        applied_files = list(proposals_dir.glob("applied_*.json"))
        
        all_files = session_files + html_files + applied_files
        
        if not all_files:
            print("❌ 정리할 파일이 없습니다.")
            return
        
        deleted_count = 0
        
        for file_path in all_files:
            try:
                # 파일 수정 시간 확인
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_time < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"삭제됨: {file_path.name}")
                    
            except Exception as e:
                logger.error(f"파일 삭제 실패 {file_path}: {e}")
        
        print(f"🧹 정리 완료: {deleted_count}개 파일 삭제 ({days}일 이상 된 파일)")
        
        if deleted_count == 0:
            print(f"💡 {days}일 이내의 파일들만 있어서 삭제할 파일이 없습니다.")
        
    except Exception as e:
        logger.error(f"세션 정리 실패: {e}")

async def run_performance_benchmark():
    """성능 벤치마크 실행"""
    try:
        print("🚀 리팩토링 시스템 성능 벤치마크 시작")
        print("="*60)
        
        system = AutoRefactoringSystem()
        analyzer = CodeQualityAnalyzer()
        
        # 1. 분석 성능 테스트
        print("📊 1단계: 코드 품질 분석 성능 테스트")
        analysis_start = asyncio.get_event_loop().time()
        
        report = await analyzer.analyze_project()
        
        analysis_end = asyncio.get_event_loop().time()
        analysis_time = analysis_end - analysis_start
        
        print(f"   ✅ 분석 완료: {len(report.file_metrics)}개 파일")
        print(f"   ⏱️ 분석 시간: {analysis_time:.2f}초")
        print(f"   📈 분석 속도: {len(report.file_metrics)/analysis_time:.1f} 파일/초")
        
        # 2. 제안 생성 성능 테스트
        print("\n🔧 2단계: 리팩토링 제안 생성 성능 테스트")
        proposal_start = asyncio.get_event_loop().time()
        
        proposals = await system.generate_refactoring_proposals(report)
        
        proposal_end = asyncio.get_event_loop().time()
        proposal_time = proposal_end - proposal_start
        
        print(f"   ✅ 제안 생성: {len(proposals)}개")
        print(f"   ⏱️ 생성 시간: {proposal_time:.2f}초")
        print(f"   📈 생성 속도: {len(proposals)/proposal_time:.1f} 제안/초")
        
        # 3. 메모리 사용량 체크
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"\n💾 메모리 사용량:")
            print(f"   📊 RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
            print(f"   📊 VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
        except ImportError:
            print("\n💾 메모리 정보: psutil 패키지가 필요합니다")
        
        # 4. 전체 성능 요약
        total_time = analysis_time + proposal_time
        
        print(f"\n📈 전체 성능 요약:")
        print(f"   ⏱️ 총 처리 시간: {total_time:.2f}초")
        print(f"   📊 전체 처리 속도: {len(report.file_metrics)/total_time:.1f} 파일/초")
        print(f"   🎯 제안 효율성: {len(proposals)/len(report.file_metrics):.1f} 제안/파일")
        
        # 5. 성능 등급 평가
        if total_time < 30:
            grade = "🚀 매우 빠름"
        elif total_time < 60:
            grade = "⚡ 빠름"
        elif total_time < 120:
            grade = "🔄 보통"
        else:
            grade = "🐌 느림"
        
        print(f"   🏆 성능 등급: {grade}")
        
        # 6. 최적화 권장사항
        print(f"\n💡 최적화 권장사항:")
        if len(report.file_metrics) > 100:
            print("   - 대용량 프로젝트: 배치 처리 크기 조정 고려")
        if proposal_time > analysis_time * 2:
            print("   - Gemini API 응답 시간 최적화 필요")
        if len(proposals) < len(report.file_metrics) * 0.5:
            print("   - 제안 생성 로직 개선 고려")
        
        print(f"\n🎉 벤치마크 완료!")
        
    except Exception as e:
        logger.error(f"벤치마크 실패: {e}")
        raise

def print_usage():
    """사용법 출력"""
    print("🤖 AI 완전자동 리팩토링 시스템")
    print("="*50)
    print("\n📋 사용법:")
    print("  python auto_refactoring_system.py refactor                    # 완전자동 리팩토링 실행")
    print("  python auto_refactoring_system.py apply <session_id>          # 인터랙티브 적용")
    print("  python auto_refactoring_system.py auto <session_id>           # AI 초안전 자동 적용")
    print("  python auto_refactoring_system.py summary <session_id>        # 세션 요약")
    print("  python auto_refactoring_system.py list                        # 세션 목록")
    print("  python auto_refactoring_system.py clean                       # 오래된 세션 정리")
    print("  python auto_refactoring_system.py benchmark                   # 성능 벤치마크")
    print("\n🎯 추천 워크플로우:")
    print("  1. python auto_refactoring_system.py refactor                 # 자동 분석 및 적용")
    print("  2. python auto_refactoring_system.py list                     # 결과 확인")
    print("  3. python auto_refactoring_system.py apply <session_id>       # 추가 적용 (필요시)")
    print("\n💡 예시:")
    print("  python auto_refactoring_system.py refactor")
    print("  python auto_refactoring_system.py auto 20250616_113140")
    print("  python auto_refactoring_system.py apply 20250616_113140")

def print_interactive_menu():
    """인터랙티브 메뉴 출력"""
    print("🤖 AI 완전자동 리팩토링 시스템")
    print("="*50)
    print("\n🎯 무엇을 하시겠습니까?")
    print("1. 🚀 완전자동 리팩토링 실행 (분석 + 제안 + 자동적용)")
    print("2. 📋 기존 세션 목록 보기")
    print("3. 🔧 기존 세션 수동 적용")
    print("4. 🤖 기존 세션 AI 자동 적용")
    print("5. 📊 성능 벤치마크 실행")
    print("6. 🧹 오래된 세션 정리")
    print("0. 종료")
    
    while True:
        try:
            choice = input("\n선택하세요 (0-6): ").strip()
            
            if choice == '1':
                print("🚀 완전자동 리팩토링을 시작합니다...")
                asyncio.run(run_full_auto_refactoring())
                break
            elif choice == '2':
                list_sessions()
                break
            elif choice == '3':
                session_id = input("세션 ID를 입력하세요: ").strip()
                if session_id:
                    asyncio.run(interactive_refactoring_apply(session_id))
                break
            elif choice == '4':
                session_id = input("세션 ID를 입력하세요: ").strip()
                if session_id:
                    asyncio.run(auto_apply_safe_refactoring_from_session(session_id))
                break
            elif choice == '5':
                print("📊 성능 벤치마크를 시작합니다...")
                asyncio.run(run_performance_benchmark())
                break
            elif choice == '6':
                days = input("몇 일 이상 된 파일을 삭제하시겠습니까? (기본값: 30): ").strip()
                days = int(days) if days.isdigit() else 30
                clean_old_sessions(days)
                break
            elif choice == '0':
                print("👋 프로그램을 종료합니다.")
                break
            else:
                print("❗ 잘못된 선택입니다. 0-6 중에서 선택하세요.")
                
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류가 발생했습니다: {e}")
            break

# 추가 유틸리티 함수들
async def apply_function_splitting(self, content: str, proposal: RefactoringProposal) -> str:
    """함수 분할 수정 적용"""
    try:
        prompt = f"""
다음 Python 코드의 긴 함수를 분할해주세요.

원본 코드:
```python
{content}
```

분할 제안: {proposal.description}

요구사항:
1. 긴 함수를 논리적 단위로 분할
2. 기존 기능과 전략 로직은 그대로 유지
3. 함수명은 기능을 명확히 표현
4. 완전한 파일 내용을 반환

수정된 전체 코드만 반환해주세요 (설명 없이):
"""
        
        response = await self.gemini.safe_generate_content(prompt)
        
        if response:
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                return code_match.group(1)
            else:
                return response.strip()
        
        return content
        
    except Exception as e:
        logger.error(f"함수 분할 수정 실패: {e}")
        return content

async def apply_complexity_reduction(self, content: str, proposal: RefactoringProposal) -> str:
    """복잡도 감소 수정 적용"""
    try:
        prompt = f"""
다음 Python 코드의 복잡도를 줄여주세요.

원본 코드:
```python
{content}
```

복잡도 감소 제안: {proposal.description}

요구사항:
1. 중첩된 조건문 단순화
2. 반복 패턴 함수로 추출
3. 전략 로직과 알고리즘은 절대 변경하지 마세요
4. 완전한 파일 내용을 반환

수정된 전체 코드만 반환해주세요 (설명 없이):
"""
        
        response = await self.gemini.safe_generate_content(prompt)
        
        if response:
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                return code_match.group(1)
            else:
                return response.strip()
        
        return content
        
    except Exception as e:
        logger.error(f"복잡도 감소 수정 실패: {e}")
        return content

# IntelligentProposalApplicator 클래스에 누락된 메서드들 추가
IntelligentProposalApplicator.apply_function_splitting = apply_function_splitting
IntelligentProposalApplicator.apply_complexity_reduction = apply_complexity_reduction

# 필요한 import 추가
from datetime import timedelta

# 최종 실행부 완성
if __name__ == "__main__":
    try:
        # 로깅 레벨 설정
        if "--debug" in sys.argv:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("🐛 디버그 모드 활성화")
        
        # 메인 로직 실행
        main()
        
    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"프로그램 실행 실패: {e}")
        print(f"❌ 오류가 발생했습니다: {e}")
        print("💡 --debug 옵션을 사용하여 상세 로그를 확인하세요.")
        sys.exit(1)

def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "refactor":
            asyncio.run(run_full_auto_refactoring())
        elif command == "apply":
            if len(sys.argv) < 3:
                print("사용법:")
                print("  python auto_refactoring_system.py apply <session_id>")
                print("  python auto_refactoring_system.py apply <session_id> <indices>")
                sys.exit(1)
            
            session_id = sys.argv[2]
            
            if len(sys.argv) > 3:
                approved_indices = [int(x.strip()) for x in sys.argv[3].split(',')]
                asyncio.run(apply_approved_refactoring(session_id, approved_indices))
            else:
                asyncio.run(interactive_refactoring_apply(session_id))
        elif command == "auto":
            if len(sys.argv) < 3:
                print("사용법: python auto_refactoring_system.py auto <session_id>")
                sys.exit(1)
            
            session_id = sys.argv[2]
            asyncio.run(auto_apply_safe_refactoring_from_session(session_id))
        elif command == "summary":
            if len(sys.argv) < 3:
                print("사용법: python auto_refactoring_system.py summary <session_id>")
                sys.exit(1)
            
            session_id = sys.argv[2]
            show_session_summary(session_id)
        elif command == "list":
            list_sessions()
        elif command == "clean":
            days = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 30
            clean_old_sessions(days)
        elif command == "benchmark":
            asyncio.run(run_performance_benchmark())
        else:
            print_usage()
    else:
        print_interactive_menu() 