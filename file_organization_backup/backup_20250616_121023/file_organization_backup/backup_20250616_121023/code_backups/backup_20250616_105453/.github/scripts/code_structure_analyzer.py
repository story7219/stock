"""
AI 기반 코드 구조 분석 및 리팩토링 계획 생성
"""

import os
import ast
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import google.generativeai as genai

@dataclass
class ModuleAnalysis:
    """모듈 분석 결과"""
    file_path: str
    lines_of_code: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    responsibilities: List[str]
    coupling_score: float
    cohesion_score: float

class CodeStructureAnalyzer:
    """코드 구조 분석기"""
    
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.modules = []
        self.restructure_plan = {}
    
    def analyze_project_structure(self) -> Dict[str, Any]:
        """프로젝트 전체 구조 분석"""
        python_files = list(Path('.').rglob('*.py'))
        python_files = [f for f in python_files if not str(f).startswith('.git')]
        
        analysis_results = {
            'total_files': len(python_files),
            'modules': [],
            'structure_issues': [],
            'recommendations': [],
            'metrics': {}
        }
        
        # 각 파일 분석
        for file_path in python_files:
            module_analysis = self.analyze_module(str(file_path))
            if module_analysis:
                analysis_results['modules'].append(module_analysis.__dict__)
                self.modules.append(module_analysis)
        
        # 구조적 문제 탐지
        structure_issues = self.detect_structure_issues()
        analysis_results['structure_issues'] = structure_issues
        
        # 메트릭 계산
        metrics = self.calculate_structure_metrics()
        analysis_results['metrics'] = metrics
        
        return analysis_results
    
    def analyze_module(self, file_path: str) -> ModuleAnalysis:
        """개별 모듈 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # 책임 분석 (키워드 기반)
            responsibilities = self.analyze_responsibilities(content, functions, classes)
            
            # 결합도/응집도 계산
            coupling_score = self.calculate_coupling(imports, len(functions) + len(classes))
            cohesion_score = self.calculate_cohesion(functions, classes, content)
            
            return ModuleAnalysis(
                file_path=file_path,
                lines_of_code=len(content.split('\n')),
                functions=functions,
                classes=classes,
                imports=imports,
                responsibilities=responsibilities,
                coupling_score=coupling_score,
                cohesion_score=cohesion_score
            )
            
        except Exception as e:
            print(f"❌ 모듈 분석 실패 ({file_path}): {e}")
            return None
    
    def analyze_responsibilities(self, content: str, functions: List[str], classes: List[str]) -> List[str]:
        """모듈의 책임 분석"""
        responsibilities = []
        
        # 키워드 기반 책임 분류
        responsibility_keywords = {
            'trading_strategy': ['strategy', 'signal', 'indicator', 'analysis', 'fibonacci', 'scout'],
            'data_collection': ['data', 'fetch', 'download', 'api', 'websocket', 'price'],
            'order_execution': ['order', 'buy', 'sell', 'execute', 'trade', 'position'],
            'portfolio_management': ['portfolio', 'balance', 'asset', 'allocation', 'risk'],
            'logging_monitoring': ['log', 'monitor', 'alert', 'notification', 'telegram'],
            'configuration': ['config', 'setting', 'env', 'parameter'],
            'utility': ['util', 'helper', 'common', 'tool']
        }
        
        content_lower = content.lower()
        all_names = functions + classes
        
        for responsibility, keywords in responsibility_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                responsibilities.append(responsibility)
            
            # 함수/클래스 이름에서도 체크
            if any(any(keyword in name.lower() for keyword in keywords) for name in all_names):
                if responsibility not in responsibilities:
                    responsibilities.append(responsibility)
        
        return responsibilities if responsibilities else ['unknown']
    
    def calculate_coupling(self, imports: List[str], total_entities: int) -> float:
        """결합도 계산 (낮을수록 좋음)"""
        if total_entities == 0:
            return 0.0
        
        external_imports = len([imp for imp in imports if not imp.startswith('.')])
        return min(1.0, external_imports / max(1, total_entities))
    
    def calculate_cohesion(self, functions: List[str], classes: List[str], content: str) -> float:
        """응집도 계산 (높을수록 좋음)"""
        if not functions and not classes:
            return 0.0
        
        # 간단한 응집도 계산: 공통 키워드 비율
        all_names = functions + classes
        if not all_names:
            return 0.0
        
        # 가장 빈번한 키워드 찾기
        keywords = {}
        for name in all_names:
            words = name.lower().split('_')
            for word in words:
                keywords[word] = keywords.get(word, 0) + 1
        
        if not keywords:
            return 0.0
        
        max_frequency = max(keywords.values())
        cohesion = max_frequency / len(all_names)
        
        return min(1.0, cohesion)
    
    def detect_structure_issues(self) -> List[Dict[str, Any]]:
        """구조적 문제 탐지"""
        issues = []
        
        # 1. 거대한 파일 탐지
        for module in self.modules:
            if module.lines_of_code > 500:
                issues.append({
                    'type': 'large_file',
                    'severity': 'high',
                    'file': module.file_path,
                    'description': f"파일이 {module.lines_of_code}줄로 너무 큽니다.",
                    'suggestion': "기능별로 여러 파일로 분리하세요."
                })
        
        # 2. 다중 책임 모듈 탐지
        for module in self.modules:
            if len(module.responsibilities) > 3:
                issues.append({
                    'type': 'multiple_responsibilities',
                    'severity': 'medium',
                    'file': module.file_path,
                    'description': f"모듈이 {len(module.responsibilities)}개의 책임을 가집니다.",
                    'responsibilities': module.responsibilities,
                    'suggestion': "단일 책임 원칙에 따라 모듈을 분리하세요."
                })
        
        # 3. 높은 결합도 탐지
        for module in self.modules:
            if module.coupling_score > 0.7:
                issues.append({
                    'type': 'high_coupling',
                    'severity': 'medium',
                    'file': module.file_path,
                    'description': f"결합도가 {module.coupling_score:.2f}로 높습니다.",
                    'suggestion': "의존성을 줄이고 인터페이스를 통한 느슨한 결합을 고려하세요."
                })
        
        # 4. 낮은 응집도 탐지
        for module in self.modules:
            if module.cohesion_score < 0.3:
                issues.append({
                    'type': 'low_cohesion',
                    'severity': 'medium',
                    'file': module.file_path,
                    'description': f"응집도가 {module.cohesion_score:.2f}로 낮습니다.",
                    'suggestion': "관련된 기능들을 함께 그룹화하세요."
                })
        
        return issues
    
    def calculate_structure_metrics(self) -> Dict[str, float]:
        """구조 메트릭 계산"""
        if not self.modules:
            return {}
        
        total_loc = sum(m.lines_of_code for m in self.modules)
        avg_coupling = sum(m.coupling_score for m in self.modules) / len(self.modules)
        avg_cohesion = sum(m.cohesion_score for m in self.modules) / len(self.modules)
        
        # 책임 분산도 계산
        all_responsibilities = []
        for module in self.modules:
            all_responsibilities.extend(module.responsibilities)
        
        unique_responsibilities = set(all_responsibilities)
        responsibility_distribution = len(unique_responsibilities) / len(self.modules) if self.modules else 0
        
        return {
            'total_lines_of_code': total_loc,
            'average_file_size': total_loc / len(self.modules),
            'average_coupling': avg_coupling,
            'average_cohesion': avg_cohesion,
            'responsibility_distribution': responsibility_distribution,
            'modularity_score': (avg_cohesion - avg_coupling + 1) / 2  # 0-1 스케일
        }
    
    async def generate_restructure_plan(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """AI를 통한 리팩토링 계획 생성"""
        
        prompt = f"""
다음은 Python 자동매매 시스템의 코드 구조 분석 결과입니다:

{json.dumps(analysis_results, ensure_ascii=False, indent=2)}

이 분석 결과를 바탕으로 다음과 같은 역할별 모듈 구조로 리팩토링 계획을 세워주세요:

## 목표 구조:
1. **strategies/** - 매매 전략 모듈
   - scout_strategy.py (척후병 전략)
   - fibonacci_strategy.py (피보나치 전략)
   - technical_analyzer.py (기술적 분석)

2. **data/** - 데이터 수집 및 관리
   - market_data_collector.py (시장 데이터 수집)
   - websocket_manager.py (실시간 데이터)
   - data_validator.py (데이터 검증)

3. **trading/** - 주문 실행 및 포트폴리오
   - order_executor.py (주문 실행)
   - portfolio_manager.py (포트폴리오 관리)
   - risk_manager.py (리스크 관리)

4. **monitoring/** - 로깅 및 모니터링
   - logger.py (로깅 시스템)
   - telegram_notifier.py (알림)
   - performance_monitor.py (성능 모니터링)

5. **core/** - 핵심 시스템
   - trader.py (메인 트레이더)
   - config.py (설정 관리)
   - exceptions.py (예외 처리)

다음 형식으로 리팩토링 계획을 제시해주세요:

```json
{{
  "restructure_needed": true/false,
  "target_structure": {{
    "폴더명/파일명": {{
      "description": "파일 설명",
      "responsibilities": ["책임1", "책임2"],
      "source_files": ["현재_파일1.py", "현재_파일2.py"],
      "functions_to_move": ["함수명1", "함수명2"],
      "classes_to_move": ["클래스명1", "클래스명2"],
      "interfaces": ["제공할_인터페이스1", "제공할_인터페이스2"]
    }}
  }},
  "migration_steps": [
    "단계별 마이그레이션 계획"
  ],
  "benefits": [
    "예상되는 개선 효과"
  ]
}}
```

JSON 형식으로만 답변해주세요.
"""
        
        try:
            response = await self.model.generate_content_async(prompt)
            plan_text = response.text.strip()
            
            # JSON 추출
            if '```json' in plan_text:
                json_start = plan_text.find('```json') + 7
                json_end = plan_text.find('```', json_start)
                plan_text = plan_text[json_start:json_end].strip()
            
            return json.loads(plan_text)
            
        except Exception as e:
            print(f"❌ 리팩토링 계획 생성 실패: {e}")
            return {
                "restructure_needed": False,
                "error": str(e)
            }

async def main():
    """메인 실행 함수"""
    print("🏗️ 코드 구조 분석 시작...")
    
    analyzer = CodeStructureAnalyzer()
    
    # 프로젝트 구조 분석
    analysis_results = analyzer.analyze_project_structure()
    
    # AI 리팩토링 계획 생성
    restructure_plan = await analyzer.generate_restructure_plan(analysis_results)
    
    # 결과 저장
    with open('structure_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(f"""# 🏗️ 코드 구조 분석 리포트

## 📊 전체 분석 결과

- **총 파일 수**: {analysis_results['total_files']}개
- **구조적 이슈**: {len(analysis_results['structure_issues'])}개
- **모듈화 점수**: {analysis_results['metrics'].get('modularity_score', 0):.2f}/1.0

## 🔍 발견된 구조적 문제

{chr(10).join([f"- **{issue['type']}** ({issue['severity']}): {issue['description']}" for issue in analysis_results['structure_issues']])}

## 🎯 리팩토링 계획

{json.dumps(restructure_plan, ensure_ascii=False, indent=2)}
""")
    
    with open('restructure_plan.json', 'w', encoding='utf-8') as f:
        json.dump(restructure_plan, f, ensure_ascii=False, indent=2)
    
    with open('analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    # 리팩토링 필요 여부 저장
    restructure_needed = restructure_plan.get('restructure_needed', False)
    with open('restructure_needed.txt', 'w') as f:
        f.write(str(restructure_needed).lower())
    
    print(f"✅ 분석 완료! 리팩토링 필요: {restructure_needed}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 