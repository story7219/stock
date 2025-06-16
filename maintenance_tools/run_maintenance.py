"""
AI 기반 코드 품질 관리 도구 실행기
- 'analyze' : 코드 분석 후 제안 생성
- 'apply' : 생성된 제안을 코드에 적용
"""
import typer
from pathlib import Path
import json
import logging
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from maintenance_tools.auto_refactor_proposer import AutoRefactorProposer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = typer.Typer(help="🤖 AI 기반 코드 품질 관리 도구")
PROPOSALS_FILE = Path("refactoring_proposals.json")

@app.command()
def analyze():
    """AI를 사용해 코드 품질을 분석하고 리팩토링 제안을 생성합니다."""
    logging.info("🚀 AI 코드 분석 및 리팩토링 제안 생성을 시작합니다...")
    proposer = AutoRefactorProposer()
    proposals = proposer.run()
    
    if not proposals:
        logging.info("🎉 분석 완료. 새로운 리팩토링 제안이 없습니다.")
        return

    proposals_as_dicts = [p.to_dict() for p in proposals]
    with open(PROPOSALS_FILE, "w", encoding="utf-8") as f:
        json.dump(proposals_as_dicts, f, ensure_ascii=False, indent=2)
    
    logging.info(f"✅ {len(proposals)}개의 리팩토링 제안을 {PROPOSALS_FILE}에 저장했습니다.")

@app.command()
def apply():
    """'refactoring_proposals.json' 파일에 따라 코드 변경사항을 실제로 적용합니다."""
    if not PROPOSALS_FILE.exists():
        logging.warning(f"⚠️ 제안 파일({PROPOSALS_FILE})을 찾을 수 없습니다.")
        return
        
    with open(PROPOSALS_FILE, "r", encoding="utf-8") as f:
        proposals_data = json.load(f)
    
    if not proposals_data:
        logging.info("ℹ️ 제안 파일이 비어있어 적용할 변경사항이 없습니다.")
        return

    logging.info(f"⚙️ {len(proposals_data)}개의 리팩토링 제안을 적용합니다...")
    for proposal in proposals_data:
        try:
            target_file = Path(proposal['file_path'])
            refactored_code = proposal['refactored_code']
            target_file.write_text(refactored_code, encoding='utf-8')
            logging.info(f"✅ '{target_file}' 적용 완료. 이유: {proposal.get('explanation', 'N/A')}")
        except Exception as e:
            logging.error(f"❌ '{proposal['file_path']}' 적용 실패: {e}")

    logging.info("🎉 모든 리팩토링 제안 적용 완료!")

if __name__ == "__main__":
    app() 