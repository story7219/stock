"""
코드 관리 자동화 시스템의 메인 실행 파일
- AI 코드 품질 분석, 리팩토링 제안, 파일 정리를 실행하는 단일 진입점
"""
import argparse
import logging
import json
import os
from datetime import datetime

# .env 로드를 위해 config.py를 먼저 임포트
try:
    import config
except ImportError:
    # config.py가 아직 상위 폴더에 있는 경우를 대비
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import config

from auto_quality_analyzer import CodeQualityAnalyzer
from auto_refactor_proposer import RefactorProposer
from auto_file_organizer import FileOrganizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_analysis(args):
    """AI 코드 품질 분석 실행"""
    logging.info("🚀 AI 코드 품질 분석을 시작합니다.")
    analyzer = CodeQualityAnalyzer()
    report = analyzer.analyze_directory("..") # 상위 폴더를 분석
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"quality_report_{timestamp}.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    logging.info(f"✅ 분석 완료. 결과가 '{report_path}'에 저장되었습니다.")
    for file, res in report.items():
        print(f"\n📄 파일: {file}")
        print(f"  - 복잡도: {res['complexity']:.2f}, 유지보수성: {res['maintainability']:.2f}")
        print(f"  - AI 코멘트: {res['ai_comment']}")


def run_refactor(args):
    """AI 리팩토링 제안 실행"""
    logging.info("🚀 AI 리팩토링 제안 생성을 시작합니다.")
    try:
        with open(args.report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
    except FileNotFoundError:
        logging.error(f"리포트 파일을 찾을 수 없습니다: {args.report_file}")
        return

    proposer = RefactorProposer()
    proposals = proposer.generate_proposals(report)
    
    if not proposals:
        logging.info("🎉 모든 파일이 양호합니다. 리팩토링 제안이 없습니다.")
        return

    logging.info("\n--- 🤖 AI 리팩토링 제안 ---")
    for file_path, proposal in proposals.items():
        print(f"\n\n====================\n📄 파일: {file_path}\n====================")
        print(proposal)
    
    logging.info("\n위 제안을 검토하고 수동으로 적용하거나, 자동 적용 시스템을 개발하여 연동하세요.")
    logging.info("승인/거부 기능은 이 시스템의 다음 단계입니다.")


def run_organize(args):
    """파일 자동 정리 실행"""
    logging.info("🚀 파일 자동 정리를 시작합니다.")
    organizer = FileOrganizer()
    # 이 스크립트가 있는 폴더 기준이므로, 상위 폴더에서 실행해야 함.
    organizer.organize_project_files("..")


def main():
    parser = argparse.ArgumentParser(description="코드 관리 자동화 시스템")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 분석 커맨드
    p_analyze = subparsers.add_parser('analyze', help='AI로 전체 코드 품질을 분석하고 리포트를 생성합니다.')
    p_analyze.set_defaults(func=run_analysis)

    # 리팩토링 제안 커맨드
    p_refactor = subparsers.add_parser('refactor', help='분석 리포트를 바탕으로 AI 리팩토링을 제안합니다.')
    p_refactor.add_argument('report_file', help='분석 리포트 JSON 파일 경로')
    p_refactor.set_defaults(func=run_refactor)

    # 파일 정리 커맨드
    p_organize = subparsers.add_parser('organize', help='프로젝트 파일을 카테고리별로 자동 정리합니다.')
    p_organize.set_defaults(func=run_organize)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    # 이 스크립트는 maintenance_tools 폴더 안에서 실행되어야 합니다.
    # 예: cd maintenance_tools
    #     python run_maintenance.py analyze
    main() 