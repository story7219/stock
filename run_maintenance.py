import difflib # 코드 차이를 비교하기 위한 라이브러리
import json
import os
import logging
import argparse
from datetime import datetime

def run_refactor(args):
    """AI 리팩토링 제안을 JSON 파일로 생성"""
    logging.info("🚀 AI 리팩토링 제안 파일 생성을 시작합니다.")
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

    proposal_path = "refactoring_proposals.json"
    with open(proposal_path, 'w', encoding='utf-8') as f:
        json.dump(proposals, f, indent=4, ensure_ascii=False)
    
    logging.info(f"✅ AI 리팩토링 제안을 '{proposal_path}' 파일에 저장했습니다.")
    logging.info("이 파일을 다운로드하여 `apply` 명령어로 반자동 리팩토링을 진행하세요.")

def run_apply(args):
    """사용자 승인을 받아 리팩토링을 적용"""
    logging.info("🚀 반자동 리팩토링 적용을 시작합니다.")
    try:
        with open(args.proposal_file, 'r', encoding='utf-8') as f:
            proposals = json.load(f)
    except FileNotFoundError:
        logging.error(f"제안 파일을 찾을 수 없습니다: {args.proposal_file}")
        return

    for file_path, new_code in proposals.items():
        # 프로젝트 루트 경로 기준으로 파일 경로 재구성
        absolute_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', file_path))
        
        if not os.path.exists(absolute_file_path):
            logging.warning(f"파일을 찾을 수 없어 건너뜁니다: {absolute_file_path}")
            continue

        with open(absolute_file_path, 'r', encoding='utf-8') as f:
            original_code = f.read()

        # diff 생성 및 출력
        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
        print("\n" + "="*80)
        print(f"📄 리팩토링 제안: {file_path}")
        print("="*80)
        
        has_diff = False
        for line in diff:
            has_diff = True
            if line.startswith('+'):
                print(f"\033[92m{line.strip()}\033[0m")  # 녹색
            elif line.startswith('-'):
                print(f"\033[91m{line.strip()}\033[0m")  # 빨간색
            else:
                print(line.strip())
        
        if not has_diff:
            print("코드 변경 사항이 없습니다.")
            continue

        # 사용자 입력
        while True:
            choice = input("👉 이 변경사항을 적용하시겠습니까? [y]es, [n]o, [s]kip, [q]uit: ").lower()
            if choice in ['y', 'n', 's', 'q']:
                break
        
        if choice == 'y':
            with open(absolute_file_path, 'w', encoding='utf-8') as f:
                f.write(new_code)
            print(f"✅ '{file_path}'에 변경사항이 적용되었습니다.")
        elif choice == 'n':
            print(f"❌ '{file_path}' 변경을 거부했습니다.")
        elif choice == 's':
            print(f"⏭️ '{file_path}' 변경을 건너뛰었습니다.")
        elif choice == 'q':
            print("반자동 리팩토링을 종료합니다.")
            return

def apply_changes_non_interactive(proposal_file):
    """(자동화용) 제안 파일의 내용을 실제 파일에 적용"""
    logging.info(f"비대화형 리팩토링 적용 시작: '{proposal_file}'")
    try:
        with open(proposal_file, 'r', encoding='utf-8') as f:
            proposals = json.load(f)
    except FileNotFoundError:
        logging.error(f"제안 파일을 찾을 수 없음: {proposal_file}")
        return False # 실패를 나타내기 위해 False 반환

    if not proposals:
        logging.info("적용할 제안이 없습니다.")
        return False # 변경사항 없음을 나타내기 위해 False 반환

    for file_path, new_code in proposals.items():
        # 프로젝트 루트 경로 기준으로 파일 경로 재구성
        absolute_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', file_path))
        if not os.path.exists(absolute_file_path):
            logging.warning(f"파일을 찾을 수 없어 건너뜁니다: {absolute_file_path}")
            continue
        
        logging.info(f"'{absolute_file_path}' 파일에 변경사항 적용 중...")
        with open(absolute_file_path, 'w', encoding='utf-8') as f:
            f.write(new_code)
    
    logging.info("모든 변경사항 적용 완료.")
    return True # 성공을 나타내기 위해 True 반환

def run_organize(args):
    """파일 자동 정리 실행"""
    # ... (기존과 동일)

def main():
    parser = argparse.ArgumentParser(description="코드 관리 자동화 시스템")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 분석
    p_analyze = subparsers.add_parser('analyze', help='AI로 전체 코드 품질을 분석하고 리포트를 생성합니다.')
    p_analyze.set_defaults(func=run_analysis)

    # 제안 생성
    p_refactor = subparsers.add_parser('refactor', help='분석 리포트를 바탕으로 AI 리팩토링 제안 파일을 생성합니다.')
    p_refactor.add_argument('report_file', help='분석 리포트 JSON 파일 경로')
    p_refactor.set_defaults(func=run_refactor)

    # (수동) 제안 적용
    p_apply = subparsers.add_parser('apply', help='(수동) 제안 파일을 바탕으로 반자동 리팩토링을 적용합니다.')
    p_apply.add_argument('proposal_file', help='리팩토링 제안 JSON 파일 경로')
    p_apply.set_defaults(func=p_apply.set_defaults(func=run_apply))

    # (자동) 제안 적용
    p_apply_auto = subparsers.add_parser('apply_auto', help='(자동화용) 제안 파일을 비대화형으로 즉시 적용합니다.')
    p_apply_auto.add_argument('proposal_file', help='리팩토링 제안 JSON 파일 경로')
    p_apply_auto.set_defaults(func=lambda args: apply_changes_non_interactive(args.proposal_file))

    # 파일 정리
    p_organize = subparsers.add_parser('organize', help='프로젝트 파일을 카테고리별로 자동 정리합니다.')
    p_organize.set_defaults(func=run_organize)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main() 