# -*- coding: utf-8 -*-
"""
데이터베이스 관리 모듈 (비동기 SQLite)
- 분석 결과(개별 종목, 포트폴리오, 리포트)를 영속적으로 저장하고 관리합니다.
"""
import aiosqlite
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    aiosqlite를 사용하여 비동기적으로 데이터베이스를 관리하는 클래스입니다.
    """
    def __init__(self, db_path="k_stock_analysis.db"):
        """
        DatabaseManager를 초기화합니다.
        :param db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self._db = None

    async def connect(self):
        """데이터베이스에 연결하고 테이블을 초기화합니다."""
        if self._db:
            return
        try:
            self._db = await aiosqlite.connect(self.db_path)
            await self._initialize_tables()
            logger.info(f"🗄️ 데이터베이스 '{self.db_path}'에 성공적으로 연결되었습니다.")
        except aiosqlite.Error as e:
            logger.error(f"❌ 데이터베이스 연결 실패: {e}")
            self._db = None
            raise

    async def close(self):
        """데이터베이스 연결을 닫습니다."""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("🗄️ 데이터베이스 연결이 닫혔습니다.")

    async def _initialize_tables(self):
        """필요한 테이블이 없는 경우 생성합니다."""
        async with self._db.cursor() as cursor:
            # 개별 종목 분석 결과 저장 테이블
            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS individual_stock_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                investment_score INTEGER,
                analysis_timestamp TEXT NOT NULL,
                analysis_data TEXT NOT NULL,
                report_id INTEGER,
                FOREIGN KEY (report_id) REFERENCES analysis_reports(id)
            )
            """)
            # 포트폴리오 추천 이력 저장 테이블
            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_data TEXT NOT NULL,
                analysis_timestamp TEXT NOT NULL,
                report_id INTEGER,
                FOREIGN KEY (report_id) REFERENCES analysis_reports(id)
            )
            """)
            # 최종 리포트 정보 저장 테이블
            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
        await self._db.commit()
        logger.info("✅ 데이터베이스 테이블 초기화 완료.")

    async def save_report(self, report_path: str) -> int:
        """
        생성된 리포트 정보를 저장하고 ID를 반환합니다.
        :param report_path: 저장된 리포트 파일의 경로
        :return: 새로 생성된 리포트의 ID
        """
        timestamp = datetime.now().isoformat()
        async with self._db.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO analysis_reports (report_path, created_at) VALUES (?, ?)",
                (report_path, timestamp)
            )
            await self._db.commit()
            return cursor.lastrowid

    async def save_individual_analysis(self, analysis_result: dict, report_id: int):
        """
        개별 종목 분석 결과를 저장합니다.
        :param analysis_result: GeminiAnalyzer에서 반환된 분석 결과 딕셔너리
        :param report_id: 연관된 리포트의 ID
        """
        async with self._db.cursor() as cursor:
            await cursor.execute("""
            INSERT INTO individual_stock_analysis 
            (stock_code, investment_score, analysis_timestamp, analysis_data, report_id)
            VALUES (?, ?, ?, ?, ?)
            """, (
                analysis_result['stock_code'],
                analysis_result['investment_score'],
                analysis_result['analysis_timestamp'],
                json.dumps(analysis_result, ensure_ascii=False),
                report_id
            ))
        await self._db.commit()
        logger.debug(f"💾 종목 [{analysis_result['stock_code']}] 분석 결과 DB 저장 완료.")
    
    async def save_portfolio_recommendation(self, portfolio_data: dict, report_id: int):
        """
        포트폴리오 추천 결과를 저장합니다.
        :param portfolio_data: GeminiAnalyzer에서 반환된 포트폴리오 딕셔너리
        :param report_id: 연관된 리포트의 ID
        """
        timestamp = datetime.now().isoformat()
        async with self._db.cursor() as cursor:
            await cursor.execute("""
            INSERT INTO portfolio_history (portfolio_data, analysis_timestamp, report_id)
            VALUES (?, ?, ?)
            """, (
                json.dumps(portfolio_data, ensure_ascii=False),
                timestamp,
                report_id
            ))
        await self._db.commit()
        logger.debug("💾 포트폴리오 추천 결과 DB 저장 완료.")

if __name__ == '__main__':
    # 모듈 직접 실행 시 테스트 코드
    async def test_db_manager():
        logging.basicConfig(level=logging.INFO)
        db_manager = DatabaseManager(db_path="test_analysis.db")
        try:
            await db_manager.connect()
            
            # 1. 테스트 리포트 정보 저장
            report_id = await db_manager.save_report("reports/test_report_01.txt")
            print(f"테스트 리포트 저장 완료, ID: {report_id}")

            # 2. 테스트 개별 종목 분석 결과 저장
            sample_analysis = {
                "stock_code": "005930",
                "investment_score": 85,
                "analysis_timestamp": datetime.now().isoformat(),
                "gemini_raw_response": "매우 긍정적인 분석 결과..."
            }
            await db_manager.save_individual_analysis(sample_analysis, report_id)
            print("테스트 개별 종목 분석 저장 완료.")

            # 3. 테스트 포트폴리오 결과 저장
            sample_portfolio = {
                "portfolio_summary": "안정성과 성장성을 겸비한 포트폴리오",
                "top_10_portfolio": [{"stock_code": "005930", "weight": 20}]
            }
            await db_manager.save_portfolio_recommendation(sample_portfolio, report_id)
            print("테스트 포트폴리오 저장 완료.")

        except Exception as e:
            print(f"테스트 중 오류 발생: {e}")
        finally:
            await db_manager.close()

    asyncio.run(test_db_manager()) 