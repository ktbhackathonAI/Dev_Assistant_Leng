from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()


class Settings:
    # SQLAlchemy가 사용할 데이터베이스 연결 URL
    BACK_URL = os.getenv("BACK_URL")
    FRONT_URL = os.getenv("FRONT_URL")

# 설정 객체 인스턴스 생성
settings = Settings()
