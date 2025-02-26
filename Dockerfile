# 1. Python 3.13 이미지 사용
FROM python:3.13

# 2. 기본 작업 디렉토리 설정 제거 (현재 디렉토리 사용)
# WORKDIR /app  # ❌ 제거

# 3. 필수 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 환경 변수 파일 추가
# COPY .env .env

# 5. 애플리케이션 코드 복사
COPY . .

# 6. FastAPI 실행 (현재 디렉토리 기준)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]