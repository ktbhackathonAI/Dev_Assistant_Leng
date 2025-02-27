import logging
import uvicorn

from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어 임포트
from config import settings  # 환경 변수 로드
from fastapi import FastAPI, HTTPException
from config import settings
from pydantic import BaseModel
from typing import Optional, List, Dict
from fastapi.responses import StreamingResponse
import asyncio
from prompt_engineering import CodeGenerator


app = FastAPI(
    title="Python Code Generator API (LangChain)",
    description="LangChain + Gemini 기반 Python 코드 생성 API",
    version="2.0.0"
)
    
origins = [
    settings.BACK_URL,      # "http://frontend:8000"
    settings.FRONT_URL,      # "http://frontend:8000"
]
# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 허용된 출처 목록
    allow_credentials=True,         # 인증 정보(쿠키, Authorization 헤더 등) 허용
    allow_methods=["*"],            # 모든 HTTP 메서드 허용 (GET, POST, PUT 등)
    allow_headers=["*"],            # 모든 헤더 허용
)

# 요청 데이터 모델 정의
class Room(BaseModel):
    id: int
    name: Optional[str] = None  # None 허용하도록 수정
    created_at: Optional[str] = None

class MessageHistory(BaseModel):  # 메시지 히스토리용 새로운 모델
    content: str
    sender: str
    created_at: str

class Message(BaseModel):
    content: str
    role: str

class RequestData(BaseModel):
    room: Room
    message_history: List[MessageHistory]  # 새로운 모델로 변경
    new_message: Message
    type: Optional[str] = "makecode"

# 기본 엔드포인트: 루트 경로
@app.get("/")
async def root():
    return {"message": "Welcome to the RAG System API"}


@app.post("/generate-code")
async def generate_code_api(data: RequestData):
    """LangChain 기반 비동기 코드 생성 API"""
    try:
        key, value = await CodeGenerator.run_code_generation(data)
        return {key: value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))