import logging
import uvicorn

from fastapi import FastAPI, HTTPException
from prompt_engineering import CodeGenerator, CodeRequest


app = FastAPI(
    title="Python Code Generator API (LangChain)",
    description="LangChain + Gemini 기반 Python 코드 생성 API",
    version="2.0.0"
)


@app.post("/generate-code")
async def generate_code_api(request: CodeRequest):
    """LangChain 기반 비동기 코드 생성 API"""
    try:
        key, value = await CodeGenerator.run_code_generation(request)
        return {key: value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  