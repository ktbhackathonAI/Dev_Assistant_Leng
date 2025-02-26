import os
import re
import io
import sys
import ast
import logging
import asyncio
import warnings
import traceback
import json  

from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# from rag import RAGRetriever

# 환경 변수 로드 (예: API 키)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 로그 설정 (오류만 출력)
logging.basicConfig(level=logging.ERROR)

# LangChain LLM (Gemini 모델) 초기화
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# 코드 스타일 옵션 정의 (PEP8, Google, NoneStyle)
class CodeStyle(str, Enum):
    PEP8 = "PEP8"
    Google = "Google"
    NoneStyle = "None"

# 코드 구조 옵션 정의 (함수형, 클래스형)
class CodeStructure(str, Enum):
    Functional = "functional"
    ClassBased = "class-based"

# 코드 요청 정보를 담는 데이터 모델 (Pydantic 사용)
class CodeRequest(BaseModel):
    description: str             # 생성할 코드에 대한 설명
    style: CodeStyle = CodeStyle.PEP8         # 코드 스타일 (기본값: PEP8)
    include_comments: bool = True             # 주석 포함 여부 (기본값: True)
    structure: CodeStructure = CodeStructure.Functional  # 코드 구조 (기본값: 함수형)

# 코드 생성기를 담당하는 클래스
class CodeGenerator:
    """Python 코드 생성기 (RAG 미적용)"""

    @classmethod
    # 비동기 함수: 코드 생성 실행 및 결과 파일 저장
    async def run_code_generation(cls, request: CodeRequest):
        """
        코드 생성 요청을 실행하고,
        생성된 Python 코드를 파일과 JSON 형식의 설명 파일로 저장하며 출력하는 함수.
        """
        # 비동기적으로 코드 생성 수행
        result = await CodeGenerator.generate_code(request)
        
        # 원하는 폴더 경로 설정 (서버의 특정 폴더)
        base_folder_path = "/root/docker/generate_projects"
        
        # 폴더가 존재하지 않으면 생성
        if not os.path.exists(base_folder_path):
            os.makedirs(base_folder_path)

        # 프로젝트 폴더 개수 확인 후 새로운 프로젝트 폴더 번호 할당
        existing_projects = [name for name in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, name))]
        project_counter = len(existing_projects) + 1
        project_folder_path = os.path.join(base_folder_path, f"project{project_counter}")

        # 새 프로젝트 폴더 생성
        os.makedirs(project_folder_path)

        
        # 마크다운 블록을 파싱하여 파일별로 저장
        md_pattern = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)
        files_text = md_pattern.findall(result)
        description_text = re.sub(md_pattern, "", result).strip()

        # print(files_text)
        # 파일 저장
        for file_text in files_text:
            name_pattern = re.compile(r"# ([^\n]+)\n([\s\S]*)", re.DOTALL)
            code_match = re.search(name_pattern, file_text)

            content, filename = code_match.group(0), code_match.group(1)
           
            # Python 코드 파일 경로
            code_save_path = os.path.join(project_folder_path, filename)
            directory = os.path.dirname(code_save_path)  # 폴더 경로만 추출

            # 🔹 폴더가 없으면 생성 (이미 존재하면 무시)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            # 생성된 Python 코드를 지정된 폴더에 "generated_code.py" 파일로 저장
            with open(code_save_path, "w", encoding="utf-8") as py_file:
                py_file.write(content)
        
        
        # Description 파일 경로
        description_save_path = os.path.join(project_folder_path, 'description.json')

        # 생성된 설명 메시지를 지정된 폴더에 "generated_description.json" 파일로 JSON 형식으로 저장
        with open(description_save_path, "w", encoding="utf-8") as json_file:
            json.dump({"description": description_text}, json_file, ensure_ascii=False, indent=2)

        return project_folder_path

    @classmethod
    async def generate_code(cls, request: CodeRequest, model: str = "gemini-1.5-flash") -> dict:
        """
        비동기 방식으로 Gemini API를 호출하여 코드를 생성하는 함수.
        1. 요청 정보를 바탕으로 프롬프트 생성
        2. LLM 호출하여 응답 받기
        3. 응답을 코드 부분과 설명 부분으로 분리
        4. 코드 부분의 오류를 검증 및 수정하여 최종 코드를 반환
        """
        # 요청 정보를 바탕으로 프롬프트 생성
        prompt = cls._generate_prompt(request)
        # LLM을 비동기적으로 호출 (동기 함수를 executor로 실행)
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: llm.invoke(prompt)
        )
        # LLM의 응답에서 content를 추출 (없으면 "코드 생성 실패" 메시지)
        full_response = response.content if hasattr(response, 'content') else "코드 생성 실패"
        return full_response
        # 응답을 코드 부분과 설명 부분으로 분리
        # code_part, description_part = cls._split_response_content(full_response)
        # return {"code": code_part, "description": description_part.strip()}

    @classmethod
    def _generate_prompt(cls, request: CodeRequest) -> str:
        """
        LangChain의 PromptTemplate을 사용하여 최적화된 프롬프트를 생성하는 함수.
        요청에 포함된 설명, 코드 스타일, 주석 포함 여부, 코드 구조 정보를 템플릿에 채워서 반환.
        """
        include_comments_text = "포함" if request.include_comments else "제외"
        structure_text = "함수형" if request.structure == CodeStructure.Functional else "클래스형"

        # rag_prompt = RAGRetriever.search_similar_terms(request.description)

        template = PromptTemplate(
            input_variables=["description", "style", "include_comments", "structure"],
            template="""
            너는 Python 코드 생성을 전문으로 하는 AI야.
            사용자 입력에 해당하는 기능을 구현해야 해.

            사용자 입력:
            "{description}"

            기능을 구현하기 위해 아래 작업 순서를 반드시 따라야 해.

            작업 순서
            1️⃣ **프로젝트 폴더 구조 설계**
                - root 디렉토리를 기반으로 해당 기능을 배포할 수 있는 전체 코드 구조를 우선적으로 설계해야 해.
                - 프로젝트 폴더 구조는 출력하지 않아야 해.
            2️⃣ **각 파일 별 코드 구현**
                - 각 파일에 해당하는 기능의 코드를 구현해야 해.
                - 파일 별로 markdown 코드 블록(```python ... ```) 안에 파일 경로, 코드 구조를 출력해야 해(requirements.txt 파일 포함).
                - 전체 코드 구조 출력이 끝난 후에 **코드 설명**을 출력해야 해.

            이 부분은 매우 중요해. Python 코드를 줄 때 반드시 이 형식을 지켜야 해!!!
            사용자가 요청한 대로 코드가 올바르게 실행될 수 있도록 코드를 작성해야 해.

            🛠️ 필수 요구 사항
            Python 문법 오류(SyntaxError)가 없어야 해.
            실행 시 런타임 오류(RuntimeError)가 발생하지 않아야 해.
            각 파일 별 기능을 참조할 시 오류(ImportError)가 발생하지 않아야 해.
            사용자 입력에 배포 프레임워크가 특정되어 있지 않으면 기본값으로 FastAPI를 사용해서 배포해야 해.
            코드의 논리는 정확해야 하며, 예상된 출력이 나와야 해.

            🎨 코드 스타일 & 구조
            코드 스타일: {style}
            주석 포함 여부: {include_comments}
            코드 구조: {structure}

            📌 📢 중요한 출력 형식 요구 사항
            출력된 코드는 시작과 끝에 불필요한 텍스트 없이 바로 실행 가능해야 해.
            예제 코드가 필요한 경우, Python 주석(#)을 사용하여 추가해야 해.
            불필요한 설명 없이 순수한 Python 코드만 출력해.
            백점 만점의 점수로 평가됩니다.

            🎯 코드 생성 요청: 이제 Python 코드와 설명을 생성해. 설명은 한국어로 작성해야 해.
            """
        )
        # 템플릿에 요청 정보를 채워 최종 프롬프트 생성
        return template.format(
            description=request.description,
            style=request.style.value,
            include_comments=include_comments_text,
            structure=structure_text
            # rag_prompt=rag_prompt
        )

    @staticmethod
    def _split_response_content(response_content: str) -> (str, str):
        """
        응답 문자열에서 첫번째 markdown 코드 블록을 코드 부분으로 추출하고,
        나머지 부분은 설명으로 취급하는 함수.
        - 만약 markdown 코드 블록이 없으면, 전체 응답을 코드로 간주.
        """
        code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", response_content, re.DOTALL)
        if code_match:
            code_part = code_match.group(1)
            description_part = response_content.replace(code_match.group(0), "")
            return code_part, description_part
        return response_content, ""

    @staticmethod
    def _remove_markdown_code_blocks(code: str) -> str:
        """
        마크다운 코드 블록(예: ```python ... ```)을 제거하여 순수한 코드만 남기는 함수.
        """
        cleaned_code = re.sub(r"```(python)?\n?", "", code)
        cleaned_code = re.sub(r"```\n?", "\n", cleaned_code)
        return cleaned_code.strip()