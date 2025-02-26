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
        python_code = result['code']
        messages = result['description']
        
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

        # 기본 파일 이름
        base_code_filename = "generated_code.py"
        base_description_filename = "generated_description.json"
        
        # Python 코드 파일 경로
        code_save_path = os.path.join(project_folder_path, base_code_filename)

        # 설명 파일 경로
        description_save_path = os.path.join(project_folder_path, base_description_filename)

        # 코드 파일이 이미 존재하면 숫자 붙여서 새 이름 생성
        counter = 1
        while os.path.exists(code_save_path):
            code_save_path = os.path.join(project_folder_path, f"generated_code_{counter}.py")
            counter += 1

        # 설명 파일이 이미 존재하면 숫자 붙여서 새 이름 생성
        counter = 1
        while os.path.exists(description_save_path):
            description_save_path = os.path.join(project_folder_path, f"generated_description_{counter}.json")
            counter += 1
        
        # 생성된 Python 코드를 지정된 폴더에 "generated_code.py" 파일로 저장
        with open(code_save_path, "w", encoding="utf-8") as py_file:
            py_file.write(python_code)
        
        # 생성된 설명 메시지를 지정된 폴더에 "generated_description.json" 파일로 JSON 형식으로 저장
        with open(description_save_path, "w", encoding="utf-8") as json_file:
            json.dump({"description": messages}, json_file, ensure_ascii=False, indent=2)
        
        # 생성된 코드와 설명을 콘솔에 출력
        print("=== Python Code ===")
        print(python_code)
        print("\n=== Description ===")
        print(messages)
        return project_folder_path, messages

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
        # 응답을 코드 부분과 설명 부분으로 분리
        code_part, description_part = cls._split_response_content(full_response)
        # 코드 부분의 오류를 검증 및 수정
        # validated_code = cls._validate_and_fix_code_until_no_error(code_part)
        # 최종 코드와 설명을 딕셔너리 형태로 반환
        return {"code": code_part, "description": description_part.strip()}

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
            내게 무언가 답변할 때마다 항상 두 부분으로 나누어 출력해야 해.
            먼저, markdown 코드 블록(예: python 과 ``` 안에) 안에 Python 코드를 출력하고,
            그 다음에 코드에 대한 설명을 출력해.

            이 부분은 매우 중요해. Python 코드를 줄 때 반드시 이 형식을 지켜야 해!!!
            사용자가 요청한 대로 코드가 올바르게 실행될 수 있도록 코드를 작성해야 해.

            🛠️ 필수 요구 사항
            Python 문법 오류(SyntaxError)가 없어야 해.
            실행 시 런타임 오류(RuntimeError)가 발생하지 않아야 해.
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
            "{description}"            
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
    
    # @classmethod
    # def _validate_and_fix_code_until_no_error(cls, code: str, max_attempts: int = 5) -> str:
    #     """
    #     코드가 오류 없이 실행될 때까지 반복적으로 검사 및 수정하는 함수.
    #     최대 max_attempts 번 시도하며, 매 시도마다 발생한 오류 메시지를 누적하여 LLM을 통해 코드 수정 요청.
    #     """
    #     error_messages = []  # 이전 오류 메시지들을 저장하는 리스트
    #     for attempt in range(max_attempts):
    #         # 문법 오류 검사
    #         syntax_error = cls._check_syntax_error(code)
    #         # 실행하여 런타임 오류 검사 및 출력 캡쳐
    #         runtime_error, execution_output = cls._execute_and_capture_output(code)

    #         # 문법 및 런타임 오류가 없으면 수정된 코드를 반환
    #         if not syntax_error and not runtime_error:
    #             return code
            
    #         # 발생한 오류 메시지 생성
    #         error_message = f"Attempt {attempt+1} 오류 발생:\n"
    #         if syntax_error:
    #             error_message += f"Syntax Error: {syntax_error}\n"
    #         if runtime_error:
    #             error_message += f"Runtime Error: {runtime_error}\n"
            
    #         logging.warning(f"⚠️ {error_message.strip()}")
    #         error_messages.append(error_message)
    #         # 누적된 오류 메시지를 바탕으로 LLM에게 코드 수정 요청
    #         code = cls._fix_code_with_llm(code, error_messages)
    #     # 최대 시도 횟수를 초과하면 실패 메시지 반환
    #     return "코드 수정 실패"

    # @staticmethod
    # def _check_syntax_error(code: str) -> str:
    #     """
    #     Python 코드의 문법 오류(SyntaxError)를 검사하는 함수.
    #     - 오류가 없으면 None을 반환
    #     - 오류가 발생하면 오류 메시지를 반환
    #     """
    #     try:
    #         ast.parse(code)
    #         return None
    #     except SyntaxError as e:
    #         return f"{e.msg} (라인: {e.lineno}, 컬럼: {e.offset})"

    # @staticmethod
    # def _execute_and_capture_output(code: str) -> tuple:
    #     """
    #     코드를 실행하여 실행 중 발생하는 오류와 출력 결과를 캡쳐하는 함수.
    #     - 정상 실행 시: (None, 출력 결과)를 반환
    #     - 오류 발생 시: (오류 메시지, 출력 결과)를 반환
    #     """
    #     captured_output = io.StringIO()
    #     captured_error = io.StringIO()

    #     sys.stdout = captured_output  # 표준 출력 리디렉션
    #     sys.stderr = captured_error  # 표준 에러 리디렉션

    #     logging.warning(f"코드 :  {code}")
        
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("error")
    #         try:
    #             exec(code, globals())  # 🔹 실행 환경을 실제 환경과 유사하게 설정
    #             execution_output = captured_output.getvalue()
    #             execution_error = captured_error.getvalue()

    #             logging.warning("✅ 실행 완료, 출력 결과:\n" + execution_output)
    #             if execution_error:
    #                 logging.error("⚠️ 실행 중 오류 발생 (stderr):\n" + execution_error)

    #             return None, captured_output.getvalue()  # 실행 오류 없음
    #         except ValueError as ve:
    #             error_traceback = traceback.format_exc()
    #             logging.error(f"❌ [ValueError] {ve}\n{error_traceback}")
    #             return f"[ValueError] {ve}\n{error_traceback}", captured_output.getvalue()
    #         except TypeError as te:
    #             error_traceback = traceback.format_exc()
    #             logging.error(f"❌ [TypeError] {te}\n{error_traceback}")
    #             return f"[TypeError] {te}\n{error_traceback}", captured_output.getvalue()
    #         except IndexError as ie:
    #             error_traceback = traceback.format_exc()
    #             logging.error(f"❌ [IndexError] {ie}\n{error_traceback}")
    #             return f"[IndexError] {ie}\n{error_traceback}", captured_output.getvalue()
    #         except KeyError as ke:
    #             error_traceback = traceback.format_exc()
    #             logging.error(f"❌ [KeyError] {ke}\n{error_traceback}")
    #             return f"[KeyError] {ke}\n{error_traceback}", captured_output.getvalue()
    #         except ZeroDivisionError as zde:
    #             error_traceback = traceback.format_exc()
    #             logging.error(f"❌ [ZeroDivisionError] {zde}\n{error_traceback}")
    #             return f"[ZeroDivisionError] {zde}\n{error_traceback}", captured_output.getvalue()
    #         except Warning as w:
    #             error_traceback = traceback.format_exc()
    #             logging.error(f"⚠️ [Warning] {w}\n{error_traceback}")
    #             return f"[Warning] {w}\n{error_traceback}", captured_output.getvalue()
    #         except Exception as e:
    #             error_traceback = traceback.format_exc()
    #             logging.error(f"❌ [Unknown Error] {e}\n{error_traceback}")
    #             return f"[Unknown Error] {e}\n{error_traceback}", captured_output.getvalue()
    #         finally:
    #             sys.stdout = sys.__stdout__  # 표준 출력 복원
    #             sys.stderr = sys.__stderr__  # 표준 에러 복원

    # @classmethod
    # def _fix_code_with_llm(cls, code: str, error_messages: list) -> str:
    #     """
    #     누적된 오류 메시지를 기반으로 LLM에게 코드 수정 요청을 하는 함수.
    #     - LLM 응답에서 마크다운 코드 블록을 제거하여 수정된 코드를 반환함.
    #     """
    #     error_context = "\n".join(error_messages)
    #     prompt = f"""
    #     ### Python 코드 오류 수정 요청
    #     아래 코드에서 문법 및 실행 오류를 수정해줘.

    #     ### 수정 목표:
    #     1. 코드가 실행될 때 문법 오류(SyntaxError)가 발생하지 않아야 함.
    #     2. 실행 중 RuntimeError가 발생하지 않아야 함.
    #     3. 기존 코드의 논리 구조를 최대한 유지하면서 오류를 해결할 것.

    #     ### 출력 형식 요구사항:
    #     - 출력된 코드는 실행 가능한 순수한 Python 코드여야 하며, 불필요한 텍스트가 없어야 함.

    #     ### 이전 오류 메시지:
    #     {error_context}

    #     ### 코드 수정 요청:
    #     ```python
    #     {code}
    #     """
    #     response = llm.invoke(prompt)
    #     generated_code = response.content if hasattr(response, 'content') else "코드 수정 실패"
    #     cleaned_code = cls._remove_markdown_code_blocks(generated_code)
    #     return cleaned_code