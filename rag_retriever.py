import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



class SentenceTransformerWrapper:
    """FAISS에서 사용할 수 있도록 `SentenceTransformer`을 래핑한 클래스"""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts: list) -> list:
        """문서 리스트를 벡터로 변환"""
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list:
        """쿼리를 벡터로 변환"""
        return self.model.encode([text], normalize_embeddings=True).tolist()[0]
    

class RAGRetriever:
    """벡터 스토어에서 유사한 개발 용어를 검색하는 RAG 기능"""

    json_data = {
        "FastAPI": {
            "structure": [
                "app/",
                "│── main.py              # FastAPI 앱 실행",
                "│── requirements.txt     # 필요한 패키지 목록",
                "│",
                "├── routers/             # 라우터(API 엔드포인트) 폴더",
                "│   ├── users.py         # 사용자 관련 API",
                "│   ├── items.py         # 아이템 관련 API",
                "│",
                "└── models/              # Pydantic 모델 정의 폴더",
                "    ├── user_model.py    # 사용자 데이터 모델",
                "    ├── item_model.py    # 아이템 데이터 모델"
            ]
        },
        "Django": {
            "structure": [
                "project_name/",
                "│── manage.py            # Django 관리 명령어 실행 파일",
                "│── requirements.txt     # 필요한 패키지 목록",
                "│",
                "├── project_name/        # Django 프로젝트 폴더",
                "│   ├── __init__.py",
                "│   ├── settings.py      # 프로젝트 설정 파일",
                "│   ├── urls.py         # 프로젝트 URL 라우팅",
                "│   ├── wsgi.py         # WSGI 서버 진입점",
                "│   ├── asgi.py         # ASGI 서버 진입점 (선택)",
                "│",
                "├── apps/                # Django 앱 폴더",
                "│   ├── app_name/",
                "│   │   ├── migrations/  # 마이그레이션 파일 폴더",
                "│   │   ├── models.py   # 데이터베이스 모델 정의",
                "│   │   ├── views.py    # 뷰 로직",
                "│   │   ├── urls.py     # 앱 별 URL 라우팅",
                "│   │   ├── serializers.py # REST API 직렬화 (선택)",
                "│",
                "└── templates/           # HTML 템플릿 폴더 (선택)"
            ]
        },
        "Flask": {
            "structure": [
                "project_name/",
                "│── app.py               # Flask 앱 실행 파일",
                "│── requirements.txt     # 필요한 패키지 목록",
                "│",
                "├── static/              # 정적 파일 (CSS, JS 등)",
                "├── templates/           # HTML 템플릿 폴더",
                "│",
                "├── models.py            # 데이터 모델 정의",
                "├── views.py             # 라우터 및 API 로직",
                "├── config.py            # 설정 파일",
                "│",
                "└── instance/            # 환경 설정 파일 저장 폴더 (선택)"
            ]
        }
    }

    embedding_model = SentenceTransformerWrapper("intfloat/multilingual-e5-large")

    @classmethod
    def search_similar_terms(cls, query: str):
        """쿼리와 가장 유사한 개발 프레임워크 검색"""
        framework_keys = list(cls.json_data.keys())

        user_embedding = np.array(cls.embedding_model.embed_query(query)).reshape(1, -1)

        framework_embeddings = np.array([cls.embedding_model.embed_query(fw) for fw in framework_keys])
            
        # 유사도 계산 (코사인 유사도)
        similarities = cosine_similarity(user_embedding, framework_embeddings)[0]

        # 가장 유사한 프레임워크 찾기
        best_match_index = np.argmax(similarities)
        best_framework = framework_keys[best_match_index]

        folder_structure = "\n".join(cls.json_data[best_framework]['structure'])

        return best_framework, folder_structure