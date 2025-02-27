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

    with open("./framework_folder_structures.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
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

        return folder_structure