import os
import json
import logging

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


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
    
    pdf_vector_store_path = "./vector_store/faiss_pdf"
    pdf_path = "./vector_store/vector_store_pdf.pdf"
    json_vector_store_path = "./vector_store/faiss_json"
    json_path = "./vector_store/framework_folder_structures.json"
    embedding_model = SentenceTransformerWrapper("intfloat/multilingual-e5-large")
    vector_store = None  # 벡터 스토어 저장
    index = None  # 벡터 인덱스 저장

    @classmethod
    def search_similar_terms(cls, query: str, top_k: int = 3, type: str = 'pdf'):
        """쿼리와 가장 유사한 개발 용어 검색"""
        if cls.vector_store is None:
            logging.info("벡터 스토어 검색중..")
        if type == 'pdf':
            cls._load_vector_store_pdf()
        elif type == 'json':
            cls._load_vector_store_json()
        results = cls.vector_store.similarity_search(query, k=top_k)
        return [res.page_content for res in results]
    
    @classmethod
    def _load_vector_store_pdf(cls):
        """벡터 스토어 로드"""
        if os.path.exists(cls.pdf_vector_store_path):
            cls.vector_store = FAISS.load_local(cls.pdf_vector_store_path, cls.embedding_model.embed_query, allow_dangerous_deserialization=True)
            cls.index = cls.vector_store.index
            logging.info("✅ 벡터 스토어 로드 완료")
        elif os.path.exists(cls.pdf_path):
            logging.error("⚠ 사전 구축된 벡터 스토어를 찾을 수 없습니다. 신규 벡터 스토어를 구축합니다.")
            docs = cls._build_docs_pdf()

            texts = [doc.page_content for doc in docs]
            embeddings = cls.embedding_model.embed_documents(texts)

            text_embedding_pairs = list(zip(texts, embeddings))

            cls.vector_store = FAISS.from_embeddings(
                text_embeddings=text_embedding_pairs,
                embedding=cls.embedding_model.embed_query
            )

            cls.vector_store.save_local(cls.pdf_vector_store_path)
        else:
            logging.error("❌ PDF를 찾을 수 없습니다. 개발 용어 PDF가 필요합니다.")
            raise FileNotFoundError("Vector store not found. Ensure it is pre-built.")


    @classmethod
    def _build_docs_pdf(cls):
        loader = PyPDFLoader(cls.pdf_path)
        pages = loader.load_and_split()

        full_text = "".join([page.page_content for page in pages])

        # 줄바꿈 기준으로 split하는 Text Splitter 생성
        text_splitter = CharacterTextSplitter(
            separator="\n \n",  # 줄 단위로 분할
            chunk_size=500,    # 한 줄씩 개념을 나누도록 설정
            chunk_overlap=0
        )

        # 문서 데이터 분할
        docs = text_splitter.create_documents([full_text])

        for doc in docs:
            doc.page_content = doc.page_content.replace("\n", "")

        return docs

   
    @classmethod
    def _load_vector_store_json(cls):
        """벡터 스토어 로드"""
        if os.path.exists(cls.json_vector_store_path):
            cls.vector_store = FAISS.load_local(cls.json_vector_store_path, cls.embedding_model.embed_query, allow_dangerous_deserialization=True)
            cls.index = cls.vector_store.index
            logging.info("✅ 벡터 스토어 로드 완료")
        elif os.path.exists(cls.json_path):
            logging.error("⚠ 사전 구축된 벡터 스토어를 찾을 수 없습니다. 신규 벡터 스토어를 구축합니다.")
            docs = cls._build_docs_json()

            # 4️⃣ FAISS Vector Store에 저장
            vector_store = FAISS.from_documents(docs, cls.embedding_model)

            # 5️⃣ FAISS Vector Store 저장 (필요 시)
            vector_store.save_local(cls.json_vector_store_path)
        else:
            logging.error("❌ PDF를 찾을 수 없습니다. 개발 용어 PDF가 필요합니다.")
            raise FileNotFoundError("Vector store not found. Ensure it is pre-built.")


    @classmethod
    def _build_docs_json(cls):
        with open(cls.json_path, "r", encoding="utf-8") as f:
            framework_data = json.load(f)

        docs = []
        for item in framework_data:
            framework_name = item["framework"]
            folder_structure = json.dumps(item["folder_structure"], indent=4, ensure_ascii=False)
            text = f"Framework: {framework_name}\nFolder Structure:\n{folder_structure}"
            docs.append(Document(page_content=text, metadata={"framework": framework_name}))

        return docs