# 이 스크립트는 모델을 다운로드하여 캐시를 생성하기 위한 용도입니다.
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import os

print("--- 모델 캐싱 시작 ---")

# .env 파일에서 모델 이름을 가져오거나 기본값을 사용합니다.
embedding_model = os.getenv("EMBEDDING_MODEL", 'dragonkue/snowflake-arctic-embed-l-v2.0-ko')
reranker_model = os.getenv("RERANKER_MODEL", 'BAAI/bge-reranker-v2-m3')

print(f"임베딩 모델 다운로드: {embedding_model}")
SentenceTransformer(embedding_model)

print(f"리랭커 모델 다운로드: {reranker_model}")
FlagReranker(reranker_model)

print("--- 모델 캐싱 완료 ---")