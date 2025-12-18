import os
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager  # <--- 1. 임포트 추가

from RAG.controller.simply_rag_controller import RAGRouter
from RAG.repository.simple_rag_repository_impl import RAGRepositoryImpl
# from RAG.repository.simple_rag_repository_impl import RAGRepositoryImpl # 사용하지 않으므로 주석 처리
from VLRag.controller.vl_rag_controller import VlRAGRouter
from VLRag.repository.vl_rag_repository_impl import VlRAGRepositoryImpl
from ocr_vl_rag.controller.ocr_rag_controller import OcrRAGRouter
from ocr_vl_rag.repository.ocr_rag_repository_impl import OcrRAGRepositoryImpl

load_dotenv()


# 2. lifespan 컨텍스트 매니저 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 애플리케이션 시작 시 실행될 코드 ---
    print("--- 🚀 FastAPI 애플리케이션 시작 (lifespan) ---")
    # RAGRepositoryImpl.getInstance()
    # VlRAGRepositoryImpl.getInstance()
    OcrRAGRepositoryImpl.getInstance()# 모델 및 리소스 초기화
    print("✅ VlRAG Repository 초기화 완료.")

    yield  # 이 시점에서 애플리케이션이 실행됩니다.

    # --- 애플리케이션 종료 시 실행될 코드 ---
    print("--- 🌙 FastAPI 애플리케이션 종료 ---")
    # 예: 데이터베이스 연결 종료, 리소스 해제 등의 로직


# 3. FastAPI 앱에 lifespan 등록
app = FastAPI(debug=True, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(RAGRouter)
app.include_router(VlRAGRouter)
app.include_router(OcrRAGRouter)

if __name__ == "__main__":
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", 8080))
    uvicorn.run(app, host=host, port=port, log_level="debug")