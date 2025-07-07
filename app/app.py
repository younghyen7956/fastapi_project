import os
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from RAG.controller.simply_rag_controller import RAGRouter
from RAG.repository.simple_rag_repository_impl import RAGRepositoryImpl

load_dotenv()
app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(RAGRouter)

@app.on_event("startup")
async def on_startup():
    print("--- 🚀 FastAPI 애플리케이션 시작 ---")
    # 애플리케이션 시작 시 RAG 리포지토리 인스턴스를 초기화하여
    # ChromaDB 서버에 정상적으로 연결되는지 확인합니다.
    RAGRepositoryImpl.getInstance()
    print("✅ RAG Repository 초기화 완료.")

if __name__ == "__main__":
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", 8000))
    uvicorn.run(app, host=host, port=port, log_level="debug")