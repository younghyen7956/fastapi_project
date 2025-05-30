import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import subprocess  # 추가된 부분

from RAG.controller.simply_rag_controller import RAGRouter
from RAG.repository.simple_rag_repository_impl import RAGRepositoryImpl

load_dotenv()
app = FastAPI(debug=True)

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(RAGRouter)


@app.on_event("startup")
async def on_startup():
    # 현재 파일의 위치를 기준으로 상위 폴더 경로 설정
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent  # 이 파일이 main.py이고 프로젝트 루트 바로 아래 있다면 .parent로 충분할 수 있습니다. 구조에 맞게 조정하세요.

    chroma_db_path = project_root / "chroma_db"
    vector_db_script_path = project_root / "vector_db_insert_data.py"

    if not chroma_db_path.exists() or not any(chroma_db_path.iterdir()):  # 폴더가 없거나, 비어있을 경우
        print(f"'{chroma_db_path}' 폴더가 존재하지 않거나 비어있습니다.")
        if vector_db_script_path.exists():
            print(f"'{vector_db_script_path}' 스크립트를 실행합니다.")
            try:
                # 스크립트 실행 (venv 환경을 사용한다면 python 경로를 해당 venv의 python으로 지정하는 것이 좋습니다)
                # 예: python_executable = project_root / ".venv" / "bin" / "python" (Linux/macOS)
                # python_executable = project_root / ".venv" / "Scripts" / "python.exe" (Windows)
                # 여기서는 시스템의 기본 python을 사용한다고 가정합니다.
                result = subprocess.run(["python", str(vector_db_script_path)], capture_output=True, text=True,
                                        check=True)
                print("스크립트 실행 성공:")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print("스크립트 실행 중 오류 발생:")
                print(e.stderr)
            except FileNotFoundError:
                print(f"오류: 'python' 인터프리터를 찾을 수 없습니다. PATH를 확인하거나 가상환경의 python 경로를 명시해주세요.")
        else:
            print(f"오류: '{vector_db_script_path}' 스크립트 파일을 찾을 수 없습니다.")
    else:
        print(f"'{chroma_db_path}' 폴더가 이미 존재합니다. 별도 작업을 수행하지 않습니다.")

    repo = RAGRepositoryImpl.getInstance()
    # (원한다면 임베딩 캐시나 인덱스 로드도 이곳에서)
    print("✅ Startup: DB logic checked.")  # 메시지 약간 수정


if __name__ == "__main__":
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", 8000))
    uvicorn.run(app, host=host, port=port, log_level="debug")