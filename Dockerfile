# 1. 베이스 이미지 선택
FROM python:3.10

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 환경 변수 설정
ENV PYTHONUNBUFFERED 1
# Hugging Face 모델 캐시 폴더 지정 (권장)
ENV SENTENCE_TRANSFORMERS_HOME /app/cache

# 4. 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('dragonkue/BGE-m3-ko')"
# ------------------------------------------------------------------

# 5. 소스 코드 복사
COPY . .

# 6. 포트 노출
EXPOSE 8080

# 7. 서버 실행
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]