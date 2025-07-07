# 1. 베이스 이미지 선택
# Python 3.11 슬림 버전을 사용하여 이미지 크기를 최적화합니다.
FROM python:3.10

# 2. 작업 디렉토리 설정
# 컨테이너 내에서 모든 작업이 이루어질 기본 폴더를 /app으로 설정합니다.
WORKDIR /app

# 3. 환경 변수 설정
# Python이 로그를 버퍼링 없이 즉시 출력하도록 설정하여 Docker 로그에서 실시간 확인이 가능하게 합니다.
ENV PYTHONUNBUFFERED 1

# 4. 의존성 설치
# 먼저 requirements.txt 파일만 복사하여 의존성을 설치합니다.
# 이렇게 하면 소스 코드가 변경되어도 의존성이 변경되지 않았다면 Docker 캐시를 활용하여 빌드 속도를 높일 수 있습니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY precache_models.py .
RUN python precache_models.py

# 5. 소스 코드 복사
# 프로젝트의 모든 파일을 컨테이너의 /app 디렉토리로 복사합니다.
COPY . .

# 6. 포트 노출
# 애플리케이션이 8000번 포트를 사용하므로, 컨테이너의 8000번 포트를 외부에 노출하도록 설정합니다.
EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8282"]