import os
import zipfile
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings  # 타입 임포트 확인
import numpy as np
import inspect

# 원본 파일과 임시 복구 파일 경로
fixed = Path.cwd() / "건축_데이터_fixed.xlsx"

# 2) 복구된 파일 읽기
df = pd.read_excel(
    fixed,
    sheet_name="Page",
    engine="openpyxl"  # 이제는 정상적으로 로드될 것입니다.
)

# ex_df = pd.read_excel(
#     fixed,
#     sheet_name="data",
#     engine="openpyxl"  # 이제는 정상적으로 로드될 것입니다.
# )

# 사용할 컬럼 인덱스
cols = [0, 1, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26]

# 1) 3행의 NaN은 4행 값으로 채워서 헤더 이름 획득
header = df.iloc[3, cols].fillna(df.iloc[4, cols]).tolist()

# 2) 바로 위(2행)의 값 (여기선 '설계자 답변', '검증위원 평가' 등) 준비
parent = df.iloc[2, cols].astype(str).tolist()

# 3) 고유 이름 생성
counts = {}
unique_names = []

for i, name in enumerate(header):
    # ① 만약 '내용' 컬럼이면, 이전 이름 뒤에 ' 내용' 추가
    if name == "내용":
        if unique_names:
            unique_names.append(f"{unique_names[-1]} 내용")
        else:
            # 만약 첫 컬럼이 '내용'이라면 parent로 대체
            unique_names.append(f"{parent[i]} 내용")
        continue

    # ② 일반 컬럼명 중복 처리: 두 번째부터 _1, _2 …
    counts[name] = counts.get(name, 0) + 1
    if counts[name] == 1:
        unique_names.append(name)
    else:
        unique_names.append(f"{name}_{counts[name] - 1}")

# print(unique_names)

fix_data = df.iloc[5:, cols].reset_index(drop=True)
fix_data.columns = unique_names
fix_data = fix_data.iloc[:-1, :]

desired_cols = [
    "연번", "설계단계", "ID번호", "공종",
    "검증위원", "도면명", "도면번호",
    "세부분류", "참조용"
]

# 2) metadata_cols는 원하는 것만
metadata_cols = [c for c in fix_data.columns if c in desired_cols]

# 3) document용 combined_text에는 **모든** 컬럼 사용
#    fix_data에 combined_text 컬럼이 아직 없다고 가정
all_cols = [c for c in fix_data.columns]  # combined_text 전이라면, fix_data.columns 그대로


def make_combined_text_all(row: pd.Series) -> str:
    parts = []
    for col in all_cols:
        v = row[col]
        if pd.isna(v) or str(v).strip() == "":
            continue
        parts.append(f"{col}: {v}")
    return "\n".join(parts)


fix_data["combined_text"] = fix_data.apply(make_combined_text_all, axis=1)

# 4) documents, metadatas, ids 준비
documents = fix_data["combined_text"].tolist()
all_metadatas = fix_data[metadata_cols].to_dict(orient="records")
all_ids = [str(i) for i in df.index[5:-1]]

# Cell 6을 아래 코드로 대체하세요

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings  # 타입 임포트 확인
from sentence_transformers import SentenceTransformer
import numpy as np
import shutil  # DB 폴더 관리를 위해
import inspect  # 시그니처 검사용

sbert_model_instance = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko")

# 2. ChromaDB EmbeddingFunction 프로토콜을 따르는 클래스 정의
print("2. 커스텀 임베딩 함수 클래스 정의 중...")


class StrictEmbeddingFunction(EmbeddingFunction):  # 클래스 이름 변경
    def __init__(self, model: SentenceTransformer):
        self.sbert_model = model  # 내부 변수 이름 명확히

    def __call__(self, input_documents: Documents) -> Embeddings:  # 파라미터 이름 변경
        if isinstance(input_documents, str):
            texts = [input_documents]
        elif isinstance(input_documents, list) and all(isinstance(doc, str) for doc in input_documents):
            texts = input_documents
        else:
            try:
                texts = [str(doc) for doc in input_documents]
                print(f"주의: 입력 문서가 문자열 또는 문자열 리스트가 아니어서 변환됨 (변환 후 첫번째 요소 타입: {type(texts[0]) if texts else 'N/A'})")
            except Exception as conversion_error:
                raise ValueError(
                    f"임베딩 함수 입력은 문자열 또는 문자열 리스트여야 합니다. 받은 입력: {input_documents}, 변환 오류: {conversion_error}")

        if not texts:  # 비어있는 리스트 처리
            return []
        embeddings_np = self.sbert_model.encode(texts, convert_to_numpy=True)

        result = embeddings_np.tolist()
        return result


embedding_function_instance = StrictEmbeddingFunction(model=sbert_model_instance)  # 인스턴스 생성


db_host = os.getenv("CHROMA_DB_HOST", "chromadb")
db_port = int(os.getenv("CHROMA_DB_PORT", 8000))

# 4. ChromaDB 클라이언트 생성
client = chromadb.HttpClient(host=db_host, port=db_port)
collection_name = os.getenv("CHROMA_COLLECTION", "construction_new")
# 5. 컬렉션 생성 또는 가져오기
print("5. 컬렉션 생성/로드 시도 중...")
try:
    if hasattr(embedding_function_instance, '__call__'):
        print(
            f"get_or_create_collection에 전달될 함수의 __call__ 시그니처: {inspect.signature(embedding_function_instance.__call__).parameters.keys()}")
    else:
        print(f"get_or_create_collection에 전달될 함수는 __call__ 속성이 없습니다.")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine",
                  "hnsw:construction_ef": 100,
                  "hnsw:M": 16,
                  "hnsw:search_ef": 100,  # get_or_create_collection 시에는 보통 생성 시 설정 따름
                  "hnsw:num_threads": 4, },  # 단순화된 메타데이터
        embedding_function=embedding_function_instance  # 클래스 인스턴스 전달
    )

    if 'documents' in locals() and 'all_metadatas' in locals() and 'all_ids' in locals():
        if documents and all_ids:  # documents와 ids가 비어있지 않은지 확인
            collection.add(
                documents=documents,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print("db insert")
        else:
            print("추가할 문서 또는 ID가 비어있습니다.")
    else:
        print("오류: 'documents'")

except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"타입: {type(embedding_function_instance)}")
    if hasattr(embedding_function_instance, '__call__'):
        print(f"__call__ 시그니처: {inspect.signature(embedding_function_instance.__call__).parameters.keys()}")
    else:
        print("해당 객체에는 __call__ 메소드가 없습니다.")

# (The last two code cells in the notebook were empty, so they don't add any code here)