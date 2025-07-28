import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import torch
from tqdm import tqdm
import traceback
import zipfile
import uuid

# Qdrant 클라이언트 및 모델 임포트
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel

# --- 1. 새로운 데이터 로드 및 전처리 (기존과 동일) ---
print("--- 1. 데이터 로드 및 전처리 시작 ---")
load_dotenv()

orig_path = Path.cwd() / "건축 데이터.xlsx"
fixed_path = Path.cwd() / "건축_데이터_fixed.xlsx"

if not orig_path.exists():
    raise FileNotFoundError(f"엑셀 원본 파일을 찾을 수 없습니다: {orig_path}")

print("--- 엑셀 파일 복구 중... ---")
with zipfile.ZipFile(orig_path, 'r') as zin, zipfile.ZipFile(fixed_path, 'w') as zout:
    for item in zin.infolist():
        if item.filename != "xl/styles.xml":
            data = zin.read(item.filename)
            zout.writestr(item, data)

print("--- 복구된 엑셀 파일 로드 중... ---")
df = pd.read_excel(fixed_path, sheet_name="Page", engine="openpyxl")

cols_to_use = [0, 1, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26]

header = df.iloc[3, cols_to_use].fillna(df.iloc[4, cols_to_use]).tolist()
counts = {}
unique_names = []
for i, name in enumerate(header):
    if name == "내용":
        if unique_names:
            unique_names.append(f"{unique_names[-1]} 내용")
        else:
            unique_names.append(f"{df.iloc[2, cols_to_use].astype(str).tolist()[i]} 내용")
        continue
    counts[name] = counts.get(name, 0) + 1
    unique_names.append(f"{name}_{counts[name]}" if counts[name] > 1 else name)

fix_data = df.iloc[5:, cols_to_use].copy()
fix_data.columns = unique_names
fix_data = fix_data.iloc[:-1, :].reset_index(drop=True)
fix_data = fix_data.fillna('')
print("--- 데이터프레임 전처리 완료 ---")
print("사용될 컬럼명:", fix_data.columns.tolist())

# --- 2. 데이터를 '문서(Document)' 단위로 변환 (수정된 로직) ---
print("--- 2. 데이터를 문서 단위로 변환하는 중... ---")

content_columns = [
    col for col in fix_data.columns
    if ('의견' in col or '내용' in col) and col != '대표의견ID'
]
metadata_cols = [col for col in fix_data.columns if col not in content_columns]
print("콘텐츠 컬럼:", content_columns)
print("메타데이터 컬럼:", metadata_cols)


def create_document_from_row(row: pd.Series, content_columns: list, metadata_cols: list) -> dict:
    """
    DataFrame의 한 행(row)을 받아서 하나의 통합된 문서(document)로 만듭니다.
    """
    text_parts = []

    # --- 문서의 기본 정보 (요약 역할) ---
    gongjong = row.get('공종', '미지정')
    seolgye = row.get('설계단계', '미지정')
    jemok = row.get('제목', '제목 없음')
    domyeon_myeong = row.get('도면명', '도면명 미지정')
    gumjung = row.get('검증위원', '미지정')

    summary_header = f"문서 연번 {row.get('연번', '미지정')}은(는) {seolgye} 단계의 '{gongjong}' 공종에 대한 검토 문서입니다. 관련 도면명은 '{domyeon_myeong}'이며, 주요 제목은 '{jemok}', 검증위원은 {gumjung}입니다."
    text_parts.append(summary_header)

    # --- 각 의견/내용 필드를 본문으로 추가 ---
    for col_name in content_columns:
        content_value = row.get(col_name)
        if pd.notna(content_value) and content_value != '':
            # 각 필드 내용을 명확히 구분하기 위해 헤더를 추가합니다.
            field_text = f"\n--- {col_name} ---\n{content_value}"
            text_parts.append(field_text)

    # --- 최종 텍스트 생성 ---
    final_text = "\n".join(text_parts).strip()

    # --- 메타데이터 생성 ---
    # 메타데이터는 원본 행의 모든 정보를 포함하도록 합니다.
    metadata = row.to_dict()

    return {"text": final_text, "metadata": metadata}


all_documents_data = []
for index, row in tqdm(fix_data.iterrows(), total=len(fix_data), desc="문서 생성 중"):
    all_documents_data.append(create_document_from_row(row, content_columns, metadata_cols))

documents_to_add = [doc['text'] for doc in all_documents_data]
metadatas_to_add = [doc['metadata'] for doc in all_documents_data]

# ID는 각 행의 고유한 값인 '연번'을 기반으로 생성합니다.
ids_to_add = [
    str(uuid.uuid5(uuid.NAMESPACE_DNS, str(m.get('연번', f'row-{i}')))) for i, m in enumerate(metadatas_to_add)
]
print(f"--- 데이터 준비 완료. 총 {len(documents_to_add)}개의 문서가 생성되었습니다. ---")


# --- 3. 임베딩 모델 준비 (기존과 동일) ---
print("--- 3. 임베딩 모델 준비 중... ---")
embedding_model_name = os.getenv("EMBEDDING_MODEL", 'BAAI/bge-m3')
selected_device = 'cpu'
if torch.cuda.is_available():
    selected_device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    selected_device = 'mps'
print(f"   - 모델: {embedding_model_name}, 디바이스: {selected_device} (FlagEmbedding 사용)")
sbert_model_instance = BGEM3FlagModel(embedding_model_name, device=selected_device, use_fp16=True)

# --- 4. Qdrant DB 연결 및 데이터 삽입 (기존과 동일) ---
print("--- 4. Qdrant DB 연결 및 데이터 삽입 시작 ---")
db_host = os.getenv("QDRANT_HOST", "localhost")
db_port = int(os.getenv("QDRANT_PORT", 6333))
client = QdrantClient(host=db_host, port=db_port)
# ✨ 새 로직을 적용했으므로, 새로운 컬렉션 이름을 사용하는 것을 강력히 권장합니다.
collection_name = "construction_v1"

print(f"--- '{collection_name}' 컬렉션 존재 여부 확인 및 생성/업데이트 ---")
vector_size = 1024

if not client.collection_exists(collection_name=collection_name):
    print(f"--- 컬렉션 '{collection_name}'이(가) 존재하지 않아 새로 생성합니다. ---")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={"dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)},
        sparse_vectors_config={
            "lexical-weights": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
    )
    print(f"--- 컬렉션 '{collection_name}' 생성 완료. ---")
else:
    print(f"--- 컬렉션 '{collection_name}'이(가) 이미 존재합니다. ---")

collection_info = client.get_collection(collection_name=collection_name)
print(f"--- 컬렉션 준비 완료. 현재 포인트 수: {collection_info.points_count} ---")

if collection_info.points_count == 0:
    print("--- 컬렉션에 데이터가 없습니다. 문서 임베딩 및 데이터 업로드 중... ---")
    batch_size = 16 # 문서 크기가 커졌으므로 배치 사이즈를 조금 줄이는 것이 안정적일 수 있습니다.

    try:
        for i in tqdm(range(0, len(documents_to_add), batch_size), desc="하이브리드 데이터 업로드"):
            i_end = min(i + batch_size, len(documents_to_add))
            documents_batch = documents_to_add[i:i_end]
            metadatas_batch = metadatas_to_add[i:i_end]
            ids_batch = ids_to_add[i:i_end]

            output = sbert_model_instance.encode(
                documents_batch, return_dense=True, return_sparse=True, return_colbert_vecs=False
            )

            dense_vectors_batch = output['dense_vecs'].tolist()
            sparse_vectors_raw = output['lexical_weights']

            points_batch = []
            for j in range(len(ids_batch)):
                lexical_weights = sparse_vectors_raw[j]
                if isinstance(lexical_weights, dict):
                    indices = list(map(int, lexical_weights.keys()))
                    values = list(lexical_weights.values())
                    sparse_vector = models.SparseVector(indices=indices, values=values)

                    points_batch.append(
                        models.PointStruct(
                            id=ids_batch[j],
                            vector={"dense": dense_vectors_batch[j], "lexical-weights": sparse_vector},
                            # ✨ payload에 저장되는 text도 통합된 전체 텍스트입니다.
                            payload={"text": documents_batch[j], **metadatas_batch[j]}
                        )
                    )

            if points_batch:
                client.upsert(collection_name=collection_name, points=points_batch, wait=True)

        print("--- ✅ DB 데이터 추가 최종 완료. ---")
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"총 {collection_info.points_count}개의 문서가 '{collection_name}' 컬렉션에 추가되었습니다.")
    except Exception as e:
        print(f"--- 💥 DB 데이터 추가 중 오류 발생: {e}")
        traceback.print_exc()
else:
    print(f"--- 컬렉션 '{collection_name}'에 이미 데이터가 있습니다. 추가를 건너뜀. ---")

print("\n--- 필터 목록을 파일로 저장하는 중... ---")
try:
    filter_data = {
        "reviewers": fix_data[fix_data['검증위원'].notna() & (fix_data['검증위원'] != '')]['검증위원'].unique().tolist(),
        "drawings": fix_data[fix_data['도면명'].notna() & (fix_data['도면명'] != '')]['도면명'].unique().tolist()
    }

    # RAG API 프로젝트 폴더에 filters.json 파일을 생성하도록 경로를 지정할 수 있습니다.
    filter_file_path = Path("filters.json")
    with open(filter_file_path, "w", encoding="utf-8") as f:
        json.dump(filter_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 필터 목록을 '{filter_file_path}' 파일로 저장했습니다.")
    print(f"   - 검증위원: {len(filter_data['reviewers'])}명")
    print(f"   - 도면명: {len(filter_data['drawings'])}개")

except Exception as e:
    print(f"--- 💥 필터 목록 파일 저장 중 오류 발생: {e}")