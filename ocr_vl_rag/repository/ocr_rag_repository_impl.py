import base64
import os
import asyncio
import json
import time
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator, Optional, List, Dict, Any
from uuid import uuid4

import numpy as np
import redis
import torch
from PIL import Image
from dotenv import load_dotenv
import easyocr
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from transformers import (
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    AutoProcessor,
)
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from ocr_vl_rag.repository.ocr_rag_repository import OcrRAGRepository


# <--- 변경: GraphState에 available_filters 추가 --->
class GraphState(TypedDict):
    query: str
    image_data: Optional[str]
    chat_history: List[Dict[str, str]]
    session_id: Optional[str]
    ocr_text: Optional[str]
    available_filters: Optional[Dict[str, Any]]  # OCR로 추출한 모든 필터 '후보'
    filters: Optional[Dict[str, Any]]  # 플래너가 선택한, 검색에 '실제 사용할' 필터
    queries_for_retrieval: List[str]
    documents: List[Document]
    k: int
    generation_instructions: Optional[str]
    generation: Any


class OcrRAGRepositoryImpl(OcrRAGRepository):
    __instance = None
    _vlm_model: Optional[AsyncLLMEngine] = None
    _vlm_processor: Optional[AutoProcessor] = None
    _embed_model_instance: Optional[SentenceTransformer] = None
    _summarizer: Optional[BartForConditionalGeneration] = None
    _summarizer_tokenizer: Optional[PreTrainedTokenizerFast] = None
    _qdrant_client: Optional[QdrantClient] = None
    _qdrant_collection_name: Optional[str] = None
    _redis_client: Optional[redis.Redis] = None
    _ocr_reader: Optional[Any] = None

    _all_id_numbers: List[str] = []
    _all_reviewers: List[str] = []
    _all_drawing_names: List[str] = []
    _all_drawing_numbers: List[str] = []

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self):
        if not hasattr(self, '_initialized_repo'):
            self._initialized_repo = True
            init_start_time = time.perf_counter()
            print("--- VlRAGRepositoryImpl: __init__ 최초 초기화 시작 ---")
            load_dotenv()
            self._initialize_models()
            self._initialize_datastores()
            self._prepare_filter_lists()
            init_end_time = time.perf_counter()
            print(
                f"--- ⏱️ VlRAGRepositoryImpl: __init__ 최초 초기화 완료. (총 소요 시간: {init_end_time - init_start_time:.4f}초) ---")

    def _initialize_models(self):
        model_init_start_time = time.perf_counter()
        print("--- VlRAGRepositoryImpl: 모델 초기화 중... ---")
        if self._vlm_model is None:
            vlm_model_name = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
            print(f"--- Loading Vision-Language Model '{vlm_model_name}' with Async vLLM... ---")
            try:
                self._vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
                engine_args = AsyncEngineArgs(
                    model=vlm_model_name,
                    quantization='awq',
                    dtype='float16',
                    enforce_eager=True,
                    trust_remote_code=True,
                    max_model_len=8192,
                    gpu_memory_utilization=0.85,
                    limit_mm_per_prompt={'image': 1}
                )
                self._vlm_model = AsyncLLMEngine.from_engine_args(engine_args)
                print("--- ✅ Vision-Language Model loaded successfully with Async vLLM. ---")
            except Exception as e:
                print(f"--- 💥 Failed to load VLM with Async vLLM: {e} ---")
                import traceback
                traceback.print_exc()

        if self._embed_model_instance is None:
            embedding_model_name = os.getenv("EMBEDDING_MODEL", 'dragonkue/snowflake-arctic-embed-l-v2.0-ko')
            self._embed_model_instance = SentenceTransformer(embedding_model_name, device='cpu')
            print(f"--- Embedding model '{embedding_model_name}' on 'cpu' loaded. ---")

        if self._summarizer is None:
            summarizer_model_name = "EbanLee/kobart-summary-v3"
            self._summarizer_tokenizer = PreTrainedTokenizerFast.from_pretrained(summarizer_model_name)
            self._summarizer = BartForConditionalGeneration.from_pretrained(summarizer_model_name)
            print(f"--- ✅ Summarization model '{summarizer_model_name}' loaded. ---")

        if self._ocr_reader is None:
            print("--- 📖 Initializing OCR Reader (EasyOCR)... ---")
            self._ocr_reader = easyocr.Reader(['ko', 'en'], gpu=True)
            print("--- ✅ EasyOCR Reader initialized. ---")

        print(f"✅ 모든 모델 초기화 완료. (총 소요 시간: {time.perf_counter() - model_init_start_time:.4f}초)")

    def _initialize_datastores(self):
        print("--- VlRAGRepositoryImpl: 데이터 저장소 초기화 중... ---")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self._qdrant_collection_name = os.getenv("QDRANT_COLLECTION", "construction_v2")
        self._qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        print(f"--- ✅ Qdrant DB 연결 완료. (Collection: '{self._qdrant_collection_name}') ---")
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        self._redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        print(f"--- ✅ Redis 서버 연결 완료. ---")

    def _prepare_filter_lists(self):
        print("--- 'filters.json' 파일에서 필터 목록을 로드하는 중... ---")
        filter_file_path = Path.cwd() / "filter.json"
        if not filter_file_path.exists():
            print(f"⚠️ '{filter_file_path.resolve()}' 파일을 찾을 수 없습니다.")
            return
        try:
            with open(filter_file_path, "r", encoding="utf-8") as f:
                filter_data = json.load(f)
                self._all_id_numbers = [str(int(num)) for num in filter_data.get("ID번호", [])]
                self._all_reviewers = filter_data.get("검증위원", [])
                self._all_drawing_names = filter_data.get("도면명", [])
                self._all_drawing_numbers = filter_data.get("도면번호", [])
            print("✅ 필터 목록 로드 완료.")
        except Exception as e:
            print(f"--- 💥 필터 목록 파일 로드 중 오류 발생: {e}")

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        try:
            stored_history = self._redis_client.get(session_id)
            return json.loads(stored_history) if stored_history else []
        except Exception as e:
            print(f"--- 💥 Redis 조회 오류 (session_id: {session_id}): {e}")
            return []

    def save_chat_history(self, session_id: str, history: List[Dict[str, str]]):
        try:
            updated_history_json = json.dumps(history, ensure_ascii=False)
            self._redis_client.set(session_id, updated_history_json, ex=86400)
        except Exception as e:
            print(f"--- 💥 Redis 저장 오류 (session_id: {session_id}): {e}")

    def _summarize_with_local_model(self, history: List[Dict[str, str]]) -> str:
        if not self._summarizer or not self._summarizer_tokenizer: return "(요약 모델 로드 실패)"
        text_to_summarize = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        if not text_to_summarize.strip(): return "(요약할 내용 없음)"
        inputs = self._summarizer_tokenizer(text_to_summarize, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self._summarizer.generate(inputs.input_ids, num_beams=4, max_length=256, early_stopping=True)
        return self._summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    async def update_chat_history(self, session_id: str, user_query: str, ai_response: str):
        history = self.get_chat_history(session_id)
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": ai_response})
        messages_to_keep = 6
        if len(history) > messages_to_keep:
            history_to_summarize, recent_history = history[:-messages_to_keep], history[-messages_to_keep:]
            summary_content = await asyncio.to_thread(self._summarize_with_local_model, history_to_summarize)
            new_history = [{"role": "system", "content": f"이전 대화 요약: {summary_content}"}] + recent_history
            history = new_history
        self.save_chat_history(session_id, history)
        print(f"--- 💾 Redis 세션 [{session_id}] 업데이트 완료 (총 메시지 수: {len(history)}개). ---")

    def _ocr_and_extract_filters_node(self, state: GraphState) -> Dict[str, Any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: OCR & Extract All Filter Candidates (Batch Processing) (시작) ---")
        image_data = state["image_data"]

        if not image_data or not self._ocr_reader:
            print("  [정보] 이미지가 없거나 OCR 리더가 없어 이 단계를 건너뜁니다.")
            return {"available_filters": None, "ocr_text": ""}

        image_bytes = base64.b64decode(image_data)
        original_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = original_image.size
        print(f"  [정보] 원본 이미지 크기: {width}x{height}")

        tile_size = 1024
        overlap = 150

        # 1. 모든 타일 이미지를 리스트에 저장
        tile_images_np = []
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                if box[2] - box[0] < overlap or box[3] - box[1] < overlap:
                    continue
                tile_image = original_image.crop(box)
                tile_images_np.append(np.array(tile_image)[:, :, ::-1])

        all_extracted_texts = set()
        if tile_images_np:
            # 2. 저장된 타일 리스트를 한 번에 OCR 처리 (배치 크기 지정)
            # EasyOCR은 이미지 리스트를 받아 배치 처리를 수행합니다.
            all_results = self._ocr_reader.readtext(tile_images_np, batch_size=8)

            # 3. 결과 취합
            for result_group in all_results:
                # readtext는 결과 그룹의 리스트를 반환할 수 있으므로 중첩 루프 사용
                for (bbox, text, prob) in result_group:
                    all_extracted_texts.add(text)

        ocr_text = " ".join(sorted(list(all_extracted_texts)))
        print(f"  [정보] OCR 추출 텍스트 (일부): {ocr_text[:200]}...")

        found_filters = {}
        filter_map = {
            "ID번호": self._all_id_numbers, "검증위원": self._all_reviewers,
            "도면명": self._all_drawing_names, "도면번호": self._all_drawing_numbers,
        }
        for field_name, keyword_list in filter_map.items():
            found_keywords = [keyword for keyword in keyword_list if keyword in ocr_text]
            if found_keywords:
                found_filters[field_name] = found_keywords

        print(f"  [출력] 추출된 필터 후보: {found_filters}")
        print(
            f"--- 🔴 Node: OCR & Extract All Filter Candidates (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return {"available_filters": found_filters or None, "ocr_text": ocr_text}

    # <--- 변경: '지능형 플래너' 역할 수행을 위해 노드 로직 전체 변경 --->
    async def _generate_query_and_select_filters_node(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Generate Query & Select Filters (Planner) (시작) ---")

        query = state["query"]
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in state["chat_history"]])
        ocr_text = state.get("ocr_text", "")
        available_filters_from_ocr = state.get("available_filters", {})  # OCR 노드에서 넘어온 필터 후보

        # --- [✨ 새로운 기능] 시작: VLM 호출 전 빠른 필터링 ---
        combined_text_for_filter_check = query + " " + ocr_text
        found_filters = {}

        filter_map = {
            "ID번호": self._all_id_numbers,
            "검증위원": self._all_reviewers,
            "도면명": self._all_drawing_names,
            "도면번호": self._all_drawing_numbers,
        }

        for field_name, keyword_list in filter_map.items():
            for keyword in keyword_list:
                if keyword in combined_text_for_filter_check:
                    if field_name not in found_filters:
                        found_filters[field_name] = []
                    # 중복 추가 방지
                    if keyword not in found_filters[field_name]:
                        found_filters[field_name].append(keyword)

        if found_filters:
            print("  [정보] 빠른 경로: 단순 매칭으로 필터 발견. VLM 호출을 건너뜁니다.")

            search_query = query
            for values in found_filters.values():
                for value in values:
                    search_query = search_query.replace(value, "")

            search_query = search_query.strip() or query

            result = {
                "queries_for_retrieval": [search_query],
                "filters": found_filters
            }
            print(f"  [출력] 생성된 검색어: {result['queries_for_retrieval']}")
            print(f"  [출력] 선택된 필터: {result['filters']}")
            print(
                f"--- 🔴 Node: Generate Query & Select Filters (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
            return result
        # --- [✨ 새로운 기능] 끝 ---

        print("  [정보] 지능적 경로: VLM으로 정교한 분석을 시도합니다.")
        parser = JsonOutputParser()
        prompt_template = """당신은 사용자의 질문 의도를 분석하여, 벡터 검색에 사용할 '검색어'와 검색 결과 범위를 좁힐 '필터'를 결정하는 검색 계획 전문가입니다.

[사용자 질문]
{query}

[이전 대화 내용]
{chat_history}

[이미지에서 추출된 정보]
- OCR 텍스트: {ocr_text}
- 사용 가능한 필터 후보: {available_filters}

[지시사항]
1.  주어진 모든 정보를 종합하여 벡터 검색에 가장 적합한 핵심 검색어를 생성해주세요.
2.  사용자 질문의 의도를 깊이 분석하여, '사용 가능한 필터 후보' 중에서 이번 검색에 사용할 필터만 정확히 선택해주세요.
3.  **[중요 규칙] 사용자의 질문이 이미지 속 특정 대상(예: 인물, 회사)에 대한 추가 정보를 찾으면서, 현재 문서 자체의 정보(예: ID, 도면명)는 제외하려는 의도로 보일 경우, 그 특정 대상에 대한 필터는 반드시 유지하고 현재 문서 관련 필터는 제외해야 합니다.**
4.  질문 내용이 이미지 정보와 관련 없다면, 필터를 사용하지 마세요.
5.  최종 결과는 반드시 JSON 형식 `{{"search_queries": ["생성된 검색어"], "filters_to_use": {{"필드명": ["값"]}}}}` 으로 반환해주세요.

[예시]
- 질문: "이 검증위원의 다른 검토 의견들을 알려줘."
- 필터 후보: `{{"ID번호": ["103387"], "검증위원": ["김진수"]}}`
- 반환: `{{"search_queries": ["김진수 위원 검증 의견"], "filters_to_use": {{"검증위원": ["김진수"]}}}}`

- 질문: "이 도면의 상세 정보를 알려줘."
- 필터 후보: `{{"ID번호": ["103387"], "검증위원": ["김진수"]}}`
- 반환: `{{"search_queries": ["103387 도면 상세 정보"], "filters_to_use": {{"ID번호": ["103387"]}}}}`

- 질문: "김진수 위원이 검토한 103387 도면에 대해 설명해줘."
- 필터 후보: `{{"ID번호": ["103387"], "검증위원": ["김진수"]}}`
- 반환: `{{"search_queries": ["김진수 위원 103387 도면 검토 의견"], "filters_to_use": {{"ID번호": ["103387"], "검증위원": ["김진수"]}}}}`

- 질문: "일반적인 아파트 단지 설계 시 주의사항은 뭐야?"
- 필터 후보: `{{"ID번호": ["103387"], "검증위원": ["김진수"]}}`
- 반환: `{{"search_queries": ["아파트 단지 설계 주의사항"], "filters_to_use": {{}}}}`

최종적으로 사용할 검색어와 필터(JSON)를 반환:"""

        analysis_prompt = PromptTemplate.from_template(prompt_template)
        final_prompt_str = analysis_prompt.format(
            query=query,
            chat_history=history_str,
            ocr_text=ocr_text,
            # OCR 단계에서 추출된 필터 후보를 VLM에 전달합니다.
            available_filters=str(available_filters_from_ocr)
        )

        messages = [{"role": "user", "content": final_prompt_str}]
        text_prompt_for_vllm = self._vlm_processor.apply_chat_template(messages, tokenize=False,
                                                                       add_generation_prompt=True)
        sampling_params = SamplingParams(temperature=0, max_tokens=1024)
        request_id = str(uuid4())
        results_generator = self._vlm_model.generate(text_prompt_for_vllm, sampling_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output is None:
            raise RuntimeError("VLM에서 검색어/필터 생성을 못했습니다.")

        json_response_str = final_output.outputs[0].text.strip()

        try:
            response_json = parser.parse(json_response_str)
            search_queries = response_json.get("search_queries", [query])
            filters_to_use = response_json.get("filters_to_use", {})
            result = {
                "queries_for_retrieval": search_queries,
                "filters": filters_to_use
            }
        except Exception as e:
            print(f"--- ⚠️ VLM 검색어/필터 생성 중 오류 발생, 기본값 사용: {e} ---")
            result = {
                "queries_for_retrieval": [query],
                "filters": {}
            }

        print(f"  [출력] 생성된 검색어: {result['queries_for_retrieval']}")
        print(f"  [출력] 선택된 필터: {result['filters']}")
        print(
            f"--- 🔴 Node: Generate Query & Select Filters (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return result

    def _retrieve_documents_node(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Retrieve Documents (Dense 검색) (시작) ---")
        queries, filters, k = state["queries_for_retrieval"], state.get("filters"), state["k"]

        qdrant_filter = None
        if filters and isinstance(filters, dict):
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list) and value:
                    conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=value)))
                elif isinstance(value, str):
                    conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
            if conditions:
                qdrant_filter = models.Filter(must=conditions)
                print(f"  [정보] Qdrant에 적용될 필터: {qdrant_filter.dict()}")

        query_vector = self._embed_model_instance.encode(queries[0]).tolist()
        search_results = self._qdrant_client.search(
            collection_name=self._qdrant_collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=k,
            with_payload=True
        )
        documents = [Document(page_content=hit.payload.get("text", ""), metadata=hit.payload) for hit in search_results]

        # <--- 디버깅 로그 수정 시작 (text 본문 출력 추가) --->
        print("\n--- 🕵️  검색된 문서 상세 정보 검증 ---")
        if not documents:
            print("  [결과] 검색된 문서가 없습니다.")
        else:
            for i, doc in enumerate(documents):
                print(f"--- [문서 {i + 1}] ---")
                # 메타데이터 출력
                retrieved_reviewer = doc.metadata.get('검증위원', 'N/A')
                retrieved_id = doc.metadata.get('ID번호', 'N/A')
                print(f"  - 메타데이터: ID번호={retrieved_id}, 검증위원={retrieved_reviewer}")
                # Text 본문 출력
                print(f"  - Text 내용: {doc.page_content}")
        print("-------------------------------------\n")
        # <--- 디버깅 로그 수정 끝 --->

        print(f"  [출력 업데이트] 최종 문서(개수): {len(documents)}")
        print(f"--- 🔴 Node: Retrieve Documents (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return {"documents": documents}

    async def _generate_rag_answer_node(self, state: GraphState) -> Dict[str, Any]:
        print("\n--- 🟢 Node: Generate RAG Answer (시작) ---")
        query = state["query"]
        documents = state["documents"]

        # `focus_entity` 관련 로직을 제거했습니다.

        instructions = state.get("generation_instructions") or "답변을 명확하고 간결하게 생성해주세요."
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in state["chat_history"]]) if state[
            "chat_history"] else "이전 대화 없음"
        context_str = "\n\n---\n\n".join([doc.page_content for doc in documents]) if documents else "참고할 문서가 없습니다."

        # <--- focus_entity를 사용하지 않는 가장 단순한 형태의 RAG 프롬프트 --->
        prompt_template_str = """당신은 주어진 [참고 문서]의 사실에만 기반하여 질문에 답변하는 정직한 건축 전문가 AI입니다.

[이전 대화 내용]
{chat_history_str}

[참고 문서]
{context_str}

[사용자 원본 요청]
{original_query}

[추가 지시사항]
{instructions}

[답변 시 핵심 규칙]
- 답변은 반드시 [참고 문서]에 있는 내용을 바탕으로, [사용자 원본 요청]에 대해 충실하게 생성해야 합니다.
- 당신의 사전 지식을 사용하거나 [참고 문서]에 없는 내용을 만들어내서는 안 됩니다.
- 만약 [참고 문서]에 [사용자 원본 요청]에 대한 답이 없다면, "주어진 정보로는 답변할 수 없습니다."라고 말하세요.

위 규칙에 따라 최종 답변을 생성해주세요:
"""

        final_prompt_str = PromptTemplate.from_template(prompt_template_str).format(
            chat_history_str=history_str,
            context_str=context_str,
            original_query=query,
            instructions=instructions
        )
        return {"generation": final_prompt_str}

    async def _generate_direct_llm_answer_node(self, state: GraphState) -> Dict[str, Any]:
        print("\n--- 🟢 Node: Generate Direct LLM Answer (시작) ---")
        query, history_str = state["query"], "\n".join(
            [f"{m['role']}: {m['content']}" for m in state["chat_history"]]) if state["chat_history"] else "이전 대화 없음"
        prompt_template = PromptTemplate.from_template(
            "당신은 친절한 대화형 AI입니다. 이전 대화 내용과 현재 사용자 질문을 바탕으로 자연스럽게 답변해주세요.\n[이전 대화 내용]\n{chat_history}\n[현재 사용자 질문]\n{query}\n답변:")
        final_prompt_str = prompt_template.format(chat_history=history_str, query=query)
        return {"generation": final_prompt_str}

    def _decide_after_retrieval(self, state: GraphState) -> str:
        print(f"\n--- 🤔 Node: Decide After Retrieval ---")
        if state.get("documents"):
            return "generate_rag_answer_node"
        else:
            print("  [결정] 검색된 문서가 없으므로 직접 LLM 답변 생성을 진행합니다.")
            return "generate_direct_llm_answer_node"

    async def generate(self, query: str, chat_history: List[Dict[str, str]], k: int = 5,
                       session_id: Optional[str] = None, image_data: Optional[str] = None) -> AsyncGenerator[str, None]:
        print(f"\n--- ✨ LangGraph Generate 시작: Query='{query[:50]}...' | 이미지 존재: {'Yes' if image_data else 'No'} ---")

        try:
            workflow = StateGraph(GraphState)

            workflow.add_node("ocr_and_extract_filters_node", self._ocr_and_extract_filters_node)
            workflow.add_node("planner_node", self._generate_query_and_select_filters_node)
            workflow.add_node("retrieve_documents_node", self._retrieve_documents_node)
            workflow.add_node("generate_rag_answer_node", self._generate_rag_answer_node)
            workflow.add_node("generate_direct_llm_answer_node", self._generate_direct_llm_answer_node)

            workflow.set_entry_point("ocr_and_extract_filters_node")
            workflow.add_edge("ocr_and_extract_filters_node", "planner_node")
            workflow.add_edge("planner_node", "retrieve_documents_node")
            workflow.add_conditional_edges(
                "retrieve_documents_node",
                self._decide_after_retrieval,
                {"generate_rag_answer_node": "generate_rag_answer_node",
                 "generate_direct_llm_answer_node": "generate_direct_llm_answer_node"}
            )
            workflow.add_edge("generate_rag_answer_node", END)
            workflow.add_edge("generate_direct_llm_answer_node", END)

            app = workflow.compile()

            initial_state = GraphState(
                query=query, image_data=image_data, chat_history=chat_history, k=k, session_id=session_id,
                ocr_text=None, available_filters=None, filters=None, queries_for_retrieval=[],
                documents=[], generation_instructions=None, generation=None
            )

            interrupt_nodes = ["generate_rag_answer_node", "generate_direct_llm_answer_node"]
            final_state = await app.ainvoke(initial_state, {"recursion_limit": 15, "interrupt_before": interrupt_nodes})

            final_prompt_to_generate = final_state.get("generation")
            if not final_prompt_to_generate:
                yield f"data: {json.dumps({'error': '최종 답변을 생성하지 못했습니다.'})}\n\n"
                return

            messages = [{"role": "user", "content": final_prompt_to_generate}]
            text_prompt = self._vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2048)
            request_id = str(uuid4())
            results_generator = self._vlm_model.generate(text_prompt, sampling_params, request_id)

            full_response = ""
            index = 0

            async for request_output in results_generator:
                new_text = request_output.outputs[0].text[index:]
                if new_text:
                    full_response += new_text
                    index = len(full_response)
                    yield f"data: {json.dumps({'token': new_text})}\n\n"

            if session_id:
                await self.update_chat_history(session_id, query, full_response)

        except Exception as e:
            print(f"--- 💥 LangGraph Generate CRITICAL ERROR: {e} ---")
            import traceback
            traceback.print_exc()
            error_message = str(e).replace("\n", " ")
            yield f"data: {json.dumps({'error': error_message})}\n\n"

        finally:
            print("--- ✨ LangGraph Generate 종료 ---")

