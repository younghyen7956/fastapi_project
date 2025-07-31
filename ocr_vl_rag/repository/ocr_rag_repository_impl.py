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
from paddleocr import PaddleOCR
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


# GraphState에 ocr_text 필드 추가
class GraphState(TypedDict):
    query: str
    image_data: Optional[str]
    chat_history: List[Dict[str, str]]
    session_id: Optional[str]
    ocr_text: Optional[str]  # OCR로 추출한 텍스트
    queries_for_retrieval: List[str]
    filters: Optional[Dict[str, Any]]
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
    _ocr_reader: Optional[PaddleOCR] = None # 타입을 PaddleOCR로 변경

    # --- 필터링을 위한 메타데이터 목록 ---
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
                    max_model_len=32768,  # 확장된 모델 최대 길이
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
            print("--- 📖 Initializing OCR Reader (PaddleOCR)... ---")
            # 제공해주신 초기화 코드를 그대로 사용합니다.
            self._ocr_reader = PaddleOCR(
                text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                use_gpu=True,  # use_gpu=True가 device="gpu:0"과 유사한 역할을 합니다.
            )
            print("--- ✅ OCR Reader initialized. ---")

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
        print("\n--- 🟢 Node: OCR & Extract Filters (시작) ---")
        image_data = state["image_data"]

        if not image_data or not self._ocr_reader:
            print("  [정보] 이미지가 없거나 OCR 리더가 없어 이 단계를 건너뜁니다.")
            return {"filters": None, "ocr_text": ""}

        image_bytes = base64.b64decode(image_data)

        # 1. PIL을 사용해 이미지 바이트를 엽니다.
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # 2. PIL 이미지를 numpy 배열로 변환합니다. (RGB 형식)
        img_rgb = np.array(pil_image)

        # 3. RGB를 BGR로 색상 채널 순서를 변경합니다. (PaddleOCR 호환용)
        img_bgr = img_rgb[:, :, ::-1]

        # 4. PaddleOCR 실행
        ocr_results = self._ocr_reader.ocr(img_bgr, cls=True)

        # 5. PaddleOCR 결과 형식에 맞게 텍스트만 추출
        extracted_texts = []
        if ocr_results and ocr_results[0]:
            for line in ocr_results[0]:
                extracted_texts.append(line[1][0])  # text 부분만 추출

        ocr_text = " ".join(extracted_texts)
        print(f"  [정보] OCR 추출 텍스트 (일부): {ocr_text[:100]}...")

        # 6. OCR 텍스트에서 필터 키워드 추출
        found_filters = {}
        filter_map = {
            "ID번호": self._all_id_numbers,
            "검증위원": self._all_reviewers,
            "도면명": self._all_drawing_names,
            "도면번호": self._all_drawing_numbers,
        }

        for field_name, keyword_list in filter_map.items():
            found_keywords = [keyword for keyword in keyword_list if keyword in ocr_text]
            if found_keywords:
                # 같은 필드에 여러 키워드가 발견될 수 있으므로 리스트로 저장
                found_filters[field_name] = found_keywords

        print(f"  [출력] 추출된 필터: {found_filters}")
        print(f"--- 🔴 Node: OCR & Extract Filters (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return {"filters": found_filters or None, "ocr_text": ocr_text}

    async def _generate_search_query_node(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Generate Search Query (Text-based) (시작) ---")
        query, history_str = state["query"], "\n".join([f"{m['role']}: {m['content']}" for m in state["chat_history"]])
        ocr_text = state.get("ocr_text", "")

        parser = JsonOutputParser()
        prompt_template = """당신은 사용자의 질문과 대화 기록, 그리고 이미지에서 추출된 OCR 텍스트를 바탕으로, 벡터 검색에 사용할 단 하나의 핵심 '검색어'와 '지시사항'을 생성하는 AI입니다.

[참고 OCR 텍스트]
{ocr_text}

[이전 대화 내용]
{chat_history}

[사용자 질문]
{query}

[지시사항]
- 사용자의 질문, 대화 기록, OCR 텍스트를 종합하여 가장 핵심적인 검색어 구문 하나와, 답변 생성 시에 참고할 지시사항을 추출해주세요.
- 최종 결과는 반드시 JSON 형식 {"search_queries": ["생성된 검색어"], "generation_instructions": "생성된 지시사항 또는 null"} 으로 반환해주세요.

JSON 출력:"""

        analysis_prompt = PromptTemplate.from_template(prompt_template)
        final_prompt_str = analysis_prompt.format(ocr_text=ocr_text, chat_history=history_str, query=query)

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
            raise RuntimeError("VLM에서 검색어 생성을 못했습니다.")

        json_response_str = final_output.outputs[0].text.strip()

        try:
            response_json = parser.parse(json_response_str)
            search_queries = response_json.get("search_queries", [query])
            if search_queries: search_queries = [search_queries[0]]
            result = {"queries_for_retrieval": search_queries,
                      "generation_instructions": response_json.get("generation_instructions")}
        except Exception as e:
            print(f"--- ⚠️ 검색어 생성 중 오류 발생, 원본 쿼리 사용: {e} ---")
            result = {"queries_for_retrieval": [query], "generation_instructions": None}

        print(f"  [출력] 생성된 검색어: {result['queries_for_retrieval']}")
        print(f"--- 🔴 Node: Generate Search Query (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return result

    def _retrieve_documents_node(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Retrieve Documents (Dense 검색) (시작) ---")
        queries, filters, k = state["queries_for_retrieval"], state.get("filters"), state["k"]

        qdrant_filter = None
        if filters and isinstance(filters, dict):
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
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
        print(f"  [출력 업데이트] 최종 문서(개수): {len(documents)}")
        print(f"--- 🔴 Node: Retrieve Documents (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return {"documents": documents}

    async def _generate_rag_answer_node(self, state: GraphState) -> Dict[str, Any]:
        print("\n--- 🟢 Node: Generate RAG Answer (시작) ---")
        query, documents, instructions = state["query"], state["documents"], state.get(
            "generation_instructions") or "답변을 명확하고 간결하게 생성해주세요."
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in state["chat_history"]]) if state[
            "chat_history"] else "이전 대화 없음"
        context_str = "\n\n---\n\n".join([doc.page_content for doc in documents]) if documents else "참고할 문서가 없습니다."

        prompt_template_str = """당신은 건축 관련 전문가 입니다. 당신의 주요 임무는 사용자의 원본 요청에 대해, '이전 대화 내용'을 참고하고 주어진 '참고 문서'와 '추가 지시사항'을 바탕으로 최종 답변을 생성하는 것입니다.

[이전 대화 내용]
{chat_history_str}

[참고 문서]
{context_str}

[사용자 원본 요청]
{original_query}

[추가 지시사항]
{instructions}

'이전 대화 내용'과 '참고 문서'를 바탕으로, '사용자 원본 요청'에 대해 '추가 지시사항'을 충실히 반영하여 최종 답변을 생성해주세요."""

        final_prompt_str = PromptTemplate.from_template(prompt_template_str).format(chat_history_str=history_str,
                                                                                    context_str=context_str,
                                                                                    original_query=query,
                                                                                    instructions=instructions)
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
        total_generate_start_time = time.perf_counter()
        print(f"\n--- ✨ LangGraph Generate 시작: Query='{query[:50]}...' | 이미지 존재: {'Yes' if image_data else 'No'} ---")

        workflow = StateGraph(GraphState)

        workflow.add_node("ocr_and_extract_filters_node", self._ocr_and_extract_filters_node)
        workflow.add_node("generate_search_query_node", self._generate_search_query_node)
        workflow.add_node("retrieve_documents_node", self._retrieve_documents_node)
        workflow.add_node("generate_rag_answer_node", self._generate_rag_answer_node)
        workflow.add_node("generate_direct_llm_answer_node", self._generate_direct_llm_answer_node)

        workflow.set_entry_point("ocr_and_extract_filters_node")
        workflow.add_edge("ocr_and_extract_filters_node", "generate_search_query_node")
        workflow.add_edge("generate_search_query_node", "retrieve_documents_node")
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
            ocr_text=None, queries_for_retrieval=[], filters=None, documents=[],
            generation_instructions=None, generation=None
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
        try:
            async for request_output in results_generator:
                new_text = request_output.outputs[0].text[index:]
                if new_text:
                    full_response += new_text
                    index = len(full_response)
                    yield f"data: {json.dumps({'token': new_text})}\n\n"
        except Exception as e:
            print(f"--- 💥 vLLM 스트리밍 중 오류 발생: {e} ---")
            yield f"data: {json.dumps({'error': '답변 생성 중 오류가 발생했습니다.'})}\n\n"

        if session_id:
            await self.update_chat_history(session_id, query, full_response)

        total_generate_end_time = time.perf_counter()
        print(f"--- ✨ LangGraph Generate 종료 (총 소요 시간: {total_generate_end_time - total_generate_start_time:.4f}초) ---")