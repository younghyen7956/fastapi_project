import os
import asyncio
import re
import time
from pathlib import Path
import json
from typing import AsyncGenerator, Optional, List, Dict, Any
from io import BytesIO
import base64
import numpy as np
import redis
from paddleocr import PaddleOCR  # [수정] easyocr -> paddleocr
from PIL import Image

import torch
from dotenv import load_dotenv
from functools import partial

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# 이 RAGRepository는 프로젝트의 추상 클래스로 가정합니다.
# 실제 프로젝트 구조에 맞게 import 경로를 조정하세요.
from RAG.repository.simple_rag_repository import RAGRepository


class GraphState(TypedDict):
    query: str
    image_data: Optional[str]  # 이미지 데이터 (base64 인코딩된 문자열)
    ocr_text: Optional[str]  # OCR로 추출된 텍스트
    chat_history: List[Dict[str, str]]
    queries_for_retrieval: List[str]
    filters: Optional[Dict[str, Any]]
    documents: List[Document]
    k: int
    generation_instructions: Optional[str]


class RAGRepositoryImpl(RAGRepository):
    __instance = None
    _initialized = False

    # --- Models & Clients ---
    _model: Optional[ChatOpenAI] = None
    _utility_llm: Optional[ChatOpenAI] = None
    _embed_model_instance: Optional[SentenceTransformer] = None
    _summarizer: Optional[BartForConditionalGeneration] = None
    _summarizer_tokenizer: Optional[PreTrainedTokenizerFast] = None
    _ocr_reader: Optional[Any] = None

    _qdrant_client: Optional[QdrantClient] = None
    _qdrant_collection_name: Optional[str] = None
    _redis_client: Optional[redis.Redis] = None

    # --- Metadata for Filtering ---
    _all_reviewers: List[str] = []
    _all_drawing_names: List[str] = []

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self):
        if not RAGRepositoryImpl._initialized:
            RAGRepositoryImpl._initialized = True
            init_start_time = time.perf_counter()
            print("--- RAGRepositoryImpl: __init__ 최초 초기화 시작 ---")
            load_dotenv()
            self._initialize_models()
            self._initialize_datastores()
            self._prepare_filter_lists()
            init_end_time = time.perf_counter()
            print(
                f"--- ⏱️ RAGRepositoryImpl: __init__ 최초 초기화 완료. (총 소요 시간: {init_end_time - init_start_time:.4f}초) ---")

    # --- Initialization Methods ---
    def _initialize_models(self):
        model_init_start_time = time.perf_counter()
        print("--- RAGRepositoryImpl: 모델 초기화 중... ---")

        RAGRepositoryImpl._model = ChatOpenAI(model=os.getenv("MAIN_LLM_MODEL", "gpt-4o-mini"), temperature=0.0,
                                              openai_api_key=os.getenv("OPENAI_API_KEY"), streaming=True)
        RAGRepositoryImpl._utility_llm = ChatOpenAI(model=os.getenv("UTILITY_LLM_MODEL", "gpt-4o-mini"),
                                                    temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"),
                                                    streaming=False)
        print("--- ✅ OpenAI LLMs initialized. ---")

        if RAGRepositoryImpl._embed_model_instance is None:
            embedding_model_name = os.getenv("EMBEDDING_MODEL", 'dragonkue/snowflake-arctic-embed-l-v2.0-ko')
            device = 'cpu'
            RAGRepositoryImpl._embed_model_instance = SentenceTransformer(embedding_model_name, device=device)
            print(f"--- Embedding model '{embedding_model_name}' on '{device}' loaded. ---")
            print("--- Embedding model warming up...")
            warmup_start_embed = time.perf_counter()
            RAGRepositoryImpl._embed_model_instance.encode("Warm-up text")
            warmup_end_embed = time.perf_counter()
            print(f"--- ✅ Embedding model warm-up complete. (소요 시간: {warmup_end_embed - warmup_start_embed:.4f}초)")

        if RAGRepositoryImpl._summarizer is None:
            summarizer_model_name = "EbanLee/kobart-summary-v3"
            print(f"--- Loading local summarization model '{summarizer_model_name}'... ---")
            try:
                RAGRepositoryImpl._summarizer_tokenizer = PreTrainedTokenizerFast.from_pretrained(summarizer_model_name)
                RAGRepositoryImpl._summarizer = BartForConditionalGeneration.from_pretrained(summarizer_model_name,
                                                                                             num_labels=2)
                print("--- ✅ Summarization model loaded. ---")
            except Exception as e:
                print(f"--- 💥 Failed to load summarization model: {e}")

        # [수정] easyocr -> paddleocr 초기화 로직
        if RAGRepositoryImpl._ocr_reader is None and PaddleOCR is not None:
            print("--- 📖 Initializing OCR Reader (PaddleOCR)... ---")
            # lang='korean'으로 한국어 모델을 지정합니다.
            RAGRepositoryImpl._ocr_reader = PaddleOCR(lang='korean', use_angle_cls=True, device="gpu")
        elif PaddleOCR is None:
            print("--- ⚠️  paddleocr 라이브러리가 설치되지 않아 OCR 기능을 비활성화합니다. ---")

        model_init_end_time = time.perf_counter()
        print(f"✅ 모든 모델 초기화 완료. (총 소요 시간: {model_init_end_time - model_init_start_time:.4f}초)")

    def _initialize_datastores(self):
        print("--- RAGRepositoryImpl: 데이터 저장소 초기화 중... ---")

        qdrant_host = os.getenv("QDRANT_HOST", "qdrant_db")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        RAGRepositoryImpl._qdrant_collection_name = os.getenv("QDRANT_COLLECTION", "construction_v2")
        RAGRepositoryImpl._qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        print(f"--- ✅ Qdrant DB 연결 완료. (Collection: '{self._qdrant_collection_name}') ---")

        redis_host = os.getenv("REDIS_HOST", "redis_db")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        RAGRepositoryImpl._redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
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
                RAGRepositoryImpl._all_reviewers = filter_data.get("reviewers", [])
                RAGRepositoryImpl._all_drawing_names = filter_data.get("drawings", [])
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
            updated_history_json = json.dumps(history)
            self._redis_client.set(session_id, updated_history_json, ex=86400)
        except Exception as e:
            print(f"--- 💥 Redis 저장 오류 (session_id: {session_id}): {e}")

    def _summarize_with_local_model(self, history: List[Dict[str, str]]) -> str:
        if not self._summarizer or not self._summarizer_tokenizer:
            return "(요약 모델 로드 실패)"

        text_to_summarize = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        inputs = self._summarizer_tokenizer(text_to_summarize, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self._summarizer.generate(inputs.input_ids, num_beams=4, max_length=256, early_stopping=True)
        return self._summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    async def update_chat_history(self, session_id: str, user_query: str, ai_response: str):
        history = self.get_chat_history(session_id)

        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": ai_response})

        turns_to_keep = 3
        messages_to_keep = turns_to_keep * 2

        if len(history) > messages_to_keep:
            print(f"--- 📝 세션 [{session_id}] 대화 요약 시작 (보존 메시지 수: {messages_to_keep}개 초과)... ---")

            history_to_summarize = history[:-messages_to_keep]
            recent_history = history[-messages_to_keep:]

            summary_content = await asyncio.to_thread(self._summarize_with_local_model, history_to_summarize)
            print(f"--- ✅ 세션 [{session_id}] 요약 완료: {summary_content[:100]}... ---")

            new_history = [{"role": "system", "content": f"이전 대화 요약: {summary_content}"}]
            new_history.extend(recent_history)

            history = new_history

        self.save_chat_history(session_id, history)
        print(f"--- 💾 Redis 세션 [{session_id}] 업데이트 완료 (총 메시지 수: {len(history)}개). ---")

    async def generate_from_text(self, query: str, chat_history: List[Dict[str, str]], k: int = 10) -> AsyncGenerator[
        str, None]:
        print("\n--- ➡️ Calling: generate_from_text ---")
        async for token in self._internal_generate(query=query, chat_history=chat_history, k=k, image_data=None):
            yield token

    async def generate_from_image(self, query: str, chat_history: List[Dict[str, str]], image_data: str, k: int = 10) -> \
    AsyncGenerator[str, None]:
        print("\n--- 📸 Calling: generate_from_image ---")
        async for token in self._internal_generate(query=query, chat_history=chat_history, k=k, image_data=image_data):
            yield token

    async def _internal_generate(self, query: str, chat_history: List[Dict[str, str]], k: int,
                                 image_data: Optional[str]) -> AsyncGenerator[str, None]:
        total_generate_start_time = time.perf_counter()
        print(f"\n--- ✨ LangGraph Generate 시작: Query='{query[:50]}...' | 이미지: {'있음' if image_data else '없음'} ---")

        output_queue = asyncio.Queue()
        workflow = StateGraph(GraphState)

        workflow.add_node("ocr_node_id", self._ocr_node_func)
        workflow.add_node("plan_retrieval_node_id", self._plan_retrieval_node_func)
        workflow.add_node("retrieve_documents_node_id", self._retrieve_documents_node_func)
        workflow.add_node("generate_rag_answer_node_id",
                          partial(self._generate_rag_answer_node_func, answer_queue=output_queue))
        workflow.add_node("generate_direct_llm_answer_node_id",
                          partial(self._generate_direct_llm_answer_node_func, answer_queue=output_queue))

        workflow.set_entry_point("ocr_node_id")
        workflow.add_edge("ocr_node_id", "plan_retrieval_node_id")
        workflow.add_edge("plan_retrieval_node_id", "retrieve_documents_node_id")

        workflow.add_conditional_edges("retrieve_documents_node_id", self._decide_after_retrieval, {
            "generate_rag_answer_node_id": "generate_rag_answer_node_id",
            "generate_direct_llm_answer_node_id": "generate_direct_llm_answer_node_id"
        })
        workflow.add_edge("generate_rag_answer_node_id", END)
        workflow.add_edge("generate_direct_llm_answer_node_id", END)

        graph_run_task = None
        try:
            app = workflow.compile()
            initial_state = GraphState(
                query=query,
                image_data=image_data,
                ocr_text=None,
                chat_history=chat_history,
                k=k,
                queries_for_retrieval=[],
                filters=None,
                documents=[],
                generation_instructions=None
            )
            graph_run_task = asyncio.create_task(app.ainvoke(initial_state, {"recursion_limit": 15}))

            async for token in self._stream_consumer(graph_run_task, output_queue):
                yield token

        except Exception as e:
            yield f"event: error\ndata: 그래프 구성 오류: {str(e)}\n\n"
        finally:
            if graph_run_task and not graph_run_task.done():
                try:
                    graph_run_task.cancel()
                    await graph_run_task
                except asyncio.CancelledError:
                    pass

            total_generate_end_time = time.perf_counter()
            print(
                f"--- ✨ LangGraph Generate 종료 (총 소요 시간: {total_generate_end_time - total_generate_start_time:.4f}초) ---")

    async def _stream_consumer(self, graph_task: asyncio.Task, queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        while True:
            try:
                token = await asyncio.wait_for(queue.get(), timeout=1.0)
                if token is None:
                    break
                yield str(token)
            except asyncio.TimeoutError:
                if graph_task.done():
                    if queue.empty():
                        break
                    continue

    # --- Graph Nodes ---

    def _ocr_node_func(self, state: GraphState) -> Dict[str, Any]:
        """(타일링 적용) 이미지가 있으면 OCR을 수행하여 텍스트를 추출합니다."""
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: OCR (Tiling with PaddleOCR) (시작) ---")  # [수정] 로그 메시지
        image_data = state.get("image_data")

        if not image_data or not self._ocr_reader:
            print("  [정보] 이미지가 없거나 OCR 리더가 초기화되지 않아 이 단계를 건너뜁니다.")
            print(f"--- 🔴 Node: OCR (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
            return {"ocr_text": None}

        try:
            image_bytes = base64.b64decode(image_data)
            original_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            width, height = original_image.size
            print(f"  [정보] 원본 이미지 크기: {width}x{height}")

            # 타일링 설정
            tile_size = 2048
            overlap = 150
            all_extracted_texts = set()

            # 이미지를 타일로 나누어 OCR 수행
            for y in range(0, height, tile_size - overlap):
                for x in range(0, width, tile_size - overlap):
                    box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                    if box[2] - box[0] < overlap or box[3] - box[1] < overlap:
                        continue

                    tile_image = original_image.crop(box)
                    img_np = np.array(tile_image)

                    # [수정] PaddleOCR 호출 및 결과 파싱 로직
                    ocr_results = self._ocr_reader.predict(img_np)

                    if ocr_results and ocr_results[0]:
                        for line in ocr_results[0]:
                            text = line[1][0]
                            all_extracted_texts.add(text)

            ocr_text = "\n".join(sorted(list(all_extracted_texts)))

            display_text = ocr_text[:150].replace('\n', ' ')
            print(f"  [정보] OCR 추출 텍스트 (일부): {display_text}...")

            print(
                f"--- 🔴 Node: OCR (Tiling with PaddleOCR) (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
            return {"ocr_text": ocr_text}
        except Exception as e:
            print(f"--- 💥 OCR 처리 중 오류 발생: {e} ---")
            return {"ocr_text": None}

    # ... (나머지 메서드들은 변경 없음) ...
    async def _plan_retrieval_node_func(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Plan Retrieval (시작) ---")

        query = state["query"]
        ocr_text = state.get("ocr_text") or ""
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"]])

        combined_text = query + " " + ocr_text
        potential_filters = {}

        filter_config = {
            "검증위원": self._all_reviewers,
            "도면명": self._all_drawing_names
        }

        for field_name, keyword_list in filter_config.items():
            for keyword in keyword_list:
                if keyword in combined_text:
                    if field_name not in potential_filters:
                        potential_filters[field_name] = []
                    if keyword not in potential_filters[field_name]:
                        potential_filters[field_name].append(keyword)

        potential_filters_str = json.dumps(potential_filters, ensure_ascii=False) if potential_filters else "없음"
        print(f"  [정보] 사전 식별된 필터 후보: {potential_filters_str}")

        print("  [정보] 지능적 경로: LLM에 최종 검색 계획 분석을 요청합니다.")
        parser = JsonOutputParser()

        analysis_prompt_template = """당신은 사용자의 의도를 파악하여 최적의 검색 계획을 수립하는 AI 검색 전략가입니다.

[분석할 정보]
1. 현재 사용자 질문: {query}
2. 대화 및 이미지 텍스트: {ocr_text}
   - (이전 대화 내용: {chat_history})
3. 사전 분석된 필터 후보: {potential_filters}

[당신의 임무]
'사전 분석된 필터 후보'가 '현재 사용자 질문'의 의도와 일치하는 경우에만 필터를 사용하고, 그렇지 않으면 무시해야 합니다.
예를 들어, 후보에 '김진수'가 있더라도 사용자가 "모든" 문서를 원하면 필터를 적용하지 마세요.
사용자가 "이 위원"에 대해 물으면 필터를 적용하세요.

이 분석을 바탕으로, 검색에 사용할 'search_queries'와 'filters'를 JSON 형식으로 최종 결정해주세요.
'filters'는 적용할 필터가 없으면 null 또는 빈 객체로 설정하세요.

{format_instructions}"""

        analysis_prompt = PromptTemplate.from_template(template=analysis_prompt_template, partial_variables={
            "format_instructions": parser.get_format_instructions()})
        chain = analysis_prompt | self._utility_llm | parser

        try:
            response_json = await chain.ainvoke({
                "query": query,
                "ocr_text": ocr_text,
                "chat_history": history_str,
                "potential_filters": potential_filters_str,
            })

            search_queries = response_json.get("search_queries", [query])
            if search_queries:
                search_queries = [search_queries[0]]

            result = {
                "queries_for_retrieval": search_queries,
                "filters": response_json.get("filters"),
                "generation_instructions": response_json.get("generation_instructions")
            }
            print(f"  [정보] LLM의 최종 결정: 검색어='{result['queries_for_retrieval']}', 필터='{result['filters']}'")

        except Exception as e:
            print(f"--- ⚠️ LLM 계획 수립 중 오류 발생, 원본 쿼리 사용: {e} ---")
            result = {"queries_for_retrieval": [query], "filters": None, "generation_instructions": None}

        print(f"--- 🔴 Node: Plan Retrieval (종료) (총 소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return result

    async def _retrieve_documents_node_func(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Retrieve Documents (Dense 검색) (시작) ---")
        queries = state["queries_for_retrieval"]
        filters = state.get("filters")
        k = state["k"]

        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=value)))
                elif isinstance(value, str):
                    conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
            if conditions:
                qdrant_filter = models.Filter(must=conditions)
                print(f"  [정보] Qdrant 필터 적용: {qdrant_filter.dict()}")

        query_vector = self._embed_model_instance.encode(queries[0]).tolist()

        search_results = self._qdrant_client.search(collection_name=self._qdrant_collection_name,
                                                    query_vector=query_vector, query_filter=qdrant_filter, limit=k,
                                                    with_payload=True)
        documents = [Document(page_content=hit.payload.get("text", ""), metadata=hit.payload) for hit in search_results]

        print(f"  [출력 업데이트] 최종 문서(개수): {len(documents)}")
        print(f"--- 🔴 Node: Retrieve Documents (종료) (총 소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return {"documents": documents}

    async def _common_answer_generation_logic(self, final_prompt_str: str, answer_queue: asyncio.Queue):
        try:
            async for token_chunk in self._model.astream(final_prompt_str):
                if token_chunk.content:
                    await answer_queue.put(token_chunk.content)
        except Exception as e:
            print(f"--- 💥 LLM 스트리밍 오류: {e} ---")
            await answer_queue.put(f"오류가 발생했습니다: {e}")
        finally:
            await answer_queue.put(None)

    async def _generate_rag_answer_node_func(self, state: GraphState, answer_queue: asyncio.Queue) -> Dict[str, Any]:
        print("\n--- 🟢 Node: Generate RAG Answer (시작) ---")
        query = state["query"]
        documents = state["documents"]
        instructions = state.get("generation_instructions") or "답변을 명확하고 간결하게 생성해주세요."
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"]]) if state[
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

        prompt = PromptTemplate.from_template(prompt_template_str)
        final_prompt_str = prompt.format(chat_history_str=history_str, context_str=context_str, original_query=query,
                                         instructions=instructions)

        await self._common_answer_generation_logic(final_prompt_str, answer_queue)
        return {}

    async def _generate_direct_llm_answer_node_func(self, state: GraphState, answer_queue: asyncio.Queue) -> Dict[
        str, Any]:
        print("\n--- 🟢 Node: Generate Direct LLM Answer (시작) ---")
        query = state["query"]
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"]]) if state[
            "chat_history"] else "이전 대화 없음"

        prompt_template = PromptTemplate.from_template(
            "당신은 친절한 대화형 AI입니다. 이전 대화 내용과 현재 사용자 질문을 바탕으로 자연스럽게 답변해주세요.\n[이전 대화 내용]\n{chat_history}\n[현재 사용자 질문]\n{query}\n답변:")
        final_prompt_str = prompt_template.format(chat_history=history_str, query=query)

        await self._common_answer_generation_logic(final_prompt_str, answer_queue)
        return {}

    def _decide_after_retrieval(self, state: GraphState) -> str:
        print(f"\n--- 🤔 Node: Decide After Retrieval ---")
        if state.get("documents"):
            print("  [결정] 검색된 문서가 있으므로 RAG 답변 생성을 진행합니다.")
            return "generate_rag_answer_node_id"
        else:
            print("  [결정] 검색된 문서가 없으므로 직접 LLM 답변 생성을 진행합니다.")
            return "generate_direct_llm_answer_node_id"