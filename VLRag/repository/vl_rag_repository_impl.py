import base64
import os
import asyncio
import json
import time
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator, Optional, List, Dict, Any
from uuid import uuid4

import redis
import torch
from PIL import Image
from dotenv import load_dotenv

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
# ## 삭제된 부분 ##: ImagePixelData import를 제거합니다.
# from vllm.multimodal import ImagePixelData

from VLRag.repository.vl_rag_repository import VlRAGRepository


class GraphState(TypedDict):
    query: str
    image_data: Optional[str]
    chat_history: List[Dict[str, str]]
    session_id: Optional[str]
    queries_for_retrieval: List[str]
    filters: Optional[Dict[str, Any]]
    documents: List[Document]
    k: int
    generation_instructions: Optional[str]
    generation: Any


class VlRAGRepositoryImpl(VlRAGRepository):
    __instance = None
    _vlm_model: Optional[AsyncLLMEngine] = None
    _vlm_processor: Optional[AutoProcessor] = None
    _embed_model_instance: Optional[SentenceTransformer] = None
    _summarizer: Optional[BartForConditionalGeneration] = None
    _summarizer_tokenizer: Optional[PreTrainedTokenizerFast] = None
    _qdrant_client: Optional[QdrantClient] = None
    _qdrant_collection_name: Optional[str] = None
    _redis_client: Optional[redis.Redis] = None
    _all_reviewers: List[str] = []
    _all_drawing_names: List[str] = []

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
                    # ## 변경된 부분 ##: 멀티모달 입력을 위한 옵션 추가
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
            print("--- 🧠 Embedding model will be loaded on CPU to save VRAM for VLM. ---")
            self._embed_model_instance = SentenceTransformer(embedding_model_name, device='cpu')
            print(f"--- Embedding model '{embedding_model_name}' on 'cpu' loaded. ---")

        if self._summarizer is None:
            summarizer_model_name = "EbanLee/kobart-summary-v3"
            print(f"--- Loading local summarization model '{summarizer_model_name}'... ---")
            self._summarizer_tokenizer = PreTrainedTokenizerFast.from_pretrained(summarizer_model_name)
            self._summarizer = BartForConditionalGeneration.from_pretrained(summarizer_model_name)
            print("--- ✅ Summarization model loaded successfully. ---")

        print(f"✅ 모든 모델 초기화 완료. (총 소요 시간: {time.perf_counter() - model_init_start_time:.4f}초)")

    def _initialize_datastores(self):
        # ... (기존 코드와 동일)
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
        # ... (기존 코드와 동일)
        print("--- 'filters.json' 파일에서 필터 목록을 로드하는 중... ---")
        filter_file_path = Path.cwd() / "filters.json"
        if not filter_file_path.exists():
            print(f"⚠️ '{filter_file_path.resolve()}' 파일을 찾을 수 없습니다. 임시 데이터를 사용합니다.")
            self._all_reviewers = ["홍길동", "이순신"]
            self._all_drawing_names = ["101동 평면도", "배관도"]
            return
        try:
            with open(filter_file_path, "r", encoding="utf-8") as f:
                filter_data = json.load(f)
                self._all_reviewers = filter_data.get("reviewers", [])
                self._all_drawing_names = filter_data.get("drawings", [])
            print("✅ 필터 목록 로드 완료.")
        except Exception as e:
            print(f"--- 💥 필터 목록 파일 로드 중 오류 발생: {e}")

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        # ... (기존 코드와 동일)
        try:
            stored_history = self._redis_client.get(session_id)
            return json.loads(stored_history) if stored_history else []
        except Exception as e:
            print(f"--- 💥 Redis 조회 오류 (session_id: {session_id}): {e}")
            return []

    def save_chat_history(self, session_id: str, history: List[Dict[str, str]]):
        # ... (기존 코드와 동일)
        try:
            updated_history_json = json.dumps(history, ensure_ascii=False)
            self._redis_client.set(session_id, updated_history_json, ex=86400)
        except Exception as e:
            print(f"--- 💥 Redis 저장 오류 (session_id: {session_id}): {e}")

    def _summarize_with_local_model(self, history: List[Dict[str, str]]) -> str:
        # ... (기존 코드와 동일)
        if not self._summarizer or not self._summarizer_tokenizer: return "(요약 모델 로드 실패)"
        text_to_summarize = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        if not text_to_summarize.strip(): return "(요약할 내용 없음)"
        inputs = self._summarizer_tokenizer(text_to_summarize, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self._summarizer.generate(inputs.input_ids, num_beams=4, max_length=256, early_stopping=True)
        return self._summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    async def update_chat_history(self, session_id: str, user_query: str, ai_response: str):
        # ... (기존 코드와 동일)
        history = self.get_chat_history(session_id)
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": ai_response})
        messages_to_keep = 6
        if len(history) > messages_to_keep:
            print(f"--- 📝 세션 [{session_id}] 대화 요약 시작...")
            history_to_summarize, recent_history = history[:-messages_to_keep], history[-messages_to_keep:]
            summary_content = await asyncio.to_thread(self._summarize_with_local_model, history_to_summarize)
            new_history = [{"role": "system", "content": f"이전 대화 요약: {summary_content}"}] + recent_history
            history = new_history
        self.save_chat_history(session_id, history)
        print(f"--- 💾 Redis 세션 [{session_id}] 업데이트 완료 (총 메시지 수: {len(history)}개). ---")

    async def _generate_search_plan_node(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Generate Search Plan (시작) ---")
        query, image_data, history_str = state["query"], state["image_data"], "\n".join(
            [f"{m['role']}: {m['content']}" for m in state["chat_history"]])
        parser = JsonOutputParser()

        prompt_object: any = None

        if image_data:
            print("    [정보] 이미지+텍스트 기반 검색 계획을 수립합니다.")
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            prompt_text_template = """당신은 이미지와 텍스트를 분석하여 JSON 형식의 작업 계획을 수립하는 AI 플래너입니다. 주어진 이미지와 사용자 질문을 바탕으로, Qdrant DB 검색에 필요한 'search_queries'와 'filters'를 추출해주세요.

[유효한 필터 목록]
- 검증위원: {valid_reviewers}
- 도면명: {valid_drawings}

[이전 대화 내용]
{chat_history}

[사용자 질문]
{query}

이미지와 모든 정보를 종합하여 JSON을 생성하세요.
{format_instructions}"""

            text_content = prompt_text_template.format(
                valid_reviewers=self._all_reviewers, valid_drawings=self._all_drawing_names,
                chat_history=history_str, query=query, format_instructions=parser.get_format_instructions()
            )

            # ## 여기가 핵심입니다 ##
            # 1. 이미지와 텍스트를 content 리스트로 구성합니다.
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_content},
                ]
            }]

            # 2. processor의 apply_chat_template을 호출하여 vLLM이 이해하는 프롬프트를 생성합니다.
            final_prompt_str = self._vlm_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 3. 이전과 동일하게 최종 객체를 구성합니다.
            prompt_object = {
                "prompt": final_prompt_str,
                "multi_modal_data": {"image": image}
            }

        else:  # 이미지가 없는 경우는 기존 방식 그대로 유지
            print("    [정보] 텍스트 기반 검색 계획을 수립합니다.")
            analysis_prompt_template = """(생략)"""
            analysis_prompt = PromptTemplate.from_template(template=analysis_prompt_template, partial_variables={
                "format_instructions": parser.get_format_instructions()})
            text_only_prompt = analysis_prompt.format(chat_history=history_str, query=query,
                                                      valid_reviewers=self._all_reviewers,
                                                      valid_drawings=self._all_drawing_names)
            messages = [{"role": "user", "content": text_only_prompt}]
            prompt_object = self._vlm_processor.apply_chat_template(messages, tokenize=False,
                                                                    add_generation_prompt=True)

        if not self._vlm_model:
            raise RuntimeError("VLM 모델이 성공적으로 초기화되지 않았습니다.")

        sampling_params = SamplingParams(temperature=0, max_tokens=1024)
        request_id = str(uuid4())

        results_generator = self._vlm_model.generate(
            prompt=prompt_object,
            sampling_params=sampling_params,
            request_id=request_id
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output is None:
            raise RuntimeError("VLM에서 응답을 생성하지 못했습니다.")

        json_response_str = final_output.outputs[0].text.strip()

        try:
            response_json = parser.parse(json_response_str)
            search_queries = response_json.get("search_queries", [query])
            if search_queries: search_queries = [search_queries[0]]
            result = {"queries_for_retrieval": search_queries, "filters": response_json.get("filters"),
                      "generation_instructions": response_json.get("generation_instructions")}
        except Exception as e:
            print(f"--- ⚠️ 쿼리 분석 중 오류 발생, 원본 쿼리 사용: {e} ---")
            result = {"queries_for_retrieval": [query], "filters": None, "generation_instructions": None}

        print(f"--- 🔴 Node: Generate Search Plan (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return result

    def _retrieve_documents_node(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Retrieve Documents (Dense 검색) (시작) ---")
        queries, filters, k = state["queries_for_retrieval"], state.get("filters"), state["k"]

        qdrant_filter = None

        if filters:
            print(f"  [정보] LLM으로부터 받은 원본 필터: {filters}")

            # LLM이 간혹 [{"key":"value"}] 형태로 리스트 안에 딕셔너리를 반환하는 경우에 대한 처리
            if isinstance(filters, list) and filters:
                filters = filters[0]

            # 최종적으로 filters가 딕셔너리인지 확인하고 Qdrant 필터 생성
            if isinstance(filters, dict):
                conditions = [
                    models.FieldCondition(key=key, match=models.MatchValue(value=val))
                    for key, val in filters.items()
                ]
                if conditions:
                    qdrant_filter = models.Filter(must=conditions)
                    print(f"  [정보] Qdrant에 적용될 필터: {qdrant_filter.dict()}")
            else:
                print(f"  --- ⚠️ 필터가 딕셔너리 형식이 아니어서 무시합니다: {filters} ---")

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
        # ... (기존 코드와 동일)
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
        # ... (기존 코드와 동일)
        print("\n--- 🟢 Node: Generate Direct LLM Answer (시작) ---")
        query, history_str = state["query"], "\n".join(
            [f"{m['role']}: {m['content']}" for m in state["chat_history"]]) if state["chat_history"] else "이전 대화 없음"
        prompt_template = PromptTemplate.from_template(
            "당신은 친절한 대화형 AI입니다. 이전 대화 내용과 현재 사용자 질문을 바탕으로 자연스럽게 답변해주세요.\n[이전 대화 내용]\n{chat_history}\n[현재 사용자 질문]\n{query}\n답변:")
        final_prompt_str = prompt_template.format(chat_history=history_str, query=query)
        return {"generation": final_prompt_str}

    def _decide_after_retrieval(self, state: GraphState) -> str:
        # ... (기존 코드와 동일)
        print(f"\n--- 🤔 Node: Decide After Retrieval ---")
        if state.get("documents"):
            print("  [결정] 문서가 검색되었으므로 RAG 답변 생성을 진행합니다.")
            return "generate_rag_answer_node"
        else:
            print("  [결정] 검색된 문서가 없으므로 직접 LLM 답변 생성을 진행합니다.")
            return "generate_direct_llm_answer_node"

    async def generate(self, query: str, chat_history: List[Dict[str, str]], k: int = 5,
                       session_id: Optional[str] = None, image_data: Optional[str] = None) -> AsyncGenerator[str, None]:
        # ... (기존 코드와 동일, 마지막 스트리밍 생성 부분은 이미 텍스트만 처리하므로 수정 필요 없음)
        total_generate_start_time = time.perf_counter()
        print(f"\n--- ✨ LangGraph Generate 시작: Query='{query[:50]}...' | 이미지 존재: {'Yes' if image_data else 'No'} ---")

        workflow = StateGraph(GraphState)
        workflow.add_node("generate_search_plan_node", self._generate_search_plan_node)
        workflow.add_node("retrieve_documents_node", self._retrieve_documents_node)
        workflow.add_node("generate_rag_answer_node", self._generate_rag_answer_node)
        workflow.add_node("generate_direct_llm_answer_node", self._generate_direct_llm_answer_node)
        workflow.set_entry_point("generate_search_plan_node")
        workflow.add_edge("generate_search_plan_node", "retrieve_documents_node")
        workflow.add_conditional_edges("retrieve_documents_node", self._decide_after_retrieval,
                                       {"generate_rag_answer_node": "generate_rag_answer_node",
                                        "generate_direct_llm_answer_node": "generate_direct_llm_answer_node"})
        workflow.add_edge("generate_rag_answer_node", END)
        workflow.add_edge("generate_direct_llm_answer_node", END)
        app = workflow.compile()

        initial_state = GraphState(query=query, image_data=image_data, chat_history=chat_history, k=k,
                                   session_id=session_id)

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