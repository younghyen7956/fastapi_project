import base64
import os
import asyncio
import re
import time
from io import BytesIO
from pathlib import Path
import json
from typing import AsyncGenerator, Optional, List, Dict, Any

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
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from VLRag.repository.vl_rag_repository import VlRAGRepository


class GraphState(TypedDict):
    query: str
    image_data: Optional[str]
    chat_history: List[Dict[str, str]]
    classification_result: Optional[str]
    queries_for_retrieval: List[str]
    filters: Optional[Dict[str, Any]]
    documents: List[Document]
    k: int
    generation_instructions: Optional[str]
    generation: Optional[str]


# --- Implementation Class ---
class VlRAGRepositoryImpl(VlRAGRepository):
    __instance = None
    _initialized = False

    # --- Models & Clients ---
    _vlm_model: Optional[Qwen2_5_VLForConditionalGeneration] = None
    _vlm_processor: Optional[AutoProcessor] = None
    _embed_model_instance: Optional[SentenceTransformer] = None
    _summarizer: Optional[BartForConditionalGeneration] = None
    _summarizer_tokenizer: Optional[PreTrainedTokenizerFast] = None
    _qdrant_client: Optional[QdrantClient] = None
    _qdrant_collection_name: Optional[str] = None
    _redis_client: Optional[redis.Redis] = None

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
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- 🖥️ 감지된 디바이스: {device} ---")
        print("--- ⚠️ 로컬 모델 로딩은 상당한 시간과 메모리가 소요될 수 있습니다. ---")

        if VlRAGRepositoryImpl._vlm_model is None:
            vlm_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            print(f"--- Loading Vision-Language Model '{vlm_model_name}'... ---")
            try:
                VlRAGRepositoryImpl._vlm_processor = AutoProcessor.from_pretrained(vlm_model_name,
                                                                                   trust_remote_code=True)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    vlm_model_name,
                    torch_dtype=torch.float16,  # bfloat16 대신 float16 사용
                    trust_remote_code=True,
                )
                # 3. 모델 전체를 감지된 디바이스로 이동
                VlRAGRepositoryImpl._vlm_model = model.to("cpu")
                print("--- ✅ Vision-Language Model loaded successfully. ---")
            except Exception as e:
                print(f"--- 💥 Failed to load VLM: {e} ---")
                import traceback
                traceback.print_exc()

        if VlRAGRepositoryImpl._embed_model_instance is None:
            embedding_model_name = os.getenv("EMBEDDING_MODEL", 'dragonkue/snowflake-arctic-embed-l-v2.0-ko')
            VlRAGRepositoryImpl._embed_model_instance = SentenceTransformer(embedding_model_name, device=device)
            print(f"--- Embedding model '{embedding_model_name}' on '{device}' loaded. ---")
            warmup_start_embed = time.perf_counter()
            VlRAGRepositoryImpl._embed_model_instance.encode("Warm-up text")
            warmup_end_embed = time.perf_counter()
            print(f"--- ✅ Embedding model warm-up complete. (소요 시간: {warmup_end_embed - warmup_start_embed:.4f}초)")

        if VlRAGRepositoryImpl._summarizer is None:
            summarizer_model_name = "EbanLee/kobart-summary-v3"
            print(f"--- Loading local summarization model '{summarizer_model_name}'... ---")
            try:
                VlRAGRepositoryImpl._summarizer_tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    summarizer_model_name)
                VlRAGRepositoryImpl._summarizer = BartForConditionalGeneration.from_pretrained(summarizer_model_name,
                                                                                               num_labels=2)
                print("--- ✅ Summarization model loaded successfully. ---")
            except Exception as e:
                print(f"--- 💥 Failed to load summarization model: {e}")

        model_init_end_time = time.perf_counter()
        print(f"✅ 모든 모델 초기화 완료. (총 소요 시간: {model_init_end_time - model_init_start_time:.4f}초)")

    def _initialize_datastores(self):
        print("--- VlRAGRepositoryImpl: 데이터 저장소 초기화 중... ---")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        VlRAGRepositoryImpl._qdrant_collection_name = os.getenv("QDRANT_COLLECTION", "construction_v2")
        VlRAGRepositoryImpl._qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        print(f"--- ✅ Qdrant DB 연결 완료. (Collection: '{self._qdrant_collection_name}') ---")

        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        VlRAGRepositoryImpl._redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        print(f"--- ✅ Redis 서버 연결 완료. ---")

    def _prepare_filter_lists(self):
        print("--- 'filters.json' 파일에서 필터 목록을 로드하는 중... ---")
        filter_file_path = Path.cwd() / "filters.json"
        if not filter_file_path.exists():
            print(f"⚠️ '{filter_file_path.resolve()}' 파일을 찾을 수 없습니다. 임시 데이터를 사용합니다.")
            VlRAGRepositoryImpl._all_reviewers = ["홍길동", "이순신"]
            VlRAGRepositoryImpl._all_drawing_names = ["101동 평면도", "배관도"]
            return
        try:
            with open(filter_file_path, "r", encoding="utf-8") as f:
                filter_data = json.load(f)
                VlRAGRepositoryImpl._all_reviewers = filter_data.get("reviewers", [])
                VlRAGRepositoryImpl._all_drawing_names = filter_data.get("drawings", [])
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
        inputs = self._summarizer_tokenizer(text_to_summarize, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self._summarizer.generate(inputs.input_ids, num_beams=4, max_length=256, early_stopping=True)
        return self._summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    async def update_chat_history(self, session_id: str, user_query: str, ai_response: str):
        history = self.get_chat_history(session_id)
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": ai_response})
        turns_to_keep, messages_to_keep = 3, 6

        if len(history) > messages_to_keep:
            print(f"--- 📝 세션 [{session_id}] 대화 요약 시작...")
            history_to_summarize, recent_history = history[:-messages_to_keep], history[-messages_to_keep:]
            summary_content = await asyncio.to_thread(self._summarize_with_local_model, history_to_summarize)
            new_history = [{"role": "system", "content": f"이전 대화 요약: {summary_content}"}] + recent_history
            history = new_history
        self.save_chat_history(session_id, history)
        print(f"--- 💾 Redis 세션 [{session_id}] 업데이트 완료 (총 메시지 수: {len(history)}개). ---")

    async def generate(self, query: str, chat_history: List[Dict[str, str]], k: int = 10,
                       image_data: Optional[str] = None) -> AsyncGenerator[str, None]:
        total_generate_start_time = time.perf_counter()
        print(f"\n--- ✨ LangGraph Generate 시작: Query='{query[:50]}...' | 이미지 존재: {'Yes' if image_data else 'No'} ---")

        workflow = StateGraph(GraphState)
        workflow.add_node("image_analysis_node", self._image_analysis_node)
        workflow.add_node("analyze_query_node", self._analyze_query_node)
        workflow.add_node("retrieve_documents_node", self._retrieve_documents_node)
        workflow.add_node("generate_rag_answer_node", self._generate_rag_answer_node)
        workflow.add_node("generate_direct_llm_answer_node", self._generate_direct_llm_answer_node)

        workflow.set_entry_point("image_analysis_node")
        workflow.add_edge("image_analysis_node", "analyze_query_node")
        workflow.add_edge("analyze_query_node", "retrieve_documents_node")
        workflow.add_conditional_edges("retrieve_documents_node", self._decide_after_retrieval, {
            "generate_rag_answer_node": "generate_rag_answer_node",
            "generate_direct_llm_answer_node": "generate_direct_llm_answer_node"
        })
        workflow.add_edge("generate_rag_answer_node", END)
        workflow.add_edge("generate_direct_llm_answer_node", END)

        app = workflow.compile()
        initial_state = GraphState(
            query=query, image_data=image_data, chat_history=chat_history, k=k,
            classification_result=None, queries_for_retrieval=[], filters=None,
            documents=[], generation_instructions=None, generation=None
        )

        try:
            final_state = await app.ainvoke(initial_state, {"recursion_limit": 15})
            final_answer = final_state.get("generation", "답변을 생성하지 못했습니다.")
            yield final_answer
        except Exception as e:
            error_message = f"그래프 실행 중 오류 발생: {str(e)}"
            print(f"--- 💥 {error_message} ---")
            yield f"event: error\ndata: {error_message}\n\n"

        total_generate_end_time = time.perf_counter()
        print(f"--- ✨ LangGraph Generate 종료 (총 소요 시간: {total_generate_end_time - total_generate_start_time:.4f}초) ---")

    async def _image_analysis_node(self, state: GraphState) -> Dict[str, Any]:
        node_start_time = time.perf_counter()
        print("\n--- 🖼️ Node: Image Analysis (시작) ---")
        query, image_data = state["query"], state["image_data"]

        if not image_data or not self._vlm_model or not self._vlm_processor:
            print("    [정보] 이미지가 없거나 VLM이 로드되지 않아 이 단계를 건너뜁니다.")
            return {}

        def _run_vlm_inference():
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")

            max_size = 1024
            if image.width > max_size or image.height > max_size:
                print(f"--- ⚠️  이미지 리사이징 수행 (원본: {image.size}) ---")
                image.thumbnail((max_size, max_size))
                print(f"--- ✅ 리사이징 완료 (결과: {image.size}) ---")

            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text",
                                                                                         "text": f"당신은 건축 도면 분석 전문가입니다. 주어진 도면 이미지와 사용자 질문을 바탕으로, 질문에 대한 답변 근거가 될 수 있는 모든 시각적 정보를 상세히 텍스트로 설명해주세요.\n\n# 사용자 질문:\n{query}"}]}]
            text_prompt = self._vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self._vlm_processor(text=[text_prompt], images=[image], return_tensors="pt")
            inputs = inputs.to(self._vlm_model.device)

            generated_ids = self._vlm_model.generate(**inputs, max_new_tokens=1024)
            response_text = self._vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            cleaned_response = response_text.split("assistant\n")[-1].strip()
            return cleaned_response

        image_description = await asyncio.to_thread(_run_vlm_inference)
        enhanced_query = f"사용자 원본 질문: \"{query}\"\n\n## 이미지 분석 결과 (도면 정보):\n{image_description}"
        print(f"    [정보] VLM 생성 설명(일부): {image_description[:150]}...")
        print(f"--- 🖼️ Node: Image Analysis (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return {"query": enhanced_query}

    async def _common_answer_generation(self, final_prompt_str: str) -> str:
        def _run_llm_inference():
            if not self._vlm_model or not self._vlm_processor:
                return "VLM 모델이 준비되지 않았습니다."
            inputs = self._vlm_processor(text=final_prompt_str, return_tensors="pt").to(self._vlm_model.device)
            generated_ids = self._vlm_model.generate(**inputs, max_new_tokens=1024)
            response_text = \
            self._vlm_processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            return response_text

        return await asyncio.to_thread(_run_llm_inference)

    async def _analyze_query_node(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Enhanced Query Analysis (시작) ---")
        query, history_str = state["query"], "\n".join([f"{m['role']}: {m['content']}" for m in state["chat_history"]])

        parser = JsonOutputParser()
        analysis_prompt_template = """당신은 사용자의 질문을 분석하여 RAG 시스템을 위한 '작업 계획'을 수립하는 AI 플래너입니다. '현재 사용자 질문'과 '이전 대화 내용'을 바탕으로 아래 3가지 요소를 JSON 형식으로 추출해주세요.

[추출할 요소]
1. "search_queries": 사용자의 질문을 벡터 검색에 가장 적합한 **단 하나의 핵심 검색어 구문(리스트 형태)**으로 변환합니다.
2. "filters": '유효한 필터 목록'에 있는 값이 질문에 언급된 경우, '필드: 값' 쌍으로 추출합니다. 없으면 null.
3. "generation_instructions": 검색 결과와 별개로, 최종 답변을 생성할 때 따라야 할 지시사항. 없으면 null.

[유효한 필터 목록]
- 검증위원: {valid_reviewers}
- 도면명: {valid_drawings}

[작업 시작]
[이전 대화 내용]
{chat_history}

[현재 사용자 질문]
{query}

{format_instructions}"""
        analysis_prompt = PromptTemplate.from_template(template=analysis_prompt_template, partial_variables={
            "format_instructions": parser.get_format_instructions()})
        final_prompt = analysis_prompt.format(
            chat_history=history_str, query=query, valid_reviewers=self._all_reviewers,
            valid_drawings=self._all_drawing_names
        )

        try:
            json_response_str = await self._common_answer_generation(final_prompt)
            response_json = parser.parse(json_response_str)
            search_queries = response_json.get("search_queries", [query])
            if search_queries: search_queries = [search_queries[0]]
            result = {"queries_for_retrieval": search_queries, "filters": response_json.get("filters"),
                      "generation_instructions": response_json.get("generation_instructions")}
        except Exception as e:
            print(f"--- ⚠️ 쿼리 분석 중 오류 발생, 원본 쿼리 사용: {e} ---")
            result = {"queries_for_retrieval": [query], "filters": None, "generation_instructions": None}

        print(f"--- 🔴 Node: Enhanced Query Analysis (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return result

    async def _retrieve_documents_node(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Retrieve Documents (Dense 검색) (시작) ---")
        queries, filters, k = state["queries_for_retrieval"], state.get("filters"), state["k"]

        qdrant_filter = None
        if filters:
            conditions = [models.FieldCondition(key=key, match=models.MatchAny(any=val)) if isinstance(val,
                                                                                                       list) else models.FieldCondition(
                key=key, match=models.MatchValue(value=val)) for key, val in filters.items()]
            if conditions: qdrant_filter = models.Filter(must=conditions)

        query_vector = self._embed_model_instance.encode(queries[0]).tolist()
        search_results = self._qdrant_client.search(
            collection_name=self._qdrant_collection_name, query_vector=query_vector, query_filter=qdrant_filter,
            limit=k, with_payload=True
        )
        documents = [Document(page_content=hit.payload.get("text", ""), metadata=hit.payload) for hit in search_results]

        print(f"  [출력 업데이트] 최종 문서(개수): {len(documents)}")
        print(f"--- 🔴 Node: Retrieve Documents (종료) (소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
        return {"documents": documents}

    async def _generate_rag_answer_node(self, state: GraphState) -> Dict[str, Any]:
        print("\n--- 🟢 Node: Generate RAG Answer (시작) ---")
        query, documents = state["query"], state["documents"]
        instructions = state.get("generation_instructions") or "답변을 명확하고 간결하게 생성해주세요."
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
        final_prompt_str = PromptTemplate.from_template(prompt_template_str).format(
            chat_history_str=history_str, context_str=context_str, original_query=query, instructions=instructions
        )

        generation = await self._common_answer_generation(final_prompt_str)
        return {"generation": generation}

    async def _generate_direct_llm_answer_node(self, state: GraphState) -> Dict[str, Any]:
        print("\n--- 🟢 Node: Generate Direct LLM Answer (시작) ---")
        query, history_str = state["query"], "\n".join(
            [f"{m['role']}: {m['content']}" for m in state["chat_history"]]) if state["chat_history"] else "이전 대화 없음"
        prompt_template = PromptTemplate.from_template(
            "당신은 친절한 대화형 AI입니다. 이전 대화 내용과 현재 사용자 질문을 바탕으로 자연스럽게 답변해주세요.\n[이전 대화 내용]\n{chat_history}\n[현재 사용자 질문]\n{query}\n답변:")
        final_prompt_str = prompt_template.format(chat_history=history_str, query=query)

        generation = await self._common_answer_generation(final_prompt_str)
        return {"generation": generation}

    def _decide_after_retrieval(self, state: GraphState) -> str:
        print(f"\n--- 🤔 Node: Decide After Retrieval ---")
        if state.get("documents"):
            print("  [결정] 문서가 검색되었으므로 RAG 답변 생성을 진행합니다.")
            return "generate_rag_answer_node"
        else:
            print("  [결정] 검색된 문서가 없으므로 직접 LLM 답변 생성을 진행합니다.")
            return "generate_direct_llm_answer_node"