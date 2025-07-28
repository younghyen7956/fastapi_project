import os
import asyncio
import re
import time
from pathlib import Path
import json
from typing import AsyncGenerator, Optional, List, Dict, Any

import redis
import tiktoken
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

# This RAGRepository is assumed to be an abstract class in your project.
# Please adjust the import path to match your actual project structure.
from RAG.repository.simple_rag_repository import RAGRepository


class GraphState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    classification_result: Optional[str]
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
                RAGRepositoryImpl._summarizer = BartForConditionalGeneration.from_pretrained(summarizer_model_name,num_labels=2)

                print("--- Summarization model warming up...")
                warmup_start_summary = time.perf_counter()
                inputs = self._summarizer_tokenizer("Warm-up text.", return_tensors="pt")
                self._summarizer.generate(inputs['input_ids'], max_length=16)
                warmup_end_summary = time.perf_counter()
                print(
                    f"--- ✅ Summarization model warm-up complete. (소요 시간: {warmup_end_summary - warmup_start_summary:.4f}초)")
            except Exception as e:
                print(f"--- 💥 Failed to load summarization model: {e}")

        model_init_end_time = time.perf_counter()
        print(f"✅ 모든 모델 초기화 완료. (총 소요 시간: {model_init_end_time - model_init_start_time:.4f}초)")

    def _initialize_datastores(self):
        print("--- RAGRepositoryImpl: 데이터 저장소 초기화 중... ---")

        qdrant_host = os.getenv("QDRANT_HOST", "qdrant_db")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        RAGRepositoryImpl._qdrant_collection_name = os.getenv("QDRANT_COLLECTION", "construction_arctic_1024_v1")
        RAGRepositoryImpl._qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        print(f"--- ✅ Qdrant DB 연결 완료. (Collection: '{self._qdrant_collection_name}') ---")

        redis_host = os.getenv("REDIS_HOST", "redis_db")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        RAGRepositoryImpl._redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        print(f"--- ✅ Redis 서버 연결 완료. ---")

    def _prepare_filter_lists(self):
        print("--- 'filters.json' 파일에서 필터 목록을 로드하는 중... ---")
        filter_file_path = Path.cwd() / "filters.json"
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

    def _format_korean_text_chunk(self, text: str) -> str:
        if not text: return ""
        text = text.strip()
        text = re.sub(r'(입니다|됩니다|습니다)\.(?!\s*(?:<br>|$))', r'\1.<br><br>', text)
        text = re.sub(r'([\.!?])(?!\s*(?:<br>|$))(?=[가-힣A-Za-z0-9\(])', r'\1<br><br>', text)
        text = re.sub(r'\n+', '<br><br>', text)
        return text.strip()

    # --- Chat History Management Methods ---
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
            self._redis_client.set(session_id, updated_history_json, ex=86400)  # 24-hour expiration
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
        # 1. 기존 대화 기록을 가져옵니다.
        history = self.get_chat_history(session_id)

        # 2. 현재 대화(user, assistant)를 기록에 먼저 추가합니다.
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": ai_response})

        turns_to_keep = 3
        messages_to_keep = turns_to_keep * 2

        # 4. 보존할 메시지 개수를 초과하는 경우에만 요약을 진행합니다.
        if len(history) > messages_to_keep:
            print(f"--- 📝 세션 [{session_id}] 대화 요약 시작 (보존 메시지 수: {messages_to_keep}개 초과)... ---")

            # 5. 요약할 부분(오래된 대화)과 보존할 부분(최신 대화)을 나눕니다.
            history_to_summarize = history[:-messages_to_keep]
            recent_history = history[-messages_to_keep:]

            # 6. 오래된 대화를 요약합니다. (비동기 처리를 위해 to_thread 사용)
            summary_content = await asyncio.to_thread(self._summarize_with_local_model, history_to_summarize)
            print(f"--- ✅ 세션 [{session_id}] 요약 완료: {summary_content[:100]}... ---")

            # 7. 새로운 대화 기록을 '요약 + 최신 대화'로 재구성합니다.
            new_history = [{"role": "system", "content": f"이전 대화 요약: {summary_content}"}]
            new_history.extend(recent_history)

            # history 변수를 새로운 기록으로 업데이트합니다.
            history = new_history

        # 8. 최종적으로 정리된 대화 기록을 Redis에 저장합니다.
        self.save_chat_history(session_id, history)
        print(f"--- 💾 Redis 세션 [{session_id}] 업데이트 완료 (총 메시지 수: {len(history)}개). ---")

    # --- Main RAG Graph Execution ---
    async def generate(self, query: str, chat_history: List[Dict[str, str]], k: int = 10) -> AsyncGenerator[str, None]:
        total_generate_start_time = time.perf_counter()
        print(f"\n--- ✨ LangGraph Generate 시작: Query='{query[:50]}...' ---")

        output_queue = asyncio.Queue()
        workflow = StateGraph(GraphState)

        workflow.add_node("analyze_query_node_id", self._analyze_query_node_func)
        workflow.add_node("retrieve_documents_node_id", self._retrieve_documents_node_func)
        workflow.add_node("generate_rag_answer_node_id",
                          partial(self._generate_rag_answer_node_func, answer_queue=output_queue))
        workflow.add_node("generate_direct_llm_answer_node_id",
                          partial(self._generate_direct_llm_answer_node_func, answer_queue=output_queue))

        workflow.set_entry_point("analyze_query_node_id")
        workflow.add_edge("analyze_query_node_id", "retrieve_documents_node_id")
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
                chat_history=chat_history,
                k=k,
                classification_result=None,
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
    async def _analyze_query_node_func(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Analyze Query (2-Step) (시작) ---")
        query = state["query"]
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"]])

        found_filters = {}
        for reviewer in self._all_reviewers:
            if reviewer in query:
                if '검증위원' not in found_filters: found_filters['검증위원'] = []
                found_filters['검증위원'].append(reviewer)
        for drawing in self._all_drawing_names:
            if drawing in query:
                if '도면명' not in found_filters: found_filters['도면명'] = []
                found_filters['도면명'].append(drawing)

        if found_filters:
            print("    [정보] 빠른 경로: 단순 매칭으로 필터 발견. LLM 호출 생략.")
            search_query = query
            for values in found_filters.values():
                for value in values:
                    search_query = search_query.replace(value, "")
            search_query = search_query.strip() or query
            return {"queries_for_retrieval": [search_query], "filters": found_filters}

        print("    [정보] 지능적 경로: LLM으로 정교한 분석 시도.")
        parser = JsonOutputParser()
        analysis_prompt_template = """당신은 사용자의 질문을 분석하여 RAG 시스템을 위한 '작업 계획'을 수립하는 AI 플래너입니다.
'현재 사용자 질문'과 '이전 대화 내용'을 바탕으로 아래 3가지 요소를 JSON 형식으로 추출해주세요.

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
        chain = analysis_prompt | self._utility_llm | parser
        try:
            response_json = await chain.ainvoke(
                {"chat_history": history_str, "query": query, "valid_reviewers": self._all_reviewers,
                 "valid_drawings": self._all_drawing_names})
            search_queries = response_json.get("search_queries", [query])
            if search_queries: search_queries = [search_queries[0]]
            result = {"queries_for_retrieval": search_queries, "filters": response_json.get("filters"),
                      "generation_instructions": response_json.get("generation_instructions")}
        except Exception as e:
            print(f"--- ⚠️ 쿼리 분석 중 오류 발생, 원본 쿼리 사용: {e} ---")
            result = {"queries_for_retrieval": [query], "filters": None, "generation_instructions": None}
        print(f"--- 🔴 Node: Analyze Query (종료) (총 소요 시간: {time.perf_counter() - node_start_time:.4f}초) ---")
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
            if conditions: qdrant_filter = models.Filter(must=conditions)

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
        return "generate_rag_answer_node_id" if state.get("documents") else "generate_direct_llm_answer_node_id"