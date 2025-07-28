import os
import asyncio
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import AsyncGenerator, Optional, List, Dict, Any
import json
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from functools import partial
from langchain_core.output_parsers import JsonOutputParser
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel  # ✨ bge-m3를 위한 올바른 라이브러리 임포트
import torch


class GraphState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    classification_result: Optional[str]
    queries_for_retrieval: List[str]
    filters: Optional[Dict[str, Any]]
    documents: List[Document]
    k: int
    generation_instructions: Optional[str]


class RAGRepositoryImpl:
    __instance = None
    _initialized = False
    _embed_model_instance: Optional[BGEM3FlagModel] = None # ✨ 타입 힌트 수정
    _model: Optional[ChatOpenAI] = None
    _utility_llm: Optional[ChatOpenAI] = None
    _global_chat_history: List[Dict[str, str]] = []
    MAX_GLOBAL_HISTORY_TURNS = 10

    _qdrant_client: Optional[QdrantClient] = None
    _qdrant_collection_name: Optional[str] = None

    _all_reviewers: List[str] = []
    _all_drawing_names: List[str] = []
    # ✨ 하이브리드 모델을 사용하므로 벡터 캐시는 제거 (단순 텍스트 쿼리 캐싱은 복잡도 증가)

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
            self._initialize_db()
            self._prepare_filter_lists()
            init_end_time = time.perf_counter()
            print(
                f"--- ⏱️ RAGRepositoryImpl: __init__ 최초 초기화 완료. (총 소요 시간: {init_end_time - init_start_time:.4f}초) ---")

    def _initialize_models(self):
        model_init_start_time = time.perf_counter()
        print("--- RAGRepositoryImpl: 모델 초기화 중... ---")
        RAGRepositoryImpl._model = ChatOpenAI(model=os.getenv("MAIN_LLM_MODEL", "gpt-4o-mini"), temperature=0.0,
                                              openai_api_key=os.getenv("OPENAI_API_KEY"), streaming=True)
        RAGRepositoryImpl._utility_llm = ChatOpenAI(model=os.getenv("UTILITY_LLM_MODEL", "gpt-4o-mini"),
                                                    temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"),
                                                    streaming=False)
        print("--- Utility LLM temperature set to 0.0 for deterministic outputs. ---")

        if RAGRepositoryImpl._embed_model_instance is None:
            embedding_model_name = os.getenv("EMBEDDING_MODEL", 'BAAI/bge-m3')
            if not embedding_model_name: raise ValueError("EMBEDDING_MODEL 환경 변수 누락")

            device = 'cpu'
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'

            RAGRepositoryImpl._embed_model_instance = BGEM3FlagModel(
                embedding_model_name, device=device, use_fp16=True
            )
            print(f"--- Embedding model '{embedding_model_name}' on '{device}' loaded. (BGEM3FlagModel) ---")

            # ✨ [추가] bge-m3 모델 웜업을 위해 dummy 인코딩 실행
            print("--- bge-m3 model warming up...")
            warmup_start = time.perf_counter()
            # 간단한 텍스트로 인코딩을 한 번 실행하여 실제 연산을 트리거합니다.
            RAGRepositoryImpl._embed_model_instance.encode("Warm-up text")
            warmup_end = time.perf_counter()
            print(f"--- ✅ bge-m3 model warm-up complete. (소요 시간: {warmup_end - warmup_start:.4f}초)")

        model_init_end_time = time.perf_counter()
        print(f"✅ LLM, Embedding 모델 초기화 완료. (소요 시간: {model_init_end_time - model_init_start_time:.4f}초)")

    def _initialize_db(self):
        db_init_start_time = time.perf_counter()
        print("--- RAGRepositoryImpl: DB 초기화 중 (Qdrant 클라이언트-서버 모드) ---")
        db_host = os.getenv("QDRANT_HOST", "localhost")
        db_port = int(os.getenv("QDRANT_PORT", 6333))
        collection_name = os.getenv("QDRANT_COLLECTION", "construction_v1")
        RAGRepositoryImpl._qdrant_client = QdrantClient(host=db_host, port=db_port)
        RAGRepositoryImpl._qdrant_collection_name = collection_name
        print(f"✅ Qdrant 서버에 연결 완료. Collection='{collection_name}'")

    def _prepare_filter_lists(self):
        print("--- 'filters.json' 파일에서 필터 목록을 로드하는 중... ---")
        filter_file_path = Path.cwd() / "filters.json"
        print(f"ℹ️ 필터 파일 탐색 경로: {filter_file_path.resolve()}")
        if not filter_file_path.exists():
            print(f"⚠️ '{filter_file_path.resolve()}' 파일을 찾을 수 없습니다. 빈 목록으로 시작합니다.")
            return
        try:
            with open(filter_file_path, "r", encoding="utf-8") as f:
                filter_data = json.load(f)
            RAGRepositoryImpl._all_reviewers = filter_data.get("reviewers", [])
            RAGRepositoryImpl._all_drawing_names = filter_data.get("drawings", [])
            print("✅ 필터 목록 로드 완료.")
            print(f"   - 검증위원: {len(RAGRepositoryImpl._all_reviewers)}명")
            print(f"   - 도면명: {len(RAGRepositoryImpl._all_drawing_names)}개")
        except Exception as e:
            print(f"--- 💥 필터 목록 파일 로드 중 오류 발생: {e}")

    def _format_korean_text_chunk(self, text: str) -> str:
        if not text: return ""
        text = text.strip()
        text = re.sub(r'(입니다|됩니다|습니다)\.(?!\s*(?:<br>|$))', r'\1.<br><br>', text)
        text = re.sub(r'([\.!?])(?!\s*(?:<br>|$))(?=[가-힣A-Za-z0-9\(])', r'\1<br><br>', text)
        text = re.sub(r'\n+', '<br><br>', text)
        return text.strip()

    async def _analyze_query_node_func(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Analyze Query (2-Step) (시작) ---")
        query = state["query"]
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"]])

        # --- 1단계: 빠른 키워드 필터링 (Fast Path) ---
        found_filters = {}
        for reviewer in RAGRepositoryImpl._all_reviewers:
            if reviewer in query:
                if '검증위원' not in found_filters: found_filters['검증위원'] = []
                found_filters['검증위원'].append(reviewer)
        for drawing in RAGRepositoryImpl._all_drawing_names:
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

            return {
                "queries_for_retrieval": [search_query],
                "filters": found_filters,
                "generation_instructions": None
            }

        # --- 2단계: LLM을 통한 정교한 분석 (Slow/Smart Path) ---
        print("    [정보] 지능적 경로: LLM으로 정교한 분석 시도.")
        parser = JsonOutputParser()
        analysis_prompt = PromptTemplate.from_template(
            """당신은 사용자의 질문을 분석하여 RAG 시스템을 위한 '작업 계획'을 수립하는 AI 플래너입니다.
            '현재 사용자 질문'과 '이전 대화 내용'을 바탕으로 아래 3가지 요소를 JSON 형식으로 추출해주세요.

            [추출할 요소]
            1. "search_queries": 사용자의 질문을 벡터 검색에 가장 적합한 **단 하나의 핵심 검색어 구문(리스트 형태)**으로 변환합니다.
            2. "filters": '유효한 필터 목록'에 있는 값이 질문에 언급된 경우, '필드: 값' 쌍으로 추출합니다. 없으면 null.
            3. "generation_instructions": 검색 결과와 별개로, 최종 답변을 생성할 때 따라야 할 지시사항. 없으면 null.

            [유효한 필터 목록]
            - 검증위원: {valid_reviewers}
            - 도면명: {valid_drawings}

            [예시]
            - 사용자 질문: "검증위원 이문찬이 제출한 단지배치도 관련 검토의견을 LIST 정리해줘"
            - 당신의 JSON 출력: {{"search_queries": ["이문찬 위원 단지배치도 검토의견"], "filters": {{"검증위원": "이문찬", "도면명": "단지배치도"}}, "generation_instructions": "검토의견을 LIST 형식으로 정리해서 보여줘"}}

            [작업 시작]
            [이전 대화 내용]
            {chat_history}
            [현재 사용자 질문]
            {query}
            {format_instructions}""",
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = analysis_prompt | RAGRepositoryImpl._utility_llm | parser
        try:
            llm_call_start_time = time.perf_counter()
            response_json = await chain.ainvoke({
                "chat_history": history_str, "query": query,
                "valid_reviewers": RAGRepositoryImpl._all_reviewers,
                "valid_drawings": RAGRepositoryImpl._all_drawing_names
            })
            search_queries = response_json.get("search_queries", [query])
            if search_queries:
                search_queries = [search_queries[0]]
            filters = response_json.get("filters")
            generation_instructions = response_json.get("generation_instructions")
        except Exception as e:
            search_queries = [query]
            filters = None
            generation_instructions = None

        return {
            "queries_for_retrieval": search_queries,
            "filters": filters,
            "generation_instructions": generation_instructions
        }

    # ✨ [최종 수정] bge-m3 모델에 맞는 하이브리드 검색 로직
    async def _retrieve_documents_node_func(self, state: GraphState) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Retrieve Documents (bge-m3 하이브리드 검색) (시작) ---")
        queries = state["queries_for_retrieval"]
        filters = state.get("filters")
        k = state["k"]

        if not queries:
            print("  [정보] 검색할 쿼리가 없어 검색을 건너뜁니다.")
            return {"documents": []}

        print(f"  [입력 상태] Queries: {queries}, Filters: {filters}, k: {k}")

        qdrant_filter = None
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                if isinstance(value, list) and len(value) > 0:
                    filter_conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=value)))
                elif isinstance(value, str):
                    filter_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
            if filter_conditions:
                qdrant_filter = models.Filter(must=filter_conditions)

        final_rrf_scores = {}
        rrf_k = 60

        for q in queries:
            print(f"  - 검색 실행 중인 쿼리: \"{q}\"")

            encode_start_time = time.perf_counter()
            output = RAGRepositoryImpl._embed_model_instance.encode(
                [q], return_dense=True, return_sparse=True, return_colbert_vecs=False
            )
            encode_end_time = time.perf_counter()
            print(f"    [측정] 쿼리 인코딩: {encode_end_time - encode_start_time:.4f}초")

            query_dense_vector = output['dense_vecs'][0].tolist()
            query_lexical_weights = output['lexical_weights'][0]
            query_sparse_vector = models.SparseVector(
                indices=list(map(int, query_lexical_weights.keys())),
                values=list(query_lexical_weights.values())
            )

            search_requests = [
                models.SearchRequest(
                    vector=models.NamedVector(name="dense", vector=query_dense_vector),
                    limit=k, with_payload=True, filter=qdrant_filter
                ),
                models.SearchRequest(
                    vector=models.NamedSparseVector(name="lexical-weights", vector=query_sparse_vector),
                    limit=k, with_payload=True, filter=qdrant_filter
                )
            ]

            search_start_time = time.perf_counter()
            dense_results, sparse_results = RAGRepositoryImpl._qdrant_client.search_batch(
                collection_name=RAGRepositoryImpl._qdrant_collection_name, requests=search_requests
            )
            search_end_time = time.perf_counter()
            print(f"    [측정] Qdrant DB 검색: {search_end_time - search_start_time:.4f}초")

            for rank, hit in enumerate(dense_results):
                if hit.id not in final_rrf_scores: final_rrf_scores[hit.id] = {'score': 0, 'payload': hit.payload}
                final_rrf_scores[hit.id]['score'] += 1 / (rrf_k + rank + 1)

            for rank, hit in enumerate(sparse_results):
                if hit.id not in final_rrf_scores: final_rrf_scores[hit.id] = {'score': 0, 'payload': hit.payload}
                final_rrf_scores[hit.id]['score'] += 1 / (rrf_k + rank + 1)

        if not final_rrf_scores:
            print("  [결과] 모든 쿼리에 대해 검색 결과가 없습니다.")
            return {"documents": []}

        sorted_results = sorted(final_rrf_scores.items(), key=lambda item: item[1]['score'], reverse=True)
        final_documents = [Document(page_content=data['payload'].get("text", ""), metadata=data['payload']) for
                           doc_id, data in sorted_results[:k]]

        print(f"  [출력 업데이트] 최종 문서(개수): {len(final_documents)}")
        node_end_time = time.perf_counter()
        print(f"--- 🔴 Node: Retrieve Documents (종료) (총 소요 시간: {node_end_time - node_start_time:.4f}초) ---")
        return {"documents": final_documents}

    async def _common_answer_generation_logic(self, final_prompt_str: str, answer_queue: asyncio.Queue):
        try:
            log_prompt = re.sub(r'\s+', ' ', final_prompt_str)[:200]
            print(f"    [LLM 스트리밍 시작] Prompt (일부): {log_prompt}...")
            ttft_start_time = time.perf_counter()
            first_token_received = False
            async for token_chunk in RAGRepositoryImpl._model.astream(final_prompt_str):
                if not first_token_received:
                    ttft_end_time = time.perf_counter()
                    print(f"    [LLM 스트리밍] 첫 토큰 수신! (소요 시간: {ttft_end_time - ttft_start_time:.4f}초)")
                    first_token_received = True
                content_to_put = token_chunk.content if hasattr(token_chunk, 'content') else str(token_chunk)
                if content_to_put: await answer_queue.put(content_to_put)
            print("    [LLM 스트리밍 정상 종료]")
        except Exception as e:
            await answer_queue.put(f"event: error\ndata: LLM 답변 생성 중 오류: {str(e)}\n\n")
        finally:
            await answer_queue.put(None)

    async def _generate_rag_answer_node_func(self, state: GraphState, answer_queue: asyncio.Queue) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Generate RAG Answer (시작) ---")
        query = state["query"]
        documents = state["documents"]
        instructions = state.get("generation_instructions") or "답변을 명확하고 간결하게 생성해주세요."
        context_str = "\n\n---\n\n".join([doc.page_content for doc in documents]) if documents else "참고할 문서가 없습니다."
        prompt_template = PromptTemplate.from_template("""당신은 건축 관련 전문가 입니다. 당신의 주요 임무는 사용자의 원본 요청에 대해, 주어진 '참고 문서'와 '추가 지시사항'을 바탕으로 최종 답변을 생성하는 것입니다.
        [참고 문서]\n{context_str}\n\n[사용자 원본 요청]\n{original_query}\n\n[추가 지시사항]\n{instructions}
        '참고 문서'를 바탕으로, '사용자 원본 요청'에 대해 '추가 지시사항'을 충실히 반영하여 최종 답변을 생성해주세요.""")
        final_prompt_str = prompt_template.format(context_str=context_str, original_query=query,
                                                  instructions=instructions)
        await self._common_answer_generation_logic(final_prompt_str, answer_queue)
        node_end_time = time.perf_counter()
        print(f"--- 🔴 Node: Generate RAG Answer (종료) (LLM 총 답변 생성 시간: {node_end_time - node_start_time:.4f}초) ---")
        return {}

    async def _generate_direct_llm_answer_node_func(self, state: GraphState, answer_queue: asyncio.Queue) -> Dict[str, any]:
        node_start_time = time.perf_counter()
        print("\n--- 🟢 Node: Generate Direct LLM Answer (시작) ---")
        query = state["query"]
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"]])
        prompt_template = PromptTemplate.from_template(
            """당신은 친절한 대화형 AI입니다. 이전 대화 내용과 현재 사용자 질문을 바탕으로 자연스럽게 답변해주세요.\n[이전 대화 내용]\n{chat_history}\n[현재 사용자 질문]\n{query}\n답변:""")
        final_prompt_str = prompt_template.format(chat_history=history_str, query=query)
        await self._common_answer_generation_logic(final_prompt_str, answer_queue)
        node_end_time = time.perf_counter()
        print(
            f"--- 🔴 Node: Generate Direct LLM Answer (종료) (LLM 총 답변 생성 시간: {node_end_time - node_start_time:.4f}초) ---")
        return {}

    def _decide_after_retrieval(self, state: GraphState) -> str:
        print(f"\n--- 🤔 Node: Decide After Retrieval ---")
        if state.get("documents"):
            print("  [결정] -> Generate RAG Answer")
            return "generate_rag_answer_node_id"
        else:
            print("  [결정] -> Generate Direct LLM Answer (문서 없음)")
            return "generate_direct_llm_answer_node_id"

    async def generate(self, query: str, k: int = 10) -> AsyncGenerator[str, None]:
        total_generate_start_time = time.perf_counter()
        print(f"\n--- ✨ LangGraph Generate 시작: Query='{query[:50]}...' ---")
        history_for_graph = list(RAGRepositoryImpl._global_chat_history)
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

        workflow.add_conditional_edges("retrieve_documents_node_id", self._decide_after_retrieval,
                                       {"generate_rag_answer_node_id": "generate_rag_answer_node_id",
                                        "generate_direct_llm_answer_node_id": "generate_direct_llm_answer_node_id"})
        workflow.add_edge("generate_rag_answer_node_id", END)
        workflow.add_edge("generate_direct_llm_answer_node_id", END)

        graph_run_task = None
        try:
            app = workflow.compile()
            initial_state = GraphState(query=query, chat_history=history_for_graph, classification_result=None,
                                       queries_for_retrieval=[], filters=None, documents=[], k=k,
                                       generation_instructions=None)
            graph_run_task = asyncio.create_task(app.ainvoke(initial_state, {"recursion_limit": 15}))
            buffer = ""
            full_ai_response_parts = []
            processed_error_event = False
            while True:
                try:
                    token = await asyncio.wait_for(output_queue.get(), timeout=1.0)
                    if token is None: break
                    if isinstance(token, str) and token.startswith("event: error"):
                        if not processed_error_event: processed_error_event = True
                        yield token
                        continue
                    buffer += str(token)
                    if any(buffer.endswith(ending) for ending in ['.', '!', '?', '다.', '요.']) or len(buffer) > 50:
                        formatted_chunk = self._format_korean_text_chunk(buffer)
                        if formatted_chunk: full_ai_response_parts.append(formatted_chunk)
                        yield formatted_chunk
                        buffer = ""
                except asyncio.TimeoutError:
                    if graph_run_task.done():
                        if output_queue.empty():
                            break
                        else:
                            continue
                    continue
            if buffer and not processed_error_event:
                formatted_chunk = self._format_korean_text_chunk(buffer)
                if formatted_chunk: full_ai_response_parts.append(formatted_chunk)
                yield formatted_chunk
        except Exception as e:
            yield f"event: error\ndata: 그래프 구성 오류: {str(e)}\n\n"
        finally:
            if graph_run_task:
                try:
                    final_graph_state = await asyncio.wait_for(graph_run_task, timeout=60.0)
                    if final_graph_state:
                        final_ai_response = "".join(full_ai_response_parts)
                        if final_ai_response and not processed_error_event:
                            RAGRepositoryImpl._global_chat_history.append({"role": "user", "content": query})
                            RAGRepositoryImpl._global_chat_history.append(
                                {"role": "assistant", "content": final_ai_response})
                            max_messages = RAGRepositoryImpl.MAX_GLOBAL_HISTORY_TURNS * 2
                            if len(RAGRepositoryImpl._global_chat_history) > max_messages:
                                RAGRepositoryImpl._global_chat_history = RAGRepositoryImpl._global_chat_history[
                                                                         -max_messages:]
                except Exception as e:
                    if not processed_error_event: yield f"event: error\ndata: 그래프 실행 중 오류: {str(e)}\n\n"
            total_generate_end_time = time.perf_counter()
            print(
                f"--- ✨ LangGraph Generate 종료 (총 소요 시간: {total_generate_end_time - total_generate_start_time:.4f}초) ---")