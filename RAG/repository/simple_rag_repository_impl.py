import os
import asyncio
import re
import time  # 로깅 및 타임스탬프 등에 사용 가능
from pathlib import Path
from dotenv import load_dotenv
from typing import AsyncGenerator, Optional, List, Dict
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker


# RAGRepository 인터페이스 (실제 프로젝트 경로에 맞게 수정)
class RAGRepository:
    async def generate(self, query: str, **kwargs) -> AsyncGenerator[str, None]:
        raise NotImplementedError


from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessageChunk
from functools import partial


# --- GraphState 정의 (client_id 제거) ---
class GraphState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    classification_result: Optional[str]
    query_for_retrieval: Optional[str]
    documents: List[Document]
    initial_k: int
    rerank_n: int


class RAGRepositoryImpl(RAGRepository):
    __instance = None
    _initialized = False

    _embed_model_instance: Optional[SentenceTransformer] = None
    _reranker_instance: Optional[FlagReranker] = None
    _model: Optional[ChatOpenAI] = None
    _utility_llm: Optional[ChatOpenAI] = None

    # --- 단일 전역 대화 기록 저장소 ---
    _global_chat_history: List[Dict[str, str]] = []
    MAX_GLOBAL_HISTORY_TURNS = 10  # 저장할 최대 대화 턴 수 (1턴 = 사용자 질문 + AI 답변)

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            print("--- RAGRepositoryImpl: __new__ 호출 ---")
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self):
        if not RAGRepositoryImpl._initialized:
            print("--- RAGRepositoryImpl: __init__ 최초 초기화 시작 ---")
            load_dotenv()
            self._initialize_models()
            self._initialize_db()
            RAGRepositoryImpl._initialized = True
            print("--- RAGRepositoryImpl: __init__ 최초 초기화 완료 ---")
        # else:
        # print("--- RAGRepositoryImpl: __init__ (이미 초기화됨) ---")

    def _initialize_models(self):
        print("--- RAGRepositoryImpl: 모델 초기화 중... ---")
        RAGRepositoryImpl._model = ChatOpenAI(model=os.getenv("MAIN_LLM_MODEL", "gpt-4o-mini"), temperature=0.0,
                                              openai_api_key=os.getenv("OPENAI_API_KEY"), streaming=True)
        RAGRepositoryImpl._utility_llm = ChatOpenAI(model=os.getenv("UTILITY_LLM_MODEL", "gpt-4o-mini"),
                                                    temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"),
                                                    streaming=False)

        if RAGRepositoryImpl._embed_model_instance is None:
            embedding_model_name = os.getenv("EMBEDDING_MODEL",'dragonkue/snowflake-arctic-embed-l-v2.0-ko')
            if not embedding_model_name: raise ValueError("EMBEDDING_MODEL 환경 변수 누락")
            RAGRepositoryImpl._embed_model_instance = SentenceTransformer(embedding_model_name)
            print(f"--- Embedding model '{embedding_model_name}' loaded. ---")

        if RAGRepositoryImpl._reranker_instance is None:
            reranker_model_name = os.getenv("RERANKER_MODEL", 'BAAI/bge-reranker-v2-m3')
            RAGRepositoryImpl._reranker_instance = FlagReranker(reranker_model_name, use_fp16=True, normalize=True)
            print(f"--- Reranker model '{reranker_model_name}' loaded. ---")
        print("✅ LLM, Embedding, Reranker 모델 초기화 완료.")

    def _initialize_db(self):
        print("--- RAGRepositoryImpl: DB 초기화 중 (클라이언트-서버 모드) ---")

        class MyEmbeddings:
            def __init__(self, model_instance: SentenceTransformer): self.model = model_instance

            def embed_documents(self, texts: List[str]) -> List[List[float]]: return self.model.encode(texts,
                                                                                                       convert_to_numpy=True).tolist()

            def embed_query(self, text: str) -> List[float]: return self.model.encode([text], convert_to_numpy=True)[
                0].tolist()

        if RAGRepositoryImpl._embed_model_instance is None:
            raise RuntimeError("Embedding model not initialized.")

        # Docker Compose 네트워크 내에서 서비스 이름(chromadb)으로 서버에 접속합니다.
        # .env 파일 등에서 호스트와 포트를 관리하는 것도 좋은 방법입니다.
        db_host = os.getenv("CHROMA_DB_HOST", "chromadb")
        db_port = int(os.getenv("CHROMA_DB_PORT", 8000))

        print(f"--- ChromaDB 서버에 연결 시도: host='{db_host}', port={db_port} ---")
        client = chromadb.HttpClient(host=db_host, port=db_port)

        collection_name = os.getenv("CHROMA_COLLECTION", "construction_new")
        self._collection = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=MyEmbeddings(RAGRepositoryImpl._embed_model_instance),
        )

        print(f"✅ ChromaDB 서버에 연결 완료. Collection='{collection_name}'")

    def _format_korean_text_chunk(self, text: str) -> str:  # 이전과 동일
        if not text: return ""
        text = text.strip()
        text = re.sub(r'(입니다|됩니다|습니다)\.(?!\s*(?:<br>|$))', r'\1.<br><br>', text)
        text = re.sub(r'([\.!?])(?!\s*(?:<br>|$))(?=[가-힣A-Za-z0-9\(])', r'\1<br><br>', text)
        text = re.sub(r'\n+', '<br><br>', text)
        text = re.sub(r'(?<=[가-힣\.!?])\s*(\d+\.)(?!\s*\d)', r'<br><br>\1', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'(<br\s*/?>\s*){3,}', '<br><br>', text)
        text = re.sub(r'^(<br\s*/?>\s*)+', '', text)
        text = re.sub(r'(<br\s*/?>\s*)+$', '', text)
        return text.strip()

    async def _classify_query_node_func(self, state: GraphState) -> Dict[str, any]:
        print("\n--- 🟢 Node: Classify Query (시작) ---")
        query = state["query"]
        chat_history = state.get("chat_history", [])
        print(f"  [입력 상태] Query: '{query}'")
        print(f"  [입력 상태] Chat History (최근 2 메시지): {chat_history[-2:] if len(chat_history) >= 2 else chat_history}")

        history_for_classification_str = ""
        if chat_history:
            relevant_history = chat_history[-(3 * 2):]
            temp_history_str = [f"{'사용자' if msg.get('role') == 'user' else 'AI'}: {msg.get('content')}" for msg in
                                relevant_history]
            history_for_classification_str = "\n".join(temp_history_str)

        classification_prompt_str = f"""당신은 사용자 질문의 의도를 분석하여 다음 세 가지 유형 중 하나로 분류하는 AI입니다. 각 유형은 따옴표 없이 단어만으로 출력해야 합니다 (예: rag_rewrite).
- "rag_rewrite": 사용자의 질문이 이전 대화와 밀접하게 연관되어 있고, 질문의 의미를 명확히 하기 위해 재작성이 필요하며, 이 재작성된 질문으로 문서를 검색해야 하는 경우. (예: "그것에 대해 더 자세히 알려줘", "위에서 말한 첫 번째 항목의 사례는?")
- "rag_direct": 사용자의 질문이 새로운 주제이거나 이전 대화와 직접적인 연속성은 없지만, 문서 검색이 필요한 경우. 질문 재작성은 불필요. (예: "새로운 건축 법규에 대해 알려줘")
- "direct_llm": 사용자의 질문이 일반적인 대화, 인사, 또는 LLM이 자체 지식으로 답변할 수 있는 간단한 내용이라 문서 검색이 불필요한 경우. (예: "안녕", "오늘 날씨 어때?", "대한민국의 수도는 어디야?")

[이전 대화 내용]
{history_for_classification_str if history_for_classification_str else "이전 대화 없음"}

[사용자 현재 질문]
{query}

분류 결과 ("rag_rewrite", "rag_direct", "direct_llm" 중 하나만 정확히 출력):"""

        response = await RAGRepositoryImpl._utility_llm.ainvoke([HumanMessage(content=classification_prompt_str)])
        classification = response.content.strip().replace('"', '')

        print(f"🔍 질문 분류 결과: '{classification}' (원본 질문: '{query}')")
        if classification not in ["rag_rewrite", "rag_direct", "direct_llm"]:
            original_classification_attempt = classification
            classification = "rag_direct"
            print(f"⚠️ 분류 결과 '{original_classification_attempt}'가 유효하지 않아 '{classification}'로 기본 설정합니다.")

        updates_to_state = {"classification_result": classification, "query_for_retrieval": query}
        print(f"  [출력 업데이트] {updates_to_state}")
        print("--- 🔴 Node: Classify Query (종료) ---")
        return updates_to_state

    async def _rewrite_query_node_func(self, state: GraphState) -> Dict[str, any]:
        print("\n--- 🟢 Node: Rewrite Query (시작) ---")
        query = state["query"]
        chat_history = state.get("chat_history", [])
        print(f"  [입력 상태] Query: '{query}', Classification: {state.get('classification_result')}")

        history_for_rewriting_str = ""
        if chat_history:
            relevant_history = chat_history[-(3 * 2):]
            temp_history_str = [f"{'사용자' if msg.get('role') == 'user' else 'AI'}: {msg.get('content')}" for msg in
                                relevant_history]
            history_for_rewriting_str = "\n".join(temp_history_str)

        rewrite_prompt_template = PromptTemplate.from_template(
            """주어진 이전 대화 내용과 후속 질문을 바탕으로, 검색 엔진이 이해하기 쉽도록 후속 질문을 독립적이고 구체적인 질문으로 재작성해주세요.
만약 후속 질문이 이전 대화에서 생성된 특정 항목(예: 체크리스트 항목들)에 관한 것이라면, 재작성된 질문에 해당 항목의 실제 내용을 명시적으로 포함시켜 주세요.

[이전 대화 내용]
{chat_history_str}

[후속 질문]
{follow_up_question}

[재작성된 검색용 질문] (이 질문은 문서 검색에 사용됩니다. 다른 부가 설명 없이 재작성된 질문만 출력해주세요.):"""
        )
        chain = rewrite_prompt_template | RAGRepositoryImpl._utility_llm
        response = await chain.ainvoke({"chat_history_str": history_for_rewriting_str, "follow_up_question": query})
        rewritten_query = response.content.strip()

        final_query_for_retrieval = rewritten_query if rewritten_query else query
        print(f"🔄 재작성된 질문: {final_query_for_retrieval}")
        updates_to_state = {"query_for_retrieval": final_query_for_retrieval}
        print(f"  [출력 업데이트] {updates_to_state}")
        print("--- 🔴 Node: Rewrite Query (종료) ---")
        return updates_to_state

    async def _retrieve_documents_node_func(self, state: GraphState) -> Dict[str, any]:
        print("\n--- 🟢 Node: Retrieve Documents (시작) ---")
        query_for_retrieval = state["query_for_retrieval"]
        initial_k = state["initial_k"]
        rerank_n = state["rerank_n"]
        print(f"  [입력 상태] Query for Retrieval: '{query_for_retrieval}', initial_k: {initial_k}, rerank_n: {rerank_n}")

        retriever = self._collection.as_retriever(
            search_type="mmr",
            search_kwargs={"k": initial_k, "fetch_k": max(initial_k, 100)}
        )
        documents = await retriever.aget_relevant_documents(query_for_retrieval)

        if documents:
            print(f"  [중간 결과] {len(documents)}개 문서 검색됨 (k={initial_k}). 재랭킹 시도 (top {rerank_n})...")
            pairs = [[query_for_retrieval, doc.page_content] for doc in documents]
            try:
                if RAGRepositoryImpl._reranker_instance is None:  # 방어 코드
                    raise RuntimeError("Reranker model not initialized.")
                scores = await asyncio.to_thread(RAGRepositoryImpl._reranker_instance.compute_score, pairs)
                ranked_docs_with_scores = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
                documents = [doc for _, doc in ranked_docs_with_scores[:rerank_n]]
                print(f"  [중간 결과] {len(documents)}개 문서 재랭킹 완료.")
            except Exception as e:
                print(f"  [경고] 재랭킹 중 오류 발생: {e}. 원본 검색 결과 상위 {rerank_n}개 사용.")
                documents = documents[:rerank_n]
        else:
            print("  [중간 결과] 검색된 문서 없음.")

        updates_to_state = {"documents": documents if documents else []}
        print(f"  [출력 업데이트] Documents (개수): {len(updates_to_state['documents'])}")
        print("--- 🔴 Node: Retrieve Documents (종료) ---")
        return updates_to_state

    async def _common_answer_generation_logic(self, final_prompt_str: str, answer_queue: asyncio.Queue):
        try:
            log_prompt = re.sub(r'\s+', ' ', final_prompt_str)[:200]
            print(f"    [LLM 스트리밍 시작] Prompt (일부): {log_prompt}...")
            async for token_chunk in RAGRepositoryImpl._model.astream(final_prompt_str):
                content_to_put = ""
                if isinstance(token_chunk, AIMessageChunk):
                    content_to_put = token_chunk.content
                elif hasattr(token_chunk, 'content'):
                    content_to_put = token_chunk.content
                elif isinstance(token_chunk, str):
                    content_to_put = token_chunk

                if content_to_put:  # 빈 문자열은 보내지 않음
                    await answer_queue.put(content_to_put)

            print("    [LLM 스트리밍 정상 종료]")
        except Exception as e:
            print(f"    [LLM 스트리밍 오류 발생]: {e}")
            error_message = str(e).replace("\n", " ")
            await answer_queue.put(f"event: error\ndata: LLM 답변 생성 중 오류: {error_message}\n\n")
        finally:
            await answer_queue.put(None)

    async def _generate_rag_answer_node_func(self, state: GraphState, answer_queue: asyncio.Queue) -> Dict[
        str, any]:
        print("\n--- 🟢 Node: Generate RAG Answer (시작) ---")
        query = state["query"]
        chat_history = state.get("chat_history", [])  # 그래프 실행 시점의 전체 히스토리 (업데이트 전)
        documents = state.get("documents", [])
        classification_result = state.get("classification_result")  # 분류 결과 가져오기

        print(
            f"  [입력 상태] Query: '{query}', Documents (개수): {len(documents)}, Classification: '{classification_result}'")

        context_str = "\n\n---\n\n".join(
            [f"[{doc.metadata.get('source', 'unknown') if doc.metadata else 'unknown'}] {doc.page_content}" for doc
             in documents]
        ) if documents else "참고할 문서가 없습니다."

        history_str = "이전 대화 없음"  # 기본값
        # "rag_rewrite"로 분류된 경우에만 이전 대화 내용을 프롬프트에 적극적으로 포함
        if classification_result == "rag_rewrite" and chat_history:
            relevant_history = chat_history[-(3 * 2):]  # 최근 3턴
            temp_history_list = [f"{'사용자' if msg.get('role') == 'user' else 'AI'}: {msg.get('content')}" for msg in
                                 relevant_history]
            if temp_history_list:  # 실제 내용이 있을 때만 join
                history_str = "\n".join(temp_history_list)

        print(f"  [Generate RAG Answer] Prompt에 사용될 history_str: '{history_str[:100]}...'")

        prompt_template = PromptTemplate.from_template(
            """당신은 건축 관련 전문가 입니다. 당신의 주요 임무는 사용자 질문에 대해 아래 제공된 '문서' 내용을 근거로 답변하는 것입니다.
다음은 이전 대화 내용과 현재 질문에 관련된 문서입니다. 이를 모두 참고하여 답변하세요.

[이전 대화 내용]
{chat_history_str}

[참고 문서]
{context_str}

[현재 사용자 질문]
{query}

답변 (반드시 '문서' 내용과 '이전 대화'의 흐름을 고려하여 답변하고, 없는 내용은 답변하지 마세요. 답변은 한국어로 해주세요.):"""
        )
        final_prompt_str = prompt_template.format(
            chat_history_str=history_str,  # 조건부로 설정된 history_str 사용
            context_str=context_str,
            query=query
        )

        await self._common_answer_generation_logic(final_prompt_str, answer_queue)
        print("--- 🔴 Node: Generate RAG Answer (스트리밍 로직 호출 완료) ---")
        return {}

    async def _generate_direct_llm_answer_node_func(self, state: GraphState, answer_queue: asyncio.Queue) -> Dict[
        str, any]:
        print("\n--- 🟢 Node: Generate Direct LLM Answer (시작) ---")
        query = state["query"]
        chat_history = state.get("chat_history", [])
        classification_result = state.get("classification_result")  # 분류 결과 가져오기
        print(f"  [입력 상태] Query: '{query}', Classification: '{classification_result}'")

        history_str = "이전 대화 없음"  # 기본값
        # "direct_llm"으로 분류된 질문에는 이전 대화 내용을 포함하지 않음 (사용자 요청 반영)
        # 만약 약간의 대화 흐름 유지를 원한다면, 여기서도 classification_result에 따라 조건부로 매우 짧은 최근 대화만 포함 가능
        # 현재는 "rag_rewrite"가 아닌 모든 경우에 "이전 대화 없음"으로 처리
        if classification_result == "rag_rewrite" and chat_history:  # 이 노드는 direct_llm 경로이므로, 이 조건은 거의 만족 안됨
            # 하지만 일관성을 위해 남겨두거나, 아예 chat_history를 안 보는 것으로 간주 가능
            relevant_history = chat_history[-(3 * 2):]
            temp_history_list = [f"{'사용자' if msg.get('role') == 'user' else 'AI'}: {msg.get('content')}" for msg in
                                 relevant_history]
            if temp_history_list:
                history_str = "\n".join(temp_history_list)

        print(f"  [Generate Direct LLM Answer] Prompt에 사용될 history_str: '{history_str[:100]}...'")

        prompt_template = PromptTemplate.from_template(
            """당신은 친절한 대화형 AI입니다. 이전 대화 내용과 현재 사용자 질문을 바탕으로 자연스럽게 답변해주세요.

[이전 대화 내용]
{chat_history_str}

[현재 사용자 질문]
{query}

답변:"""
        )
        final_prompt_str = prompt_template.format(
            chat_history_str=history_str,  # 조건부로 설정된 history_str 사용
            query=query
        )

        await self._common_answer_generation_logic(final_prompt_str, answer_queue)
        print("--- 🔴 Node: Generate Direct LLM Answer (스트리밍 로직 호출 완료) ---")
        return {}

    def _decide_next_node_after_classification(self, state: GraphState) -> str:
        classification = state["classification_result"]
        print(f"\n--- 🤔 Node: Decide Next After Classification (분류: '{classification}') ---")
        if classification == "rag_rewrite":
            print("  [결정] -> Rewrite Query")
            return "rewrite_query_node_id"
        elif classification == "rag_direct":
            print("  [결정] -> Retrieve Documents")
            return "retrieve_documents_node_id"
        elif classification == "direct_llm":
            print("  [결정] -> Generate Direct LLM Answer")
            return "generate_direct_llm_answer_node_id"
        print(f"  [경고] 알 수 없는 분류 결과 '{classification}', 안전하게 Retrieve Documents로 진행.")
        return "retrieve_documents_node_id"  # 좀 더 안전한 fallback

    def _decide_after_retrieval(self, state: GraphState) -> str:
        documents = state.get("documents", [])
        classification_result = state.get("classification_result")  # 초기 분류 참고 가능
        print(f"\n--- 🤔 Node: Decide After Retrieval (문서 개수: {len(documents)}, 초기 분류: '{classification_result}') ---")

        # 초기 의도가 RAG였거나 (rag_direct, rag_rewrite), 문서가 실제로 있는 경우 RAG 답변 생성
        if classification_result in ["rag_direct", "rag_rewrite"] or documents:
            print("  [결정] -> Generate RAG Answer")
            return "generate_rag_answer_node_id"
        else:  # 초기 의도가 direct_llm이었고 문서도 없는 경우
            print("  [결정] -> Generate Direct LLM Answer")
            return "generate_direct_llm_answer_node_id"

    async def generate(
            self,
            query: str,
            initial_k: int = 75,
            rerank_n: int = 50
            # chat_history 파라미터도 제거 (내부 _global_chat_history 사용)
    ) -> AsyncGenerator[str, None]:

        print(f"\n--- ✨ LangGraph Generate 시작: Query='{query[:50]}...' (Global History Mode) ---")

        # --- 전역 대화 기록 사용 ---
        # deepcopy를 사용하여 그래프 실행 중 _global_chat_history가 직접 변경되는 것을 방지할 수 있으나,
        # 현재 로직은 그래프 실행 전 history_for_graph를 만들고, 그래프 종료 후 _global_chat_history를 업데이트하므로 괜찮음.
        history_for_graph = list(RAGRepositoryImpl._global_chat_history)  # 현재 시점의 기록 복사
        print(f"  [Generate] Global Chat History 로드. 현재 {len(history_for_graph)}개 메시지.")

        output_queue = asyncio.Queue()
        workflow = StateGraph(GraphState)

        # 노드 추가
        workflow.add_node("classify_query_node_id", self._classify_query_node_func)
        workflow.add_node("rewrite_query_node_id", self._rewrite_query_node_func)
        workflow.add_node("retrieve_documents_node_id", self._retrieve_documents_node_func)

        rag_answer_with_queue = partial(self._generate_rag_answer_node_func, answer_queue=output_queue)
        direct_answer_with_queue = partial(self._generate_direct_llm_answer_node_func, answer_queue=output_queue)
        workflow.add_node("generate_rag_answer_node_id", rag_answer_with_queue)
        workflow.add_node("generate_direct_llm_answer_node_id", direct_answer_with_queue)

        # 엣지 및 진입점 설정
        workflow.set_entry_point("classify_query_node_id")
        workflow.add_conditional_edges(
            "classify_query_node_id",
            self._decide_next_node_after_classification,
            {
                "rewrite_query_node_id": "rewrite_query_node_id",
                "retrieve_documents_node_id": "retrieve_documents_node_id",
                "generate_direct_llm_answer_node_id": "generate_direct_llm_answer_node_id",
                END: END
            }
        )
        workflow.add_edge("rewrite_query_node_id", "retrieve_documents_node_id")
        workflow.add_conditional_edges(
            "retrieve_documents_node_id",
            self._decide_after_retrieval,
            {
                "generate_rag_answer_node_id": "generate_rag_answer_node_id",
                "generate_direct_llm_answer_node_id": "generate_direct_llm_answer_node_id"
            }
        )
        workflow.add_edge("generate_rag_answer_node_id", END)
        workflow.add_edge("generate_direct_llm_answer_node_id", END)

        try:
            app = workflow.compile()
            print("--- ✅ LangGraph 컴파일 완료 ---")
        except Exception as e:
            print(f"--- 💥 LangGraph 컴파일 중 오류: {e} ---")
            error_message = str(e).replace('\n', ' ')
            yield f"event: error\ndata: 그래프 구성 오류: {error_message}\n\n"
            return

        initial_state = GraphState(
            query=query,
            chat_history=history_for_graph,  # 로드한 전역 기록 사용
            # client_id 필드 GraphState에서 제거 (또는 None)
            classification_result=None,
            query_for_retrieval=query,
            documents=[],
            initial_k=initial_k,
            rerank_n=rerank_n
        )

        print(f"--- 🚀 LangGraph 실행 시작 (Initial Query: '{initial_state['query']}') ---")
        graph_run_task = asyncio.create_task(app.ainvoke(initial_state, {"recursion_limit": 15}))

        buffer = ""
        sentence_endings = ['.', '!', '?', '다.', '요.']
        processed_error_event = False
        full_ai_response_parts = []

        try:
            while True:
                try:
                    token = await asyncio.wait_for(output_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if graph_run_task.done():
                        await asyncio.sleep(0.01)
                        if output_queue.empty():
                            break
                        else:
                            try:
                                token = output_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                if graph_run_task.exception():
                                    print(
                                        f"--- ⚠️ Graph task 완료 (예외 발생, timeout loop): {graph_run_task.exception()} ---")
                                break
                    continue

                if token is None:
                    if buffer:
                        formatted_chunk = self._format_korean_text_chunk(buffer)
                        if formatted_chunk:
                            full_ai_response_parts.append(formatted_chunk)
                            yield formatted_chunk
                        buffer = ""
                    break

                if isinstance(token, str) and token.startswith("event: error"):
                    if not processed_error_event:
                        processed_error_event = True
                        if buffer:
                            formatted_chunk = self._format_korean_text_chunk(buffer)
                            if formatted_chunk: yield formatted_chunk
                            buffer = ""
                        print(f"---  Fehler! Yielding error event: {token[:100]}... ---")
                        yield token
                    continue

                if isinstance(token, str) and not token.startswith("event: error"):
                    buffer += token

                should_flush = any(buffer.endswith(ending) for ending in sentence_endings) or len(buffer) > 50
                if should_flush:
                    formatted_chunk = self._format_korean_text_chunk(buffer)
                    if formatted_chunk:
                        full_ai_response_parts.append(formatted_chunk)
                        yield formatted_chunk
                    buffer = ""

            if buffer and not processed_error_event:
                formatted_chunk = self._format_korean_text_chunk(buffer)
                if formatted_chunk:
                    full_ai_response_parts.append(formatted_chunk)
                    yield formatted_chunk

        finally:
            print(f"--- 스트리밍 루프 종료. Graph task 대기 중... ---")
            final_graph_state = None  # 명시적 초기화
            try:
                final_graph_state = await asyncio.wait_for(graph_run_task, timeout=60.0)

                final_ai_response = "".join(full_ai_response_parts)

                if graph_run_task.exception():
                    exc = graph_run_task.exception()
                    print(f"--- 💥 LangGraph 최종 실행 완료 (예외 발생): {exc} ---")
                    if not processed_error_event:
                        error_message = str(exc).replace('\n', ' ')
                        yield f"event: error\ndata: 그래프 실행 중 예외 발생: {error_message}\n\n"
                elif final_graph_state:  # 예외 없이 정상 종료 시
                    print(
                        f"--- ✅ LangGraph 최종 실행 완료. Final classification: {final_graph_state.get('classification_result')} ---")
                    if final_ai_response and not processed_error_event:
                        # --- 전역 대화 기록 업데이트 ---
                        RAGRepositoryImpl._global_chat_history.append({"role": "user", "content": query})
                        RAGRepositoryImpl._global_chat_history.append(
                            {"role": "assistant", "content": final_ai_response})

                        max_messages = RAGRepositoryImpl.MAX_GLOBAL_HISTORY_TURNS * 2
                        if len(RAGRepositoryImpl._global_chat_history) > max_messages:
                            RAGRepositoryImpl._global_chat_history = RAGRepositoryImpl._global_chat_history[
                                                                     -max_messages:]
                        print(
                            f"  [Generate] Global Chat History 업데이트 완료. 현재 {len(RAGRepositoryImpl._global_chat_history)}개 메시지 저장됨.")
                    elif processed_error_event:
                        print(f"  [Generate] 오류 이벤트 발생으로 Global History 업데이트 안 함.")
                    else:
                        print(f"  [Generate] AI 응답이 없거나(empty) 오류로 Global History 업데이트 안 함.")
                else:  # 예외는 없지만 final_graph_state가 None인 경우 (거의 발생 안 함)
                    print(f"--- ✅ LangGraph 최종 실행 완료 (final_graph_state가 None). ---")


            except asyncio.TimeoutError:
                print(f"--- 💥 LangGraph 실행 시간 초과 (60초). 작업 강제 취소 시도. ---")
                graph_run_task.cancel()
                try:
                    await graph_run_task
                except asyncio.CancelledError:
                    print("--- 💥 LangGraph 작업 강제 취소됨. ---")
                except Exception as e_cancel:
                    print(f"--- 💥 LangGraph 작업 취소 중 추가 오류: {e_cancel} ---")
                if not processed_error_event:
                    yield f"event: error\ndata: 그래프 처리 시간 초과\n\n"
            except Exception as e:
                print(f"--- 💥 최종 그래프 실행 대기 중 일반 오류: {e} ---")
                if not processed_error_event:
                    error_message = str(e).replace('\n', ' ')
                    yield f"event: error\ndata: 그래프 실행 중 심각한 오류: {error_message}\n\n"

                    print(f"--- ✨ LangGraph Generate 종료 (Global History Mode) ---")