import os
import asyncio
import re, time
from pathlib import Path
from dotenv import load_dotenv
from typing import AsyncGenerator, Optional,List, Dict
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from RAG.repository.simple_rag_repository import RAGRepository
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.documents import Document # Document 추가

class QueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put_nowait(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.queue.put_nowait(None)


class RAGRepositoryImpl(RAGRepository):
    __instance = None
    # metadata_field_info = [
    #     AttributeInfo(
    #         name="연번",
    #         description="문서 또는 항목의 순서나 일련번호를 나타냅니다. 주로 숫자 형식입니다.",
    #         type="integer",  # 또는 문자열일 수 있으므로, 실제 데이터 타입에 맞게 조정 필요
    #     ),
    #     AttributeInfo(
    #         name="설계단계",
    #         description="건축 프로젝트의 설계 단계를 구분하는 정보입니다. 예를 들어 '기획설계', '계획설계', '중간설계', '실시설계' 등이 될 수 있습니다.",
    #         type="string",
    #     ),
    #     AttributeInfo(
    #         name="id번호",
    #         description="각 문서나 데이터 항목에 부여된 고유 식별자입니다. 문자 또는 숫자, 혹은 조합으로 이루어질 수 있습니다.",
    #         type="string",  # 숫자만 있더라도 문자열로 처리하는 것이 일반적입니다.
    #     ),
    #     AttributeInfo(
    #         name="공종",
    #         description="공사의 종류나 분야를 나타냅니다. 예를 들어 '건축공사', '토목공사', '전기공사', '기계설비공사', '소방공사' 등이 있습니다.",
    #         type="string",
    #     ),
    #     AttributeInfo(
    #         name="검증위원",
    #         description="문서 또는 도면을 검토하고 검증한 위원(들)의 이름입니다. 특정 전문가의 검토 자료를 찾을 때 사용될 수 있습니다. (예: '홍길동', '이순신')",
    #         type="string",
    #     ),
    #     AttributeInfo(
    #         name="도면명",
    #         description="도면의 공식적인 이름 또는 제목입니다. 예를 들어 '1층 평면도', '구조 단면 상세도' 등이 될 수 있습니다.",
    #         type="string",
    #     ),
    #     AttributeInfo(
    #         name="도면 번호",  # '도면 번호'와 같이 공백이 있는 필드명도 가능
    #         description="도면에 부여된 고유한 식별 번호 또는 코드입니다. (예: 'A-001', 'S-102-Rev.A')",
    #         type="string",
    #     ),
    #     AttributeInfo(
    #         name="세부 분류",
    #         description="문서나 항목을 보다 상세하게 분류하는 카테고리 정보입니다. '공종'보다 더 구체적인 분류에 사용될 수 있습니다. (예: '외벽 마감재', '조명 설비 상세')",
    #         type="string",
    #     ),
    #     AttributeInfo(
    #         name="참조용이야",  # 필드명이 "참조용이야" 그대로 사용
    #         description="문서가 참고 자료인지, 또는 특별한 참고 사항이 있는지 여부를 나타내는 부가 정보 필드입니다. '예/아니오' 또는 특정 키워드로 정보를 담을 수 있습니다.",
    #         type="string",  # 또는 'boolean' 타입이 적절할 수 있습니다. (예: "True", "False")
    #     ),
    #     # 만약 다른 필드가 있다면 여기에 추가 정의할 수 있습니다.
    #     # 예: AttributeInfo(name="작성일", description="문서가 작성된 날짜 (YYYY-MM-DD 형식)", type="string"),
    # ]
    # document_content_description = "건축 설계, 시공, 감리, 안전, 법규 관련 전문 문서 및 도면 데이터"  # 좀 더 구체적으로 수정
    def __new__(cls):
        if cls.__instance is None:
            load_dotenv()
            cls.__instance = super().__new__(cls)
            cls.__instance.connectDB()
            cls.__instance.get_model()
        return cls.__instance

    @classmethod
    def getInstance(cls):
        return cls.__new__(cls)

    def get_model(self) -> None:
        self._model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            streaming=True,
            callback_manager=None
        )

        # self._query_constructor_model = ChatOpenAI(
        #     model="gpt-4o-mini",
        #     temperature=0.0,
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),
        #     streaming=False,  # SelfQuery LLM은 보통 스트리밍 불필요
        # )
        # print("✅ LLM Models (generation & query_constructor) loaded.")

    def connectDB(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        persist_dir = project_root / "chroma_db"

        embed_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))

        class MyEmbeddings:
            def __init__(self, m): self.model = m

            def embed_documents(self, texts):
                return self.model.encode(texts, convert_to_numpy=True).tolist()

            def embed_query(self, text):
                return self.model.encode([text], convert_to_numpy=True)[0].tolist()

        self._collection = Chroma(
            persist_directory=str(persist_dir),
            collection_name=os.getenv("CHROMA_COLLECTION", "construction"),
            embedding_function=MyEmbeddings(embed_model),
        )

        print(f"✅ Chroma DB connected at {persist_dir}, collection={self._collection._collection_name}")

    def _format_korean_text_chunk(self, text: str) -> str:
        """한국어 텍스트를 HTML로 포맷팅 (문장마다 <br><br> 삽입)"""
        if not text:
            return text

        # 0) '입니다/됩니다/습니다.' 뒤에 줄바꿈
        text = re.sub(
            r'(입니다|됩니다|습니다)\.(?=[가-힣A-Za-z0-9])',
            r'\1.<br><br>',
            text
        )

        # 1) 마침표·물음표·느낌표 뒤에 한글/영문/숫자가 바로 오면 줄바꿈
        text = re.sub(
            r'([\.!?])(?=[가-힣A-Za-z0-9])',
            r'\1<br><br>',
            text
        )

        # 2) 숫자 목록 항목 (1. 2. 3.) 앞에도 줄바꿈
        text = re.sub(
            r'(?<=[가-힣\.])(\d+\.)',
            r'<br><br>\1',
            text
        )

        # 3) **굵은글씨** → <strong>
        text = re.sub(
            r'\*\*([^*]+)\*\*',
            r'<br><br><strong>\1</strong>',
            text
        )

        # 4) 연속된 <br> 3개 이상은 2개로 줄이기
        text = re.sub(r'(<br>){3,}', '<br><br>', text)

        # 5) 맨 앞의 불필요한 <br> 제거
        text = re.sub(r'^(<br>)+', '', text)

        return text

    async def generate(
            self,
            query: str,
            initial_k: int = 75,
            rerank_n: int = 50,
    ) -> AsyncGenerator[str, None]:
        # 요청마다 새 큐 생성
        queue = asyncio.Queue()
        handler = QueueCallbackHandler(queue)

        # 1차 검색
        # retriever = self._collection.as_retriever(search_kwargs={"k": initial_k, "fetch_k": 20 })
        retriever = self._collection.as_retriever(search_kwargs={"k": initial_k})
        # retriever = self._collection.as_retriever(search_type="mmr",
        #                                           search_kwargs={"k": initial_k,"fetch_k":100})
        start_time_retrieval = time.time()
        init_docs = retriever.get_relevant_documents(query)
        end_time_retrieval = time.time()

        # print(f"총 검색된 초기 문서 개수 (init_docs): {len(init_docs)}")
        # lee_gil_ho_docs_count = 0
        # for i, doc in enumerate(init_docs):
        #     # ★★★ 중요: '검증위원'은 실제 메타데이터에 저장된 필드명으로 변경해야 합니다.
        #     # 예를 들어, 'author', '작성자', '위원명' 등일 수 있습니다.
        #     author_name = doc.metadata.get("검증위원")  # 또는 doc.metadata.get("author") 등
        #
        #     is_lee_gil_ho_doc = (author_name == "이문찬")
        #     if is_lee_gil_ho_doc:
        #         lee_gil_ho_docs_count += 1
        #
        #     # 처음 5개 문서와 '이길호' 위원 문서를 최대 10개까지 출력 (너무 많으면 터미널이 길어짐)
        #     if i < 5 or (is_lee_gil_ho_doc and lee_gil_ho_docs_count <= 20):
        #         print(f"\n[문서 {i + 1}]")
        #         print(f"  - 내용 (앞 100자): {doc.page_content[:100]}...")
        #         print(f"  - 메타데이터: {doc.metadata}")
        #         if is_lee_gil_ho_doc:
        #             print(f"  ⭐ 이길호 위원 문서입니다.")
        #
        # print(f"\ninit_docs 내 '이길호' 위원 문서 총 개수: {lee_gil_ho_docs_count}")
        # print("--- [DEBUG] init_docs 내용 확인 끝 ---\n")
        # print(f"⏱️ ChromaDB 검색 시간: {end_time_retrieval - start_time_retrieval:.4f} 초")
        # if not init_docs:
        #     yield "event: error\ndata: 검색된 문서가 없습니다.\n\n"
        #     return


        # configured_generation_model = self._model.with_config(
        #     {"callbacks": [handler]}
        # )
        #
        # start_time_retrieval = time.time()
        # init_docs: List[Document] = []
        #
        # try:
        #     # 1. SelfQueryRetriever 설정 및 문서 검색 (from_llm 사용)
        #     # self._collection이 Chroma 인스턴스(VectorStore 타입)여야 합니다.
        #     self_query_retriever = SelfQueryRetriever.from_llm(
        #         llm=self._query_constructor_model,
        #         vectorstore=self._collection,  # Chroma 인스턴스를 직접 전달
        #         document_contents=self.document_content_description,
        #         metadata_field_info=self.metadata_field_info,
        #         verbose=True,
        #         # 참고: from_llm 사용 시 search_kwargs를 통한 k값 제어는
        #         #       내부적으로 vectorstore.as_retriever()를 호출할 때 기본값을 따르거나,
        #         #       다른 매개변수(예: chain_kwargs)를 통해 설정해야 할 수 있습니다.
        #         #       우선 'vectorstore' 오류 해결에 집중합니다.
        #     )
        #
        #     # 비동기 문서 검색 실행
        #     init_docs = await self_query_retriever.ainvoke(query)
        #
        # except Exception as e:
        #     print(f"SelfQueryRetriever 문서 검색 중 오류 발생: {e}")
        #     import traceback  # 상세한 스택 트레이스를 위해 임포트
        #     traceback.print_exc()  # 전체 스택 트레이스 출력
        #     error_message = str(e).replace("\n", " ")
        #     yield f"event: error\ndata: 문서 검색 중 오류가 발생했습니다 ({type(e).__name__}): {error_message}\n\n"
        #     return  # 오류 발생 시 함수 종료
        # 재랭킹
        reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, normalize=True)
        pairs = [[query, doc.page_content] for doc in init_docs]
        scores = reranker.compute_score(pairs)
        ranked = sorted(zip(scores, init_docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:rerank_n]]
        if not top_docs:
            yield "event: error\ndata: 재랭킹 후 문서가 없습니다.\n\n"
            return

        # print(f"\n[DEBUG] rerank_n 값: {rerank_n}")
        # print(f"[DEBUG] 재랭킹 후 top_docs 개수 (최대 {rerank_n}개): {len(top_docs)}")
        #
        # lee_gil_ho_in_top_docs_count = 0
        # print("--- [DEBUG] top_docs 내 '이문찬' 위원 문서 확인 시작 ---")
        # for i, doc in enumerate(top_docs):
        #     author_name = doc.metadata.get("검증위원")
        #     is_lee_gil_ho_doc = (author_name == "이문찬")
        #     if is_lee_gil_ho_doc:
        #         lee_gil_ho_in_top_docs_count += 1
        #         print(f"\n[TOP DOC - 이문찬 위원 문서 #{lee_gil_ho_in_top_docs_count}] (원래 init_docs에서의 순서는 다를 수 있음)")
        #         print(f"  - 내용 (앞 100자): {doc.page_content[:100]}...")
        #         print(f"  - 메타데이터: {doc.metadata}")
        # print(f"\n[DEBUG] top_docs 내 '이문찬' 위원 문서 총 개수: {lee_gil_ho_in_top_docs_count}")
        # print("--- [DEBUG] top_docs 내 '이문찬' 위원 문서 확인 끝 ---\n")

        # 프롬프트 조합
        context = "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
            for doc in top_docs
        )
        # context = "\n\n---\n\n".join(
        #     f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        #     for doc in init_docs
        # )
        prompt = PromptTemplate(
            template=(
                "당신은 건축 관련 전문가 입니다. 당신의 주요 임무는 사용자 질문에 대해 아래 제공된 '문서' 내용을 근거로 답변하는 것입니다.\n"
                "--- 문서 시작 ---\n"
                "{context}\n"
                "--- 문서 끝 ---\n\n"
                "질문: {question}\n\n"
                "답변 (반드시 '문서' 내용에 근거하고, 없는 내용은 답변하지 마세요):"
            ),
            input_variables=["context", "question"],
        ).format(context=context, question=query)

        llm_call_task = None  # finally 블록에서 사용하기 위해 변수 선언
        try:
            # LLM 호출을 별도의 태스크로 만들어 백그라운드에서 실행
            # apredict 또는 agenerate 사용 가능. 여기서는 apredict 사용 예시
            llm_call_task = asyncio.create_task(
                self._model.apredict(prompt, callbacks=[handler])  # ★★★ 핵심 변경점 ★★★
            )

            # llm_call_task = asyncio.create_task(
            #     configured_generation_model.apredict(prompt, callbacks=[handler])  # ★★★ 핵심 변경점 ★★★
            # )

            buffer = ""
            # 한국어 문장 종결 어미 추가 고려 ('다.', '요.' 등)
            sentence_endings = ['.', '!', '?', '다.', '요.']

            while True:
                token = await queue.get()
                if token is None:  # 스트림 종료 신호
                    if buffer:  # 남아있는 버퍼 처리
                        formatted_chunk = self._format_korean_text_chunk(buffer)
                        yield f"data: {formatted_chunk}\n\n"
                    break  # 큐 소비 루프 종료

                buffer += token

                # 현재 버퍼의 끝이 문장 종결 어미 중 하나로 끝나거나 버퍼 길이가 일정 이상일 때 플러시
                # _format_korean_text_chunk 함수가 HTML 포맷팅을 처리
                should_flush = any(buffer.endswith(ending) for ending in sentence_endings) or len(buffer) > 50

                if should_flush:
                    formatted_chunk = self._format_korean_text_chunk(buffer)
                    if formatted_chunk:  # 빈 데이터 전송 방지
                        yield f"data: {formatted_chunk}\n\n"
                    buffer = ""  # 버퍼 초기화

            # LLM 호출 태스크가 완료될 때까지 대기 (예외 발생 시 처리 위함)
            await llm_call_task

        except Exception as e:
            print(f"LLM 답변 생성 또는 스트리밍 중 오류 발생: {e}")
            # 클라이언트에게 에러 이벤트 전송
            error_message = str(e).replace("\n", " ")  # 개행문자 제거
            yield f"event: error\ndata: 답변 생성 중 오류가 발생했습니다: {error_message}\n\n"
            # 이미 None이 큐에 들어갔거나, 여기서 루프가 종료되므로 추가적인 queue.put은 불필요할 수 있음
            # 단, 핸들러의 on_llm_error에서 None을 넣지 않았다면 여기서 넣어주는 것이 안전
            if not queue.empty():  # 방어적으로 None을 넣어 루프 확실히 종료
                queue.put_nowait(None)

        finally:
            # llm_call_task가 생성되었고 아직 끝나지 않았다면 취소
            if llm_call_task and not llm_call_task.done():
                llm_call_task.cancel()
                try:
                    await llm_call_task  # 취소 요청 후 완료 대기
                except asyncio.CancelledError:
                    print("LLM 호출 태스크가 취소되었습니다.")
                except Exception as e:  # 취소 중 다른 예외 발생 가능성
                    print(f"LLM 호출 태스크 취소 중 예외 발생: {e}")
