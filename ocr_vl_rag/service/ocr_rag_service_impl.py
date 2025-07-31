import json
from typing import AsyncGenerator, Optional, List, Dict

from RAG.repository.simple_rag_repository_impl import RAGRepositoryImpl
from ocr_vl_rag.service.ocr_rag_service import OcrRAGService


class OcrRAGServiceImpl(OcrRAGService):
    __instance = None
    _repo: Optional[RAGRepositoryImpl] = None

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
        if not hasattr(self, '_initialized_service'):
            print("--- RAGServiceImpl: __init__ 최초 초기화 ---")
            self._repo = RAGRepositoryImpl.getInstance()
            self._initialized_service = True

    # ✨ session_id를 받도록 시그니처 수정
    async def text_Generate(self, query: str, session_id: str) -> AsyncGenerator[str, None]:
        print(f"--- RAGServiceImpl.text_Generate CALLED for session: '{session_id}' ---")

        # 1. 서비스가 리포지토리를 통해 이전 대화 기록을 가져옵니다.
        session_history = self._repo.get_chat_history(session_id)

        # 2. 리포지토리의 generate 함수를 호출하여 답변 생성을 시작합니다.
        response_generator = self._repo.generate(query, session_history)

        # 3. 답변을 스트리밍하면서, 동시에 전체 답변을 수집합니다.
        full_response_parts = []
        try:
            async for token in response_generator:
                full_response_parts.append(token)
                yield token
        finally:
            # 4. 스트리밍이 끝나면, 서비스가 리포지토리를 통해 대화 기록 업데이트를 요청합니다.
            full_ai_response = "".join(full_response_parts)
            if full_ai_response:
                await self._repo.update_chat_history(
                    session_id=session_id,
                    user_query=query,
                    ai_response=full_ai_response
                )