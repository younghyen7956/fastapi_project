import json
from typing import AsyncGenerator, Optional, List, Dict

# 레포지토리 클래스명을 VlRAGRepositoryImpl로 수정합니다.
from VLRag.repository.vl_rag_repository_impl import VlRAGRepositoryImpl
from VLRag.service.vl_rag_service import VlRAGService


class VlRAGServiceImpl(VlRAGService):
    __instance = None
    _repo: Optional[VlRAGRepositoryImpl] = None # <--- 변경: 레포지토리 타입 명시

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
            print("--- VlRAGServiceImpl: __init__ 최초 초기화 ---")
            # 레포지토리 클래스명을 VlRAGRepositoryImpl로 수정합니다.
            self._repo = VlRAGRepositoryImpl.getInstance()
            self._initialized_service = True

    # ✨ image_data를 받도록 시그니처 수정
    async def text_Generate(
        self,
        query: str,
        session_id: str,
        image_data: Optional[str] = None # <--- 변경: image_data 파라미터 추가
    ) -> AsyncGenerator[str, None]:
        print(f"--- VlRAGServiceImpl.text_Generate CALLED for session: '{session_id}' | 이미지 존재: {'Yes' if image_data else 'No'} ---")

        # 1. 서비스가 리포지토리를 통해 이전 대화 기록을 가져옵니다.
        session_history = self._repo.get_chat_history(session_id)

        # 2. 리포지토리의 generate 함수 호출 시 image_data를 전달합니다.
        response_generator = self._repo.generate(
            query=query,
            chat_history=session_history,
            image_data=image_data # <--- 변경: image_data 전달
        )

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