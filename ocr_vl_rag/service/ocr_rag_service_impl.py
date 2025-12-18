from typing import Optional, AsyncGenerator

# 1. Repository 임포트 경로와 클래스 이름을 수정합니다.
from ocr_vl_rag.repository.ocr_rag_repository_impl import OcrRAGRepositoryImpl
from ocr_vl_rag.service.ocr_rag_service import OcrRAGService


class OcrRAGServiceImpl(OcrRAGService):
    __instance = None
    # 2. _repo의 타입 힌트를 OcrRAGRepositoryImpl로 변경합니다.
    _repo: Optional[OcrRAGRepositoryImpl] = None

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
            # 3. 초기화 시 OcrRAGRepositoryImpl 인스턴스를 가져오도록 수정합니다.
            print("--- OcrRAGServiceImpl: __init__ 최초 초기화 ---")
            self._repo = OcrRAGRepositoryImpl.getInstance()
            self._initialized_service = True
            print("--- ✅ OcrRAG Repository 연결 완료. ---")

    # ✨✨✨ 핵심 수정 부분 ✨✨✨
    async def text_Generate(
            self,
            query: str,
            session_id: str,
            image_data: Optional[str] = None  # 4. image_data 파라미터를 추가합니다.
    ) -> AsyncGenerator[str, None]:
        """
        컨트롤러로부터 받은 인자들을 Repository 계층으로 전달하고,
        스트리밍 결과를 그대로 반환합니다.
        대화 기록 관리는 Repository가 담당합니다.
        """

        print(
            f"--- OcrRAGServiceImpl.text_Generate CALLED for session: '{session_id}' (Image: {'Yes' if image_data else 'No'}) ---")

        # 5. Repository의 generate 메서드에 모든 파라미터를 전달합니다.
        #    이제 대화 기록을 수동으로 관리할 필요가 없습니다.
        async for token in self._repo.generate(
                query=query,
                session_id=session_id,
                image_data=image_data,
                chat_history=self._repo.get_chat_history(session_id)  # chat_history를 직접 가져와 전달
        ):
            yield token