from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

class VlRAGService(ABC):
    @abstractmethod
    async def text_Generate(
        self,
        query: str,
        session_id: str,
        image_data: Optional[str] = None # <--- 변경: image_data 파라미터 추가
    ) -> AsyncGenerator[str, None]:
        pass