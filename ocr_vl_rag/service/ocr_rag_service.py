from abc import ABC, abstractmethod
from typing import AsyncGenerator

class OcrRAGService(ABC):
    @abstractmethod
    async def text_Generate(self, query: str, session_id: str) -> AsyncGenerator[str, None]:
        pass