from abc import ABC, abstractmethod
from typing import AsyncGenerator

class RAGRepository(ABC):

    @abstractmethod
    async def generate(self, query: str, k: int) -> AsyncGenerator[str, None]:
        pass