from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Optional


class VlRAGRepository(ABC):

    @abstractmethod
    async def generate(self, query: str, chat_history: List[Dict[str, str]], k: int = 10,
                       image_data: Optional[str] = None) -> AsyncGenerator[str, None]:
        pass