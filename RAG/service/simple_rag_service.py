from abc import ABC, abstractmethod


class RAGService(ABC):
    @abstractmethod
    def text_Generate(self,query: str):
        pass