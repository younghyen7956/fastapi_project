from abc import ABC, abstractmethod


class RAGRepository(ABC):

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def connectDB(self):
        pass

    @abstractmethod
    def generate(self, query: str) -> dict:
        pass