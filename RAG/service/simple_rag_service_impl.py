# from RAG.repository.simple_rag_repository_impl import RAGRepositoryImpl # 경로 확인
# from RAG.service.simple_rag_service import RAGService # 경로 확인
from typing import AsyncGenerator, List, Dict, Optional # 추가

# RAGRepositoryImpl 임포트 (경로는 실제 프로젝트에 맞게 수정)
from RAG.repository.simple_rag_repository_impl import RAGRepositoryImpl 

# RAGService 인터페이스가 있다면 그것을 상속, 없다면 object 상속
class RAGService: # 임시 RAGService 인터페이스 정의
    async def text_Generate(self, query: str, initial_k: int, rerank_n: int,) -> AsyncGenerator[str, None]:
        raise NotImplementedError


class RAGServiceImpl(RAGService):
    __instance = None
    _repo: Optional[RAGRepositoryImpl] = None # 타입 힌트 추가

    def __new__(cls, *args, **kwargs): # 싱글턴 인수 처리
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            # __init__에서 repo 초기화
        return cls.__instance

    def __init__(self):
        if not hasattr(self, '_initialized_service'): # 서비스 초기화 플래그
            print("--- RAGServiceImpl: __init__ 최초 초기화 ---")
            self._repo = RAGRepositoryImpl.getInstance() # 레포지토리 인스턴스 생성 및 저장
            self._initialized_service = True
        else:
            print("--- RAGServiceImpl: __init__ (이미 초기화됨) ---")


    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls() # __new__ 와 __init__ 호출
        return cls.__instance

    # generate 메서드 시그니처에 모든 파라미터 추가
    async def text_Generate(self, 
                            query: str, 
                            initial_k: int, 
                            rerank_n: int,
                           ) -> AsyncGenerator[str, None]:
        print(f"--- RAGServiceImpl.text_Generate CALLED with query: '{query}' ---")
        # RAGRepositoryImpl.generate 호출 시 모든 파라미터 전달
        async for token in self._repo.generate(
            query=query,
            initial_k=initial_k,
            rerank_n=rerank_n,
        ):
            yield token