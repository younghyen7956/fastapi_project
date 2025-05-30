from fastapi import APIRouter, Depends, HTTPException, Request  # Request 추가
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, List, Dict, Optional  # Optional, List, Dict 추가
from pydantic import BaseModel  # ValidationError 제거 (FastAPI가 자동 처리)
import asyncio

# 서비스 임포트 (경로 확인 필요)
from RAG.service.simple_rag_service_impl import RAGServiceImpl

# from RAG.service.simple_rag_service import RAGService # 인터페이스, 여기선 구현체 사용

RAGRouter = APIRouter()


# 요청 바디 모델 확장
class StreamRequestBody(BaseModel):
    query: str
    initial_k: int = 75
    rerank_n: int = 50


def injectSearchService() -> RAGServiceImpl:
    return RAGServiceImpl.getInstance()


@RAGRouter.post(
    "/stream",
    # response_class=StreamingResponse, # StreamingResponse는 직접 반환하므로 명시 불필요
    summary="POST 방식으로 query, chat_history 등을 JSON 바디로 받아 토큰 스트리밍",
)
async def stream_rag(
        body: StreamRequestBody,  # 수정된 요청 바디 모델 사용
        request: Request,  # 클라이언트 연결 끊김 감지용
        service: RAGServiceImpl = Depends(injectSearchService),
) -> StreamingResponse:
    """
    클라이언트에서 POST /stream 으로
    {
      "query": "...",
      "chat_history": [ {"role": "user", "content": "..."}, {"role": "ai", "content": "..."} ],
      "initial_k": 75,
      "rerank_n": 50
    }
    형태의 JSON 바디를 보내면 LLM 토큰을 SSE 형식으로 실시간 전송합니다.
    """
    print(f"--- ✅ FastAPI /stream endpoint CALLED with query: '{body.query}' ---")

    # FastAPI가 body 유효성 검사를 자동으로 수행하므로 try-except 불필요

    async def event_generator() -> AsyncGenerator[str, None]:
        print("--- ✨ FastAPI event_generator started ---")
        try:
            processed_any_data = False
            # 서비스 호출 시 모든 파라미터 전달
            async for chunk_from_service in service.text_Generate(
                    query=body.query,
                    initial_k=body.initial_k,
                    rerank_n=body.rerank_n,
            ):
                if await request.is_disconnected():  # 클라이언트 연결 끊김 확인
                    print("--- ⚠️ Client disconnected, stopping event_generator. ---")
                    break

                processed_any_data = True
                if isinstance(chunk_from_service, str) and chunk_from_service.startswith("event: error"):
                    # print(f"--- ➡️ FastAPI yielding ERROR event (already formatted by RAG): {chunk_from_service[:100]}... ---")
                    yield chunk_from_service  # 이미 완전한 SSE 오류 이벤트 형식
                elif isinstance(chunk_from_service, str):
                    sse_line = f"data: {chunk_from_service}\n\n"
                    yield sse_line
                else:
                    print(f"--- ⚠️ FastAPI received unexpected chunk type from service: {type(chunk_from_service)} ---")

            if not processed_any_data and not await request.is_disconnected():
                print(
                    "--- ⚠️ FastAPI event_generator: service.text_Generate produced no data. Sending keep-alive or empty message. ---")
                yield "event: ping\ndata: keep-alive\n\n"  # 또는 빈 데이터 메시지

        except asyncio.CancelledError:  # 클라이언트 연결 종료 시 발생 가능
            print("--- ⚠️ FastAPI event_generator: Task was cancelled (client likely disconnected). ---")
        except Exception as e:
            print(f"--- 💥 FastAPI event_generator CRITICAL ERROR: {e} ---")
            import traceback
            traceback.print_exc()  # 전체 스택 트레이스 출력
            error_message = str(e).replace("\n", " ")
            yield f"event: error\ndata: Critical error in stream: {error_message}\n\n"
        finally:
            print("--- 🏁 FastAPI event_generator finished ---")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        # X-Accel-Buffering for Nginx
    )
