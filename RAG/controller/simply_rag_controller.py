from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from pydantic import BaseModel
import asyncio
import json  # json 임포트가 필요할 수 있습니다.

# 서비스 임포트
from RAG.service.simple_rag_service_impl import RAGServiceImpl

RAGRouter = APIRouter()


# 요청 바디 모델 확장
class StreamRequestBody(BaseModel):
    query: str
    session_id: str  # ✨ 수정된 부분

def injectSearchService() -> RAGServiceImpl:
    return RAGServiceImpl.getInstance()


@RAGRouter.post(
    "/stream",
    summary="POST 방식으로 query, session_id 등을 JSON 바디로 받아 토큰 스트리밍",
)
async def stream_rag(
        body: StreamRequestBody,
        request: Request,
        service: RAGServiceImpl = Depends(injectSearchService),
) -> StreamingResponse:
    print(f"--- ✅ FastAPI /stream endpoint CALLED with query: '{body.query}' for session: {body.session_id} ---")

    async def event_generator() -> AsyncGenerator[str, None]:
        print("--- ✨ FastAPI event_generator started ---")
        try:
            # 서비스 호출 시 모든 파라미터 전달
            async for token in service.text_Generate(
                    query=body.query,
                    session_id=body.session_id,  # ✨ 수정된 부분
            ):
                if await request.is_disconnected():
                    print("--- ⚠️ Client disconnected, stopping event_generator. ---")
                    break

                # FastAPI 에서는 데이터를 json 형식으로 감싸서 보내는 것이 안정적입니다.
                yield f"data: {json.dumps({'token': token})}\n\n"

        except asyncio.CancelledError:
            print("--- ⚠️ FastAPI event_generator: Task was cancelled (client likely disconnected). ---")
        except Exception as e:
            print(f"--- 💥 FastAPI event_generator CRITICAL ERROR: {e} ---")
            import traceback
            traceback.print_exc()
            error_message = str(e).replace("\n", " ")
            yield f"event: error\ndata: {json.dumps({'error': error_message})}\n\n"
        finally:
            print("--- 🏁 FastAPI event_generator finished ---")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )