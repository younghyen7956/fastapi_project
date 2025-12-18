from fastapi import APIRouter, Depends, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Optional
import asyncio
import json
import base64  # base64 인코딩을 위해 추가합니다.
from ocr_vl_rag.service.ocr_rag_service_impl import OcrRAGServiceImpl

OcrRAGRouter = APIRouter()


# StreamRequestBody 모델은 더 이상 필요하지 않습니다.

def injectSearchService() -> OcrRAGServiceImpl:
    return OcrRAGServiceImpl.getInstance()


@OcrRAGRouter.post(
    "/stream_ocr_vl",
    summary="POST 방식으로 query, session_id, image 등을 Form-data로 받아 토큰 스트리밍",
)
async def stream_rag(
        request: Request,
        # JSON 본문(body) 대신 Form 데이터와 File을 직접 받습니다.
        query: str = Form(...),
        session_id: str = Form(...),
        # 이미지는 선택적으로 받을 수 있도록 Optional[UploadFile]로 설정합니다.
        image: Optional[UploadFile] = File(None),
        service: OcrRAGServiceImpl = Depends(injectSearchService),
) -> StreamingResponse:
    print(f"--- ✅ FastAPI /stream endpoint CALLED with query: '{query}' for session: {session_id} ---")

    # 이미지가 첨부되었는지 확인하고, base64로 인코딩합니다.
    image_data_b64: Optional[str] = None
    if image:
        print(f"--- 🖼️ Received image: {image.filename} ({image.content_type}) ---")
        image_bytes = await image.read()
        image_data_b64 = base64.b64encode(image_bytes).decode("utf-8")
        print("--- ✅ Image encoded to base64 successfully. ---")

    async def event_generator() -> AsyncGenerator[str, None]:
        print("--- ✨ FastAPI event_generator started ---")
        try:
            # 서비스 호출 시 모든 파라미터를 전달합니다.
            # 서비스의 text_Generate 메서드가 image_data를 받을 수 있도록 수정해야 합니다.
            async for token_data in service.text_Generate(
                    query=query,
                    session_id=session_id,
                    image_data=image_data_b64,  # 인코딩된 이미지 데이터 전달
            ):
                if await request.is_disconnected():
                    print("--- ⚠️ Client disconnected, stopping event_generator. ---")
                    break

                # Repository의 generate 메서드가 JSON 문자열을 yield 하므로 그대로 전달합니다.
                yield token_data

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