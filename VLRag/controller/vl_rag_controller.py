from fastapi import APIRouter, Depends, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Optional
import asyncio
import json
import base64  # <--- 변경: 이미지 인코딩을 위해 추가

from VLRag.service.vl_rag_service_impl import VlRAGServiceImpl

VlRAGRouter = APIRouter()

def injectSearchService() -> VlRAGServiceImpl:
    return VlRAGServiceImpl.getInstance()


@VlRAGRouter.post(
    "/stream_vl",
    summary="POST 방식으로 query, session_id, image 등을 Form/File으로 받아 토큰 스트리밍",
)
async def stream_rag(
        request: Request,
        query: str = Form(...),
        session_id: str = Form(...),
        image: Optional[UploadFile] = File(None),  # 이미지는 선택적으로 받음
        service: VlRAGServiceImpl = Depends(injectSearchService),
) -> StreamingResponse:
    print(f"--- ✅ FastAPI /stream endpoint CALLED with query: '{query}' for session: {session_id} ---")

    # --- 변경: 업로드된 이미지 처리 로직 추가 ---
    image_data: Optional[str] = None
    if image:
        print(f"--- 🖼️ Received image: {image.filename} ({image.content_type}) ---")
        # 비동기적으로 파일 내용을 읽습니다.
        contents = await image.read()
        # 읽은 내용을 Base64로 인코딩합니다.
        image_data = base64.b64encode(contents).decode('utf-8')
        print("--- ✅ Image successfully encoded to Base64 ---")

    # ----------------------------------------

    async def event_generator() -> AsyncGenerator[str, None]:
        print("--- ✨ FastAPI event_generator started ---")
        try:
            # --- 변경: 서비스 호출 시 image_data 전달 ---
            async for token in service.text_Generate(
                    query=query,
                    session_id=session_id,
                    image_data=image_data,  # 인코딩된 이미지 데이터 전달
            ):
                # ----------------------------------------
                if await request.is_disconnected():
                    print("--- ⚠️ Client disconnected, stopping event_generator. ---")
                    break

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