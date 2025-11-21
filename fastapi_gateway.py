"""
FastAPI Gateway - REST API to gRPC Bridge
Frontend gọi REST API → Gateway forward tới gRPC server
"""
import os
import io
import uuid
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import librosa

from grpc_client_optimized import OptimizedEmoTalkClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global gRPC client
grpc_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management - khởi tạo gRPC client khi start"""
    global grpc_client
    grpc_server = os.getenv("GRPC_SERVER", "localhost:50051")
    logger.info(f"Connecting to gRPC server: {grpc_server}")
    grpc_client = OptimizedEmoTalkClient(grpc_server)
    
    # Health check
    if grpc_client.health_check():
        logger.info("✅ gRPC server is healthy")
    else:
        logger.error("❌ gRPC server is not healthy")
    
    yield
    
    logger.info("Shutting down...")
    grpc_client.close()

app = FastAPI(
    title="EmoTalk Gateway API",
    description="REST API Gateway to gRPC Server - For Frontend Integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    grpc_status: str

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "EmoTalk Gateway API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "process_audio_stream": "/api/v1/process-audio-stream",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    grpc_status = "unknown"
    try:
        if grpc_client and grpc_client.health_check():
            grpc_status = "healthy"
        else:
            grpc_status = "unhealthy"
    except Exception as e:
        grpc_status = f"error: {str(e)}"
        logger.error(f"gRPC health check failed: {e}")
    
    return HealthResponse(
        status="healthy",
        service="EmoTalk Gateway API",
        version="1.0.0",
        grpc_status=grpc_status
    )

@app.post("/api/v1/process-audio-stream")
async def process_audio_stream(file: UploadFile = File(...)):
    """
    Process audio với streaming response (SSE - Server-Sent Events)
    Gateway forward request tới gRPC server và stream response về frontend
    
    Fixed parameters:
    - emotion_level: 1
    - person_id: 0
    - post_processing: True
    - chunk_duration: 25.0s
    - chunk_overlap: 0.5s
    """
    request_id = str(uuid.uuid4())
    request_start_time = time.time()
    
    # Fixed parameters
    emotion_level = 1
    person_id = 0
    post_processing = True
    chunk_duration = 25.0
    chunk_overlap = 0.5
    
    try:
        # Đọc file
        audio_bytes = await file.read()
        
        # Load audio để validate và get duration
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        audio_duration = len(audio_array) / sr
        
        logger.info(f"[{request_id}] Gateway: Starting stream processing via gRPC")
        logger.info(f"[{request_id}] File: {file.filename}, duration: {audio_duration:.2f}s")
        
        async def generate():
            """Generator để stream chunks từ gRPC server"""
            chunk_count = 0
            
            try:
                # Callback để xử lý từng chunk từ gRPC
                def on_chunk(chunk_data):
                    nonlocal chunk_count
                    chunk_count += 1
                    
                    frames = chunk_data['frames']
                    logger.info(f"[{request_id}] Gateway: Received chunk {chunk_count} "
                               f"from gRPC ({len(frames)} frames)")
                    
                    # Convert về format SSE cho frontend
                    import json
                    return f"data: {json.dumps(frames)}\n\n"
                
                # Tạo temp file cho gRPC client
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                try:
                    # Process audio via gRPC với callback
                    for chunk_data in grpc_client.process_audio_file_generator(
                        tmp_path,
                        request_id=request_id,
                        emotion_level=emotion_level,
                        person_id=person_id,
                        post_processing=post_processing,
                        chunk_duration=chunk_duration,
                        chunk_overlap=chunk_overlap
                    ):
                        chunk_count += 1
                        frames = chunk_data['frames']
                        
                        logger.info(f"[{request_id}] Gateway: Streaming chunk {chunk_count} "
                                   f"({len(frames)} frames)")
                        
                        # Stream về frontend
                        import json
                        yield f"data: {json.dumps(frames)}\n\n"
                        
                        if chunk_data['is_final']:
                            total_time = time.time() - request_start_time
                            logger.info(f"[{request_id}] Gateway: ✅ Completed in {total_time:.3f}s, "
                                       f"{chunk_count} chunks")
                            break
                
                finally:
                    # Cleanup temp file
                    import os
                    os.unlink(tmp_path)
                    
            except Exception as e:
                logger.error(f"[{request_id}] Gateway error: {e}", exc_info=True)
                import json
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Gateway error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    grpc_server = os.getenv("GRPC_SERVER", "localhost:50051")
    
    logger.info(f"Starting FastAPI Gateway on {host}:{port}")
    logger.info(f"Connecting to gRPC server: {grpc_server}")
    
    uvicorn.run(
        "fastapi_gateway:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
