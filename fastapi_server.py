"""
REST API với FastAPI - Để frontend và các service khác sử dụng
"""
import os
import io
import uuid
import time
import base64
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import librosa

from emotalk_processor import EmoTalkProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global processor instance
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management - khởi tạo model khi start"""
    global processor
    logger.info("Initializing EmoTalk processor...")
    model_path = os.getenv("MODEL_PATH", "./pretrain_model/EmoTalk.pth")
    device = os.getenv("DEVICE", "cuda")
    processor = EmoTalkProcessor(model_path=model_path, device=device)
    logger.info("EmoTalk processor initialized successfully")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="EmoTalk API",
    description="API để xử lý audio và tạo 52 blendshapes với streaming chunks",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware để frontend có thể call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AudioProcessRequest(BaseModel):
    audio_base64: str = Field(..., description="Audio data encoded in base64 (WAV format, 16kHz)")
    emotion_level: int = Field(1, ge=0, le=1, description="Emotion level (0 or 1)")
    person_id: int = Field(0, ge=0, le=23, description="Person ID (0-23)")
    post_processing: bool = Field(True, description="Enable post-processing (smoothing, blinking)")
    chunk_duration: float = Field(25.0, gt=0, description="Chunk duration in seconds")
    chunk_overlap: float = Field(0.5, ge=0, description="Chunk overlap in seconds")

class BlendshapeFrame(BaseModel):
    timecode: float = Field(..., description="Timecode in seconds")
    frame_number: int = Field(..., description="Frame number")
    blendshapes: List[float] = Field(..., description="52 blendshape values")

class BlendshapeChunk(BaseModel):
    request_id: str
    chunk_index: int
    start_time: float
    end_time: float
    frames: List[BlendshapeFrame]
    is_final: bool

class ProcessResponse(BaseModel):
    request_id: str
    status: str
    message: str
    total_chunks: Optional[int] = None
    total_frames: Optional[int] = None
    duration: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

# In-memory storage cho processing status (trong production dùng Redis)
processing_status = {}

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "EmoTalk API",
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
    return HealthResponse(
        status="healthy",
        service="EmoTalk API",
        version="1.0.0"
    )



@app.post("/api/v1/process-audio-stream")
async def process_audio_stream(
    file: UploadFile = File(...)
):
    """
    Process audio với streaming response (SSE - Server-Sent Events)
    Stream từng chunk về client ngay khi xử lý xong
    
    Fixed parameters:
    - emotion_level: 1
    - person_id: 0
    - post_processing: True
    - chunk_duration: 25.0s
    - chunk_overlap: 0.5s
    """
    request_id = str(uuid.uuid4())
    request_start_time = time.perf_counter()  # Dùng perf_counter thay vì time.time()
    
    # Fixed parameters
    emotion_level = 1
    person_id = 0
    post_processing = True
    chunk_duration = 25.0
    chunk_overlap = 0.5
    
    try:
        # Đọc file
        audio_bytes = await file.read()
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        audio_duration = len(audio_array)/sr
        logger.info(f"[{request_id}] Starting stream processing: {file.filename}, duration: {audio_duration:.2f}s")
        
        async def generate():
            """Generator để stream chunks"""
            chunk_count = 0
            
            for chunk_result in processor.process_full_audio(
                audio_array,
                emotion_level=emotion_level,
                person_id=person_id,
                post_processing=post_processing,
                chunk_duration=chunk_duration,
                overlap=chunk_overlap,
                sample_rate=sr
            ):
                blendshapes = chunk_result['blendshapes']
                fps = 30.0
                
                frames = []
                chunk_start_time = chunk_result['start_time']  # Timecode trong audio
                for frame_idx in range(blendshapes.shape[0]):
                    timecode = chunk_start_time + (frame_idx / fps)
                    frames.append({
                        'timecode': round(timecode, 6),
                        'blendshapes': blendshapes[frame_idx].tolist()
                    })
                
                chunk_count += 1
                
                logger.info(f"[{request_id}] Streaming chunk {chunk_count}: {chunk_result['start_time']:.2f}s-{chunk_result['end_time']:.2f}s, {len(frames)} frames")
                
                import json
                # Trả về array frames trực tiếp, không wrap trong object
                yield f"data: {json.dumps(frames)}\n\n"
                
                if chunk_result['is_final']:
                    total_time = time.perf_counter() - request_start_time  # Dùng perf_counter
                    logger.info(f"[{request_id}] ✅ Stream completed - Total: {total_time:.3f}s, Audio: {audio_duration:.2f}s, Chunks: {chunk_count}")
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error in streaming: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Status endpoint and background processing removed - use sync endpoints only

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "fastapi_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
