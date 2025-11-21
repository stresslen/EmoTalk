# EmoTalk Service - Production Deployment

API service hoàn chỉnh để xử lý audio và tạo 52 blendshapes với streaming.

## 🚀 Quick Start

### 1. Build và chạy với Docker Compose

```bash
# Khởi động tất cả services
docker-compose up -d

# Xem logs
docker-compose logs -f emotalk-api

# Stop services
docker-compose down
```

### 2. Sử dụng chỉ Kafka (không có API container)

```bash
# Khởi động chỉ Kafka
docker-compose up -d zookeeper kafka kafka-ui

# Chạy API locally
python fastapi_server.py
```

## 📡 API Endpoints

### REST API (FastAPI) - Port 8000

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Process Audio File (Sync)
```bash
curl -X POST "http://localhost:8000/api/v1/process-audio-file" \
  -F "file=@./audio/test.wav" \
  -F "emotion_level=1" \
  -F "person_id=0" \
  -F "post_processing=true" \
  -F "chunk_duration=25.0" \
  -F "chunk_overlap=0.5"
```

#### 3. Process Audio với Base64 (Async)
```bash
curl -X POST "http://localhost:8000/api/v1/process-audio" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_base64": "BASE64_ENCODED_AUDIO",
    "emotion_level": 1,
    "person_id": 0,
    "post_processing": true,
    "chunk_duration": 25.0,
    "chunk_overlap": 0.5
  }'
```

#### 4. Process Audio Stream (SSE)
```bash
curl -N "http://localhost:8000/api/v1/process-audio-stream" \
  -F "file=@./audio/test.wav" \
  -F "emotion_level=1"
```

#### 5. Check Status (cho async requests)
```bash
curl http://localhost:8000/api/v1/status/{request_id}
```

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🐳 Docker Services

### Services trong docker-compose:

1. **zookeeper** (port 2181) - Kafka coordinator
2. **kafka** (port 9092, 9094) - Message queue
3. **kafka-ui** (port 8080) - Kafka management UI
4. **emotalk-api** (port 8000) - REST API service
5. **nginx** (port 80, 443) - Reverse proxy (optional)

### Monitoring

- Kafka UI: http://localhost:8080
- API Health: http://localhost:8000/health
- API Docs: http://localhost:8000/docs

## 🔧 Configuration

### Environment Variables

```bash
DEVICE=cuda                              # cuda hoặc cpu
MODEL_PATH=/app/pretrain_model/EmoTalk.pth
PORT=8000
HOST=0.0.0.0
KAFKA_SERVERS=localhost:9092
```

### Volume Mounts

```yaml
volumes:
  - ./pretrain_model:/app/pretrain_model  # Model files
  - ./result:/app/result                  # Output results
  - ./audio:/app/audio                    # Input audio files
```

## 📝 API Usage Examples

### Python Client Example

```python
import requests
import base64

# 1. Upload file và xử lý sync
with open('audio.wav', 'rb') as f:
    files = {'file': f}
    data = {
        'emotion_level': 1,
        'person_id': 0,
        'post_processing': True,
        'chunk_duration': 25.0,
        'chunk_overlap': 0.5
    }
    response = requests.post(
        'http://localhost:8000/api/v1/process-audio-file',
        files=files,
        data=data
    )
    result = response.json()
    print(f"Total frames: {result['total_frames']}")
    print(f"Duration: {result['duration']}s")
    
    # Save blendshapes
    frames = result['frames']
    for frame in frames:
        timecode = frame['timecode']
        blendshapes = frame['blendshapes']  # 52 values
        print(f"Frame {frame['frame_number']}: {timecode:.3f}s")

# 2. Process với base64 (async)
with open('audio.wav', 'rb') as f:
    audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()

response = requests.post(
    'http://localhost:8000/api/v1/process-audio',
    json={
        'audio_base64': audio_base64,
        'emotion_level': 1,
        'person_id': 0,
        'post_processing': True
    }
)
result = response.json()
request_id = result['request_id']

# Check status
import time
while True:
    status_response = requests.get(
        f'http://localhost:8000/api/v1/status/{request_id}'
    )
    status = status_response.json()
    print(f"Status: {status['status']}, Progress: {status['progress']}%")
    
    if status['status'] in ['completed', 'failed']:
        break
    time.sleep(1)

if status['status'] == 'completed':
    frames = status['frames']
    print(f"Completed! Total frames: {len(frames)}")

# 3. Streaming
import sseclient
import json

with open('audio.wav', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/api/v1/process-audio-stream',
        files=files,
        stream=True
    )
    
    client = sseclient.SSEClient(response)
    for event in client.events():
        chunk_data = json.loads(event.data)
        print(f"Chunk {chunk_data['chunk_index']}: {len(chunk_data['frames'])} frames")
        if chunk_data['is_final']:
            print("Processing completed!")
            break
```

### JavaScript/Frontend Example

```javascript
// 1. Upload file
const formData = new FormData();
formData.append('file', audioFile);
formData.append('emotion_level', 1);
formData.append('person_id', 0);

fetch('http://localhost:8000/api/v1/process-audio-file', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(`Total frames: ${data.total_frames}`);
    console.log(`Duration: ${data.duration}s`);
    
    // Process frames
    data.frames.forEach(frame => {
        const blendshapes = frame.blendshapes; // 52 values
        const timecode = frame.timecode;
        // Apply blendshapes to 3D model
    });
});

// 2. Streaming với EventSource
const eventSource = new EventSource(
    'http://localhost:8000/api/v1/process-audio-stream?' + 
    new URLSearchParams({ file: 'audio.wav' })
);

eventSource.onmessage = (event) => {
    const chunkData = JSON.parse(event.data);
    console.log(`Chunk ${chunkData.chunk_index} received`);
    
    // Process frames in real-time
    chunkData.frames.forEach(frame => {
        applyBlendshapes(frame.blendshapes, frame.timecode);
    });
    
    if (chunkData.is_final) {
        console.log('Processing completed!');
        eventSource.close();
    }
};
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Process audio file
curl -X POST "http://localhost:8000/api/v1/process-audio-file" \
  -F "file=@./audio/angry1.wav" \
  -F "emotion_level=1" \
  -F "person_id=0"

# Streaming
curl -N "http://localhost:8000/api/v1/process-audio-stream" \
  -F "file=@./audio/angry1.wav"
```

## 🔒 Production Deployment

### 1. Với Nginx (recommended)

```bash
# Uncomment nginx service trong docker-compose.yml
docker-compose up -d

# API sẽ available tại:
# - http://your-domain.com/api/v1/*
# - http://your-domain.com/docs
```

### 2. SSL/HTTPS Setup

```bash
# Generate SSL certificate
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem

# Update nginx.conf và restart
docker-compose restart nginx
```

### 3. Environment-specific configs

```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.yml up -d
```

## 📊 Response Format

### BlendshapeFrame
```json
{
  "timecode": 0.033,
  "frame_number": 1,
  "blendshapes": [0.1, 0.2, ..., 0.5]  // 52 values
}
```

### ProcessResponse (Sync)
```json
{
  "request_id": "uuid",
  "status": "completed",
  "total_chunks": 3,
  "total_frames": 150,
  "duration": 5.0,
  "frames": [...]
}
```

### StatusResponse (Async)
```json
{
  "status": "processing",
  "progress": 45.5,
  "message": "Processing chunk 2",
  "total_chunks": 3,
  "total_frames": 67
}
```

## 🛠️ Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python fastapi_server.py

# Run with reload (development)
uvicorn fastapi_server:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Test với sample audio
python -c "
import requests
with open('audio/angry1.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/process-audio-file',
        files={'file': f}
    )
    print(response.json())
"
```

## 🐛 Troubleshooting

### GPU không available
```bash
# Kiểm tra NVIDIA driver
nvidia-smi

# Kiểm tra Docker có access GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Kafka connection errors
```bash
# Kiểm tra Kafka running
docker-compose ps

# Xem logs
docker-compose logs kafka

# Restart Kafka
docker-compose restart kafka
```

### Model loading errors
```bash
# Kiểm tra model file
ls -lh pretrain_model/EmoTalk.pth

# Chạy với CPU nếu GPU không available
DEVICE=cpu python fastapi_server.py
```

## 📦 Build và Push Docker Image

```bash
# Build image
docker build -t emotalk-api:latest .

# Tag for registry
docker tag emotalk-api:latest your-registry/emotalk-api:latest

# Push to registry
docker push your-registry/emotalk-api:latest
```

## License

Theo license của EmoTalk gốc
