# EmoTalk - Emotion-Driven 3D Facial Animation

Real-time emotion-driven facial animation system using EmoTalk AI model with optimized gRPC backend and REST API gateway.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-orange.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🎯 Features

- ✅ **Real-time Processing**: Stream blendshapes as chunks are processed
- ✅ **High Performance**: gRPC backend with queue system for concurrent requests
- ✅ **REST API Gateway**: Easy integration with web frontends
- ✅ **GPU Accelerated**: CUDA support for fast inference
- ✅ **Production Ready**: Docker deployment with health checks
- ✅ **Scalable**: Queue system prevents request loss, handles bursts
- ✅ **52 Blendshapes**: Full facial animation control @ 30fps

## 🏗️ Architecture

```
┌──────────────┐
│   Frontend   │ (Browser/App)
└──────┬───────┘
       │ HTTP/REST + SSE
       ▼
┌──────────────────┐
│  API Gateway     │ (FastAPI, Port 8000)
│  - REST API      │
│  - SSE Streaming │
└──────┬───────────┘
       │ gRPC (Internal)
       ▼
┌──────────────────┐
│  gRPC Server     │ (Port 50051)
│  - Request Queue │
│  - Worker Pool   │
│  - AI Processing │
└──────────────────┘
```

### Why gRPC + Gateway?

- **gRPC Backend**: High performance, binary protocol, HTTP/2 multiplexing
- **REST Gateway**: Browser compatibility, easy integration
- **Queue System**: No missed requests, handles concurrent load
- **6x faster** than pure REST/SSE approach

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support (or CPU mode)
- Python 3.10+ (for local development)

### 1. Clone Repository

```bash
git clone https://github.com/stresslen/Audio2Face.git
cd Audio2Face/EmoTalk
```

### 2. Download Model

Place the pretrained model in `pretrain_model/EmoTalk.pth`

### 3. Deploy with Docker

```bash
# Start gRPC server only
docker compose -f docker-compose.grpc.yml up -d

# Or start full stack (gRPC + Gateway)
docker compose -f docker-compose.full.yml up -d
```

### 4. Test

```bash
# Quick test
./test_service.sh

# Or manually with curl
curl -X POST http://localhost:8000/api/v1/process-audio-stream \
  -F "file=@audio/sample.wav" \
  --no-buffer
```

## 📡 API Usage

### REST API (Port 8000)

**Endpoint:** `POST /api/v1/process-audio-stream`

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/process-audio-stream \
  -F "file=@audio.wav" \
  --no-buffer
```

**Response:** Server-Sent Events (SSE)
```javascript
data: [
  {
    "timecode": 0.000000,
    "blendshapes": [0.378, 0.376, 0.038, ..., 0.521]  // 52 values
  },
  {
    "timecode": 0.033333,
    "blendshapes": [0.381, 0.379, 0.039, ..., 0.523]
  },
  ...
]
```

### Frontend Integration

```javascript
async function processAudio(audioFile) {
    const formData = new FormData();
    formData.append('file', audioFile);
    
    const response = await fetch('http://localhost:8000/api/v1/process-audio-stream', {
        method: 'POST',
        body: formData
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const frames = JSON.parse(line.slice(6));
                
                // Update 3D model with blendshapes
                frames.forEach(frame => {
                    updateFacialAnimation(frame.timecode, frame.blendshapes);
                });
            }
        }
    }
}
```

### Python Client (gRPC)

```python
from grpc_client_optimized import OptimizedEmoTalkClient

with OptimizedEmoTalkClient('localhost:50051') as client:
    # Process audio file
    chunks = client.process_audio_file(
        'audio.wav',
        emotion_level=1,
        person_id=0
    )
    
    for chunk in chunks:
        for frame in chunk['frames']:
            print(f"Time: {frame['timecode']}s")
            print(f"Blendshapes: {frame['blendshapes'][:5]}...")
```

## 🐳 Docker Deployment

### Configuration

**docker-compose.grpc.yml** - gRPC server only:
```yaml
services:
  emotalk_grpc:
    ports:
      - "50051:50051"
    environment:
      - DEVICE=cuda          # or 'cpu'
      - NUM_WORKERS=2        # Number of concurrent workers
      - QUEUE_SIZE=100       # Max queue size
```

**docker-compose.full.yml** - Full stack (gRPC + Gateway):
```yaml
services:
  grpc_server:
    ports:
      - "50051:50051"
  api_gateway:
    ports:
      - "8000:8000"
    environment:
      - GRPC_SERVER=grpc_server:50051
```

### Commands

```bash
# Start services
docker compose -f docker-compose.full.yml up -d

# View logs
docker compose -f docker-compose.full.yml logs -f

# Check status
docker compose -f docker-compose.full.yml ps

# Stop services
docker compose -f docker-compose.full.yml down
```

## 🔧 Configuration

### Environment Variables

**gRPC Server:**
- `MODEL_PATH`: Path to pretrained model (default: `./pretrain_model/EmoTalk.pth`)
- `DEVICE`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `NUM_WORKERS`: Number of worker threads (default: `2`)
- `QUEUE_SIZE`: Maximum queue size (default: `100`)

**API Gateway:**
- `PORT`: API port (default: `8000`)
- `HOST`: Listen address (default: `0.0.0.0`)
- `GRPC_SERVER`: gRPC server address (default: `localhost:50051`)

### Processing Parameters

Fixed parameters optimized for best results:
- `emotion_level`: 1 (emotion-driven)
- `person_id`: 0 (default speaker)
- `post_processing`: True (smoothing + blinking)
- `chunk_duration`: 25.0s
- `chunk_overlap`: 0.5s

## 📊 Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Processing Speed | ~0.5-0.8x realtime (GPU) |
| Latency | 2-4s for first chunk |
| Throughput | 2-4 concurrent requests |
| Output | 30 fps, 52 blendshapes |

### Scaling

- **1 Worker**: Sequential processing, no missed requests
- **2 Workers**: 2x throughput (recommended for single GPU)
- **4 Workers**: 4x throughput (requires multiple GPUs or powerful GPU)

### Queue System

- **FIFO Queue**: Process requests in order
- **Non-blocking**: Reject when full, prevent OOM
- **Metrics**: Track queue size, wait time, success rate

## 📁 Project Structure

```
EmoTalk/
├── grpc_server_optimized.py    # gRPC server with queue system
├── grpc_client_optimized.py    # gRPC client library
├── fastapi_gateway.py          # REST API gateway
├── fastapi_server.py           # Legacy FastAPI server (SSE)
├── emotalk_processor.py        # EmoTalk AI processor
├── model.py                    # Model architecture
├── wav2vec.py                  # Audio feature extraction
├── utils.py                    # Utilities
├── proto/
│   └── emotalk.proto          # gRPC protocol definition
├── docker-compose.grpc.yml    # Docker: gRPC only
├── docker-compose.full.yml    # Docker: Full stack
├── Dockerfile.grpc            # Dockerfile for gRPC server
├── Dockerfile.gateway         # Dockerfile for API gateway
├── test_service.sh            # Test script
├── pretrain_model/
│   └── EmoTalk.pth           # Pretrained model (download separately)
└── audio/                     # Sample audio files
```

## 🧪 Testing

### Quick Test

```bash
./test_service.sh
```

### Manual Test

```bash
# Health check
curl http://localhost:8000/health

# Process audio
curl -X POST http://localhost:8000/api/v1/process-audio-stream \
  -F "file=@audio/sample.wav" \
  --no-buffer
```

### Python Test

```python
from grpc_client_optimized import OptimizedEmoTalkClient

client = OptimizedEmoTalkClient('localhost:50051')
assert client.health_check(), "Server not healthy"

chunks = client.process_audio_file('audio/sample.wav')
print(f"Received {len(chunks)} chunks")
```

## 🛠️ Development

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate gRPC code
python -m grpc_tools.protoc \
    -I./proto \
    --python_out=. \
    --grpc_python_out=. \
    ./proto/emotalk.proto

# Start gRPC server
python grpc_server_optimized.py --port 50051 --device cuda

# Start API gateway (in another terminal)
export GRPC_SERVER=localhost:50051
python fastapi_gateway.py
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📚 Documentation

- [Gateway Architecture](GATEWAY_ARCHITECTURE.md) - Detailed architecture guide
- [gRPC Optimization](README_GRPC_OPTIMIZED.md) - gRPC implementation details
- [Quick Start Guide](QUICKSTART_GRPC.md) - Deployment quickstart
- [Solution Summary](SOLUTION_SUMMARY.md) - Technical decisions and comparisons

## 🔍 Monitoring

### Health Checks

```bash
# API Gateway
curl http://localhost:8000/health

# gRPC Server (from container)
docker exec emotalk_grpc_optimized python -c "
from grpc_client_optimized import OptimizedEmoTalkClient
client = OptimizedEmoTalkClient('localhost:50051')
print('Healthy' if client.health_check() else 'Unhealthy')
"
```

### Logs

```bash
# All services
docker compose -f docker-compose.full.yml logs -f

# Specific service
docker compose -f docker-compose.full.yml logs -f api_gateway
docker compose -f docker-compose.full.yml logs -f grpc_server

# Stats (auto-reported every 30s)
docker logs emotalk_grpc_optimized | grep "📊 Stats"
```

## 🚨 Troubleshooting

### Queue Full

**Error:** "Server is busy, queue is full"

**Solution:**
```yaml
# Increase queue size or workers
environment:
  - QUEUE_SIZE=200
  - NUM_WORKERS=4
```

### Out of Memory

**Solution:** Reduce workers or use CPU mode
```yaml
environment:
  - NUM_WORKERS=1
  - DEVICE=cpu
```

### Connection Refused

**Check:**
```bash
# Verify services are running
docker compose -f docker-compose.full.yml ps

# Check network
docker network inspect emotalk_emotalk_network
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- EmoTalk model research and development
- Built with FastAPI, gRPC, PyTorch
- Audio processing with librosa and wav2vec2

## 📞 Support

- Issues: [GitHub Issues](https://github.com/stresslen/Audio2Face/issues)
- Documentation: See documentation files in repo

---

**Version:** 1.0.0  
**Last Updated:** 2025-11-21  
**Status:** ✅ Production Ready
