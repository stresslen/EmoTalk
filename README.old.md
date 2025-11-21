# EmoTalk API

EmoTalk is a real-time facial animation API that converts audio speech into 52 ARKit-compatible blendshapes for facial animation. Built with FastAPI and PyTorch, it provides streaming support for low-latency animation generation.

## Features

- 🎙️ **Audio to Blendshapes**: Convert speech audio to 52 ARKit blendshapes
- ⚡ **Real-time Streaming**: Server-Sent Events (SSE) for chunk-by-chunk processing
- 🐳 **Docker Ready**: Full Docker and Docker Compose support with CUDA
- 📊 **Performance Logging**: Detailed timing metrics for monitoring
- 🔄 **Simple Output**: Clean JSON with only `timecode` and `blendshapes` arrays

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for GPU inference)
- Python 3.10+ (for local development)

### 1. Clone Repository

```bash
git clone https://github.com/stresslen/EmoTalk.git
cd EmoTalk
```

### 2. Download Model

Download the pretrained EmoTalk model and place it in `pretrain_model/`:

```bash
# Place your EmoTalk.pth file here
pretrain_model/EmoTalk.pth
```

### 3. Run with Docker

```bash
# Start all services
docker compose up -d

# Check logs
docker logs emotalk-api -f

# Wait for model to load (30-60 seconds)
# When you see "EmoTalk processor initialized successfully", it's ready
```

### 4. Test API

```bash
# Health check
curl http://localhost:8000/health

# Test with audio file
curl -X POST http://localhost:8000/api/v1/process-audio-stream \
  -F "file=@audio/angry1.wav"
```

## API Documentation

### Endpoint

**POST** `/api/v1/process-audio-stream`

Processes audio file and streams blendshapes in real-time.

### Request

- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `file`: Audio file (WAV format, 16kHz recommended)

### Response

**Content-Type**: `text/event-stream`

Streams chunks of blendshape data:

```json
data: [
  {
    "timecode": 0.0,
    "blendshapes": [0.464, 0.459, 0.045, ...]
  },
  {
    "timecode": 0.033333,
    "blendshapes": [0.465, 0.460, 0.046, ...]
  }
]
```

### Fixed Parameters

The API uses optimized default parameters:
- **emotion_level**: 1
- **person_id**: 0  
- **post_processing**: true (includes smoothing and blinking)
- **chunk_duration**: 25.0 seconds
- **chunk_overlap**: 0.5 seconds

### Example with Python

```python
import requests

url = "http://localhost:8000/api/v1/process-audio-stream"

with open("audio/test.wav", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files, stream=True)
    
    for line in response.iter_lines():
        if line and line.startswith(b'data: '):
            data = line[6:]  # Remove 'data: ' prefix
            frames = json.loads(data)
            for frame in frames:
                timecode = frame['timecode']
                blendshapes = frame['blendshapes']  # 52 values
                print(f"Frame at {timecode}s: {len(blendshapes)} blendshapes")
```

## Performance Logs

The API logs detailed timing information:

```
[request_id] Starting stream processing: test.wav, duration: 3.69s
Split audio (3.69s) into 1 chunks (chunk_size: 25.0s, overlap: 0.5s)
✓ Chunk 1/1: Process=0.240s, Audio=3.69s, RTF=0.07x, Frames=300
[request_id] ✅ Stream completed - Total: 0.270s, Audio: 3.69s, Chunks: 1
```

**RTF (Real-Time Factor)**: Processing time / Audio duration
- RTF < 1.0 = Faster than real-time ✅
- RTF = 1.0 = Real-time speed
- RTF > 1.0 = Slower than real-time ❌

## Development

### Local Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server (CPU mode)
DEVICE=cpu python fastapi_server.py

# Run tests
python test_api.py
```

### Docker Build

```bash
# Build image
docker compose build

# Run with GPU
docker compose up -d

# Run with CPU only
docker compose -f docker-compose.yml up -d
```

### Project Structure

```
EmoTalk/
├── fastapi_server.py          # Main API server
├── emotalk_processor.py       # Audio processing & model inference
├── model.py                   # EmoTalk PyTorch model
├── wav2vec.py                 # Wav2Vec2 feature extractor
├── utils.py                   # Utility functions
├── test_api.py                # API testing script
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker image (CUDA)
├── Dockerfile.cpu             # Docker image (CPU only)
├── docker-compose.yml         # Docker Compose config
├── pretrain_model/            # Model weights directory
│   └── EmoTalk.pth           # (Download separately)
└── audio/                     # Sample audio files
```

## Configuration

### Environment Variables

- `PORT`: API server port (default: 8000)
- `HOST`: API server host (default: 0.0.0.0)
- `DEVICE`: Device for inference: `cuda` or `cpu` (default: cuda)
- `MODEL_PATH`: Path to model file (default: ./pretrain_model/EmoTalk.pth)

### Docker Compose

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - DEVICE=cuda
  - PORT=8000
  - MODEL_PATH=/app/pretrain_model/EmoTalk.pth
```

## Troubleshooting

### Model Loading Takes Long Time

The model needs to download Wav2Vec2 weights (~1GB) on first run. Subsequent starts will be faster.

### Connection Reset / Server Not Ready

Wait 30-60 seconds after `docker compose up` for the model to fully load. Check logs:

```bash
docker logs emotalk-api --tail 50
```

Look for: `INFO:fastapi_server:EmoTalk processor initialized successfully`

### Out of Memory

Reduce chunk size or use CPU mode:

```bash
DEVICE=cpu python fastapi_server.py
```

### CUDA Not Available

Make sure:
1. NVIDIA drivers are installed
2. `nvidia-docker2` is installed
3. GPU is accessible: `nvidia-smi`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/api/v1/process-audio-stream` | POST | Process audio (streaming) |
| `/docs` | GET | Interactive API documentation |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work, please cite:

```bibtex
@misc{emotalk2024,
  title={EmoTalk: Speech-Driven Emotional 3D Face Animation},
  author={Your Name},
  year={2024},
  url={https://github.com/stresslen/EmoTalk}
}
```

## Support

- 📧 Email: stresslen@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/stresslen/EmoTalk/issues)
- 📖 Documentation: [API_README.md](API_README.md)

---

**Status**: Production Ready ✅
