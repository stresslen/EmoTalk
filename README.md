# EmoTalk - Audio to 52 Blendshapes

Hệ thống xử lý audio và tạo 52 blendshapes cho facial animation với gRPC + Queue system.

## Đóng góp của tôi

- Tối ưu hóa hiệu suất gRPC server với streaming và batch processing
- Xây dựng FastAPI Gateway với hệ thống queue để xử lý concurrent requests
- Triển khai Docker containerization cho deployment
- Cải thiện xử lý audio với normalization và feature extraction
- Tích hợp logging và monitoring system

## Quick Deploy

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU với CUDA support
- Model file: `pretrain_model/EmoTalk.pth`

### Deploy

```bash
./deploy.sh
```

