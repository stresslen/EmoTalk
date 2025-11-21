# EmoTalk API với Kafka và gRPC

API xử lý audio đầu vào và trả về 52 blendshapes với timecode theo stream chunks.

## Tính năng

- ✅ **Kafka Integration**: Nhận và xử lý liên tiếp nhiều audio trong thời gian ngắn, đảm bảo không bị miss thông tin
- ✅ **Sequential Processing**: Xử lý tuần tự từng audio một, đảm bảo thứ tự và chất lượng
- ✅ **gRPC Streaming**: Stream blendshapes theo chunks với timecode
- ✅ **Chunking với Overlap**: Chia audio thành chunks 25s với overlap nhỏ để tránh miss thông tin
- ✅ **52 Blendshapes**: Output đầy đủ 52 blendshapes cho mỗi frame (30 fps)
- ✅ **Post-processing**: Smoothing và blinking tự động

## Kiến trúc

```
Multiple Clients (gRPC)
    ↓ ↓ ↓ (nhiều audio liên tiếp)
gRPC Server → Kafka Queue (ordered, persistent)
                    ↓
            Kafka Consumer (sequential processing)
                    ↓
            EmoTalk Processor (xử lý từng audio)
                    ↓
            Stream Blendshapes (chunks 25s + overlap)
```

### Quy trình xử lý:

1. **Nhận audio liên tiếp**: Client gửi nhiều audio requests trong thời gian ngắn
2. **Kafka Queue**: Tất cả requests được lưu vào queue theo thứ tự
3. **Sequential Processing**: Consumer xử lý từng audio một, đảm bảo:
   - Không bị miss bất kỳ request nào
   - Xử lý đúng thứ tự (FIFO)
   - Chỉ commit offset sau khi hoàn thành
4. **Streaming Response**: Blendshapes được stream về client ngay khi từng chunk hoàn thành

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Khởi động Kafka (Docker)

```bash
docker-compose up -d
```

Kiểm tra Kafka UI tại: http://localhost:8080

### 3. Generate gRPC code

```bash
chmod +x generate_grpc.sh
./generate_grpc.sh
```

## Sử dụng

### 1. Khởi động gRPC Server

```bash
python grpc_server.py --port 50051 --device cuda --model_path ./pretrain_model/EmoTalk.pth
```

Tham số:
- `--port`: Port để listen (mặc định: 50051)
- `--device`: Device để chạy model (cuda/cpu)
- `--model_path`: Đường dẫn đến pretrained model
- `--kafka_servers`: Kafka bootstrap servers (mặc định: localhost:9092)

### 2. Test với Client

```bash
python grpc_client.py \
    --audio ./audio/angry1.wav \
    --emotion_level 1 \
    --person_id 0 \
    --chunk_duration 25.0 \
    --chunk_overlap 0.5 \
    --output ./result/output.npy
```

Tham số:
- `--audio`: Đường dẫn file audio (WAV format, 16kHz)
- `--emotion_level`: Mức độ cảm xúc (0 hoặc 1)
- `--person_id`: ID người (0-23)
- `--chunk_duration`: Độ dài mỗi chunk (giây, mặc định: 25.0)
- `--chunk_overlap`: Độ dài overlap (giây, mặc định: 0.5)
- `--output`: File output để lưu kết quả (NPY format)
- `--no_post_processing`: Tắt post-processing

### 3. Khởi động Kafka Consumer (Optional)

Nếu muốn xử lý riêng biệt thông qua Kafka:

```bash
python kafka_consumer.py
```

### 4. Test Kafka Producer (Optional)

```bash
python kafka_producer.py
```

## API Response Format

### BlendshapeResponse (Stream)

```protobuf
message BlendshapeResponse {
  string request_id = 1;           // ID của request
  int32 chunk_index = 2;           // Index của chunk hiện tại
  repeated BlendshapeFrame frames = 3;  // Danh sách frames
  bool is_final = 4;               // Có phải chunk cuối không
  string error_message = 5;        // Thông báo lỗi nếu có
}
```

### BlendshapeFrame

```protobuf
message BlendshapeFrame {
  double timecode = 1;             // Timecode của frame (seconds)
  int32 frame_number = 2;          // Số thứ tự frame
  repeated float blendshapes = 3;  // 52 giá trị blendshapes
}
```

## Xử lý Chunking

- Audio được chia thành các chunks với độ dài configurable (mặc định: 25s)
- Mỗi chunk có overlap nhỏ (mặc định: 0.5s) để tránh miss thông tin ở biên
- Blendshapes được stream ngay khi mỗi chunk được xử lý xong
- Timecode được tính chính xác cho từng frame

## Đảm bảo không miss thông tin khi nhận liên tiếp nhiều audio

### 1. Kafka Queue Management
- **Persistent Queue**: Tất cả audio requests được lưu trữ trong Kafka
- **Manual Commit**: Consumer chỉ commit offset sau khi xử lý thành công hoàn toàn
- **Replication**: Kafka có thể cấu hình replication để đảm bảo data safety

### 2. Sequential Processing
- **FIFO Queue**: Xử lý theo thứ tự First-In-First-Out
- **Single Consumer Thread**: Xử lý từng audio một, không bị race condition
- **Blocking Processing**: Consumer đợi xử lý xong audio hiện tại mới lấy audio tiếp theo

### 3. Audio Chunking
- **Chunk Overlap (0.5s mặc định)**: Overlap giữa các chunks để đảm bảo continuity
- **No Frame Loss**: Mỗi frame audio đều được xử lý đầy đủ

### 4. Error Handling
- **Retry Logic**: Tự động retry khi gặp lỗi tạm thời
- **Error Tracking**: Log chi tiết mọi lỗi để debug
- **Dead Letter Queue**: Có thể implement DLQ cho các request thất bại nhiều lần

## Monitoring

- Kafka UI: http://localhost:8080
- Xem logs của server/consumer để theo dõi processing

## Tắt services

```bash
docker-compose down
```

## Ví dụ sử dụng trong Python

```python
from grpc_client import EmoTalkClient
import numpy as np

# Tạo client
client = EmoTalkClient(host='localhost', port=50051)

# Process audio
results = client.process_audio_file(
    audio_path='./audio/test.wav',
    emotion_level=1,
    person_id=0,
    post_processing=True,
    chunk_duration=25.0,
    chunk_overlap=0.5
)

# Extract blendshapes
all_blendshapes = []
for chunk in results:
    for frame in chunk['frames']:
        timecode = frame['timecode']
        blendshapes = frame['blendshapes']  # 52 values
        all_blendshapes.append(blendshapes)

# Convert to numpy array
blendshapes_array = np.array(all_blendshapes)
print(f"Shape: {blendshapes_array.shape}")  # (num_frames, 52)

# Save results
np.save('output.npy', blendshapes_array)

client.close()
```

## Troubleshooting

### Kafka connection errors
- Đảm bảo Docker containers đang chạy: `docker-compose ps`
- Kiểm tra logs: `docker-compose logs kafka`

### gRPC errors
- Kiểm tra server đang chạy và port đúng
- Kiểm tra file audio format (phải là WAV, 16kHz)

### CUDA out of memory
- Giảm `chunk_duration` để xử lý chunks nhỏ hơn
- Sử dụng `--device cpu` nếu cần

## License

Theo license của EmoTalk gốc
