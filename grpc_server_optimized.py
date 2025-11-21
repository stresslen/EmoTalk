"""
Optimized gRPC Server với Queue System
- Xử lý nhiều requests đồng thời
- Queue để đảm bảo không miss request
- Worker pool để xử lý tuần tự nhưng hiệu quả
"""
import grpc
from concurrent import futures
import logging
import threading
import queue
import time
import uuid
import numpy as np
import librosa
import io
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Import generated protobuf files
import emotalk_pb2
import emotalk_pb2_grpc

from emotalk_processor import EmoTalkProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AudioRequest:
    """Request wrapper để quản lý trong queue"""
    request_id: str
    audio_array: np.ndarray
    emotion_level: int
    person_id: int
    post_processing: bool
    chunk_duration: float
    chunk_overlap: float
    response_queue: queue.Queue
    context: Any  # gRPC context
    timestamp: float


class RequestQueue:
    """
    Queue system để quản lý requests
    - FIFO queue để xử lý tuần tự
    - Thread-safe
    - Hỗ trợ priority nếu cần
    """
    def __init__(self, maxsize=100):
        self.queue = queue.Queue(maxsize=maxsize)
        self.active_requests = {}  # request_id -> AudioRequest
        self.lock = threading.Lock()
        self._stopped = False
        
        # Metrics
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.queue_times = deque(maxlen=100)  # Track queue wait times
    
    def put(self, request: AudioRequest, block=True, timeout=None):
        """Thêm request vào queue"""
        try:
            self.queue.put(request, block=block, timeout=timeout)
            with self.lock:
                self.active_requests[request.request_id] = request
                self.total_requests += 1
            logger.info(f"[{request.request_id}] Request queued. Queue size: {self.queue.qsize()}")
            return True
        except queue.Full:
            logger.error(f"[{request.request_id}] Queue is full! Cannot accept request.")
            return False
    
    def get(self, block=True, timeout=None):
        """Lấy request từ queue"""
        try:
            request = self.queue.get(block=block, timeout=timeout)
            
            # Tính queue wait time
            wait_time = time.time() - request.timestamp
            self.queue_times.append(wait_time)
            
            logger.info(f"[{request.request_id}] Request dequeued. "
                       f"Wait time: {wait_time:.3f}s, Queue size: {self.queue.qsize()}")
            return request
        except queue.Empty:
            return None
    
    def mark_completed(self, request_id: str, success=True):
        """Đánh dấu request đã hoàn thành"""
        with self.lock:
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            if success:
                self.completed_requests += 1
            else:
                self.failed_requests += 1
    
    def get_stats(self):
        """Lấy thống kê queue"""
        with self.lock:
            avg_queue_time = sum(self.queue_times) / len(self.queue_times) if self.queue_times else 0
            return {
                'queue_size': self.queue.qsize(),
                'active_requests': len(self.active_requests),
                'total_requests': self.total_requests,
                'completed_requests': self.completed_requests,
                'failed_requests': self.failed_requests,
                'avg_queue_wait_time': avg_queue_time,
                'success_rate': self.completed_requests / self.total_requests if self.total_requests > 0 else 0
            }
    
    def stop(self):
        """Dừng queue"""
        self._stopped = True


class ProcessorWorker(threading.Thread):
    """
    Worker thread để xử lý requests từ queue
    Mỗi worker xử lý tuần tự các requests được assign
    """
    def __init__(self, worker_id: int, processor: EmoTalkProcessor, 
                 request_queue: RequestQueue):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.processor = processor
        self.request_queue = request_queue
        self._stopped = False
        
        # Metrics
        self.requests_processed = 0
        self.total_processing_time = 0.0
        
        logger.info(f"Worker {worker_id} initialized")
    
    def run(self):
        """Main worker loop"""
        logger.info(f"Worker {self.worker_id} started")
        
        while not self._stopped:
            try:
                # Lấy request từ queue với timeout
                request = self.request_queue.get(block=True, timeout=1.0)
                
                if request is None:
                    continue
                
                # Kiểm tra client còn kết nối không
                if not request.context.is_active():
                    logger.warning(f"[{request.request_id}] Client disconnected, skipping")
                    self.request_queue.mark_completed(request.request_id, success=False)
                    continue
                
                # Xử lý request
                self._process_request(request)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}", exc_info=True)
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def _process_request(self, request: AudioRequest):
        """Xử lý một request"""
        request_id = request.request_id
        start_time = time.time()
        
        try:
            logger.info(f"[{request_id}] Worker {self.worker_id} processing...")
            
            # Clear GPU cache trước khi xử lý
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process audio với chunking
            chunk_count = 0
            for chunk_result in self.processor.process_full_audio(
                request.audio_array,
                emotion_level=request.emotion_level,
                person_id=request.person_id,
                post_processing=request.post_processing,
                chunk_duration=request.chunk_duration,
                overlap=request.chunk_overlap,
                sample_rate=16000
            ):
                # Kiểm tra client còn kết nối
                if not request.context.is_active():
                    logger.warning(f"[{request_id}] Client disconnected during processing")
                    break
                
                # Tạo response và put vào response queue
                frames = []
                blendshapes = chunk_result['blendshapes']
                start_time_chunk = chunk_result['start_time']
                fps = 30.0
                
                for frame_idx in range(blendshapes.shape[0]):
                    timecode = start_time_chunk + (frame_idx / fps)
                    frame_msg = emotalk_pb2.BlendshapeFrame(
                        timecode=timecode,
                        frame_number=frame_idx,
                        blendshapes=blendshapes[frame_idx].tolist()
                    )
                    frames.append(frame_msg)
                
                response = emotalk_pb2.BlendshapeResponse(
                    request_id=request_id,
                    chunk_index=chunk_result['chunk_index'],
                    frames=frames,
                    is_final=chunk_result['is_final']
                )
                
                # Put vào response queue
                request.response_queue.put(response)
                chunk_count += 1
                
                logger.info(f"[{request_id}] Chunk {chunk_count} sent "
                           f"({chunk_result['start_time']:.2f}s-{chunk_result['end_time']:.2f}s, "
                           f"{len(frames)} frames)")
            
            # Signal completion
            request.response_queue.put(None)
            
            processing_time = time.time() - start_time
            self.requests_processed += 1
            self.total_processing_time += processing_time
            
            logger.info(f"[{request_id}] ✅ Completed by worker {self.worker_id} "
                       f"in {processing_time:.3f}s, {chunk_count} chunks")
            
            # Clear GPU cache sau khi xử lý xong
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.request_queue.mark_completed(request_id, success=True)
            
        except Exception as e:
            logger.error(f"[{request_id}] Error processing: {e}", exc_info=True)
            
            # Send error response
            error_response = emotalk_pb2.BlendshapeResponse(
                request_id=request_id,
                error_message=str(e)
            )
            request.response_queue.put(error_response)
            request.response_queue.put(None)
            
            self.request_queue.mark_completed(request_id, success=False)
    
    def stop(self):
        """Dừng worker"""
        self._stopped = True
    
    def get_stats(self):
        """Lấy thống kê worker"""
        avg_time = (self.total_processing_time / self.requests_processed 
                   if self.requests_processed > 0 else 0)
        return {
            'worker_id': self.worker_id,
            'requests_processed': self.requests_processed,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': avg_time
        }


class EmoTalkServicer(emotalk_pb2_grpc.EmoTalkServiceServicer):
    """
    gRPC Servicer với queue system
    """
    def __init__(self, request_queue: RequestQueue, num_workers=2):
        self.request_queue = request_queue
        self.workers = []
        self.num_workers = num_workers
        
        logger.info(f"EmoTalkServicer initialized with {num_workers} workers")
    
    def start_workers(self, processor: EmoTalkProcessor):
        """Khởi động worker pool"""
        for i in range(self.num_workers):
            worker = ProcessorWorker(i, processor, self.request_queue)
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {len(self.workers)} workers")
    
    def stop_workers(self):
        """Dừng tất cả workers"""
        logger.info("Stopping workers...")
        for worker in self.workers:
            worker.stop()
        
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        logger.info("All workers stopped")
    
    def ProcessAudio(self, request, context):
        """
        Xử lý audio request với queue system
        """
        request_id = request.request_id or str(uuid.uuid4())
        
        logger.info(f"[{request_id}] Received ProcessAudio request")
        
        try:
            # Parse audio data
            audio_bytes = request.audio_data
            if not audio_bytes:
                error_msg = "Audio data is empty"
                logger.error(f"[{request_id}] {error_msg}")
                yield emotalk_pb2.BlendshapeResponse(
                    request_id=request_id,
                    error_message=error_msg
                )
                return
            
            # Convert bytes to numpy array
            try:
                audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            except Exception as e:
                error_msg = f"Failed to parse audio: {e}"
                logger.error(f"[{request_id}] {error_msg}")
                yield emotalk_pb2.BlendshapeResponse(
                    request_id=request_id,
                    error_message=error_msg
                )
                return
            
            # Validate parameters
            emotion_level = request.emotion_level if request.emotion_level in [0, 1] else 1
            person_id = request.person_id if 0 <= request.person_id <= 23 else 0
            post_processing = request.post_processing
            chunk_duration = request.chunk_duration if request.chunk_duration > 0 else 25.0
            chunk_overlap = request.chunk_overlap if request.chunk_overlap >= 0 else 0.5
            
            audio_duration = len(audio_array) / 16000
            logger.info(f"[{request_id}] Audio: {audio_duration:.2f}s, "
                       f"emotion={emotion_level}, person={person_id}")
            
            # Tạo response queue
            response_queue = queue.Queue()
            
            # Tạo AudioRequest và put vào queue
            audio_request = AudioRequest(
                request_id=request_id,
                audio_array=audio_array,
                emotion_level=emotion_level,
                person_id=person_id,
                post_processing=post_processing,
                chunk_duration=chunk_duration,
                chunk_overlap=chunk_overlap,
                response_queue=response_queue,
                context=context,
                timestamp=time.time()
            )
            
            # Put vào request queue
            if not self.request_queue.put(audio_request, block=False):
                error_msg = "Server is busy, queue is full. Please try again later."
                logger.error(f"[{request_id}] {error_msg}")
                yield emotalk_pb2.BlendshapeResponse(
                    request_id=request_id,
                    error_message=error_msg
                )
                return
            
            # Stream responses từ response queue
            while True:
                try:
                    response = response_queue.get(timeout=60.0)  # Timeout 60s
                    
                    if response is None:
                        # Processing completed
                        break
                    
                    yield response
                    
                except queue.Empty:
                    logger.error(f"[{request_id}] Response timeout")
                    break
                except Exception as e:
                    logger.error(f"[{request_id}] Error streaming response: {e}")
                    break
            
        except Exception as e:
            error_msg = f"Error processing audio: {e}"
            logger.error(f"[{request_id}] {error_msg}", exc_info=True)
            yield emotalk_pb2.BlendshapeResponse(
                request_id=request_id,
                error_message=error_msg
            )
    
    def HealthCheck(self, request, context):
        """Health check endpoint"""
        return emotalk_pb2.HealthCheckResponse(
            status=emotalk_pb2.HealthCheckResponse.SERVING
        )
    
    def GetStats(self, request, context):
        """
        Lấy thống kê server (custom endpoint)
        """
        queue_stats = self.request_queue.get_stats()
        worker_stats = [w.get_stats() for w in self.workers]
        
        logger.info(f"Stats requested - Queue: {queue_stats}")
        
        # Trả về stats dưới dạng JSON string trong error_message field
        # (hoặc định nghĩa message type riêng trong proto)
        import json
        stats = {
            'queue': queue_stats,
            'workers': worker_stats
        }
        
        return emotalk_pb2.BlendshapeResponse(
            request_id="stats",
            error_message=json.dumps(stats, indent=2)
        )


def serve(port=50051, model_path="./pretrain_model/EmoTalk.pth", 
          device="cuda", num_workers=2, queue_size=100):
    """
    Khởi động optimized gRPC server
    
    Args:
        port: Port để listen
        model_path: Đường dẫn đến pretrained model
        device: Device để chạy model
        num_workers: Số lượng worker threads
        queue_size: Kích thước tối đa của queue
    """
    # Khởi tạo processor (shared giữa workers)
    logger.info("Initializing EmoTalk processor...")
    processor = EmoTalkProcessor(model_path=model_path, device=device)
    
    # Khởi tạo request queue
    logger.info(f"Initializing request queue (size={queue_size})...")
    request_queue = RequestQueue(maxsize=queue_size)
    
    # Khởi tạo servicer
    servicer = EmoTalkServicer(request_queue, num_workers=num_workers)
    
    # Khởi động workers
    servicer.start_workers(processor)
    
    # Tạo gRPC server với timeout settings
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.so_reuseport', 1),
            ('grpc.use_local_subchannel_pool', 1),
            ('grpc.keepalive_time_ms', 30000),  # 30s
            ('grpc.keepalive_timeout_ms', 10000),  # 10s
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 5000),
        ]
    )
    
    # Add servicer
    emotalk_pb2_grpc.add_EmoTalkServiceServicer_to_server(servicer, server)
    
    # Listen
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    logger.info(f"🚀 Optimized gRPC server started on port {port}")
    logger.info(f"   - Workers: {num_workers}")
    logger.info(f"   - Queue size: {queue_size}")
    logger.info(f"   - Device: {device}")
    
    # Stats reporter thread
    def report_stats():
        while True:
            time.sleep(30)  # Report every 30s
            stats = request_queue.get_stats()
            logger.info(f"📊 Stats - Queue: {stats['queue_size']}, "
                       f"Active: {stats['active_requests']}, "
                       f"Completed: {stats['completed_requests']}, "
                       f"Success rate: {stats['success_rate']:.2%}, "
                       f"Avg wait: {stats['avg_queue_wait_time']:.3f}s")
    
    stats_thread = threading.Thread(target=report_stats, daemon=True)
    stats_thread.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        servicer.stop_workers()
        request_queue.stop()
        server.stop(0)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized EmoTalk gRPC Server')
    parser.add_argument('--port', type=int, default=50051, 
                       help='Port to listen on')
    parser.add_argument('--model_path', type=str, 
                       default='./pretrain_model/EmoTalk.pth',
                       help='Path to pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run model (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of worker threads')
    parser.add_argument('--queue_size', type=int, default=100,
                       help='Maximum queue size')
    
    args = parser.parse_args()
    
    serve(
        port=args.port,
        model_path=args.model_path,
        device=args.device,
        num_workers=args.num_workers,
        queue_size=args.queue_size
    )
