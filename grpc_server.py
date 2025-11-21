"""
gRPC Server - Xử lý audio requests và stream blendshapes
"""
import grpc
from concurrent import futures
import logging
import threading
import queue
import numpy as np
import librosa
import io

# Import generated protobuf files
import emotalk_pb2
import emotalk_pb2_grpc

from emotalk_processor import EmoTalkProcessor
from kafka_producer import AudioRequestProducer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmoTalkServicer(emotalk_pb2_grpc.EmoTalkServiceServicer):
    def __init__(self, processor, kafka_producer):
        """
        Khởi tạo gRPC servicer
        
        Args:
            processor: EmoTalkProcessor instance
            kafka_producer: AudioRequestProducer instance
        """
        self.processor = processor
        self.kafka_producer = kafka_producer
        self.active_streams = {}  # request_id -> queue
        self.lock = threading.Lock()
        
        logger.info("EmoTalkServicer initialized")
    
    def ProcessAudio(self, request, context):
        """
        Xử lý audio request và stream blendshapes
        
        Args:
            request: AudioRequest message
            context: gRPC context
            
        Yields:
            BlendshapeResponse messages
        """
        request_id = request.request_id
        if not request_id:
            request_id = str(uuid.uuid4())
        
        logger.info(f"Received ProcessAudio request: {request_id}")
        
        try:
            # Parse audio data
            audio_bytes = request.audio_data
            if not audio_bytes:
                error_msg = "Audio data is empty"
                logger.error(error_msg)
                yield emotalk_pb2.BlendshapeResponse(
                    request_id=request_id,
                    error_message=error_msg
                )
                return
            
            # Convert bytes to numpy array (giả sử audio là WAV format)
            try:
                audio_array, sample_rate = librosa.load(
                    io.BytesIO(audio_bytes), 
                    sr=16000
                )
            except Exception as e:
                error_msg = f"Failed to parse audio: {e}"
                logger.error(error_msg)
                yield emotalk_pb2.BlendshapeResponse(
                    request_id=request_id,
                    error_message=error_msg
                )
                return
            
            # Validate parameters
            emotion_level = request.emotion_level
            if emotion_level not in [0, 1]:
                emotion_level = 1
            
            person_id = request.person_id
            if person_id < 0 or person_id > 23:
                person_id = 0
            
            post_processing = request.post_processing
            chunk_duration = request.chunk_duration if request.chunk_duration > 0 else 25.0
            chunk_overlap = request.chunk_overlap if request.chunk_overlap > 0 else 0.5
            
            logger.info(f"Processing audio: {len(audio_array)/16000:.2f}s, "
                       f"emotion_level={emotion_level}, person_id={person_id}, "
                       f"post_processing={post_processing}, "
                       f"chunk_duration={chunk_duration}s, overlap={chunk_overlap}s")
            
            # Process audio with chunking
            for chunk_result in self.processor.process_full_audio(
                audio_array,
                emotion_level=emotion_level,
                person_id=person_id,
                post_processing=post_processing,
                chunk_duration=chunk_duration,
                overlap=chunk_overlap,
                sample_rate=16000
            ):
                # Tạo BlendshapeFrame messages
                frames = []
                blendshapes = chunk_result['blendshapes']
                start_time = chunk_result['start_time']
                fps = 30.0  # EmoTalk outputs 30 fps
                
                for frame_idx in range(blendshapes.shape[0]):
                    timecode = start_time + (frame_idx / fps)
                    
                    frame_msg = emotalk_pb2.BlendshapeFrame(
                        timecode=timecode,
                        frame_number=frame_idx,
                        blendshapes=blendshapes[frame_idx].tolist()
                    )
                    frames.append(frame_msg)
                
                # Tạo BlendshapeResponse
                response = emotalk_pb2.BlendshapeResponse(
                    request_id=request_id,
                    chunk_index=chunk_result['chunk_index'],
                    frames=frames,
                    is_final=chunk_result['is_final']
                )
                
                logger.info(f"Streaming chunk {chunk_result['chunk_index']} "
                           f"with {len(frames)} frames")
                
                yield response
                
                # Check if client disconnected
                if context.is_active() is False:
                    logger.warning(f"Client disconnected: {request_id}")
                    break
            
            logger.info(f"Completed processing request: {request_id}")
            
        except Exception as e:
            error_msg = f"Error processing audio: {e}"
            logger.error(error_msg, exc_info=True)
            yield emotalk_pb2.BlendshapeResponse(
                request_id=request_id,
                error_message=error_msg
            )
    
    def HealthCheck(self, request, context):
        """
        Health check endpoint
        
        Args:
            request: HealthCheckRequest
            context: gRPC context
            
        Returns:
            HealthCheckResponse
        """
        logger.debug("Health check request received")
        return emotalk_pb2.HealthCheckResponse(
            status=emotalk_pb2.HealthCheckResponse.SERVING
        )


def serve(port=50051, model_path="./pretrain_model/EmoTalk.pth", device="cuda",
          kafka_bootstrap_servers=['localhost:9092']):
    """
    Khởi động gRPC server
    
    Args:
        port: Port để listen
        model_path: Đường dẫn đến pretrained model
        device: Device để chạy model
        kafka_bootstrap_servers: Kafka brokers
    """
    # Khởi tạo processor
    logger.info("Initializing EmoTalk processor...")
    processor = EmoTalkProcessor(model_path=model_path, device=device)
    
    # Khởi tạo Kafka producer
    logger.info("Initializing Kafka producer...")
    kafka_producer = AudioRequestProducer(bootstrap_servers=kafka_bootstrap_servers)
    
    # Tạo gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
        ]
    )
    
    # Add servicer
    emotalk_pb2_grpc.add_EmoTalkServiceServicer_to_server(
        EmoTalkServicer(processor, kafka_producer),
        server
    )
    
    # Listen
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    logger.info(f"gRPC server started on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)
        kafka_producer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EmoTalk gRPC Server')
    parser.add_argument('--port', type=int, default=50051, help='Port to listen on')
    parser.add_argument('--model_path', type=str, default='./pretrain_model/EmoTalk.pth',
                       help='Path to pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run model (cuda/cpu)')
    parser.add_argument('--kafka_servers', type=str, default='localhost:9092',
                       help='Kafka bootstrap servers (comma-separated)')
    
    args = parser.parse_args()
    
    kafka_servers = args.kafka_servers.split(',')
    
    serve(
        port=args.port,
        model_path=args.model_path,
        device=args.device,
        kafka_bootstrap_servers=kafka_servers
    )
