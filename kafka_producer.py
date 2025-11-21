"""
Kafka Producer - Nhận audio requests và đưa vào Kafka queue
"""
import json
import uuid
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioRequestProducer:
    def __init__(self, bootstrap_servers=['localhost:9092'], topic='audio-requests'):
        """
        Khởi tạo Kafka Producer - Nhận và queue nhiều audio requests liên tiếp
        
        Producer này hỗ trợ:
        - Nhận nhiều audio requests liên tiếp trong thời gian ngắn
        - Gửi vào Kafka queue để xử lý tuần tự
        - Đảm bảo thứ tự thông qua message key (request_id)
        
        Args:
            bootstrap_servers: Danh sách Kafka brokers
            topic: Topic để publish messages
        """
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Đảm bảo message được ghi vào tất cả replicas
            retries=3,
            max_in_flight_requests_per_connection=1,  # Đảm bảo thứ tự
            compression_type='gzip'
        )
        logger.info(f"Kafka Producer initialized for topic: {topic}")
    
    def send_audio_request(self, audio_data, emotion_level=1, person_id=0, 
                          post_processing=True, chunk_duration=25.0, 
                          chunk_overlap=0.5, request_id=None):
        """
        Gửi audio request vào Kafka queue
        
        Args:
            audio_data: Audio data dạng bytes
            emotion_level: Mức độ cảm xúc (0 hoặc 1)
            person_id: ID người (0-23)
            post_processing: Có sử dụng post-processing không
            chunk_duration: Độ dài mỗi chunk (giây)
            chunk_overlap: Độ dài overlap (giây)
            request_id: ID của request (tự động tạo nếu không có)
            
        Returns:
            request_id: ID của request đã gửi
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Tạo message
        message = {
            'request_id': request_id,
            'audio_data': audio_data.hex(),  # Convert bytes to hex string
            'emotion_level': emotion_level,
            'person_id': person_id,
            'post_processing': post_processing,
            'chunk_duration': chunk_duration,
            'chunk_overlap': chunk_overlap,
            'timestamp': time.time()
        }
        
        try:
            # Gửi message với key là request_id để đảm bảo ordering
            future = self.producer.send(
                self.topic,
                key=request_id,
                value=message
            )
            
            # Đợi confirmation
            record_metadata = future.get(timeout=10)
            logger.info(f"Message sent successfully - Request ID: {request_id}, "
                       f"Partition: {record_metadata.partition}, "
                       f"Offset: {record_metadata.offset}")
            
            return request_id
            
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    def send_audio_request_with_callback(self, audio_data, emotion_level=1, 
                                        person_id=0, post_processing=True,
                                        chunk_duration=25.0, chunk_overlap=0.5,
                                        request_id=None, callback=None):
        """
        Gửi audio request với callback function
        
        Args:
            callback: Function được gọi khi message được gửi thành công hoặc lỗi
                     callback(request_id, error)
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        message = {
            'request_id': request_id,
            'audio_data': audio_data.hex(),
            'emotion_level': emotion_level,
            'person_id': person_id,
            'post_processing': post_processing,
            'chunk_duration': chunk_duration,
            'chunk_overlap': chunk_overlap,
            'timestamp': time.time()
        }
        
        def on_send_success(record_metadata):
            logger.info(f"Message sent - Request ID: {request_id}, "
                       f"Partition: {record_metadata.partition}, "
                       f"Offset: {record_metadata.offset}")
            if callback:
                callback(request_id, None)
        
        def on_send_error(excp):
            logger.error(f"Failed to send message - Request ID: {request_id}, "
                        f"Error: {excp}")
            if callback:
                callback(request_id, excp)
        
        # Gửi async với callback
        self.producer.send(
            self.topic,
            key=request_id,
            value=message
        ).add_callback(on_send_success).add_errback(on_send_error)
        
        return request_id
    
    def flush(self):
        """Đảm bảo tất cả messages đã được gửi"""
        self.producer.flush()
    
    def close(self):
        """Đóng producer"""
        self.producer.close()
        logger.info("Kafka Producer closed")


if __name__ == "__main__":
    # Test producer
    import numpy as np
    
    producer = AudioRequestProducer()
    
    # Tạo test audio data (giả lập)
    test_audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds
    audio_bytes = test_audio.tobytes()
    
    try:
        request_id = producer.send_audio_request(
            audio_data=audio_bytes,
            emotion_level=1,
            person_id=0,
            post_processing=True,
            chunk_duration=25.0,
            chunk_overlap=0.5
        )
        print(f"Test message sent with request_id: {request_id}")
        
        producer.flush()
        
    finally:
        producer.close()
