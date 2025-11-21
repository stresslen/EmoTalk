"""
Kafka Consumer - Xử lý audio requests từ Kafka queue tuần tự
"""
import json
import logging
import threading
import queue
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioRequestConsumer:
    def __init__(self, bootstrap_servers=['localhost:9092'], 
                 topic='audio-requests',
                 group_id='emotalk-processors',
                 processor_callback=None):
        """
        Khởi tạo Kafka Consumer - Xử lý tuần tự nhiều audio requests liên tiếp
        
        Consumer này được thiết kế để:
        - Nhận nhiều audio requests trong thời gian ngắn
        - Xử lý tuần tự từng request một (FIFO)
        - Đảm bảo không miss bất kỳ request nào
        
        Args:
            bootstrap_servers: Danh sách Kafka brokers
            topic: Topic để subscribe
            group_id: Consumer group ID
            processor_callback: Callback function để xử lý message
                               callback(request_data) -> None
        """
        self.topic = topic
        self.group_id = group_id
        self.processor_callback = processor_callback
        self.running = False
        self.consumer_thread = None
        self.processing_queue = queue.Queue()
        
        # Khởi tạo consumer với cấu hình đảm bảo không miss message
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='earliest',  # Đọc từ đầu nếu không có offset
            enable_auto_commit=False,  # Manual commit để đảm bảo xử lý thành công
            max_poll_records=1,  # Xử lý từng message một để đảm bảo tuần tự
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            max_poll_interval_ms=300000  # 5 phút cho việc xử lý
        )
        
        logger.info(f"Kafka Consumer initialized - Topic: {topic}, Group: {group_id}")
    
    def start(self):
        """Bắt đầu consume messages"""
        if self.running:
            logger.warning("Consumer is already running")
            return
        
        self.running = True
        
        # Thread để consume messages từ Kafka
        self.consumer_thread = threading.Thread(target=self._consume_loop, daemon=False)
        self.consumer_thread.start()
        
        # Thread để xử lý messages tuần tự
        self.processor_thread = threading.Thread(target=self._process_loop, daemon=False)
        self.processor_thread.start()
        
        logger.info("Consumer started")
    
    def _consume_loop(self):
        """Loop để consume messages từ Kafka"""
        logger.info("Starting consume loop")
        
        try:
            while self.running:
                # Poll messages
                msg_pack = self.consumer.poll(timeout_ms=1000, max_records=1)
                
                if not msg_pack:
                    continue
                
                for topic_partition, messages in msg_pack.items():
                    for message in messages:
                        logger.info(f"Received message - Partition: {message.partition}, "
                                  f"Offset: {message.offset}, "
                                  f"Key: {message.key}")
                        
                        # Đưa message vào queue để xử lý tuần tự
                        self.processing_queue.put({
                            'message': message,
                            'topic_partition': topic_partition
                        })
                        
        except Exception as e:
            logger.error(f"Error in consume loop: {e}", exc_info=True)
        finally:
            logger.info("Consume loop ended")
    
    def _process_loop(self):
        """Loop để xử lý messages tuần tự"""
        logger.info("Starting process loop")
        
        try:
            while self.running or not self.processing_queue.empty():
                try:
                    # Lấy message từ queue (timeout 1s)
                    item = self.processing_queue.get(timeout=1.0)
                    message = item['message']
                    topic_partition = item['topic_partition']
                    
                    request_data = message.value
                    request_id = request_data['request_id']
                    
                    logger.info(f"Processing request: {request_id}")
                    
                    try:
                        # Parse audio data từ hex string
                        audio_bytes = bytes.fromhex(request_data['audio_data'])
                        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                        
                        # Chuẩn bị data để xử lý
                        processed_data = {
                            'request_id': request_id,
                            'audio': audio_array,
                            'emotion_level': request_data['emotion_level'],
                            'person_id': request_data['person_id'],
                            'post_processing': request_data['post_processing'],
                            'chunk_duration': request_data['chunk_duration'],
                            'chunk_overlap': request_data['chunk_overlap'],
                            'timestamp': request_data['timestamp']
                        }
                        
                        # Gọi callback để xử lý
                        if self.processor_callback:
                            self.processor_callback(processed_data)
                        
                        # Commit offset sau khi xử lý thành công
                        self.consumer.commit_async({
                            topic_partition: message.offset + 1
                        })
                        
                        logger.info(f"Successfully processed request: {request_id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing request {request_id}: {e}", 
                                   exc_info=True)
                        # Có thể implement retry logic hoặc dead letter queue ở đây
                        
                    finally:
                        self.processing_queue.task_done()
                        
                except queue.Empty:
                    continue
                    
        except Exception as e:
            logger.error(f"Error in process loop: {e}", exc_info=True)
        finally:
            logger.info("Process loop ended")
    
    def stop(self):
        """Dừng consumer"""
        logger.info("Stopping consumer...")
        self.running = False
        
        # Đợi xử lý hết messages trong queue
        if hasattr(self, 'processing_queue'):
            self.processing_queue.join()
        
        # Đợi threads kết thúc
        if self.consumer_thread:
            self.consumer_thread.join(timeout=10)
        if self.processor_thread:
            self.processor_thread.join(timeout=10)
        
        # Commit cuối cùng và đóng consumer
        try:
            self.consumer.commit()
            self.consumer.close()
        except Exception as e:
            logger.error(f"Error closing consumer: {e}")
        
        logger.info("Consumer stopped")


if __name__ == "__main__":
    # Test consumer
    def test_processor(request_data):
        """Test callback function"""
        print(f"\n{'='*50}")
        print(f"Processing request: {request_data['request_id']}")
        print(f"Audio shape: {request_data['audio'].shape}")
        print(f"Emotion level: {request_data['emotion_level']}")
        print(f"Person ID: {request_data['person_id']}")
        print(f"Post processing: {request_data['post_processing']}")
        print(f"Chunk duration: {request_data['chunk_duration']}")
        print(f"Chunk overlap: {request_data['chunk_overlap']}")
        print(f"{'='*50}\n")
        
        # Giả lập xử lý mất thời gian
        import time
        time.sleep(2)
    
    consumer = AudioRequestConsumer(
        processor_callback=test_processor
    )
    
    try:
        consumer.start()
        print("Consumer is running. Press Ctrl+C to stop...")
        
        # Keep running
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping consumer...")
        consumer.stop()
