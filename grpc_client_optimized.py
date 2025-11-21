"""
Optimized gRPC Client
- Hỗ trợ multiple concurrent requests
- Async streaming
- Connection pooling
"""
import grpc
import asyncio
import logging
import time
import io
from typing import List, Callable, Optional
import numpy as np

import emotalk_pb2
import emotalk_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedEmoTalkClient:
    """
    Optimized gRPC client với connection pooling
    """
    def __init__(self, server_address='localhost:50051', 
                 max_concurrent_streams=10):
        """
        Khởi tạo client
        
        Args:
            server_address: Địa chỉ server
            max_concurrent_streams: Số stream đồng thời tối đa
        """
        self.server_address = server_address
        self.max_concurrent_streams = max_concurrent_streams
        
        # Channel options để tối ưu
        self.channel_options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
        ]
        
        # Tạo channel và stub
        self.channel = grpc.insecure_channel(
            server_address,
            options=self.channel_options
        )
        self.stub = emotalk_pb2_grpc.EmoTalkServiceStub(self.channel)
        
        logger.info(f"Client connected to {server_address}")
    
    def process_audio_file(self, 
                          audio_path: str,
                          request_id: Optional[str] = None,
                          emotion_level: int = 1,
                          person_id: int = 0,
                          post_processing: bool = True,
                          chunk_duration: float = 25.0,
                          chunk_overlap: float = 0.5,
                          callback: Optional[Callable] = None) -> List[dict]:
        """
        Xử lý audio file và stream results
        
        Args:
            audio_path: Đường dẫn đến file audio
            request_id: ID của request (optional)
            emotion_level: Mức độ cảm xúc (0 hoặc 1)
            person_id: ID người (0-23)
            post_processing: Có sử dụng post-processing không
            chunk_duration: Độ dài mỗi chunk (seconds)
            chunk_overlap: Độ dài overlap (seconds)
            callback: Function để callback khi nhận chunk
            
        Returns:
            List các chunks (nếu không có callback)
        """
        start_time = time.time()
        
        # Đọc audio file
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        logger.info(f"Processing {audio_path} ({len(audio_data)} bytes)")
        
        # Tạo request
        request = emotalk_pb2.AudioRequest(
            request_id=request_id or "",
            audio_data=audio_data,
            emotion_level=emotion_level,
            person_id=person_id,
            post_processing=post_processing,
            chunk_duration=chunk_duration,
            chunk_overlap=chunk_overlap
        )
        
        # Stream responses
        chunks = []
        chunk_count = 0
        
        try:
            for response in self.stub.ProcessAudio(request):
                # Check for errors
                if response.error_message:
                    logger.error(f"Server error: {response.error_message}")
                    raise Exception(response.error_message)
                
                chunk_count += 1
                
                # Convert response to dict
                chunk_data = {
                    'request_id': response.request_id,
                    'chunk_index': response.chunk_index,
                    'is_final': response.is_final,
                    'frames': []
                }
                
                for frame in response.frames:
                    chunk_data['frames'].append({
                        'timecode': frame.timecode,
                        'frame_number': frame.frame_number,
                        'blendshapes': list(frame.blendshapes)
                    })
                
                logger.info(f"Received chunk {chunk_count}: "
                           f"{len(chunk_data['frames'])} frames, "
                           f"is_final={chunk_data['is_final']}")
                
                # Callback
                if callback:
                    callback(chunk_data)
                else:
                    chunks.append(chunk_data)
                
                if response.is_final:
                    break
            
            total_time = time.time() - start_time
            logger.info(f"✅ Processing completed in {total_time:.3f}s, "
                       f"{chunk_count} chunks received")
            
            return chunks if not callback else None
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()}: {e.details()}")
            raise
    
    def process_audio_file_generator(self,
                                     audio_path: str,
                                     request_id: Optional[str] = None,
                                     emotion_level: int = 1,
                                     person_id: int = 0,
                                     post_processing: bool = True,
                                     chunk_duration: float = 25.0,
                                     chunk_overlap: float = 0.5):
        """
        Generator version - yield chunks instead of callback
        For use in FastAPI gateway streaming
        """
        # Đọc audio file
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        # Tạo request
        request = emotalk_pb2.AudioRequest(
            request_id=request_id or "",
            audio_data=audio_data,
            emotion_level=emotion_level,
            person_id=person_id,
            post_processing=post_processing,
            chunk_duration=chunk_duration,
            chunk_overlap=chunk_overlap
        )
        
        # Stream responses
        try:
            for response in self.stub.ProcessAudio(request):
                # Check for errors
                if response.error_message:
                    raise Exception(response.error_message)
                
                # Convert response to dict
                chunk_data = {
                    'request_id': response.request_id,
                    'chunk_index': response.chunk_index,
                    'is_final': response.is_final,
                    'frames': []
                }
                
                for frame in response.frames:
                    chunk_data['frames'].append({
                        'timecode': frame.timecode,
                        'blendshapes': list(frame.blendshapes)
                    })
                
                yield chunk_data
                
                if response.is_final:
                    break
                    
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()}: {e.details()}")
            raise
    
    def health_check(self) -> bool:
        """
        Kiểm tra server health
        
        Returns:
            True nếu server healthy
        """
        try:
            request = emotalk_pb2.HealthCheckRequest()
            response = self.stub.HealthCheck(request, timeout=5.0)
            
            is_serving = (response.status == 
                         emotalk_pb2.HealthCheckResponse.SERVING)
            
            logger.info(f"Health check: {'SERVING' if is_serving else 'NOT_SERVING'}")
            return is_serving
            
        except grpc.RpcError as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def close(self):
        """Đóng connection"""
        self.channel.close()
        logger.info("Client connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


async def process_multiple_files_async(client: OptimizedEmoTalkClient,
                                       audio_files: List[str],
                                       **kwargs):
    """
    Xử lý nhiều files đồng thời với asyncio
    
    Args:
        client: OptimizedEmoTalkClient instance
        audio_files: List các đường dẫn audio files
        **kwargs: Arguments cho process_audio_file
    
    Returns:
        List kết quả
    """
    import concurrent.futures
    
    loop = asyncio.get_event_loop()
    
    # Tạo thread pool để xử lý đồng thời
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(audio_files)
    ) as executor:
        
        tasks = []
        for audio_file in audio_files:
            task = loop.run_in_executor(
                executor,
                client.process_audio_file,
                audio_file,
                None,  # request_id
                kwargs.get('emotion_level', 1),
                kwargs.get('person_id', 0),
                kwargs.get('post_processing', True),
                kwargs.get('chunk_duration', 25.0),
                kwargs.get('chunk_overlap', 0.5),
                None  # callback
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results


def main():
    """Test client"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized EmoTalk gRPC Client')
    parser.add_argument('audio_file', type=str, 
                       help='Path to audio file')
    parser.add_argument('--server', type=str, default='localhost:50051',
                       help='Server address')
    parser.add_argument('--emotion_level', type=int, default=1,
                       choices=[0, 1], help='Emotion level')
    parser.add_argument('--person_id', type=int, default=0,
                       help='Person ID (0-23)')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Test client
    with OptimizedEmoTalkClient(args.server) as client:
        # Health check
        if not client.health_check():
            logger.error("Server is not healthy")
            return
        
        # Process audio
        chunks = client.process_audio_file(
            args.audio_file,
            emotion_level=args.emotion_level,
            person_id=args.person_id,
            post_processing=True
        )
        
        # Save output if specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(chunks, f, indent=2)
            logger.info(f"Saved output to {args.output}")
        
        # Print summary
        total_frames = sum(len(chunk['frames']) for chunk in chunks)
        logger.info(f"Total: {len(chunks)} chunks, {total_frames} frames")


if __name__ == "__main__":
    main()
