"""
gRPC Client - Test client để gửi audio và nhận blendshapes
"""
import grpc
import logging
import time
import sys

import emotalk_pb2
import emotalk_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmoTalkClient:
    def __init__(self, host='localhost', port=50051):
        """
        Khởi tạo gRPC client
        
        Args:
            host: gRPC server host
            port: gRPC server port
        """
        self.channel = grpc.insecure_channel(
            f'{host}:{port}',
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
            ]
        )
        self.stub = emotalk_pb2_grpc.EmoTalkServiceStub(self.channel)
        logger.info(f"Connected to gRPC server at {host}:{port}")
    
    def process_audio_file(self, audio_path, emotion_level=1, person_id=0,
                          post_processing=True, chunk_duration=25.0,
                          chunk_overlap=0.5, request_id=None):
        """
        Gửi audio file và nhận blendshapes
        
        Args:
            audio_path: Đường dẫn đến file audio (WAV format, 16kHz)
            emotion_level: Mức độ cảm xúc (0 hoặc 1)
            person_id: ID người (0-23)
            post_processing: Có sử dụng post-processing không
            chunk_duration: Độ dài mỗi chunk (seconds)
            chunk_overlap: Độ dài overlap (seconds)
            request_id: ID của request (tự động tạo nếu không có)
            
        Returns:
            results: List các chunk results
        """
        import uuid
        
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Đọc audio file
        try:
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
        except Exception as e:
            logger.error(f"Failed to read audio file: {e}")
            return None
        
        logger.info(f"Processing audio file: {audio_path}")
        logger.info(f"Request ID: {request_id}")
        logger.info(f"Audio size: {len(audio_data)} bytes")
        
        # Tạo request
        request = emotalk_pb2.AudioRequest(
            request_id=request_id,
            audio_data=audio_data,
            emotion_level=emotion_level,
            person_id=person_id,
            post_processing=post_processing,
            chunk_duration=chunk_duration,
            chunk_overlap=chunk_overlap
        )
        
        # Gửi request và nhận stream responses
        results = []
        total_frames = 0
        
        try:
            start_time = time.time()
            
            for response in self.stub.ProcessAudio(request):
                if response.error_message:
                    logger.error(f"Error from server: {response.error_message}")
                    return None
                
                chunk_index = response.chunk_index
                num_frames = len(response.frames)
                total_frames += num_frames
                
                logger.info(f"Received chunk {chunk_index}: {num_frames} frames")
                
                # Lưu kết quả
                chunk_result = {
                    'chunk_index': chunk_index,
                    'frames': []
                }
                
                for frame in response.frames:
                    chunk_result['frames'].append({
                        'timecode': frame.timecode,
                        'frame_number': frame.frame_number,
                        'blendshapes': list(frame.blendshapes)
                    })
                
                results.append(chunk_result)
                
                if response.is_final:
                    logger.info("Received final chunk")
                    break
            
            elapsed_time = time.time() - start_time
            logger.info(f"Processing completed in {elapsed_time:.2f}s")
            logger.info(f"Total frames received: {total_frames}")
            
            return results
            
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e.code()} - {e.details()}")
            return None
    
    def health_check(self):
        """
        Kiểm tra health của server
        
        Returns:
            bool: True nếu server healthy
        """
        try:
            request = emotalk_pb2.HealthCheckRequest(service="emotalk")
            response = self.stub.HealthCheck(request)
            
            if response.status == emotalk_pb2.HealthCheckResponse.SERVING:
                logger.info("Server is healthy")
                return True
            else:
                logger.warning(f"Server status: {response.status}")
                return False
                
        except grpc.RpcError as e:
            logger.error(f"Health check failed: {e.code()} - {e.details()}")
            return False
    
    def close(self):
        """Đóng connection"""
        self.channel.close()
        logger.info("Client connection closed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='EmoTalk gRPC Client')
    parser.add_argument('--host', type=str, default='localhost',
                       help='gRPC server host')
    parser.add_argument('--port', type=int, default=50051,
                       help='gRPC server port')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to audio file (WAV format, 16kHz)')
    parser.add_argument('--emotion_level', type=int, default=1,
                       help='Emotion level (0 or 1)')
    parser.add_argument('--person_id', type=int, default=0,
                       help='Person ID (0-23)')
    parser.add_argument('--no_post_processing', action='store_true',
                       help='Disable post-processing')
    parser.add_argument('--chunk_duration', type=float, default=25.0,
                       help='Chunk duration in seconds')
    parser.add_argument('--chunk_overlap', type=float, default=0.5,
                       help='Chunk overlap in seconds')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results (NPY format)')
    
    args = parser.parse_args()
    
    # Tạo client
    client = EmoTalkClient(host=args.host, port=args.port)
    
    try:
        # Health check
        if not client.health_check():
            logger.error("Server is not healthy")
            sys.exit(1)
        
        # Process audio
        results = client.process_audio_file(
            audio_path=args.audio,
            emotion_level=args.emotion_level,
            person_id=args.person_id,
            post_processing=not args.no_post_processing,
            chunk_duration=args.chunk_duration,
            chunk_overlap=args.chunk_overlap
        )
        
        if results is None:
            logger.error("Failed to process audio")
            sys.exit(1)
        
        # Save results if output specified
        if args.output:
            import numpy as np
            
            # Merge all frames
            all_blendshapes = []
            for chunk in results:
                for frame in chunk['frames']:
                    all_blendshapes.append(frame['blendshapes'])
            
            blendshapes_array = np.array(all_blendshapes)
            np.save(args.output, blendshapes_array)
            logger.info(f"Results saved to {args.output}")
            logger.info(f"Shape: {blendshapes_array.shape}")
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total chunks: {len(results)}")
        total_frames = sum(len(chunk['frames']) for chunk in results)
        print(f"Total frames: {total_frames}")
        if total_frames > 0:
            duration = results[-1]['frames'][-1]['timecode']
            print(f"Duration: {duration:.2f} seconds")
        print("="*50 + "\n")
        
    finally:
        client.close()


if __name__ == "__main__":
    main()
