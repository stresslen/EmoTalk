"""
EmoTalk Processor - Xử lý audio và tạo blendshapes với chunking
"""
import torch
import librosa
import numpy as np
from scipy.signal import savgol_filter
import random
import logging
import time
from model import EmoTalk
import argparse

logging.basicConfig(level=logging.INFO)
logging.getLogger('transformers').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class EmoTalkProcessor:
    def __init__(self, model_path="./pretrain_model/EmoTalk.pth", device="cuda"):
        """
        Khởi tạo EmoTalk processor
        
        Args:
            model_path: Đường dẫn đến pretrained model
            device: Device để chạy model (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Khởi tạo model
        args = argparse.Namespace(
            bs_dim=52,
            feature_dim=832,
            period=30,
            device=self.device,
            max_seq_len=5000,
            batch_size=1
        )
        
        self.model = EmoTalk(args)
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device), weights_only=True),
            strict=False
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Eye blinking patterns
        self.eye_patterns = [
            np.array([0.36537236, 0.950235724, 0.95593375, 0.916715622, 0.367256105, 0.119113259, 0.025357503]),
            np.array([0.234776169, 0.909951985, 0.944758058, 0.777862132, 0.191071674, 0.235437036, 0.089163929]),
            np.array([0.870040774, 0.949833691, 0.949418545, 0.695911646, 0.191071674, 0.072576277, 0.007108896]),
            np.array([0.000307991, 0.556701422, 0.952656746, 0.942345619, 0.425857186, 0.148335218, 0.017659493])
        ]
        
        logger.info("EmoTalk model loaded successfully")
    
    def process_audio_chunk(self, audio_array, emotion_level=1, person_id=0, 
                          post_processing=True):
        """
        Xử lý một chunk audio và trả về blendshapes
        
        Args:
            audio_array: Audio array (numpy array)
            emotion_level: Mức độ cảm xúc (0 hoặc 1)
            person_id: ID người (0-23)
            post_processing: Có sử dụng post-processing không
            
        Returns:
            blendshapes: Array shape (num_frames, 52)
        """
        try:
            chunk_start = time.perf_counter()
            
            # Convert to tensor
            tensor_start = time.perf_counter()
            audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0).to(self.device)
            level_tensor = torch.tensor([emotion_level]).to(self.device)
            person_tensor = torch.tensor([person_id]).to(self.device)
            tensor_time = time.perf_counter() - tensor_start
            
            # Predict
            predict_start = time.perf_counter()
            with torch.no_grad():
                prediction = self.model.predict(audio_tensor, level_tensor, person_tensor)
                prediction = prediction.squeeze().detach().cpu().numpy()
                
                # Xóa tensors khỏi GPU ngay sau khi dùng
                del audio_tensor, level_tensor, person_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            predict_time = time.perf_counter() - predict_start
            
            # Post-processing
            post_start = time.perf_counter()
            if post_processing:
                output = self._apply_post_processing(prediction)
            else:
                output = prediction
            post_time = time.perf_counter() - post_start
            
            total_time = time.perf_counter() - chunk_start
            logger.debug(f"Chunk processing time: {total_time:.3f}s (tensor: {tensor_time:.3f}s, predict: {predict_time:.3f}s, post: {post_time:.3f}s)")
            
            return output
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            raise
    
    def _apply_post_processing(self, prediction):
        """
        Áp dụng post-processing (smoothing và blinking)
        
        Args:
            prediction: Raw prediction array (num_frames, 52)
            
        Returns:
            output: Processed array (num_frames, 52)
        """
        output = np.zeros((prediction.shape[0], prediction.shape[1]))
        
        # Smoothing với Savitzky-Golay filter
        for i in range(prediction.shape[1]):
            output[:, i] = savgol_filter(prediction[:, i], 5, 2)
        
        # Reset eye channels
        output[:, 8] = 0
        output[:, 9] = 0
        
        # Add blinking
        i = random.randint(0, 60)
        while i < output.shape[0] - 7:
            eye_num = random.randint(0, 3)
            eye_pattern = self.eye_patterns[eye_num]
            output[i:i + 7, 8] = eye_pattern
            output[i:i + 7, 9] = eye_pattern
            time1 = random.randint(60, 180)
            i = i + time1
        
        return output
    
    def split_audio_into_chunks(self, audio_array, sample_rate=16000, 
                               chunk_duration=25.0, overlap=0.5):
        """
        Chia audio thành các chunks với overlap
        
        Args:
            audio_array: Audio array đầy đủ
            sample_rate: Sample rate của audio (Hz)
            chunk_duration: Độ dài mỗi chunk (seconds)
            overlap: Độ dài overlap giữa các chunks (seconds)
            
        Returns:
            chunks: List các tuples (start_time, end_time, audio_chunk)
        """
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        start = 0
        
        while start < len(audio_array):
            end = min(start + chunk_samples, len(audio_array))
            chunk = audio_array[start:end]
            
            # KHÔNG pad - giữ nguyên độ dài thực của audio
            # Chunk cuối có thể ngắn hơn chunk_duration, model vẫn xử lý được
            
            start_time = start / sample_rate
            end_time = end / sample_rate
            
            chunks.append((start_time, end_time, chunk))
            
            if end >= len(audio_array):
                break
                
            start += step_samples
        
        total_duration = len(audio_array) / sample_rate
        logger.info(f"Split audio ({total_duration:.2f}s) into {len(chunks)} chunks "
                   f"(chunk_size: {chunk_duration}s, overlap: {overlap}s)")
        
        return chunks
    
    def process_full_audio(self, audio_array, emotion_level=1, person_id=0,
                          post_processing=True, chunk_duration=25.0, 
                          overlap=0.5, sample_rate=16000):
        """
        Xử lý toàn bộ audio với chunking
        
        Args:
            audio_array: Audio array đầy đủ
            emotion_level: Mức độ cảm xúc
            person_id: ID người
            post_processing: Có sử dụng post-processing không
            chunk_duration: Độ dài mỗi chunk (seconds)
            overlap: Độ dài overlap (seconds)
            sample_rate: Sample rate của audio
            
        Yields:
            chunk_result: Dict chứa thông tin chunk và blendshapes
                {
                    'chunk_index': int,
                    'start_time': float,
                    'end_time': float,
                    'blendshapes': array,
                    'is_final': bool
                }
        """
        # Resample nếu cần
        if sample_rate != 16000:
            logger.info(f"Resampling audio from {sample_rate}Hz to 16000Hz")
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Tính toán thời lượng audio thực tế
        actual_audio_duration = len(audio_array) / sample_rate
        
        # Chia thành chunks
        chunks = self.split_audio_into_chunks(
            audio_array, 
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap=overlap
        )
        
        # Xử lý từng chunk
        for idx, (start_time, end_time, chunk) in enumerate(chunks):
            chunk_process_start = time.perf_counter()
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)} "
                       f"({start_time:.2f}s - {end_time:.2f}s)")
            
            # Process chunk
            blendshapes = self.process_audio_chunk(
                chunk,
                emotion_level=emotion_level,
                person_id=person_id,
                post_processing=post_processing
            )
            
            chunk_process_time = time.perf_counter() - chunk_process_start
            chunk_duration_sec = len(chunk) / sample_rate
            rtf = chunk_process_time / chunk_duration_sec if chunk_duration_sec > 0 else 0
            logger.info(f"✓ Chunk {idx + 1}/{len(chunks)}: Process={chunk_process_time:.3f}s, Audio={chunk_duration_sec:.2f}s, RTF={rtf:.2f}x, Frames={blendshapes.shape[0]}")
            
            # Yield result
            yield {
                'chunk_index': idx,
                'start_time': start_time,
                'end_time': end_time,
                'blendshapes': blendshapes,
                'is_final': (idx == len(chunks) - 1),
                'processing_time': chunk_process_time
            }


if __name__ == "__main__":
    # Test processor
    processor = EmoTalkProcessor(device="cpu")
    
    # Load test audio
    test_audio_path = "./audio/angry1.wav"
    audio_array, sr = librosa.load(test_audio_path, sr=16000)
    
    print(f"Audio loaded: {len(audio_array) / sr:.2f} seconds")
    
    # Process with chunking
    for chunk_result in processor.process_full_audio(
        audio_array,
        emotion_level=1,
        person_id=0,
        post_processing=True,
        chunk_duration=25.0,
        overlap=0.5
    ):
        print(f"\nChunk {chunk_result['chunk_index']}:")
        print(f"  Time: {chunk_result['start_time']:.2f}s - {chunk_result['end_time']:.2f}s")
        print(f"  Blendshapes shape: {chunk_result['blendshapes'].shape}")
        print(f"  Is final: {chunk_result['is_final']}")
