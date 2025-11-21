#!/usr/bin/env python3
"""
Script test API EmoTalk - Kiểm tra các endpoints và đo thời gian xử lý
"""
import requests
import json
import time
import os
from pathlib import Path

# Config
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TEST_AUDIO_FILE = "./audio/angry1.wav"

def print_separator(title=""):
    """In dòng phân cách"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)

def test_health_check():
    """Test health check endpoint"""
    print_separator("TEST 1: Health Check")
    
    try:
        start = time.time()
        response = requests.get(f"{API_BASE_URL}/health")
        elapsed = time.time() - start
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed:.3f}s")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "Health check failed"
        assert response.json()["status"] == "healthy", "Service not healthy"
        
        print("✅ PASSED")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_root_endpoint():
    """Test root endpoint"""
    print_separator("TEST 2: Root Endpoint")
    
    try:
        start = time.time()
        response = requests.get(f"{API_BASE_URL}/")
        elapsed = time.time() - start
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed:.3f}s")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "Root endpoint failed"
        
        print("✅ PASSED")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_process_audio_file():
    """Test process audio file endpoint (sync)"""
    print_separator("TEST 3: Process Audio File (Sync)")
    
    if not os.path.exists(TEST_AUDIO_FILE):
        print(f"❌ Test audio file not found: {TEST_AUDIO_FILE}")
        return False
    
    try:
        # Đọc audio file
        with open(TEST_AUDIO_FILE, "rb") as f:
            files = {"file": ("test.wav", f, "audio/wav")}
            
            print(f"Sending request to: {API_BASE_URL}/api/v1/process-audio-file")
            print(f"Fixed Parameters: emotion_level=1, person_id=0, post_processing=True, chunk_duration=25.0, chunk_overlap=0.5")
            
            start = time.time()
            response = requests.post(
                f"{API_BASE_URL}/api/v1/process-audio-file",
                files=files,
                timeout=300  # 5 minutes timeout
            )
            elapsed = time.time() - start
            
            print(f"\nStatus Code: {response.status_code}")
            print(f"Total Request Time: {elapsed:.3f}s")
            
            if response.status_code == 200:
                result = response.json()
                
                # Kiểm tra response structure
                frames = result.get("frames", [])
                total_frames = len(frames)
                
                print(f"\n📊 Response:")
                print(f"   Total Frames: {total_frames}")
                print(f"   Total Request Time: {elapsed:.3f}s")
                
                # Kiểm tra sample frame đầu tiên
                if frames:
                    first_frame = frames[0]
                    last_frame = frames[-1]
                    
                    print(f"\n   First Frame:")
                    print(f"      Timecode: {first_frame['timecode']}s")
                    print(f"      Blendshapes Count: {len(first_frame['blendshapes'])}")
                    print(f"      Blendshapes Sample (first 5): {[round(x, 4) for x in first_frame['blendshapes'][:5]]}")
                    
                    print(f"\n   Last Frame:")
                    print(f"      Timecode: {last_frame['timecode']}s")
                    
                    duration = last_frame['timecode']
                    rtf = elapsed / duration if duration > 0 else 0
                    print(f"\n   Performance:")
                    print(f"      Audio Duration: {duration:.2f}s")
                    print(f"      Processing Time: {elapsed:.2f}s")
                    print(f"      Real-time Factor: {rtf:.2f}x")
                
                assert total_frames > 0, "No frames generated"
                
                print("\n✅ PASSED")
                return True
            else:
                print(f"Error Response: {response.text}")
                print("❌ FAILED")
                return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_process_audio_stream():
    """Test process audio stream endpoint (SSE)"""
    print_separator("TEST 4: Process Audio Stream (SSE)")
    
    if not os.path.exists(TEST_AUDIO_FILE):
        print(f"❌ Test audio file not found: {TEST_AUDIO_FILE}")
        return False
    
    try:
        with open(TEST_AUDIO_FILE, "rb") as f:
            files = {"file": ("test.wav", f, "audio/wav")}
            
            print(f"Sending request to: {API_BASE_URL}/api/v1/process-audio-stream")
            print(f"Fixed Parameters: emotion_level=1, person_id=0, post_processing=True, chunk_duration=25.0, chunk_overlap=0.5")
            
            start = time.time()
            response = requests.post(
                f"{API_BASE_URL}/api/v1/process-audio-stream",
                files=files,
                stream=True,
                timeout=300
            )
            
            print(f"\nStatus Code: {response.status_code}")
            
            if response.status_code == 200:
                chunk_count = 0
                total_frames = 0
                first_chunk_time = None
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = json.loads(line[6:])
                            chunk_count += 1
                            frames = data.get('frames', [])
                            frame_count = len(frames)
                            total_frames += frame_count
                            
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - start
                            
                            print(f"\n📦 Chunk {chunk_count}:")
                            print(f"   Frames: {frame_count}")
                            if frames:
                                print(f"   First timecode: {frames[0]['timecode']}s")
                                print(f"   Last timecode: {frames[-1]['timecode']}s")
                            print(f"   Is Final: {data.get('is_final')}")
                
                elapsed = time.time() - start
                
                print(f"\n📊 Stream Performance:")
                print(f"   Total Time: {elapsed:.2f}s")
                print(f"   Time to First Chunk: {first_chunk_time:.2f}s")
                print(f"   Total Chunks: {chunk_count}")
                print(f"   Total Frames: {total_frames}")
                
                assert chunk_count > 0, "No chunks received"
                
                print("\n✅ PASSED")
                return True
            else:
                print(f"Error Response: {response.text}")
                print("❌ FAILED")
                return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Chạy tất cả tests"""
    print("\n" + "🚀 " + "="*76)
    print("  EMOTALK API TEST SUITE")
    print("="*80)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Test Audio File: {TEST_AUDIO_FILE}")
    print("="*80)
    
    results = []
    
    # Test 1: Health Check
    results.append(("Health Check", test_health_check()))
    
    # Test 2: Root Endpoint
    results.append(("Root Endpoint", test_root_endpoint()))
    
    # Test 3: Process Audio File (Sync)
    results.append(("Process Audio File", test_process_audio_file()))
    
    # Test 4: Process Audio Stream (SSE)
    results.append(("Process Audio Stream", test_process_audio_stream()))
    
    # Summary
    print_separator("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed\n")
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name:.<50} {status}")
    
    print("\n" + "="*80 + "\n")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = run_all_tests()
    sys.exit(exit_code)
