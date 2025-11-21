#!/bin/bash

# Simple test script for gRPC service

echo "🧪 Testing EmoTalk gRPC Service"
echo "================================"
echo ""

# Check if audio file exists
if [ ! -f "audio/angry1.wav" ]; then
    echo "❌ Audio file not found: audio/angry1.wav"
    echo "Please add an audio file to the audio/ directory"
    exit 1
fi

echo "✅ Audio file found: audio/angry1.wav"
echo ""

# Test with Python client
echo "🔄 Sending test request..."
python3 -c "
import sys
sys.path.insert(0, '/root/EmoTalk')

from grpc_client_optimized import OptimizedEmoTalkClient

try:
    # Connect to server
    client = OptimizedEmoTalkClient('localhost:50051')
    
    # Health check
    print('Checking server health...')
    if not client.health_check():
        print('❌ Server is not healthy')
        sys.exit(1)
    
    print('✅ Server is healthy')
    print('')
    
    # Process audio
    print('Processing audio file...')
    chunks = client.process_audio_file(
        '/root/EmoTalk/audio/angry1.wav',
        emotion_level=1,
        person_id=0
    )
    
    # Print results
    total_frames = sum(len(chunk['frames']) for chunk in chunks)
    print('')
    print('✅ Processing completed!')
    print(f'   Chunks: {len(chunks)}')
    print(f'   Total frames: {total_frames}')
    print(f'   Duration: ~{total_frames/30:.2f}s @ 30fps')
    
    # Show first chunk info
    if chunks:
        first_chunk = chunks[0]
        first_frame = first_chunk['frames'][0]
        print('')
        print('Sample frame:')
        print(f'   Timecode: {first_frame[\"timecode\"]}s')
        print(f'   Blendshapes: {len(first_frame[\"blendshapes\"])} values')
        print(f'   First 5 values: {first_frame[\"blendshapes\"][:5]}')
    
    client.close()
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✅ Test completed successfully!"
    echo "================================"
else
    echo ""
    echo "================================"
    echo "❌ Test failed!"
    echo "================================"
    exit 1
fi
