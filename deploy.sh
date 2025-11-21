#!/bin/bash

# Quick Deploy Script - Deploy everything in one command

set -e

echo "🚀 EmoTalk Quick Deploy"
echo "======================="
echo ""

# Stop and remove old containers
echo "🔄 Stopping old containers..."
docker stop emotalk-api emotalk-kafka emotalk-zookeeper 2>/dev/null || true
docker rm emotalk-api emotalk-kafka emotalk-zookeeper 2>/dev/null || true
docker compose -f docker-compose.full.yml down 2>/dev/null || true

# Build and start
echo "🏗️  Building and starting services..."
docker compose -f docker-compose.full.yml up -d --build

# Wait for services
echo "⏳ Waiting for services to be ready..."
sleep 15

# Check status
echo ""
echo "📊 Service Status:"
docker compose -f docker-compose.full.yml ps

# Test health
echo ""
echo "🏥 Health Check:"
echo -n "  API Gateway: "
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Healthy"
else
    echo "❌ Not responding"
fi

echo -n "  gRPC Server: "
if docker exec emotalk_grpc_server python3 -c "from grpc_client_optimized import OptimizedEmoTalkClient; import sys; client = OptimizedEmoTalkClient('localhost:50051'); sys.exit(0 if client.health_check() else 1)" 2>/dev/null; then
    echo "✅ Healthy"
else
    echo "⚠️  Starting (wait a bit more)"
fi

echo ""
echo "✅ Deployment completed!"
echo ""
echo "📡 Endpoints:"
echo "  - API Gateway (HTTP): http://localhost:8000"
echo "  - gRPC Server: localhost:50051"
echo ""
echo "🧪 Test commands:"
echo "  - Test API: curl http://localhost:8000/health"
echo "  - Upload audio: curl -X POST http://localhost:8000/api/v1/process-audio-stream -F 'file=@audio/angry1.wav' --no-buffer"
echo "  - View logs: docker compose -f docker-compose.full.yml logs -f"
echo ""
