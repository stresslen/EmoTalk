#!/bin/bash
# Quick deployment script

set -e

echo "Starting EmoTalk gRPC Service..."
echo ""

# Check docker
if ! docker --version > /dev/null 2>&1; then
    echo "Error: Docker not found"
    exit 1
fi

# Stop existing containers
echo "Stopping existing containers..."
docker compose -f docker-compose.grpc.yml down 2>/dev/null || true

# Build and start
echo "Building and starting service..."
docker compose -f docker-compose.grpc.yml up -d --build

# Wait for service
echo ""
echo "Waiting for service to be ready..."
sleep 10

# Check status
docker compose -f docker-compose.grpc.yml ps

echo ""
echo "Service started!"
echo "Test with: ./test_service.sh"
