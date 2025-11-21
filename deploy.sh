#!/bin/bash

# EmoTalk Service Deployment Script

set -e

echo "=========================================="
echo "EmoTalk Service Deployment"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if pretrained model exists
print_info "Checking pretrained model..."
if [ ! -f "./pretrain_model/EmoTalk.pth" ]; then
    print_error "Pretrained model not found at ./pretrain_model/EmoTalk.pth"
    print_warn "Please download the model and place it in ./pretrain_model/"
    exit 1
fi
print_info "Pretrained model found ✓"

# Check Docker installation
print_info "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_info "Docker found ✓"

# Check Docker Compose
print_info "Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_info "Docker Compose found ✓"

# Ask for deployment type
echo ""
echo "Select deployment type:"
echo "1) GPU (CUDA required)"
echo "2) CPU only"
echo "3) Full stack (Kafka + API GPU)"
echo "4) Full stack (Kafka + API CPU)"
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        PROFILE="gpu"
        print_info "Deploying with GPU support..."
        ;;
    2)
        PROFILE="cpu"
        print_info "Deploying with CPU only..."
        ;;
    3)
        print_info "Deploying full stack with GPU..."
        docker-compose -f docker-compose.production.yml --profile gpu up -d
        print_info "Services started successfully!"
        print_info "API: http://localhost:8000"
        print_info "API Docs: http://localhost:8000/docs"
        print_info "Kafka UI: http://localhost:8080"
        exit 0
        ;;
    4)
        print_info "Deploying full stack with CPU..."
        docker-compose -f docker-compose.production.yml --profile cpu up -d
        print_info "Services started successfully!"
        print_info "API: http://localhost:8000"
        print_info "API Docs: http://localhost:8000/docs"
        print_info "Kafka UI: http://localhost:8080"
        exit 0
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

# Build Docker image
print_info "Building Docker image..."
if [ "$PROFILE" == "gpu" ]; then
    docker build -t emotalk-api:latest -f Dockerfile .
else
    docker build -t emotalk-api:latest -f Dockerfile.cpu .
fi

print_info "Docker image built successfully ✓"

# Run container
print_info "Starting container..."
docker run -d \
    --name emotalk-api \
    -p 8000:8000 \
    -v $(pwd)/pretrain_model:/app/pretrain_model:ro \
    -v $(pwd)/result:/app/result \
    -v $(pwd)/audio:/app/audio:ro \
    $([ "$PROFILE" == "gpu" ] && echo "--gpus all") \
    emotalk-api:latest

print_info "Container started successfully ✓"

# Wait for service to be ready
print_info "Waiting for service to be ready..."
sleep 10

# Health check
print_info "Performing health check..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        print_info "Service is healthy ✓"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "Service failed to start"
        docker logs emotalk-api
        exit 1
    fi
    sleep 2
done

echo ""
echo "=========================================="
echo "Deployment completed successfully!"
echo "=========================================="
echo ""
echo "Service endpoints:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Health: http://localhost:8000/health"
echo ""
echo "Useful commands:"
echo "  - View logs: docker logs -f emotalk-api"
echo "  - Stop service: docker stop emotalk-api"
echo "  - Remove container: docker rm emotalk-api"
echo "  - Restart: docker restart emotalk-api"
echo ""
