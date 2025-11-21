#!/bin/bash

# Script để generate gRPC code từ protobuf

echo "Generating gRPC code from protobuf..."

python -m grpc_tools.protoc \
    -I./proto \
    --python_out=. \
    --grpc_python_out=. \
    ./proto/emotalk.proto

echo "Done! Generated files:"
echo "  - emotalk_pb2.py (message definitions)"
echo "  - emotalk_pb2_grpc.py (service definitions)"
