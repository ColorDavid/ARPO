#!/bin/bash

model=/home/dongyinpeng/mnt/models/UI-TARS-1.5-7B
model_name=ui-tars
num_images=16

port=9000

# Function to clean up processes on exit
cleanup() {
    echo "Stopping all processes..."
    pkill -P $$  # Kill all child processes of this script
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM to run cleanup function
trap cleanup SIGINT SIGTERM

# Start processes
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
        --served-model-name $model_name \
        --model $model \
        --limit-mm-per-prompt image=$num_images \
        -tp=1 \
        --port $((9000 + i)) &
done
# i=0
# CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
#     --served-model-name $model_name \
#     --model $model \
#     --limit-mm-per-prompt image=$num_images \
#     -tp=1 \
#     --port $((9000)) &

# Wait to keep the script running
wait