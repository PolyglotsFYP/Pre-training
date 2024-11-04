#!/bin/bash

if [ $# -ne 1 ]; then
    echo "run_pt.sh <your_huggingface_token>"
    exit 1
fi

TOKEN=$1

# pip install -r requirements.txt
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

huggingface-cli login --token $TOKEN

chmod +x ./pre-training/run_pt.sh

cd pre-training && ./run_pt.sh
