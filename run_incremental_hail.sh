#!/bin/bash
cd HAIL-FFIA
# run_fish_incremental.sh
modality=$1

if [ -z "$modality" ]; then
    echo "Usage: $0 <modality>"
    echo "modality: audio, visual, audio-visual"
    echo ""
    echo "Examples:"
    echo "  $0 audio-visual"
    echo "  $0 audio"
    echo "  $0 visual"
    exit 1
fi

if [ "$modality" != "audio" ] && [ "$modality" != "visual" ] && [ "$modality" != "audio-visual" ]; then
    echo "Error: modality must be 'audio', 'visual', or 'audio-visual'"
    exit 1
fi

echo "Fish Incremental Learning"
echo "========================"
echo "Modality: $modality"
echo "6 fish species: Red_Tilapia → Tilapia → Jade_Perch → Black_Perch → Lotus_Carp → Sunfish"
echo "4 intensity classes: None, Weak, Medium, Strong"
echo ""

# Training parameters
num_classes=4
max_epoches=100
lr=1e-3
weight_decay=1e-4
train_batch_size=64
infer_batch_size=32
num_workers=8

echo "Training parameters:"
echo "  Classes: $num_classes"
echo "  Epochs: $max_epoches"
echo "  Learning rate: $lr"
echo "  Batch size: $train_batch_size"
echo ""

# Create directories
mkdir -p "./save/Fish/$modality"
mkdir -p "./save/fig/Fish/$modality"

CUDA_VISIBLE_DEVICES=0 python -u train_fish_incremental.py \
    --dataset Fish \
    --modality $modality \
    --train_batch_size $train_batch_size \
    --infer_batch_size $infer_batch_size \
    --num_workers $num_workers \
    --max_epoches $max_epoches \
    --num_classes $num_classes \
    --lr $lr \
    --weight_decay $weight_decay \
    --seed 42 \
    --upper_bound False