#!/bin/bash
# Quick start script for training iVideoGPT on Moving MNIST
# Run with: bash run_mnist_training.sh

echo "=================================="
echo "iVideoGPT Training on Moving MNIST"
echo "=================================="

# Configuration
TOKENIZER_PATH="thuml/ivideogpt-oxe-64-act-free"
OUTPUT_DIR="./output/mnist_ivideogpt"
NUM_SEQUENCES=10000
SEQ_LEN=20
CONTEXT_LENGTH=2
BATCH_SIZE=8
NUM_EPOCHS=10
LEARNING_RATE=1e-4

echo ""
echo "Training Configuration:"
echo "  Tokenizer: $TOKENIZER_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Sequences: $NUM_SEQUENCES"
echo "  Seq Length: $SEQ_LEN frames"
echo "  Context: $CONTEXT_LENGTH frames"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python train_mnist_ivideogpt.py \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_sequences $NUM_SEQUENCES \
    --num_val_sequences 1000 \
    --seq_len $SEQ_LEN \
    --context_length $CONTEXT_LENGTH \
    --num_digits 2 \
    --frame_size 64 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 100 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1 \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_steps 500 \
    --dataloader_num_workers 4 \
    --generate_samples \
    --num_samples 4 \
    --seed 42

echo ""
echo "=================================="
echo "Training completed!"
echo "Check results in: $OUTPUT_DIR"
echo "View logs: tensorboard --logdir $OUTPUT_DIR"
echo "=================================="
