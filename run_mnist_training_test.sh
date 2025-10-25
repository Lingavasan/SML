#!/bin/bash
# Quick test training - just verify everything works
# Should complete in ~10 minutes on GPU

echo "=============================================="
echo "iVideoGPT Training - QUICK TEST"
echo "=============================================="

# Minimal configuration for stable training
TOKENIZER_PATH="thuml/ivideogpt-oxe-64-act-free"
OUTPUT_DIR="./output/mnist_test_tiny"
NUM_SEQUENCES=50  # Small but not too small
SEQ_LEN=10       # Standard sequence length
CONTEXT_LENGTH=2
BATCH_SIZE=2     # Smaller batch for stability
NUM_EPOCHS=2     # Two epochs for better convergence
LEARNING_RATE=1e-5  # Lower learning rate for stability
GRAD_ACCUM=2     # Some gradient accumulation
NUM_WORKERS=2    # More workers for faster loading
SAVE_STEPS=5     # Regular checkpoints
EVAL_STEPS=5     # Regular evaluation
MAX_GRAD_NORM=1.0  # Standard gradient clipping

echo ""
echo "Quick Test Configuration:"
echo "  Sequences: $NUM_SEQUENCES (very small)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Expected time: ~10 minutes on GPU, ~2 hours on CPU"
echo ""

cd "$(dirname "$0")"

python train_mnist_ivideogpt.py \
--tokenizer_path "$TOKENIZER_PATH" \
--output_dir "$OUTPUT_DIR" \
--num_sequences $NUM_SEQUENCES \
--num_val_sequences 2 \
--seq_len $SEQ_LEN \
--context_length $CONTEXT_LENGTH \
--num_digits 1 \
--frame_size 64 \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--num_train_epochs $NUM_EPOCHS \
--learning_rate $LEARNING_RATE \
--weight_decay 0.001 \
--lr_scheduler_type linear \
--num_warmup_steps 10 \
--max_grad_norm $MAX_GRAD_NORM \
--gradient_accumulation_steps $GRAD_ACCUM \
--logging_steps 1 \
--save_steps $SAVE_STEPS \
--eval_steps $EVAL_STEPS \
--dataloader_num_workers $NUM_WORKERS \
--num_samples 1 \
--seed 42 \
--generate_samples \
--mixed_precision no
