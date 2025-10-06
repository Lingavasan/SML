#!/bin/bash
# Quick test training - just verify everything works
# Should complete in ~10 minutes on GPU

echo "=============================================="
echo "iVideoGPT Training - QUICK TEST"
echo "=============================================="

# Minimal configuration for testing
TOKENIZER_PATH="thuml/ivideogpt-oxe-64-act-free"
OUTPUT_DIR="./output/mnist_test"
NUM_SEQUENCES=100  # Tiny dataset
SEQ_LEN=20
CONTEXT_LENGTH=2
BATCH_SIZE=8
NUM_EPOCHS=2  # Just 2 epochs
LEARNING_RATE=1e-4
GRAD_ACCUM=1
NUM_WORKERS=4
SAVE_STEPS=25  # Save once during training
EVAL_STEPS=25

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
  --num_val_sequences 20 \
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
  --num_warmup_steps 10 \
  --max_grad_norm 1.0 \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --logging_steps 5 \
  --save_steps $SAVE_STEPS \
  --eval_steps $EVAL_STEPS \
  --dataloader_num_workers $NUM_WORKERS \
  --num_samples 2 \
  --seed 42 \
  --mixed_precision fp16 \
  --generate_samples
