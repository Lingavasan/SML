#!/usr/bin/env python3
"""
Training script for iVideoGPT with Moving MNIST dataset.

This script trains iVideoGPT (video prediction transformer) using the Moving MNIST
dataset with bouncing digits. It handles:
- Data loading with Moving MNIST
- VQ-VAE tokenization
- Transformer training
- Checkpointing and evaluation
- Sample generation

Usage:
    python train_mnist_ivideogpt.py \
        --tokenizer_path "thuml/ivideogpt-oxe-64-act-free" \
        --output_dir "./output/mnist_ivideogpt" \
        --num_sequences 10000 \
        --seq_len 20 \
        --per_device_train_batch_size 8 \
        --num_train_epochs 10
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import imageio

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    get_scheduler,
)

# Import your Moving MNIST dataset
from mnist import MovingMNIST

# Import iVideoGPT components
from ivideogpt.vq_model import CompressiveVQModel

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train iVideoGPT on Moving MNIST dataset")
    
    # Dataset parameters
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for MNIST data"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=10000,
        help="Number of training sequences to generate"
    )
    parser.add_argument(
        "--num_val_sequences",
        type=int,
        default=1000,
        help="Number of validation sequences"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=20,
        help="Number of frames per sequence"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=2,
        help="Number of context frames for prediction"
    )
    parser.add_argument(
        "--num_digits",
        type=int,
        default=2,
        help="Maximum number of digits per sequence (1-2)"
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        default=64,
        help="Size of video frames (64x64 or 256x256)"
    )
    
    # Model parameters
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="thuml/ivideogpt-oxe-64-act-free",
        help="Path to pretrained VQ-VAE tokenizer"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained GPT model (None = train from scratch with config)"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/llama/config.json",
        help="Path to model config for training from scratch"
    )
    
    # Training parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/mnist_ivideogpt",
        help="Directory to save checkpoints and outputs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device during training"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per device during evaluation"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides num_train_epochs)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before backward pass"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Run evaluation every X updates steps"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether to save checkpoints: 'epoch' or number of steps"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    # Data loading
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Evaluation
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation, no training"
    )
    parser.add_argument(
        "--generate_samples",
        action="store_true",
        help="Generate sample predictions during evaluation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate during evaluation"
    )
    
    # Mixed precision
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision training"
    )
    
    args = parser.parse_args()
    return args


def save_video_samples(frames, output_path, fps=10):
    """
    Save video frames as MP4 or GIF.
    
    Args:
        frames: Tensor of shape (T, C, H, W) or numpy array
        output_path: Path to save video
        fps: Frames per second
    """
    # Convert to numpy if tensor
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    
    # Convert from (T, C, H, W) to (T, H, W, C)
    if frames.shape[1] == 3 or frames.shape[1] == 1:
        frames = np.transpose(frames, (0, 2, 3, 1))
    
    # Convert to uint8 [0, 255]
    if frames.max() <= 1.0:
        frames = (frames * 255).astype(np.uint8)
    
    # If grayscale, convert to RGB
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    
    # Save as GIF or MP4
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.gif':
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
    else:
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    
    logger.info(f"Saved video to {output_path}")


def tokenize_video(vq_model, pixel_values, context_length):
    """
    Tokenize video frames using VQ-VAE.
    
    Args:
        vq_model: CompressiveVQModel instance
        pixel_values: Tensor of shape (B, T, C, H, W)
        context_length: Number of context frames
    
    Returns:
        tokens: Tokenized sequence
        labels: Target labels for prediction
    """
    device = pixel_values.device
    batch_size, seq_len, channels, height, width = pixel_values.shape
    
    # Encode all frames
    # Reshape to (B*T, C, H, W) for encoding
    flat_frames = pixel_values.view(-1, channels, height, width)
    
    with torch.no_grad():
        # Encode to discrete tokens
        _, _, token_indices = vq_model.encode(flat_frames)
        # token_indices shape: (B*T, num_tokens_per_frame)
    
    # Reshape back to (B, T, num_tokens_per_frame)
    num_tokens_per_frame = token_indices.shape[1]
    token_indices = token_indices.view(batch_size, seq_len, num_tokens_per_frame)
    
    # Flatten to sequence: (B, T * num_tokens_per_frame)
    token_sequence = token_indices.view(batch_size, -1)
    
    # Create input/target pairs
    # Input: context frames + all but last token
    # Target: all but first context_length frames
    context_tokens = context_length * num_tokens_per_frame
    input_ids = token_sequence[:, :-1]  # All but last token
    labels = token_sequence[:, context_tokens:].clone()  # Predict after context
    
    # Mask context tokens in labels (don't compute loss on them)
    if context_tokens > 0:
        labels = F.pad(labels, (context_tokens - 1, 0), value=-100)
    
    return input_ids, labels


def evaluate_model(model, vq_model, eval_dataloader, accelerator, args, global_step=0):
    """
    Evaluate the model on validation set.
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    logger.info("Running evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            # batch shape: (B, T, C, H, W)
            pixel_values = batch.to(accelerator.device)
            
            # Tokenize
            input_ids, labels = tokenize_video(vq_model, pixel_values, args.context_length)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    metrics = {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
    }
    
    logger.info(f"Eval Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    # Generate sample predictions if requested
    if args.generate_samples and accelerator.is_main_process:
        generate_predictions(model, vq_model, eval_dataloader, accelerator, args, global_step)
    
    model.train()
    return metrics


def generate_predictions(model, vq_model, eval_dataloader, accelerator, args, global_step=0):
    """
    Generate sample video predictions.
    """
    logger.info("Generating sample predictions...")
    
    model.eval()
    vq_model.eval()
    
    # Get a batch
    batch = next(iter(eval_dataloader))
    pixel_values = batch[:args.num_samples].to(accelerator.device)
    
    batch_size, seq_len, channels, height, width = pixel_values.shape
    
    with torch.no_grad():
        # Use context frames
        context_frames = pixel_values[:, :args.context_length]
        
        # Tokenize context
        input_ids, _ = tokenize_video(vq_model, context_frames, context_length=0)
        
        # Generate tokens for remaining frames
        num_tokens_per_frame = input_ids.shape[1] // args.context_length
        num_predict_tokens = (seq_len - args.context_length) * num_tokens_per_frame
        
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=num_predict_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=100,
            pad_token_id=0
        )
        
        # Decode generated tokens back to frames
        # Reshape to (B, T, num_tokens_per_frame)
        all_tokens = generated_ids[:, :seq_len * num_tokens_per_frame]
        all_tokens = all_tokens.view(batch_size, seq_len, num_tokens_per_frame)
        
        # Decode frame by frame
        predicted_frames = []
        for t in range(seq_len):
            frame_tokens = all_tokens[:, t]  # (B, num_tokens_per_frame)
            decoded_frame = vq_model.decode_tokens(frame_tokens)
            predicted_frames.append(decoded_frame)
        
        predicted_video = torch.stack(predicted_frames, dim=1)  # (B, T, C, H, W)
    
    # Save samples
    sample_dir = Path(args.output_dir) / "samples" / f"step_{global_step}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(args.num_samples, batch_size)):
        # Save ground truth
        gt_path = sample_dir / f"sample_{i}_gt.gif"
        save_video_samples(pixel_values[i], gt_path)
        
        # Save prediction
        pred_path = sample_dir / f"sample_{i}_pred.gif"
        save_video_samples(predicted_video[i], pred_path)
        
        # Save context + prediction side by side
        combined = torch.cat([pixel_values[i], predicted_video[i]], dim=-1)  # Concatenate width-wise
        combined_path = sample_dir / f"sample_{i}_combined.gif"
        save_video_samples(combined, combined_path)
    
    logger.info(f"Saved {args.num_samples} sample predictions to {sample_dir}")
    
    model.train()


def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["project_dir"] = args.output_dir
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=ProjectConfiguration(**accelerator_log_kwargs),
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save args
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    accelerator.wait_for_everyone()
    
    # ===== 1. Load VQ-VAE Tokenizer =====
    logger.info(f"Loading VQ-VAE tokenizer from {args.tokenizer_path}")
    vq_model = CompressiveVQModel.from_pretrained(args.tokenizer_path)
    vq_model.eval()  # Keep in eval mode (frozen)
    vq_model = vq_model.to(accelerator.device)
    
    # ===== 2. Load or Initialize GPT Model =====
    if args.model_path:
        logger.info(f"Loading pretrained model from {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
    else:
        logger.info(f"Initializing model from config: {args.model_config}")
        config = AutoConfig.from_pretrained(args.model_config)
        model = AutoModelForCausalLM.from_config(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # ===== 3. Create Datasets =====
    logger.info("Creating Moving MNIST datasets...")
    
    train_dataset = MovingMNIST(
        root=args.data_root,
        train=True,
        seq_len=args.seq_len,
        num_digits=args.num_digits,
        frame_size=args.frame_size,
        num_sequences=args.num_sequences,
        rgb_output=True,  # Required for iVideoGPT
        download=True
    )
    
    val_dataset = MovingMNIST(
        root=args.data_root,
        train=False,  # Use test set for validation
        seq_len=args.seq_len,
        num_digits=args.num_digits,
        frame_size=args.frame_size,
        num_sequences=args.num_val_sequences,
        rgb_output=True,
        download=True
    )
    
    logger.info(f"Training sequences: {len(train_dataset)}")
    logger.info(f"Validation sequences: {len(val_dataset)}")
    
    # ===== 4. Create DataLoaders =====
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )
    
    # ===== 5. Setup Optimizer and Scheduler =====
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # ===== 6. Prepare with Accelerator =====
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # ===== 7. Training Loop =====
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (with parallel, dist, accum) = {args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show progress bar on main process
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None and os.path.isdir(args.resume_from_checkpoint):
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
            training_difference = os.path.splitext(path)[0]
            resume_step = int(training_difference.replace("step_", ""))
            completed_steps = resume_step
            starting_epoch = resume_step // num_update_steps_per_epoch
            progress_bar.update(completed_steps)
    
    # Eval before training if requested
    if args.eval_only:
        logger.info("Running evaluation only (no training)...")
        metrics = evaluate_model(model, vq_model, eval_dataloader, accelerator, args, completed_steps)
        return
    
    model.train()
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            # batch shape: (B, T, C, H, W)
            pixel_values = batch
            
            # Tokenize video
            with torch.no_grad():
                input_ids, labels = tokenize_video(vq_model, pixel_values, args.context_length)
            
            # Forward pass
            with accelerator.accumulate(model):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                
                # Clip gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
                # Logging
                if completed_steps % args.logging_steps == 0:
                    avg_loss = total_loss / args.logging_steps
                    logger.info(f"Step {completed_steps}: loss = {avg_loss:.4f}, lr = {lr_scheduler.get_last_lr()[0]:.6f}")
                    
                    if accelerator.is_main_process:
                        accelerator.log({
                            "train_loss": avg_loss,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        }, step=completed_steps)
                    
                    total_loss = 0
                
                # Evaluation
                if completed_steps % args.eval_steps == 0:
                    metrics = evaluate_model(model, vq_model, eval_dataloader, accelerator, args, completed_steps)
                    
                    if accelerator.is_main_process:
                        accelerator.log(metrics, step=completed_steps)
                
                # Save checkpoint
                if completed_steps % args.save_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    logger.info(f"Saved checkpoint to {output_dir}")
            
            if completed_steps >= args.max_train_steps:
                break
        
        # End of epoch evaluation
        if epoch < args.num_train_epochs - 1:
            metrics = evaluate_model(model, vq_model, eval_dataloader, accelerator, args, completed_steps)
            
            if accelerator.is_main_process:
                accelerator.log(metrics, step=completed_steps)
    
    # ===== 8. Save Final Model =====
    logger.info("Training completed!")
    
    if accelerator.is_main_process:
        output_dir = os.path.join(args.output_dir, "final_model")
        accelerator.unwrap_model(model).save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        logger.info(f"Saved final model to {output_dir}")
    
    # Final evaluation
    metrics = evaluate_model(model, vq_model, eval_dataloader, accelerator, args, completed_steps)
    logger.info("Final evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
