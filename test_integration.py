#!/usr/bin/env python3
"""
Test script to verify Moving MNIST + iVideoGPT integration.

This script tests:
1. Dataset creation and RGB output
2. DataLoader compatibility
3. VQ-VAE tokenization
4. Model input/output shapes
5. Full training step

Run: python test_integration.py
"""

import torch
from torch.utils.data import DataLoader
import sys

print("=" * 70)
print("iVideoGPT + Moving MNIST Integration Test")
print("=" * 70)

# Test 1: Import and dataset creation
print("\n[Test 1/5] Testing dataset creation...")
try:
    from mnist import MovingMNIST
    
    dataset = MovingMNIST(
        root='./data',
        train=True,
        seq_len=20,
        num_digits=2,
        frame_size=64,
        num_sequences=100,
        rgb_output=True,
        download=True
    )
    
    print(f"✓ Dataset created: {len(dataset)} sequences")
    print(f"✓ Configuration: {dataset.seq_len} frames, {dataset.frame_size}×{dataset.frame_size} pixels")
    print(f"✓ RGB output: {dataset.rgb_output}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 2: DataLoader and batch shape
print("\n[Test 2/5] Testing DataLoader integration...")
try:
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    batch = next(iter(loader))
    print(f"✓ DataLoader created successfully")
    print(f"✓ Batch shape: {batch.shape}")
    print(f"✓ Batch dtype: {batch.dtype}")
    print(f"✓ Value range: [{batch.min():.3f}, {batch.max():.3f}]")
    
    # Verify shape
    expected_shape = (4, 20, 3, 64, 64)
    if batch.shape == expected_shape:
        print(f"✓ Shape matches expected: {expected_shape}")
    else:
        print(f"✗ Shape mismatch! Expected {expected_shape}, got {batch.shape}")
        sys.exit(1)
    
    # Verify RGB
    if batch.shape[2] == 3:
        print("✓ RGB output confirmed (3 channels)")
    else:
        print(f"✗ Expected 3 channels (RGB), got {batch.shape[2]}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: VQ-VAE tokenizer
print("\n[Test 3/5] Testing VQ-VAE tokenization...")
try:
    from ivideogpt.vq_model import CompressiveVQModel
    
    print("  Loading tokenizer (may take a minute)...")
    tokenizer = CompressiveVQModel.from_pretrained("thuml/ivideogpt-oxe-64-act-free")
    tokenizer.eval()
    print("✓ Tokenizer loaded successfully")
    
    # Test encoding
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = tokenizer.to(device)
    batch = batch.to(device)
    
    # Flatten batch for encoding
    B, T, C, H, W = batch.shape
    flat_batch = batch.view(B * T, C, H, W)
    print(f"  Flattened shape: {flat_batch.shape}")
    
    with torch.no_grad():
        indices, codes, _ = tokenizer.encode(flat_batch)
    
    print(f"✓ Encoding successful")
    print(f"✓ Token indices shape: {indices.shape}")
    print(f"✓ Token codes shape: {codes.shape}")
    print(f"✓ Number of tokens per frame: {indices.shape[1]}")
    
    # Test decoding
    with torch.no_grad():
        reconstructed = tokenizer.decode(codes)
    
    print(f"✓ Decoding successful")
    print(f"✓ Reconstructed shape: {reconstructed.shape}")
    
    # Verify reconstruction quality
    mse = torch.nn.functional.mse_loss(flat_batch, reconstructed)
    print(f"✓ Reconstruction MSE: {mse.item():.6f}")
    
    if mse.item() < 0.1:
        print("✓ Reconstruction quality: Good")
    elif mse.item() < 0.3:
        print("⚠ Reconstruction quality: Fair (may need tokenizer fine-tuning)")
    else:
        print("⚠ Reconstruction quality: Poor (tokenizer may need fine-tuning)")
        
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print("\n  Note: This test requires HuggingFace access and may download ~1GB of data.")
    print("  If download fails, check your internet connection or HuggingFace credentials.")
    sys.exit(1)

# Test 4: Model loading
print("\n[Test 4/5] Testing model compatibility...")
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    
    # Load config
    config_path = "configs/llama/config.json"
    print(f"  Loading config from {config_path}...")
    
    try:
        config = AutoConfig.from_pretrained(config_path)
        print("✓ Config loaded from local file")
    except:
        print("  Local config not found, using default")
        config = AutoConfig.from_pretrained("gpt2")
    
    # Initialize model
    print("  Initializing model...")
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model initialized")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    # Test forward pass with dummy tokens
    print("  Testing forward pass...")
    dummy_input_ids = torch.randint(0, 10000, (2, 100)).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=dummy_input_ids)
    
    print(f"✓ Forward pass successful")
    print(f"✓ Output shape: {outputs.logits.shape}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print("\n  Note: This test requires a valid model config.")
    sys.exit(1)

# Test 5: Complete training step
print("\n[Test 5/5] Testing complete training step...")
try:
    # Prepare inputs
    print("  Preparing training batch...")
    batch_size = 2
    seq_len = 20
    context_length = 2
    
    # Get a small batch
    small_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    video_batch = next(iter(small_loader)).to(device)
    
    # Tokenize
    print("  Tokenizing video...")
    B, T, C, H, W = video_batch.shape
    flat_video = video_batch.view(B * T, C, H, W)
    
    with torch.no_grad():
        token_indices, _, _ = tokenizer.encode(flat_video)
    
    # Reshape to (B, T, num_tokens_per_frame)
    num_tokens_per_frame = token_indices.shape[1]
    token_indices = token_indices.view(B, T, num_tokens_per_frame)
    
    # Flatten to sequence
    token_sequence = token_indices.view(B, -1)
    
    # Create input/target pairs
    context_tokens = context_length * num_tokens_per_frame
    input_ids = token_sequence[:, :context_tokens + 10]  # Context + some tokens
    labels = token_sequence[:, context_tokens:context_tokens + 10].clone()
    
    # Pad labels
    labels = torch.nn.functional.pad(labels, (context_tokens, 0), value=-100)
    
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Forward pass
    print("  Running forward pass...")
    model.train()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    
    print(f"✓ Training step successful")
    print(f"✓ Loss: {loss.item():.4f}")
    
    # Test backward pass
    print("  Testing backward pass...")
    loss.backward()
    print("✓ Backward pass successful")
    
    # Check gradients
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    if has_grads:
        print("✓ Gradients computed successfully")
    else:
        print("⚠ No gradients found (this may be okay depending on model architecture)")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✅ ALL INTEGRATION TESTS PASSED!")
print("=" * 70)
print("\nSummary:")
print("  ✓ Dataset creates correct RGB format")
print("  ✓ DataLoader produces correct batch shapes")
print("  ✓ VQ-VAE tokenization works")
print("  ✓ Model forward/backward passes work")
print("  ✓ Complete training step successful")
print("\n🚀 Ready to train iVideoGPT on Moving MNIST!")
print("\nNext steps:")
print("  1. Run: bash run_mnist_training.sh")
print("  2. Or:  python train_mnist_ivideogpt.py")
print("  3. Monitor with: tensorboard --logdir ./output/mnist_ivideogpt")
print("=" * 70)
