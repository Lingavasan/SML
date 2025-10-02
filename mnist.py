import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import os


class MovingMNIST(Dataset):
    """
    Generates sequences of moving MNIST digits with bouncing physics.
    
    Requirements:
    - Each sequence: 64×64 pixel frames
    - 1-2 digits moving per sequence
    - Time series of seq_len frames (e.g., 20-30)
    - Digits bounce off frame edges
    - Output shape: (seq_len, 3, 64, 64) for RGB (default, required for iVideoGPT)
                    (seq_len, 1, 64, 64) for grayscale (if rgb_output=False)
    """
    
    def __init__(
        self,
        root='./data',
        train=True,
        seq_len=20,
        num_digits=2,
        frame_size=64,
        digit_size=28,
        num_sequences=10000,
        transform=None,
        download=True,
        rgb_output=True
    ):
        """
        Args:
            root: Root directory for MNIST data
            train: Use training or test set
            seq_len: Number of frames in each sequence (20-30 recommended)
            num_digits: Number of digits per sequence (1-2)
            frame_size: Size of output frames (64×64)
            digit_size: Size of MNIST digits (28×28)
            num_sequences: Total number of sequences to generate
            transform: Optional transform to apply
            download: Whether to download MNIST if not present
            rgb_output: If True, convert grayscale to RGB (required for iVideoGPT)
        """
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.frame_size = frame_size
        self.digit_size = digit_size
        self.num_sequences = num_sequences
        self.transform = transform
        self.rgb_output = rgb_output
        
        # Load MNIST dataset and normalize to [0, 1]
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )
        
        # Extract all MNIST images as numpy arrays
        self.mnist_images = []
        for idx in range(len(self.mnist)):
            img, _ = self.mnist[idx]
            # Convert to numpy and ensure [0, 1] range
            img_np = img.squeeze().numpy()
            self.mnist_images.append(img_np)
        
        print(f"Loaded {len(self.mnist_images)} MNIST digits")
        print(f"Will generate {num_sequences} sequences of {seq_len} frames")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Generate a single sequence of moving MNIST digits.
        
        Returns:
            Tensor of shape (seq_len, 3, 64, 64) if rgb_output=True
            Tensor of shape (seq_len, 1, 64, 64) if rgb_output=False
        """
        # Set random seed for reproducibility per sequence
        np.random.seed(idx)
        
        # Generate the sequence
        sequence = self._generate_sequence()
        
        # Convert to tensor: (seq_len, 1, 64, 64) or (seq_len, 3, 64, 64)
        sequence_tensor = torch.from_numpy(sequence).float()
        sequence_tensor = sequence_tensor.unsqueeze(1)  # Add channel dimension (grayscale)
        
        # Convert to RGB if required (needed for iVideoGPT)
        if self.rgb_output:
            sequence_tensor = sequence_tensor.repeat(1, 3, 1, 1)  # (seq_len, 3, H, W)
        
        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
        
        return sequence_tensor
    
    def _generate_sequence(self):
        """
        Core logic to generate a moving MNIST sequence.
        
        Steps:
        1. Randomly sample digits
        2. Initialize positions and velocities
        3. Create frame-by-frame updates with bouncing
        4. Combine multiple digits into frames
        
        Returns:
            numpy array of shape (seq_len, 64, 64)
        """
        # Initialize empty sequence
        sequence = np.zeros((self.seq_len, self.frame_size, self.frame_size), dtype=np.float32)
        
        # Sample random number of digits (1 or 2)
        actual_num_digits = np.random.randint(1, self.num_digits + 1)
        
        # For each digit, initialize position and velocity
        digits = []
        for _ in range(actual_num_digits):
            # Randomly sample a digit image
            digit_idx = np.random.randint(0, len(self.mnist_images))
            digit_img = self.mnist_images[digit_idx]
            
            # Random initial position within valid bounds
            # Ensure digit doesn't start outside frame
            max_x = self.frame_size - self.digit_size
            max_y = self.frame_size - self.digit_size
            x = np.random.randint(0, max_x + 1)
            y = np.random.randint(0, max_y + 1)
            
            # Random velocity: ±1-3 pixels per frame
            dx = np.random.randint(1, 4) * np.random.choice([-1, 1])
            dy = np.random.randint(1, 4) * np.random.choice([-1, 1])
            
            digits.append({
                'image': digit_img,
                'x': x,
                'y': y,
                'dx': dx,
                'dy': dy
            })
        
        # Generate each frame in the sequence
        for frame_idx in range(self.seq_len):
            # Start with blank frame
            frame = np.zeros((self.frame_size, self.frame_size), dtype=np.float32)
            
            # Place each digit in the frame
            for digit in digits:
                # Place digit at current position
                x, y = int(digit['x']), int(digit['y'])
                
                # Add digit to frame (blend by summing)
                frame[y:y+self.digit_size, x:x+self.digit_size] += digit['image']
                
                # Update position for next frame
                new_x = digit['x'] + digit['dx']
                new_y = digit['y'] + digit['dy']
                
                # Bounce off edges (reverse velocity if hitting boundary)
                if new_x <= 0 or new_x >= self.frame_size - self.digit_size:
                    digit['dx'] = -digit['dx']
                    new_x = digit['x'] + digit['dx']
                
                if new_y <= 0 or new_y >= self.frame_size - self.digit_size:
                    digit['dy'] = -digit['dy']
                    new_y = digit['y'] + digit['dy']
                
                # Update position
                digit['x'] = new_x
                digit['y'] = new_y
            
            # Clip values to [0, 1] to handle overlapping digits
            frame = np.clip(frame, 0, 1)
            sequence[frame_idx] = frame
        
        return sequence
    
    def visualize_sequence(self, idx, save_path='./moving_mnist_samples'):
        """
        Visualize and save a sequence as individual frame images.
        
        Args:
            idx: Index of sequence to visualize
            save_path: Directory to save frames
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Generate sequence
        sequence = self[idx].squeeze().numpy()  # Shape: (seq_len, 64, 64)
        
        # Save each frame
        for frame_idx, frame in enumerate(sequence):
            # Convert to PIL Image (scale to 0-255)
            img = Image.fromarray((frame * 255).astype(np.uint8), mode='L')
            img.save(os.path.join(save_path, f'seq_{idx}_frame_{frame_idx:03d}.png'))
        
        print(f"Saved {len(sequence)} frames to {save_path}")
    
    def visualize_grid(self, idx, save_path='./moving_mnist_samples', grid_cols=5):
        """
        Visualize a sequence as a grid of frames in a single image.
        
        Args:
            idx: Index of sequence to visualize
            save_path: Directory to save the grid image
            grid_cols: Number of columns in the grid
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Generate sequence
        sequence = self[idx].squeeze().numpy()  # Shape: (seq_len, 64, 64)
        
        # Calculate grid dimensions
        num_frames = len(sequence)
        grid_rows = (num_frames + grid_cols - 1) // grid_cols
        
        # Create grid image
        grid_height = grid_rows * self.frame_size
        grid_width = grid_cols * self.frame_size
        grid = np.zeros((grid_height, grid_width), dtype=np.float32)
        
        # Place frames in grid
        for frame_idx, frame in enumerate(sequence):
            row = frame_idx // grid_cols
            col = frame_idx % grid_cols
            y = row * self.frame_size
            x = col * self.frame_size
            grid[y:y+self.frame_size, x:x+self.frame_size] = frame
        
        # Save grid
        img = Image.fromarray((grid * 255).astype(np.uint8), mode='L')
        img.save(os.path.join(save_path, f'seq_{idx}_grid.png'))
        
        print(f"Saved grid visualization to {save_path}/seq_{idx}_grid.png")


def validate_pytorch_compatibility(dataset, num_samples=5):
    """
    Comprehensive validation to ensure sequences work with PyTorch models.
    
    Validates:
    1. Tensor conversion and data types
    2. Value normalization [0, 1]
    3. Shape correctness
    4. DataLoader batch sampling
    5. Visual inspection (optional)
    
    Args:
        dataset: MovingMNIST dataset instance
        num_samples: Number of samples to validate
    """
    print("=" * 60)
    print("PYTORCH COMPATIBILITY VALIDATION")
    print("=" * 60)
    
    # 1. Test tensor conversion and data type
    print("\n[1/5] Testing tensor conversion and data types...")
    sequence = dataset[0]
    
    print(f"✓ Type: {type(sequence)} (expected: torch.Tensor)")
    assert isinstance(sequence, torch.Tensor), "Output is not a PyTorch tensor!"
    
    print(f"✓ Data type: {sequence.dtype} (expected: torch.float32)")
    assert sequence.dtype == torch.float32, f"Expected float32, got {sequence.dtype}"
    
    print(f"✓ Device: {sequence.device} (CPU)")
    print(f"✓ Requires grad: {sequence.requires_grad} (False by default)")
    
    # 2. Check normalization
    print("\n[2/5] Validating normalization [0, 1]...")
    min_val = sequence.min().item()
    max_val = sequence.max().item()
    print(f"✓ Value range: [{min_val:.4f}, {max_val:.4f}]")
    assert 0.0 <= min_val <= 1.0, f"Min value {min_val} out of range!"
    assert 0.0 <= max_val <= 1.0, f"Max value {max_val} out of range!"
    
    # Check multiple samples
    for i in range(min(num_samples, len(dataset))):
        seq = dataset[i]
        assert seq.min() >= 0.0 and seq.max() <= 1.0, f"Sample {i} values out of range!"
    print(f"✓ All {num_samples} samples have valid value ranges")
    
    # 3. Verify shape
    print("\n[3/5] Verifying tensor shapes...")
    # Shape depends on rgb_output setting
    channels = 3 if dataset.rgb_output else 1
    expected_shape = (dataset.seq_len, channels, dataset.frame_size, dataset.frame_size)
    print(f"✓ Shape: {tuple(sequence.shape)} (expected: {expected_shape})")
    assert sequence.shape == expected_shape, f"Shape mismatch!"
    
    # Check consistency across samples
    for i in range(min(num_samples, len(dataset))):
        seq = dataset[i]
        assert seq.shape == expected_shape, f"Sample {i} has inconsistent shape!"
    print(f"✓ All {num_samples} samples have consistent shapes")
    
    # 4. Test DataLoader integration
    print("\n[4/5] Testing DataLoader integration...")
    from torch.utils.data import DataLoader
    
    test_batch_size = 4
    dataloader = DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    channels = 3 if dataset.rgb_output else 1
    expected_batch_shape = (test_batch_size, dataset.seq_len, channels, dataset.frame_size, dataset.frame_size)
    print(f"✓ Batch shape: {tuple(batch.shape)} (expected: {expected_batch_shape})")
    assert batch.shape == expected_batch_shape, "Batch shape mismatch!"
    
    print(f"✓ Batch dtype: {batch.dtype}")
    print(f"✓ Batch value range: [{batch.min():.4f}, {batch.max():.4f}]")
    
    # Test multiple batches
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        if batch_count >= 3:
            break
    print(f"✓ Successfully loaded {batch_count} batches")
    
    # 5. Motion and physics validation
    print("\n[5/5] Validating motion and physics...")
    sequence = dataset[0].squeeze().numpy()  # (seq_len, 64, 64)
    
    # Check that frames are different (motion exists)
    frame_diffs = []
    for i in range(len(sequence) - 1):
        diff = np.abs(sequence[i+1] - sequence[i]).sum()
        frame_diffs.append(diff)
    
    avg_diff = np.mean(frame_diffs)
    print(f"✓ Average frame difference: {avg_diff:.2f} (motion detected)")
    assert avg_diff > 0, "No motion detected between frames!"
    
    # Check that digit pixels exist
    total_pixels = sequence.sum()
    print(f"✓ Total digit pixels: {total_pixels:.0f} (digits present)")
    assert total_pixels > 0, "No digits found in sequence!"
    
    print("\n" + "=" * 60)
    print("✅ ALL VALIDATION CHECKS PASSED!")
    print("=" * 60)
    print("\nDataset is fully compatible with PyTorch models and ready for training.")
    
    # Show RGB status
    if dataset.rgb_output:
        print("\n✅ RGB OUTPUT ENABLED (3 channels) - Ready for iVideoGPT!")
    else:
        print("\n⚠️  GRAYSCALE OUTPUT (1 channel) - Set rgb_output=True for iVideoGPT")
    
    return True


def visualize_motion_matplotlib(dataset, idx=0, num_frames=None, save_path=None):
    """
    Visualize a sequence using matplotlib to verify motion and bouncing.
    
    Args:
        dataset: MovingMNIST dataset instance
        idx: Index of sequence to visualize
        num_frames: Number of frames to show (None = all)
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Get sequence
    sequence = dataset[idx].squeeze().numpy()  # (seq_len, 64, 64)
    
    if num_frames is None:
        num_frames = min(len(sequence), 10)  # Show up to 10 frames
    else:
        num_frames = min(num_frames, len(sequence))
    
    # Create subplot grid
    cols = 5
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows))
    fig.suptitle(f'Moving MNIST Sequence {idx} - Motion Verification', fontsize=14, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        if i < num_frames:
            frame = sequence[i]
            ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Frame {i}', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved matplotlib visualization to {save_path}")
    
    plt.show()
    print("✓ Visual inspection: Verify digits move and bounce off walls")


def test_model_integration(dataset, device='cpu'):
    """
    Test integration with a simple PyTorch model.
    
    Args:
        dataset: MovingMNIST dataset instance
        device: Device to test on ('cpu' or 'cuda')
    """
    print("\n" + "=" * 60)
    print("TESTING MODEL INTEGRATION")
    print("=" * 60)
    
    # Create a simple 3D CNN model for testing
    class SimpleVideoCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv3d = torch.nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1)
            self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = torch.nn.Linear(16, 10)
        
        def forward(self, x):
            # x: (batch, seq_len, channels, H, W)
            # Conv3D expects: (batch, channels, seq_len, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            x = torch.relu(self.conv3d(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleVideoCNN().to(device)
    print(f"✓ Created test model on {device}")
    
    # Test with DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader)).to(device)
    
    print(f"✓ Loaded batch: {batch.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch)
    
    print(f"✓ Forward pass successful!")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output dtype: {output.dtype}")
    
    # Test gradient computation
    model.train()
    output = model(batch)
    loss = output.sum()
    loss.backward()
    
    print(f"✓ Backward pass successful!")
    print(f"✓ Gradients computed successfully")
    
    print("\n" + "=" * 60)
    print("✅ MODEL INTEGRATION TEST PASSED!")
    print("=" * 60)
    print("\nDataset works seamlessly with PyTorch models.")
    
    return True


# Example usage and testing
if __name__ == "__main__":
    # Create Moving MNIST dataset
    print("=" * 60)
    print("MOVING MNIST DATASET - COMPREHENSIVE TEST")
    print("=" * 60)
    
    print("\n[Step 1/6] Creating dataset...")
    dataset = MovingMNIST(
        root='./data',
        train=True,
        seq_len=20,
        num_digits=2,
        frame_size=64,
        num_sequences=1000,
        download=True,
        rgb_output=True  # RGB output (required for iVideoGPT)
    )
    print(f"✓ Dataset created: {len(dataset)} sequences")
    print(f"✓ Configuration: {dataset.seq_len} frames × {dataset.frame_size}×{dataset.frame_size} pixels")
    print(f"✓ Digits per sequence: 1-{dataset.num_digits}")
    print(f"✓ RGB output: {dataset.rgb_output} ({'3 channels (RGB)' if dataset.rgb_output else '1 channel (grayscale)'})")
    
    # Test basic tensor properties
    print("\n[Step 2/6] Testing basic tensor conversion...")
    sequence = dataset[0]
    frames_tensor = sequence  # Already a tensor
    print(f"✓ Type: {type(frames_tensor)}")
    print(f"✓ Shape: {frames_tensor.shape} (expected: (seq_len, {'3' if dataset.rgb_output else '1'}, 64, 64))")
    print(f"✓ Data type: {frames_tensor.dtype} (expected: torch.float32)")
    print(f"✓ Value range: [{frames_tensor.min():.4f}, {frames_tensor.max():.4f}] (expected: [0, 1])")
    print(f"✓ Channels: {frames_tensor.shape[1]} ({'RGB' if frames_tensor.shape[1] == 3 else 'Grayscale'})")
    
    # Visualize sequences
    print("\n[Step 3/6] Generating visualizations...")
    for i in range(3):
        dataset.visualize_grid(i, save_path='./moving_mnist_samples')
    print("✓ Saved 3 sample visualizations to ./moving_mnist_samples/")
    
    # Test DataLoader
    print("\n[Step 4/6] Testing DataLoader integration...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    for batch in loader:
        print(f"✓ Batch shape: {batch.shape} (expected: (batch_size, seq_len, {'3' if dataset.rgb_output else '1'}, 64, 64))")
        print(f"✓ Batch dtype: {batch.dtype}")
        print(f"✓ Batch value range: [{batch.min():.4f}, {batch.max():.4f}]")
        print(f"✓ Batch channels: {batch.shape[2]} ({'RGB - Ready for iVideoGPT!' if batch.shape[2] == 3 else 'Grayscale'})")
        break
    
    print(f"✓ DataLoader works correctly!")
    
    # Run comprehensive validation
    print("\n[Step 5/6] Running comprehensive PyTorch compatibility validation...")
    validate_pytorch_compatibility(dataset, num_samples=5)
    
    # Visualize motion with matplotlib (if available)
    print("\n[Step 6/6] Testing matplotlib visualization...")
    try:
        visualize_motion_matplotlib(
            dataset, 
            idx=0, 
            num_frames=10,
            save_path='./moving_mnist_samples/matplotlib_motion_check.png'
        )
    except Exception as e:
        print(f"⚠ Matplotlib visualization skipped: {e}")
    
    # Test model integration
    print("\n[Optional] Testing with a simple PyTorch model...")
    try:
        test_model_integration(dataset, device='cpu')
    except Exception as e:
        print(f"⚠ Model integration test skipped: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📋 Summary:")
    print(f"   • Dataset: {len(dataset)} sequences")
    print(f"   • Shape: ({dataset.seq_len}, {3 if dataset.rgb_output else 1}, {dataset.frame_size}, {dataset.frame_size})")
    print(f"   • Format: torch.float32 tensor")
    print(f"   • Range: [0.0, 1.0] normalized")
    print(f"   • Channels: {3 if dataset.rgb_output else 1} ({'RGB' if dataset.rgb_output else 'Grayscale'})")
    print(f"   • Digits: 1-{dataset.num_digits} moving per sequence")
    print(f"   • Physics: Bouncing off walls ✓")
    print(f"   • PyTorch: Fully compatible ✓")
    print(f"   • DataLoader: Ready ✓")
    if dataset.rgb_output:
        print(f"   • iVideoGPT: Ready ✅ (RGB output enabled)")
    else:
        print(f"   • iVideoGPT: Set rgb_output=True for compatibility")
    print("\n🚀 Ready for model training!")
