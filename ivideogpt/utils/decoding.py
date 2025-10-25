"""
Utilities for robust token decoding with fallback mechanisms.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

def safe_decode_tokens(
    vq_model,
    tokens: torch.Tensor,
    context_length: int,
    fallback_strategy: str = 'nearest',
    max_retries: int = 3
) -> Tuple[torch.Tensor, bool]:
    """
    Safely decode tokens with fallback mechanisms for handling invalid values.
    
    Args:
        vq_model: The VQ-VAE model used for decoding
        tokens: Tensor of token indices to decode (B, T)
        context_length: Number of context frames
        fallback_strategy: Strategy for handling invalid tokens ('nearest' or 'zero')
        max_retries: Maximum number of retry attempts
    
    Returns:
        Tuple of:
        - decoded tensor of shape (B, T, C, H, W)
        - success flag indicating if primary decoding succeeded
    """
    # Input validation
    if tokens.ndim != 2:
        tokens = tokens.view(tokens.size(0), -1)
        
    # Clone tokens to avoid modifying input
    working_tokens = tokens.clone()
    
    # First try normal decoding
    try:
        with torch.inference_mode():
            decoded = vq_model.decode_tokens(working_tokens)
            if not (torch.isnan(decoded).any() or torch.isinf(decoded).any()):
                return decoded, True
    except Exception as e:
        logger.warning(f"Primary decoding failed: {e}")
    
    # If we get here, primary decoding failed
    logger.info("Attempting fallback decoding...")
    
    # Get valid token range
    valid_range = (0, vq_model.num_vq_embeddings - 1)
    
    for attempt in range(max_retries):
        # Clean up invalid tokens based on strategy
        if fallback_strategy == 'nearest':
            # Replace invalid tokens with nearest valid token
            working_tokens = torch.clamp(working_tokens, 
                                       min=valid_range[0],
                                       max=valid_range[1])
        else:  # 'zero' strategy
            # Replace invalid tokens with 0 (pad token)
            invalid_mask = (working_tokens < valid_range[0]) | (working_tokens > valid_range[1])
            working_tokens[invalid_mask] = 0
        
        try:
            with torch.inference_mode():
                decoded = vq_model.decode_tokens(working_tokens)
                
                # Validate output
                if torch.isnan(decoded).any() or torch.isinf(decoded).any():
                    if attempt == max_retries - 1:
                        # Last attempt - return zero tensor
                        logger.warning("All decoding attempts failed, returning zeros")
                        return torch.zeros_like(decoded), False
                    continue
                
                # If we get here, decoding succeeded
                return decoded, True
                
        except Exception as e:
            logger.warning(f"Fallback attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Last attempt - return zero tensor
                shape = list(working_tokens.shape[:-1]) + list(vq_model.output_shape)
                return torch.zeros(shape, device=working_tokens.device), False
    
    # Should never get here, but just in case
    return torch.zeros_like(decoded), False


def normalize_predictions(
    predictions: torch.Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Normalize prediction tensor to valid range, handling edge cases.
    
    Args:
        predictions: Tensor to normalize
        min_val: Target minimum value
        max_val: Target maximum value 
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor
    """
    # Replace NaN/Inf with zeros
    predictions = torch.nan_to_num(predictions, nan=0.0, posinf=max_val, neginf=min_val)
    
    # Clip to remove any remaining extreme values
    predictions = torch.clamp(predictions, min_val, max_val)
    
    # Normalize to [0, 1]
    predictions = (predictions - min_val) / (max_val - min_val + eps)
    
    # Scale back to target range
    predictions = predictions * (max_val - min_val) + min_val
    
    return predictions


def verify_tokens(tokens: torch.Tensor, vq_model) -> torch.Tensor:
    """
    Verify token indices are valid and clean if needed.
    
    Args:
        tokens: Tensor of token indices
        vq_model: The VQ-VAE model (used to determine valid token range)
        
    Returns:
        Cleaned token tensor
    """
    valid_range = (0, vq_model.num_vq_embeddings - 1)
    
    # Create mask for invalid tokens
    invalid_mask = (tokens < valid_range[0]) | (tokens > valid_range[1])
    
    if invalid_mask.any():
        logger.warning(f"Found {invalid_mask.sum().item()} invalid tokens")
        # Replace invalid tokens with pad token (0)
        tokens = tokens.clone()  # Create copy to avoid modifying input
        tokens[invalid_mask] = 0
        
    return tokens
