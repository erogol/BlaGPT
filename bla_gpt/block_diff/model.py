import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class RotaryPositionalEmbedding(nn.Module):
    """
    Simplified but optimized implementation of Rotary Positional Embeddings.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Only rotate half of the dimensions
        self.rotary_dim = dim // 2
        
        # Initialize inverse frequency buffer
        freqs = 1.0 / (10000 ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer("freqs", freqs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor x.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
        
        Returns:
            Tensor with positional information
        """
        # Check and remember shape
        orig_shape = x.shape
        if len(orig_shape) == 3:
            batch_size, seq_len, dim = orig_shape
            has_head_dim = False
        elif len(orig_shape) == 4:
            batch_size, seq_len, heads, dim = orig_shape
            has_head_dim = True
            # Reshape to [batch_size, seq_len, heads*dim] for simpler processing
            x = x.reshape(batch_size, seq_len, heads * dim)
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, got shape {x.shape}")
        
        # Limit to max sequence length
        seq_len = min(seq_len, self.max_seq_len)
        
        # Create a copy of the input for output
        x_out = x.clone()
        
        # Get positional indices
        positions = torch.arange(seq_len, device=x.device).float()
        
        # Process pairs of dimensions at a time
        for dim_idx in range(0, self.rotary_dim, 2):
            if dim_idx + 1 >= dim:
                break  # Safety check for odd dimensions
                
            # Compute sinusoidal frequencies for this dimension pair
            freq = positions * self.freqs[dim_idx // 2]  # [seq_len]
            cos = torch.cos(freq)  # [seq_len]
            sin = torch.sin(freq)  # [seq_len]
            
            # Apply rotations to all batch elements at once, but one position at a time
            for pos in range(seq_len):
                # Get current position's sin and cos
                cos_pos = cos[pos].item()
                sin_pos = sin[pos].item()
                
                # Get values for this position
                x_i = x[:, pos, dim_idx].clone()
                x_i1 = x[:, pos, dim_idx + 1].clone()
                
                # Apply rotation
                x_out[:, pos, dim_idx] = x_i * cos_pos - x_i1 * sin_pos
                x_out[:, pos, dim_idx + 1] = x_i * sin_pos + x_i1 * cos_pos
        
        # Reshape back if needed
        if has_head_dim:
            x_out = x_out.reshape(batch_size, seq_len, heads, dim)
        
        return x_out


class TransformerLayer(nn.Module):
    """A single transformer layer with self-attention and feed-forward network."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, src: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class BD3LM(nn.Module):
    """
    Block Discrete Denoising Diffusion Language Model (BD3-LM) as described in the paper.
    
    This model uses a transformer architecture with block-causal attention 
    and performs diffusion within each block.
    """
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 768, 
                 nhead: int = 12, 
                 num_layers: int = 12,
                 dim_feedforward: int = 3072, 
                 dropout: float = 0.1,
                 max_seq_len: int = 1024,
                 block_size: int = 16,
                 mask_id: int = None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size
        self.mask_id = mask_id if mask_id is not None else vocab_size - 1
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rotary_pe = RotaryPositionalEmbedding(d_model, max_seq_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _create_block_causal_mask(self, seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
        """
        Creates a block-causal attention mask.
        Tokens in block b attend to tokens in blocks 1 to b.
        """
        num_blocks = (seq_len + block_size - 1) // block_size  # Ceiling division
        
        # Create block-level causal mask
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
        
        # Expand to token level
        expanded_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
        
        # Truncate to the actual sequence length
        expanded_mask = expanded_mask[:seq_len, :seq_len]
        
        # Convert to attention mask (1 = attend, 0 = don't attend)
        # Then convert to the format expected by MultiheadAttention (0 = attend, -inf = don't attend)
        attn_mask = (1 - expanded_mask) * -1e9
        
        return attn_mask
    
    def forward(self, 
                input_ids: torch.Tensor,
                target_ids: torch.Tensor,
                noise_level: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the BD3LM model.
        
        Args:
            input_ids: Input token IDs with noise applied [batch_size, seq_len]
            target_ids: Target token IDs (clean tokens) [batch_size, seq_len]
            noise_level: Noise level for each example in the batch [batch_size]
                        (between 0 and 1, where 1 = fully masked)
        
        Returns:
            Dictionary containing 'loss' and 'logits'
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # If noise level not provided, use a uniform distribution between 0 and 1
        if noise_level is None:
            noise_level = torch.rand(batch_size, device=device)
        elif isinstance(noise_level, float):
            noise_level = torch.tensor([noise_level], device=device).expand(batch_size)
        
        # Get features from the model
        x = self._forward_features(input_ids)
        
        # Get logits
        logits = self.output_projection(x)
        
        # Calculate loss
        loss = self._calculate_bd3_loss(logits, input_ids, target_ids, noise_level)
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def _calculate_bd3_loss(self, 
                           logits: torch.Tensor, 
                           input_ids: torch.Tensor, 
                           target_ids: torch.Tensor,
                           noise_level: torch.Tensor) -> torch.Tensor:
        """
        Calculate the BD3 loss as described in the paper (Eq. 8).
        
        The loss is a weighted cross-entropy loss that depends on the noise level.
        For masked tokens (input_id == mask_id), we predict the original token.
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for cross-entropy
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)
        
        # Calculate cross-entropy loss for all tokens
        cross_entropy = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        cross_entropy = cross_entropy.reshape(batch_size, seq_len)
        
        # Create mask for tokens that are masked in input
        is_masked = (input_ids == self.mask_id).float()
        
        # Weight the loss based on noise level as per Eq. 8
        # α'_t / (1 - α_t) where α_t = 1 - noise_level
        # For linear schedule, α'_t = -1, so the weight is 1 / noise_level
        noise_level = noise_level.unsqueeze(1)  # [batch_size, 1]
        weights = 1.0 / noise_level
        
        # Apply weights only to masked tokens
        weighted_loss = cross_entropy * is_masked * weights
        
        # Return mean loss
        return weighted_loss.sum() / (is_masked.sum() + 1e-8)  # Add small epsilon to avoid division by zero
    
    def generate(self, 
                prompt: Optional[torch.Tensor] = None, 
                max_len: int = 1024, 
                num_diffusion_steps: int = 1000,
                temperature: float = 1.0,
                top_p: float = 0.9,
                device: torch.device = None) -> torch.Tensor:
        """
        Generate a sequence using block diffusion.
        
        Args:
            prompt: Optional prompt tokens [batch_size, prompt_len]
            max_len: Maximum sequence length to generate
            num_diffusion_steps: Number of diffusion steps per block
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            device: Device to generate on
        
        Returns:
            Generated sequence [batch_size, seq_len]
        """
        if device is None and prompt is not None:
            device = prompt.device
        elif device is None:
            device = next(self.parameters()).device
        
        batch_size = 1 if prompt is None else prompt.shape[0]
        
        # Initialize with prompt or create an empty sequence
        if prompt is not None:
            generated = prompt
            seq_len = prompt.shape[1]
        else:
            generated = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            seq_len = 0
        
        # Calculate number of blocks needed
        total_blocks = (max_len + self.block_size - 1) // self.block_size
        
        # Generate blocks autoregressively
        for block_idx in range(seq_len // self.block_size, total_blocks):
            # Initialize this block with mask tokens
            if seq_len < max_len:
                current_block_size = min(self.block_size, max_len - seq_len)
                block = torch.full((batch_size, current_block_size), self.mask_id, 
                                  dtype=torch.long, device=device)
                
                # Append masked block to the sequence
                generated = torch.cat([generated, block], dim=1)
                seq_len += current_block_size
            
            # Create cached key-values for efficiency
            cached_kv = None
            
            # Perform diffusion within the current block
            for step in range(num_diffusion_steps):
                # Get logits for the entire sequence
                with torch.no_grad():
                    # Get the current sequence length (important for positional embeddings)
                    current_seq_len = generated.shape[1]
                    
                    # Get logits from model
                    logits = self.output_projection(
                        self._forward_features(generated)
                    )
                
                # Extract logits for the current block
                block_start = block_idx * self.block_size
                block_end = min(block_start + self.block_size, current_seq_len)
                block_logits = logits[:, block_start:block_end, :]
                
                # Apply temperature
                block_logits = block_logits / temperature
                
                # Nucleus sampling
                for i in range(batch_size):
                    for j in range(block_end - block_start):
                        pos = block_start + j
                        if pos < current_seq_len and generated[i, pos] == self.mask_id:
                            probs = F.softmax(block_logits[i, j], dim=-1)
                            
                            # Nucleus sampling (top-p)
                            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                            sorted_indices_to_keep = cumsum_probs <= top_p
                            sorted_indices_to_keep[0] = True  # Always keep the top token
                            
                            # Create new distribution from top-p tokens
                            sorted_probs[~sorted_indices_to_keep] = 0.0
                            sorted_probs = sorted_probs / sorted_probs.sum()
                            
                            # Sample from this distribution
                            sampled_idx = torch.multinomial(sorted_probs, 1).item()
                            token_id = sorted_indices[sampled_idx].item()
                            
                            # Update the token in the generated sequence
                            generated[i, pos] = token_id
                
                # Check if all masks in the block are gone
                masks_in_block = (generated[:, block_start:block_end] == self.mask_id).sum().item()
                if masks_in_block == 0:
                    break
        
        return generated
    
    def _forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Helper method to get features without computing loss."""
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # Create block-causal attention mask
        seq_len = input_ids.shape[1]
        device = input_ids.device
        attn_mask = self._create_block_causal_mask(seq_len, self.block_size, device)
        
        # Apply rotary position embeddings
        x = self.rotary_pe(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
            
        return x