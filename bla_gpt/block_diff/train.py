import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List

from model import BD3LM


class TextDataset(Dataset):
    """A simple dataset for handling tokenized text data."""
    
    def __init__(self, text_ids: torch.Tensor, block_size: int):
        self.text_ids = text_ids
        self.block_size = block_size
    
    def __len__(self):
        return len(self.text_ids) - self.block_size
    
    def __getitem__(self, idx):
        return self.text_ids[idx:idx + self.block_size]


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_masks(input_ids: torch.Tensor, mask_id: int, mask_prob: torch.Tensor) -> torch.Tensor:
    """
    Create masked input by replacing some tokens with mask_id based on mask_prob.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        mask_id: ID of the mask token
        mask_prob: Masking probability for each sample in the batch [batch_size]
    
    Returns:
        Masked input tensor [batch_size, seq_len]
    """
    batch_size, seq_len = input_ids.shape
    
    # Random mask probability per position
    random_probs = torch.rand_like(input_ids.float())
    
    # Expand mask_prob to match input_ids shape
    mask_prob_expanded = mask_prob.unsqueeze(1).expand(-1, seq_len)
    
    # Create masks where random prob < mask prob
    masks = (random_probs < mask_prob_expanded).long()
    
    # Apply masks
    masked_input = input_ids.clone()
    masked_input[masks == 1] = mask_id
    
    return masked_input


def train(
    model: BD3LM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_beta: float = 0.0,
    clip_omega: float = 1.0,
    adapt_schedule: bool = False,
    eval_steps: int = 1000,
    log_steps: int = 100
) -> Dict[str, float]:
    """
    Train the BD3LM model.
    
    Args:
        model: The BD3LM model to train
        dataloader: DataLoader with tokenized text
        optimizer: Optimizer for training
        device: Device to train on
        clip_beta: Lower bound for mask probability
        clip_omega: Upper bound for mask probability
        adapt_schedule: Whether to adapt the masking schedule during training
        eval_steps: How often to evaluate the model
        log_steps: How often to log training progress
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    log_loss = 0.0
    total_steps = 0
    best_eval_loss = float('inf')
    
    # Clipped schedule parameters
    beta = clip_beta
    omega = clip_omega
    
    # Low-discrepancy sampler for t
    def sample_t(batch_size, total_steps, current_step):
        offsets = torch.arange(batch_size) / batch_size
        base = (current_step % batch_size) / batch_size
        times = (base + offsets) % 1.0
        return beta + (omega - beta) * times
    
    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        batch_size = batch.shape[0]
        
        # Sample noise levels from clipped range [beta, omega]
        if adapt_schedule and step % eval_steps == 0 and step > 0:
            # Update schedule based on loss variance
            # This is a placeholder for the adaptive scheduling in the paper
            # In practice, you would adjust beta and omega here
            pass
        
        noise_level = sample_t(batch_size, len(dataloader), step)
        noise_level = torch.tensor(noise_level, device=device)
        
        # Create masked inputs
        masked_input = create_masks(batch, model.mask_id, noise_level)
        
        # Forward pass
        output = model(masked_input, batch, noise_level)
        loss = output['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        log_loss += loss.item()
        total_steps += 1
        
        # Log progress
        if step % log_steps == 0 and step > 0:
            avg_loss = log_loss / log_steps
            pbar.set_description(f"Loss: {avg_loss:.4f}")
            log_loss = 0.0
    
    return {
        "train_loss": total_loss / total_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a BD3LM model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to tokenized data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--block_size", type=int, default=16, help="Block size for block diffusion")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps")
    parser.add_argument("--clip_beta", type=float, default=0.3, help="Lower bound for mask probability")
    parser.add_argument("--clip_omega", type=float, default=0.8, help="Upper bound for mask probability")
    parser.add_argument("--adapt_schedule", action="store_true", help="Adapt masking schedule during training")
    parser.add_argument("--warmup_steps", type=int, default=2500, help="Learning rate warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to train on")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = torch.load(args.data_path)
    dataset = TextDataset(data, args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device(args.device)
    model = BD3LM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        block_size=args.block_size,
        mask_id=args.vocab_size - 1,  # Assuming the last token is [MASK]
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Train model
    metrics = train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        clip_beta=args.clip_beta,
        clip_omega=args.clip_omega,
        adapt_schedule=args.adapt_schedule,
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print(f"Training completed. Final loss: {metrics['train_loss']:.4f}")


if __name__ == "__main__":
    main()