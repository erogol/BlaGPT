import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt


def create_clipped_schedule(beta: float, omega: float, num_steps: int) -> torch.Tensor:
    """
    Create a clipped noise schedule that samples mask rates from U[beta, omega].
    
    Args:
        beta: Lower bound for mask probability
        omega: Upper bound for mask probability
        num_steps: Number of steps in the schedule
    
    Returns:
        Tensor of noise levels [num_steps]
    """
    return torch.linspace(beta, omega, num_steps)


def apply_noise(input_ids: torch.Tensor, mask_id: int, noise_level: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Apply noise to input tokens by masking them according to noise_level.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        mask_id: ID of the mask token
        noise_level: Noise level (between 0 and 1, where 1 = fully masked)
                     Either a scalar or a tensor of shape [batch_size]
    
    Returns:
        Noised input tokens [batch_size, seq_len]
    """
    if isinstance(noise_level, float):
        noise_level = torch.tensor([noise_level], device=input_ids.device)
    
    batch_size, seq_len = input_ids.shape
    
    # Expand noise_level to match input_ids shape
    noise_level = noise_level.view(-1, 1).expand(-1, seq_len)
    
    # Create random mask
    random_mask = torch.rand_like(input_ids.float()) < noise_level
    
    # Apply mask
    noised_ids = input_ids.clone()
    noised_ids[random_mask] = mask_id
    
    return noised_ids


def calculate_variance(model, dataloader, device, num_samples=10):
    """
    Calculate the variance of the gradient estimator for different noise schedules.
    
    Args:
        model: The BD3LM model
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        num_samples: Number of samples to use for variance estimation
    
    Returns:
        Dictionary mapping noise schedule parameters to variance
    """
    model.eval()
    variances = {}
    
    # Test different clipped schedules
    schedule_params = [
        (0.0, 0.5),  # U[0, .5]
        (0.3, 0.8),  # U[.3, .8]
        (0.5, 1.0),  # U[.5, 1]
        (0.0, 1.0),  # U[0, 1]
    ]
    
    for beta, omega in schedule_params:
        all_losses = []
        
        # Get num_samples batches
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            batch = batch.to(device)
            batch_size = batch.shape[0]
            
            # Sample noise levels from the schedule
            noise_levels = torch.linspace(beta, omega, batch_size, device=device)
            
            # Apply noise
            noised_batch = apply_noise(batch, model.mask_id, noise_levels)
            
            # Get loss
            with torch.no_grad():
                output = model(noised_batch, batch, noise_levels)
                
            all_losses.append(output['loss'].item())
        
        # Calculate variance
        variances[(beta, omega)] = np.var(all_losses)
    
    return variances


def plot_schedule_variance(variances):
    """
    Plot the variance of different noise schedules.
    
    Args:
        variances: Dictionary mapping noise schedule parameters to variance
    """
    labels = []
    values = []
    
    for (beta, omega), var in variances.items():
        labels.append(f"U[{beta}, {omega}]")
        values.append(var)
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("Variance of Different Noise Schedules")
    plt.xlabel("Noise Schedule")
    plt.ylabel("Variance")
    plt.yscale('log')
    plt.tight_layout()
    
    return plt.gcf()


def estimate_perplexity(model, dataloader, device, noise_levels=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Estimate perplexity of the model on a dataset.
    
    Args:
        model: The BD3LM model
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        noise_levels: List of noise levels to evaluate at
    
    Returns:
        Dictionary mapping noise levels to perplexity
    """
    model.eval()
    perplexities = {}
    
    for noise_level in noise_levels:
        total_loss = 0.0
        total_tokens = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            batch_size, seq_len = batch.shape
            
            # Apply noise at this level
            noised_batch = apply_noise(batch, model.mask_id, noise_level)
            
            # Get loss
            with torch.no_grad():
                output = model(noised_batch, batch, torch.tensor([noise_level], device=device))
                
            # Sum loss over all tokens
            loss = output['loss'].item() * (noised_batch == model.mask_id).sum().item()
            total_loss += loss
            total_tokens += (noised_batch == model.mask_id).sum().item()
        
        # Calculate perplexity
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        perplexities[noise_level] = perplexity
    
    return perplexities