"""
Loss functions for BlaGPT models.

This module contains various loss functions used in training:
- Standard cross-entropy loss utilities
- Z-loss regularization 
- Token Order Prediction (TOP) loss
- Multi-Token Prediction (MTP) loss utilities
"""

import torch
import torch.nn.functional as F
from attentions import soft_cap


def compute_z_loss(logits, dim=-1, eps=1e-20):
    """
    Compute z-loss regularization to prevent model from being too confident.

    Args:
        logits: Raw logits from model of shape [batch, seq_len, vocab_size]
        dim: Dimension along which to compute z-loss (usually vocab dimension)
        eps: Small constant for numerical stability

    Returns:
        z_loss: Scalar z-loss term to add to training loss
    """
    # Get log of the partition function (logsumexp)
    log_z = torch.logsumexp(logits, dim=dim, keepdim=True)

    # Compute mean of log_z squared
    z_loss = torch.square(torch.max(log_z, torch.zeros_like(log_z)))
    z_loss = torch.mean(z_loss)

    return z_loss


def construct_top_targets(seq, vocab_size, window_size):
    """
    Convert a token sequence to TOP target sequence using optimized tensor operations.
    Based on official implementation from TOP paper repository.
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len + window_size)
        vocab_size: Size of vocabulary  
        window_size: Window size for proximity scoring
        
    Returns:
        TOP target tensor of shape (batch_size, seq_len, vocab_size) with proximity scores
    """
    batch_size, total_len = seq.shape
    seq_len = total_len - window_size
    device = seq.device
    dtype = seq.dtype
    
    if seq_len <= 0:
        return torch.full((batch_size, 1, vocab_size), float('-inf'), device=device)
    
    # Initialize output tensor with -inf
    out = torch.full((batch_size, seq_len, vocab_size), float('-inf'), device=device, dtype=torch.float)
    
    # Track next occurrence positions for all tokens
    next_occurrence = torch.full((batch_size, vocab_size), total_len, device=device, dtype=torch.long)
    
    # Process sequence in reverse to find next occurrences efficiently
    for t in range(total_len - 1, -1, -1):
        # Get tokens at position t for all sequences in batch
        tokens_at_t = seq[:, t]  # Shape: (batch_size,)
        
        # Create valid mask for tokens within vocabulary
        valid_mask = (tokens_at_t >= 0) & (tokens_at_t < vocab_size)
        
        # Update next occurrence positions using one-hot encoding
        if valid_mask.any():
            # Convert tokens to one-hot for efficient updating
            token_one_hot = F.one_hot(tokens_at_t, num_classes=vocab_size).float()  # (B, V)
            
            # Update next occurrence positions where tokens are valid
            update_mask = valid_mask.unsqueeze(1)  # (B, 1)
            next_occurrence = torch.where(
                update_mask & (token_one_hot > 0), 
                t, 
                next_occurrence
            )
        
        # Compute distances and scores for output positions
        if t < seq_len:
            # Calculate distances to next occurrence
            distances = next_occurrence - t  # (B, V)
            
            # Create window mask: 0 < distance <= window_size
            window_mask = (distances > 0) & (distances <= window_size)
            
            # Compute proximity scores: window_size - distance (closer = higher score)
            scores = torch.where(window_mask, window_size - distances, torch.tensor(float('-inf'), device=device))
            
            # Assign scores to output tensor
            out[:, t, :] = scores.float()
    
    return out


def listnet_loss(y_pred, y_true):
    """
    ListNet loss from "Learning to Rank: From Pairwise Approach to Listwise Approach".
    Official implementation from TOP paper repository.
    
    Args:
        y_pred: Model predictions of shape [*, slate_length] 
        y_true: Ground truth labels of shape [*, slate_length]
        
    Returns:
        Loss value as a torch.Tensor
    """
    return torch.mean(-torch.sum(
        F.softmax(y_true, dim=-1).nan_to_num(nan=0) * 
        F.log_softmax(y_pred, dim=-1), 
        dim=-1
    ))


def compute_top_loss(model, x, idx, targets):
    """
    Compute Token Order Prediction (TOP) loss for the given model and inputs.
    
    Args:
        model: The GPT model instance
        x: Hidden states from final transformer layer of shape (batch_size, seq_len, n_embd)
        idx: Input token sequence of shape (batch_size, seq_len)
        targets: Target token sequence of shape (batch_size, seq_len)
        
    Returns:
        TOP loss scalar or None if not applicable
    """
    if not (model.config.use_top and hasattr(model, "top_head")):
        return None
        
    # Create extended sequence for TOP target construction
    window_size = min(model.config.top_window_size, targets.size(1))
    
    if window_size <= 0:
        return None
        
    # Create extended sequence: current input + future targets (for window)
    extended_seq = torch.cat([idx, targets[:, :window_size]], dim=1)
    
    # Generate TOP targets - use optimized implementation  
    from top_kernels import get_top_target_fn
    force_optimized = getattr(model.config, 'top_force_optimized', True)
    target_fn = get_top_target_fn(force_optimized=force_optimized)
    top_targets = target_fn(extended_seq, model.config.vocab_size, window_size)
    
    # Compute TOP predictions (only for positions we have targets for)
    seq_len = min(x.size(1), top_targets.size(1))
    top_logits = model.top_head(x[:, :seq_len]).float()
    
    # Apply soft capping if enabled
    if model.soft_cap > 0.0:
        top_logits = soft_cap(top_logits, model.soft_cap)
    
    # Compute TOP loss
    top_loss = listnet_loss(top_logits, top_targets[:, :seq_len])
    
    return top_loss


def construct_fsp_window_ids(targets, horizon, eot_token_id=-1):
    """
    Build the FSP bag-of-words target window ids and validity mask (Eq. 9
    of arXiv:2510.14751): a(t, tau)_i = 1[i in {x_{t+2}, ..., x_{t+tau}}].

    targets[:, i] == x_{i+1}, so the bag for position i is
    targets[:, i+1 : i+tau] (token ids x_{i+2}..x_{i+tau}).

    Args:
        targets: Next-token targets, shape (B, T); targets[:, i] == x_{i+1}.
        horizon: tau, the future window end offset (window length = tau - 1).
        eot_token_id: token id marking a document boundary. Any window
            containing this id is masked invalid rather than truncated, so
            the bag never mixes tokens from two different documents.

    Returns:
        (window_ids, valid_mask, valid_count):
            window_ids: (B, valid_count, tau - 1) target token ids per bag.
            valid_mask: (B, valid_count) bool, False where the window
                crosses a document boundary.
            valid_count: T - tau + 1, number of positions with a
                fully in-range window (0 if tau > T).
    """
    B, T = targets.shape
    valid_count = T - horizon + 1
    if valid_count <= 0:
        return None, None, 0

    device = targets.device
    offsets = torch.arange(1, horizon, device=device)  # 1 .. tau-1
    base = torch.arange(valid_count, device=device).unsqueeze(1)  # (valid_count, 1)
    window_positions = base + offsets.unsqueeze(0)  # (valid_count, tau - 1)

    window_ids = targets[:, window_positions]  # (B, valid_count, tau - 1)
    crosses_boundary = (window_ids == eot_token_id).any(dim=-1)  # (B, valid_count)
    valid_mask = ~crosses_boundary
    return window_ids, valid_mask, valid_count


def compute_fsp_loss(model, x, idx, targets):
    """
    Compute the Future Summary Prediction (FSP) auxiliary loss: a
    memory-efficient, exact rewrite of the reweighted binary cross-entropy
    bag-of-words loss from Mahajan et al., "Beyond Multi-Token Prediction:
    Pretraining LLMs with Future Summaries" (arXiv:2510.14751), Eq. 9-10.

    Target (Eq. 9): a(t, tau)_i = 1[i in {x_{t+2}, ..., x_{t+tau}}], a
    multi-hot indicator over the vocabulary of tokens appearing in the
    future window starting two steps ahead of t (skipping x_{t+1}, which
    NTP already supervises) through tau steps ahead.

    Loss (Eq. 10, uniform w(i) = 1 -- see E8_NOTE.md for why we do not use
    the paper's optional tf-idf reweighting):
        l_a = -sum_i [a_i * log sigmoid(z_i) + (1 - a_i) * log(1 - sigmoid(z_i))]
    Using log(1 - sigmoid(z)) = -softplus(z) and
    log(sigmoid(z)) - log(1 - sigmoid(z)) = z, this is algebraically exactly:
        l_a = sum_i softplus(z_i) - sum_{i in bag} z_i
    The first term is a plain reduction over the existing (dense) logits
    tensor. The second term is computed by *gathering* z at the bag's
    token ids (torch.gather) and de-duplicating repeated ids so each
    unique vocabulary id contributes at most once, matching the multi-hot
    semantics -- we never materialize a (batch, seq, vocab) multi-hot
    label tensor.

    Args:
        model: The GPT model instance (must have `fsp_head` when enabled).
        x: Hidden states from the final transformer layer, shape (B, T, n_embd).
        idx: Input token ids, shape (B, T).
        targets: Next-token targets, shape (B, T); targets[:, i] == x_{i+1}.

    Returns:
        Scalar FSP loss averaged over valid (non-masked) positions, or
        None if FSP is disabled or there are no valid positions at all.
    """
    if not (model.config.fsp_weight > 0.0 and hasattr(model, "fsp_head")):
        return None

    tau = model.config.fsp_horizon
    vocab_size = model.config.vocab_size
    eot_id = getattr(model.config, "fsp_eot_token_id", -1)

    window_ids, valid_mask, valid_count = construct_fsp_window_ids(targets, tau, eot_id)
    if valid_count <= 0 or not valid_mask.any():
        return None

    # Auxiliary head logits, only for positions that could have a valid window.
    fsp_logits = model.fsp_head(x[:, :valid_count]).float()  # (B, valid_count, V)
    if model.soft_cap > 0.0:
        fsp_logits = soft_cap(fsp_logits, model.soft_cap)

    # Dense term: sum_i softplus(z_i), no target-dependent tensor needed.
    softplus_sum = F.softplus(fsp_logits).sum(dim=-1)  # (B, valid_count)

    # Sparse/gather term: sum over the *unique* ids in the bag of z_i.
    # Gather logits at the bag's token ids (memory: B * valid_count * window_len,
    # not B * valid_count * V), then de-duplicate along the window axis by
    # sorting the ids and keeping only first occurrences.
    z_at_ids = torch.gather(fsp_logits, dim=-1, index=window_ids)  # (B, valid_count, window_len)
    sorted_ids, sort_idx = torch.sort(window_ids, dim=-1)
    z_sorted = torch.gather(z_at_ids, dim=-1, index=sort_idx)
    first_occurrence = torch.ones_like(sorted_ids, dtype=torch.bool)
    first_occurrence[..., 1:] = sorted_ids[..., 1:] != sorted_ids[..., :-1]
    pos_sum = (z_sorted * first_occurrence.float()).sum(dim=-1)  # (B, valid_count)

    # Normalize by vocab size (mean over V, not the paper's raw sum over V)
    # to keep the loss on a numeric scale comparable to NTP cross-entropy;
    # see E8_NOTE.md "Normalization" section.
    per_position_loss = (softplus_sum - pos_sum) / vocab_size

    valid_mask_f = valid_mask.float()
    denom = valid_mask_f.sum().clamp(min=1.0)
    fsp_loss = (per_position_loss * valid_mask_f).sum() / denom
    return fsp_loss