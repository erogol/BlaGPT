from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from coqpit import Coqpit
from norms import RMSNorm
from torch.utils.checkpoint import checkpoint


@dataclass
class AveyConfig(Coqpit):
    """Configuration for Avey model - defaults from the paper"""

    # Model architecture (Small model by default)
    block_size: int = 1024  # Maximum sequence length
    vocab_size: int = 50304  # Vocabulary size (same as GPT-2)
    n_embd: int = 768  # Embedding dimension
    n_layer: int = 26  # Number of layers (26 for 153M model)
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms

    # Avey-specific parameters (from paper)
    split_size: int = 64  # Size of each split
    top_k: int = 7  # Number of top splits to select
    expansion_factor: int = 4  # Expansion factor in enricher
    tail_ratio: float = 0.5  # Portion of enriched embedding for contextualizer
    max_context_width: int = 8192  # Maximum context width for contextualizer

    # Training parameters
    context_width: int = 512  # Maximum context width for training
    tie_embed_weights: bool = True  # Weight tying between input and output embeddings

    # Memory optimization parameters
    use_gradient_checkpointing: bool = (
        False  # Enable gradient checkpointing for memory savings
    )
    chunk_size: int = 64  # Chunk size for similarity computations
    dynamic_v_allocation: bool = True  # Dynamically allocate V matrix based on context

    # Optimizer configuration (following bla_gpt pattern)
    optimizer_name: str = "AdamW"
    optimizer_args: dict = None

    # Overriding default train.py parameters
    compile_model: bool = False  # Whether to compile the model with torch.compile
    # device_batch_size: int = 8  # Batch size per device

    def __post_init__(self):
        if self.optimizer_args is None:
            self.optimizer_args = {
                "betas": (0.9, 0.95),
                "eps": 1e-12,
                "weight_decay": 0.1,
            }


class Enricher(nn.Module):
    """Enricher: Position-wise neural network that expands embeddings"""

    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        self.d_model = d_model
        self.expanded_dim = d_model * expansion_factor
        # Equation 1 in paper includes bias
        self.linear = nn.Linear(d_model, self.expanded_dim, bias=True)

    def forward(self, x):
        # Apply linear transformation and ReLU² activation
        # Use in-place operation for memory efficiency
        x = self.linear(x)
        return F.relu(x, inplace=True) ** 2  # ReLU² as per paper


class OptimizedContextualizer(nn.Module):
    """Memory-optimized Contextualizer: Embedding-wise neural network with dynamic parameterization"""

    def __init__(
        self, mt_dim, max_context_width=8192, dynamic_allocation=True, chunk_size=512
    ):
        super().__init__()
        self.mt_dim = mt_dim
        self.mt_half = mt_dim // 2
        self.max_context_width = max_context_width
        self.dynamic_allocation = dynamic_allocation
        self.chunk_size = chunk_size

        if dynamic_allocation:
            # Only allocate a small initial V matrix and expand as needed
            self.register_buffer("V_cache", None)
            self.V_param = nn.Parameter(
                torch.randn(1, 1) * 0.02
            )  # Dummy parameter for initialization
        else:
            # Traditional approach - allocate full matrix
            self.V = nn.Parameter(
                torch.randn(max_context_width, max_context_width) * 0.02
            )

        # Optional biases as per Equation 2
        self.bias = nn.Parameter(torch.zeros(1, 1, self.mt_half))

    def _get_v_matrix(self, context_width):
        """Get V matrix of appropriate size, creating or expanding as needed"""
        if not self.dynamic_allocation:
            return self.V[:context_width, :context_width]

        # Check if we need to create or expand the cached V matrix
        if self.V_cache is None or self.V_cache.shape[0] < context_width:
            # Create new V matrix of required size
            device = self.V_param.device
            dtype = self.V_param.dtype

            new_size = max(
                context_width,
                self.V_cache.shape[0] * 2 if self.V_cache is not None else 64,
            )
            new_size = min(new_size, self.max_context_width)

            new_V = torch.randn(new_size, new_size, device=device, dtype=dtype) * 0.02

            # Copy existing values if we have them
            if self.V_cache is not None:
                old_size = self.V_cache.shape[0]
                new_V[:old_size, :old_size] = self.V_cache

            self.V_cache = new_V

        return self.V_cache[:context_width, :context_width]

    def _chunked_similarity_computation(
        self, x_right_norm, context_width, weights=None
    ):
        """Compute cosine similarity matrix in chunks to save memory"""
        batch_size = x_right_norm.shape[0]
        device = x_right_norm.device

        # Pre-allocate result tensor
        result = torch.zeros(
            batch_size,
            context_width,
            self.mt_half,
            device=device,
            dtype=x_right_norm.dtype,
        )

        V_slice = self._get_v_matrix(context_width)

        # Process in chunks to reduce memory usage
        for start_idx in range(0, context_width, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, context_width)
            chunk_size = end_idx - start_idx

            # Compute similarity for this chunk
            x_chunk = x_right_norm[
                :, start_idx:end_idx, :
            ]  # [batch, chunk_size, mt_half]

            # Compute cosine similarity: chunk × full
            cosine_sim_chunk = torch.bmm(
                x_chunk, x_right_norm.transpose(1, 2)
            )  # [batch, chunk_size, context_width]

            # Apply V matrix weighting
            V_chunk = V_slice[start_idx:end_idx, :].unsqueeze(
                0
            )  # [1, chunk_size, context_width]
            weighted_sim_chunk = V_chunk * cosine_sim_chunk

            # Apply ranker weights if provided
            if weights is not None:
                n_splits = weights.shape[1]
                split_size = context_width // n_splits

                weight_vector = torch.zeros(context_width, device=device)
                for i in range(n_splits):
                    s_idx = i * split_size
                    e_idx = min(s_idx + split_size, context_width)
                    weight_vector[s_idx:e_idx] = weights[
                        0, i
                    ]  # Broadcast across batch later

                # Apply weights: [batch, chunk_size, context_width] * [context_width]
                weighted_sim_chunk = weighted_sim_chunk * weight_vector.unsqueeze(
                    0
                ).unsqueeze(0)

            # Matrix multiplication: (V ⊙ similarity) × x_right
            chunk_result = torch.bmm(
                weighted_sim_chunk, x_right_norm
            )  # [batch, chunk_size, mt_half]
            result[:, start_idx:end_idx, :] = chunk_result

        return result

    def forward(self, x, weights=None):
        # x shape: [batch, context_width, mt_dim]
        batch_size, context_width, _ = x.shape

        # Split into left and right portions
        x_left = x[..., : self.mt_half]  # [batch, context, mt/2]
        x_right = x[..., self.mt_half :]  # [batch, context, mt/2]

        # Normalize for cosine similarity (N(Ztr) in the paper)
        x_right_norm = F.normalize(x_right, p=2, dim=-1)

        # Use chunked computation for large contexts
        if context_width > self.chunk_size:
            contextualized = self._chunked_similarity_computation(
                x_right_norm, context_width, weights
            )
        else:
            # Standard computation for small contexts
            cosine_sim = torch.bmm(x_right_norm, x_right_norm.transpose(1, 2))
            V_slice = self._get_v_matrix(context_width)
            weighted_sim = V_slice.unsqueeze(0) * cosine_sim

            if weights is not None:
                n_splits = weights.shape[1]
                split_size = context_width // n_splits
                weight_matrix = torch.zeros(batch_size, context_width, device=x.device)
                for i in range(n_splits):
                    start_idx = i * split_size
                    end_idx = min(start_idx + split_size, context_width)
                    weight_matrix[:, start_idx:end_idx] = weights[:, i].unsqueeze(1)

                weight_matrix_expanded = weight_matrix.unsqueeze(1).expand(
                    -1, context_width, -1
                )
                weighted_sim = weighted_sim * weight_matrix_expanded

            contextualized = torch.bmm(weighted_sim, x_right)

        # Add bias and apply gating
        contextualized = contextualized + self.bias
        contextualized = torch.sigmoid(contextualized)

        # Element-wise multiplication: Ztl ⊙ σ(...)
        output = x_left * contextualized

        return output


class Fuser(nn.Module):
    """Fuser: Combines contextualized and uncontextualized features"""

    def __init__(self, mh_dim, mt_half_dim, d_model):
        super().__init__()
        # Note: Paper mentions no bias in Equation 3
        self.linear = nn.Linear(mh_dim + mt_half_dim, d_model, bias=False)

    def forward(self, head, contextualized_tail):
        # Concatenate head (uncontextualized) and contextualized tail
        combined = torch.cat([head, contextualized_tail], dim=-1)
        return self.linear(combined)


class OptimizedRanker(nn.Module):
    """Memory-optimized Ranker: Identifies top-k most relevant splits using MaxSim"""

    def __init__(self, d_model, split_size=64, chunk_size=512):
        super().__init__()
        self.d_model = d_model
        self.split_size = split_size
        self.chunk_size = chunk_size

    def _compute_maxsim_chunked(self, current_split, previous_splits):
        """
        Memory-efficient MaxSim computation using chunking

        Args:
            current_split: [batch, split_size, d_model]
            previous_splits: [batch, n_prev_splits, split_size, d_model]

        Returns:
            scores: [batch, n_prev_splits]
        """
        batch_size, n_prev_splits, split_size, d_model = previous_splits.shape

        # Normalize once
        current_norm = F.normalize(
            current_split, p=2, dim=-1
        )  # [batch, split_size, d_model]
        previous_norm = F.normalize(
            previous_splits, p=2, dim=-1
        )  # [batch, n_prev, split_size, d_model]

        # Process in chunks to avoid large intermediate tensors
        scores = torch.zeros(batch_size, n_prev_splits, device=current_split.device)

        chunk_size = min(self.chunk_size, n_prev_splits)
        for start_idx in range(0, n_prev_splits, chunk_size):
            end_idx = min(start_idx + chunk_size, n_prev_splits)

            # Process chunk of previous splits
            prev_chunk = previous_norm[
                :, start_idx:end_idx, :, :
            ]  # [batch, chunk_size, split_size, d_model]

            # Compute similarities for this chunk
            # [batch, split_size, d_model] @ [batch, chunk_size, d_model, split_size]
            similarities = torch.matmul(
                current_norm.unsqueeze(1),  # [batch, 1, split_size, d_model]
                prev_chunk.transpose(
                    -1, -2
                ),  # [batch, chunk_size, d_model, split_size]
            )  # [batch, chunk_size, split_size, split_size]

            # Take max similarity for each token in current split
            max_sims, _ = similarities.max(dim=-1)  # [batch, chunk_size, split_size]

            # Sum max similarities for each previous split in chunk
            chunk_scores = max_sims.sum(dim=-1)  # [batch, chunk_size]
            scores[:, start_idx:end_idx] = chunk_scores

        return scores

    def forward(self, embeddings, current_split_idx, k=7):
        """
        Select top-k most relevant splits for the current split
        """
        batch_size, seq_len, _ = embeddings.shape
        device = embeddings.device

        if current_split_idx == 0:
            indices = torch.full(
                (batch_size, 1), current_split_idx, dtype=torch.long, device=device
            )
            weights = torch.ones(batch_size, 1, device=device)
            return indices, weights

        # Calculate required length and reshape to splits
        n_splits_available = current_split_idx + 1
        required_len = n_splits_available * self.split_size

        if seq_len < required_len:
            embeddings_padded = F.pad(
                embeddings, (0, 0, 0, required_len - seq_len), value=0
            )
        else:
            embeddings_padded = embeddings[:, :required_len, :]

        # Reshape to splits more efficiently
        splits = embeddings_padded.view(
            batch_size, n_splits_available, self.split_size, -1
        )

        current_split = splits[:, current_split_idx, :, :]

        if current_split_idx > 0:
            previous_splits = splits[:, :current_split_idx, :, :]
            # Use chunked computation for memory efficiency
            scores = self._compute_maxsim_chunked(current_split, previous_splits)
        else:
            scores = torch.empty(batch_size, 0, device=device)

        # Select top-k splits
        k_actual = min(k, scores.shape[1])
        if k_actual > 0:
            top_scores, top_indices = torch.topk(scores, k_actual, dim=1)

            # Normalize scores
            max_score = top_scores.max(dim=1, keepdim=True)[0]
            normalized_scores = top_scores / (max_score + 1e-8)

            # Add current split
            all_indices = torch.cat(
                [
                    top_indices,
                    torch.full(
                        (batch_size, 1),
                        current_split_idx,
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=1,
            )
            all_weights = torch.cat(
                [normalized_scores, torch.ones(batch_size, 1, device=device)], dim=1
            )
        else:
            all_indices = torch.full(
                (batch_size, 1), current_split_idx, dtype=torch.long, device=device
            )
            all_weights = torch.ones(batch_size, 1, device=device)

        return all_indices, all_weights


class AveyLayer(nn.Module):
    """Single layer of the Avey"""

    def __init__(
        self,
        d_model,
        expansion_factor=4,
        tail_ratio=0.5,
        max_context_width=8192,
        use_checkpoint=False,
        dynamic_v_allocation=True,
        chunk_size=512,
    ):
        super().__init__()
        self.d_model = d_model
        self.expanded_dim = d_model * expansion_factor
        self.tail_dim = int(self.expanded_dim * tail_ratio)
        self.head_dim = self.expanded_dim - self.tail_dim
        self.tail_half_dim = self.tail_dim // 2
        self.use_checkpoint = use_checkpoint

        # Components
        self.norm = RMSNorm(d_model)
        self.enricher = Enricher(d_model, expansion_factor)
        self.contextualizer = OptimizedContextualizer(
            self.tail_dim, max_context_width, dynamic_v_allocation, chunk_size
        )
        self.fuser = Fuser(self.head_dim, self.tail_half_dim, d_model)

    def _forward_impl(self, x, weights=None):
        """Implementation of forward pass"""
        # Normalize input
        x_norm = self.norm(x)

        # Enrich embeddings
        enriched = self.enricher(x_norm)

        # Split into head and tail
        head = enriched[..., : self.head_dim]
        tail = enriched[..., self.head_dim :]

        # Contextualize tail
        contextualized = self.contextualizer(tail, weights)

        # Fuse head and contextualized tail
        output = self.fuser(head, contextualized)

        # Residual connection
        return x + output

    def forward(self, x, weights=None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, weights, use_reentrant=False)
        else:
            return self._forward_impl(x, weights)


class Avey(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Token embeddings
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [
                        AveyLayer(
                            config.n_embd,
                            config.expansion_factor,
                            config.tail_ratio,
                            max_context_width=(config.top_k + 1) * config.split_size,
                            use_checkpoint=config.use_gradient_checkpointing,
                            dynamic_v_allocation=config.dynamic_v_allocation,
                            chunk_size=config.chunk_size,
                        )
                        for _ in range(config.n_layer)
                    ]
                ),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        # Optimized ranker
        self.ranker = OptimizedRanker(
            config.n_embd, config.split_size, config.chunk_size
        )

        # Final output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_embed_weights:
            self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _gather_splits_optimized(self, x_splits, indices, batch_size):
        """Memory-optimized split gathering"""
        device = x_splits.device
        k_plus_1 = indices.shape[1]

        # Use advanced indexing efficiently
        batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1)
        batch_indices = batch_indices.expand(batch_size, k_plus_1)

        # Gather and reshape in one operation
        gathered_splits = x_splits[batch_indices, indices]
        return gathered_splits.view(batch_size, -1, x_splits.shape[-1])

    def forward(self, idx, targets=None):
        """Forward pass through Avey model."""
        batch_size, seq_len = idx.shape
        device = idx.device

        # Token embeddings
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)

        # Calculate splits and pad if necessary
        n_splits = (seq_len + self.config.split_size - 1) // self.config.split_size
        padded_seq_len = n_splits * self.config.split_size

        if padded_seq_len > seq_len:
            x = F.pad(x, (0, 0, 0, padded_seq_len - seq_len), value=0)

        # Reshape to splits
        x_splits = x.view(batch_size, n_splits, self.config.split_size, -1)

        # Process splits sequentially with memory optimization
        all_outputs = []

        for current_split_idx in range(n_splits):
            if current_split_idx == 0:
                # First split - no previous context
                context = x_splits[:, current_split_idx].view(
                    batch_size, self.config.split_size, -1
                )
                split_weights = None
            else:
                # Get context using ranker
                x_up_to_current = x_splits[:, : current_split_idx + 1].view(
                    batch_size, -1, x_splits.shape[-1]
                )
                indices, weights = self.ranker(
                    x_up_to_current, current_split_idx, self.config.top_k
                )

                context = self._gather_splits_optimized(x_splits, indices, batch_size)
                split_weights = weights

            # Process through layers
            h = context
            for layer in self.transformer.h:
                h = layer(h, split_weights)

            # Extract current split output
            current_output = (
                h[:, -self.config.split_size :] if split_weights is not None else h
            )
            all_outputs.append(current_output)

            # Clear intermediate tensors to save memory
            del h
            if "indices" in locals():
                del indices, weights

        # Concatenate outputs and remove padding
        x = torch.cat(all_outputs, dim=1)[:, :seq_len, :]

        # Final processing
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference optimization
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
