# Techniques

This directory tries to explain various techniques implemented in this repository.

## Architectures
- [**ResFormer**](./resformer.md) - Value residual learning from first layer to all subsequent layers
- [**Composable Value Residual**](./value_residual.md) - ResFormer-style value residuals inside the normal BlaGPT attention path
- [**U-net Long Skips**](./unet_skips.md) - Mirrored early-to-late layer skip connections with learned scalar weights

## Attention Mechanisms
- [**Differential Attention v2**](./diff_attn_v2.md) - Subtracts two attention patterns to cancel noise and amplify relevant signals
- [**Gated Attention**](./gated_attention.md) - Sigmoid gates after SDPA for improved training stability
- [**Composable Gated Attention**](./composable_gated_attention.md) - Gated Attention composed on top of XSA + GOAT sink prior
- [**Exclusive Self Attention**](./exclusive_self_attention.md) - Removes the value-direction component from attention outputs
- [**GOAT Sink Prior**](./goat_sink_prior.md) - Per-head key-0 attention log-prior for learned sink control
- [**GRAPE-A Query-Gated**](./grape_a_qgate.md) - Query-gated additive position bias replacing RoPE with learned content-dependent decay
- [**Key-Dimension Attention (KDA)**](./kda.md) - Linear-complexity attention with fine-grained gating and delta rule

## Normalization
- [**GatedNorm**](./gated_norm.md) - Low-rank feature gates applied after RMSNorm
- [**HybridNorm**](./hybrid_norm.md) - QKV attention normalization plus post-normalized FFN residual path

## Memory and Embeddings
- [**Engram**](./engram.md) - Hash-based n-gram memory lookup for transformer layers
- [**STEM**](./stem.md) - Specialized token embedding modules for early layers

## Training Objectives
- [**Token Order Prediction (TOP)**](./top.md) - Auxiliary loss for improved language modeling through token ranking

## Optimizers
- [**AdaMuon**](./adamuon.md) - Adaptive Muon optimizer with second-moment estimation
- [**NorMuon**](./normuon.md) - Normalized Muon optimizer with norm-preserving adaptive scaling
- [**Aurora**](./aurora.md) - Leverage-aware spectral optimizer (Muon-family); current best optimizer (F99)
- [**Cautious Weight Decay**](./cautious_weight_decay.md) - Selective weight decay based on momentum-parameter sign alignment
