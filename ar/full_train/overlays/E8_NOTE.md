# E8: Future Summary Prediction (FSP) auxiliary loss

Source: Divyat Mahajan et al., "Beyond Multi-Token Prediction:
Pretraining LLMs with Future Summaries" (arXiv:2510.14751, 2025, rev.
2026). Handcrafted bag-of-words variant, the paper calls it "FSP-BCE".

## Horizon: `fsp_horizon = 100`

The paper's Eq. 9 defines the future summary target as a bag over
tokens `x_{t+2}, ..., x_{t+tau}`. Table 3 (8B-scale ablation) tests
FSP-BCE at `tau=12` and `tau=100`. The long window wins: GSM8K goes
from 69.9% (tau=12) to 71.4% (tau=100), and Section 3.1's own finding
states "long future summaries matter" (short-range MTP fails long-range
planning tasks). We pick `tau=100`, the paper's long-range setting.

## Two weights: `fsp_weight = 0.1` (E8a) and `fsp_weight = 1.0` (E8b)

The paper's Eq. 7 sums the NTP loss and the FSP loss with an implicit
coefficient of 1 (`L_FSP = L_NTP + E[l_a(...)]`), and the appendix
hyperparameter list (learning rate, batch size, weight decay, epochs,
grad clipping) does not name a separate scalar weight for the aux
loss. So `fsp_weight = 1.0` (E8b) is the paper-faithful, direct-sum
setting.

`fsp_weight = 0.1` (E8a) is our added, down-weighted setting, not from
the paper. We add it because our loss normalization (see below) still
leaves the FSP loss's gradient scale set by how many of the ~50k
vocabulary logits the model gets wrong, which can differ substantially
from the softmax-based NTP loss's scale, especially early in training.
Testing a lower weight alongside the paper-faithful 1.0 is a cheap way
to check whether the aux loss needs down-weighting on our data mix
before committing to a full run.

## Target construction (Eq. 9)

`a(t, tau)_i = 1[i in {x_{t+2}, ..., x_{t+tau}}]`, a multi-hot vector
over the vocabulary, built only from future tokens used strictly as
labels (see "Causality" below). Implemented in
`losses.construct_fsp_window_ids`.

## Loss and its normalization (Eq. 10)

Paper's Eq. 10 (reweighted BCE, `w(i)` e.g. tf-idf):

```
l_a = -sum_i w(i) [ a_i log(sigmoid(z_i)) + (1-a_i) log(1-sigmoid(z_i)) ]
```

**Reweighting.** We use uniform `w(i) = 1`, not tf-idf. The paper's own
3B-scale ablation (Section 4.3 / Appendix B.3) found removing tf-idf
reweighting "did not provide benefits" in most cases they tested (two
exceptions: GSM8K at tau=12, HumanEval+ at tau=10). tf-idf needs
corpus-level document-frequency statistics that this training loop does
not currently track; given the paper's own finding that it is not
reliably better, we use uniform weighting as a documented
simplification, not a hidden approximation.

**Numeric normalization (our addition, not stated in the paper).** As
literally written, Eq. 10 sums over the full vocabulary V (~50,304 for
this repo's tokenizer). At initialization (`z_i` near 0), that sum is
about `V * log(2) ≈ 34,800` nats — several thousand times larger than
the NTP cross-entropy (`≈ log(V) ≈ 10.8` nats). Using it unweighted
would dominate the total loss and destabilize training. We divide by
`V` (mean over vocabulary, not raw sum) so the FSP loss sits on a scale
comparable to NTP cross-entropy before `fsp_weight` is applied. This
normalization is applied inside `compute_fsp_loss` in
`bla_gpt/losses.py` and is stated here explicitly since the paper does
not specify one.

**Exact algebraic rewrite used for the memory-efficient implementation.**
Using `log(1 - sigmoid(z)) = -softplus(z)` and
`log(sigmoid(z)) - log(1 - sigmoid(z)) = z`:

```
l_a = sum_i softplus(z_i) - sum_{i in bag} z_i
```

The first term is a plain reduction over the auxiliary head's existing
logits tensor. The second term is computed by `torch.gather`-ing the
logits at the bag's token ids and de-duplicating repeated ids (sort +
first-occurrence mask) so a token repeated in the window still
contributes only once, matching the multi-hot semantics exactly. We
verified this against a brute-force dense multi-hot BCE computation,
including a duplicated token, in
`tests/test_e8_fsp.py::test_fsp_loss_matches_manual_bce_with_dedup`
(matches to 1e-6).

## Causality (task item 3)

Future tokens are used only as loss labels, never as backbone input.
`compute_fsp_loss` reads `targets` (next-token labels the training loop
already loads) to build the bag; the auxiliary head `fsp_head` only
consumes the causal hidden state `x` (the same post-`ln_f` hidden state
the NTP head reads, itself only a function of `x_{<=t}` through the
causal transformer). No future token is ever concatenated into, or
attended to by, the backbone.
`tests/test_e8_fsp.py::test_causality_ntp_and_fsp_head` confirms
perturbing a token suffix leaves earlier NTP logits unchanged.

## Document boundaries (task item 4)

Yes, the packed pretraining data has a usable boundary token: each
document is prefixed with the GPT-2 tiktoken `<|endoftext|>` id (50256)
during data prep (`data/fineweb.py:78-82`: `eot =
enc._special_tokens['<|endoftext|>']`; `tokens = [eot]` starts every
document before its text is appended). We store this id in the new
config field `fsp_eot_token_id` (default 50256, matching this
tokenizer) and mask out (exclude from the loss) any window whose bag
contains that id, rather than truncating the bag at the boundary. This
is simpler than partial-bag truncation and guarantees the auxiliary
target never mixes tokens from two documents. If a different tokenizer
or data pipeline is used, `fsp_eot_token_id` must be set to match, or
this masking silently does nothing (all windows counted valid) --
`tests/test_e8_fsp.py::test_construct_fsp_window_ids_no_boundary_data_returns_all_valid`
documents that exact fallback behavior.

## Val loss (task item 5)

`compute_fsp_loss` is only called from inside `if self.training:` in
`GPT.forward` (`bla_gpt/bla_gpt.py`), so validation (which calls
`model.eval()` before its batches in `train_ar.py`) always computes
plain NTP cross-entropy with no FSP term. The FSP loss is logged
separately as `loss_dict["fsp_loss"]` during training only.

## Memory and parameters (task item 6)

For this repo's production config (`ar/full_train/configs/F99_aurora_best.json`:
`n_embd=768`, `vocab_size=50304`):

- Extra parameters: `fsp_head` is `nn.Linear(768, 50304, bias=False)` =
  `768 * 50304 = 38,633,472` (~38.6M) extra parameters, only when
  `fsp_weight > 0.0`.
- Extra activation memory: the auxiliary head must produce a dense
  `(B, valid_count, V)` logits tensor -- this is unavoidable for any
  full-vocabulary auxiliary head, and is comparable in size to the
  model's existing NTP logits tensor. At `B=32`, `block_size=1024`,
  `tau=100` (`valid_count=925`), that is `32*925*50304*4 bytes ≈ 5.96
  GB` (fp32, forward only; the existing NTP logits tensor at the same
  batch/seq is `≈ 6.59 GB`, so FSP roughly doubles the "big logits
  tensor" memory floor of the current model).
- What the memory-efficient form avoids: a naive implementation would
  additionally materialize a `(B, valid_count, V)` dense multi-hot
  label tensor of the same size (another `≈ 5.96 GB`) to compute
  `F.binary_cross_entropy_with_logits`. Our gather-based loss (see
  above) replaces that with `(B, valid_count, tau-1)` gather buffers,
  `32*925*99*4*3 bytes ≈ 35 MB` -- negligible. Net effect: FSP adds
  ~6 GB of unavoidable logits memory, not ~12 GB.

## Files

- `bla_gpt/bla_gpt.py`: `GPTConfig.fsp_weight`, `fsp_horizon`,
  `fsp_eot_token_id`; `fsp_head` construction; loss wiring in
  `GPT.forward`.
- `bla_gpt/losses.py`: `construct_fsp_window_ids`, `compute_fsp_loss`.
- `tests/test_e8_fsp.py`: CPU tests (weight-0 equivalence, hand-checked
  target construction, EOT masking, dedup-vs-dense-BCE cross-check,
  gradients, inference without future labels, causality, save/load).
