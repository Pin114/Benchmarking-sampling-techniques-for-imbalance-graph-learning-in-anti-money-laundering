"""GraphENS augmentation primitives.

Implements the core building blocks of Algorithm 1 from:
  Park, J., Song, J., & Yang, E. (2022). GraphENS: Neighbor-Aware Ego Network
  Synthesis for Class-Imbalanced Node Classification. ICLR 2022.

This module holds only the pure, stateless pieces of the algorithm (node
sampling, mixing-ratio math, saliency-masked feature mixup, blended-adjacency
neighbor sampling, confidence aggregation, and the warmup path). Per-epoch
orchestration -- computing this epoch's saliency/confidence via autograd and
stitching augmented nodes onto the training graph -- lives in
GNN_features_graphens_with_predictions in experiments_supervised.py, which
closes over the cross-epoch state (o_hat, saliency, warmup counter) these
functions consume as plain tensor inputs.

Two design decisions are called out explicitly at their point of use below:
  1. Binary-classification reduction of Algorithm 1's target-class sampling
     (see sample_minor_target_pairs).
  2. Confidence aggregation implements the REFERENCE CODE's mean-then-softmax
     formula rather than Algorithm 1's literal softmax-then-mean wording
     (see aggregate_confidence). This was a confirmed, deliberate choice.
"""
from typing import List, Optional, Tuple

import torch


def build_adjacency_list(edge_index: torch.Tensor, num_nodes: int) -> List[torch.Tensor]:
    """Real-graph adjacency, built once per training run (edges are static
    across epochs -- only the synthetic nodes/edges GraphENS adds each epoch
    are not). adjacency[v] holds the ids of all u with a directed edge
    (u -> v) in edge_index, i.e. v's incoming neighbors -- for the symmetric
    real graph this is also v's undirected neighbor set.
    """
    if edge_index.numel() == 0:
        return [torch.empty(0, dtype=torch.long) for _ in range(num_nodes)]
    src = edge_index[0].detach().cpu()
    dst = edge_index[1].detach().cpu()
    order = torch.argsort(dst)
    src_sorted = src[order]
    counts = torch.zeros(num_nodes, dtype=torch.long)
    counts.index_add_(0, dst, torch.ones_like(dst))
    offsets = torch.cumsum(counts, dim=0) - counts
    adjacency = []
    for node in range(num_nodes):
        start = int(offsets[node])
        length = int(counts[node])
        adjacency.append(src_sorted[start:start + length])
    return adjacency


def sample_minor_target_pairs(
    minor_indices: torch.Tensor,
    target_class_pools: List[torch.Tensor],
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Class-/target-conditional source & destination sampling (Algorithm 1,
    item 1 / line 7-9). v_minor is sampled uniformly with replacement from
    minor_indices. v_target's CLASS is first sampled from a multinomial
    weighted by log(N_i) alone, N_i = the size of class i's training pool
    (Algorithm 1 line 7, "c_target ~ p_class(C)"; paper Section 5.2: "we
    sample a target class from the multinomial distribution with
    (log N_1, log N_2, ..., log N_C)"); given the sampled class, the node
    itself is then uniform within it (Section 5.2: "randomly select a
    target node from the training nodes of the sampled class").

    CONFIRMED DECISION: log(N_i) alone is the paper's literal formula. The
    reference implementation (models/gens.py:53, sampling_idx_individual_dst)
    instead weights by log(N_i)/N_i -- an extra division by class size that
    appears nowhere in the paper's text. Deliberately not implemented here,
    same policy as the confidence-aggregation formula and items 3-4 above.

    BINARY-CLASSIFICATION REDUCTION: this codebase's target-class
    eligibility is separately restricted to the majority class only (a
    prior, independently confirmed design decision, not part of this log(N)
    vs log(N)/N choice) -- callers pass target_class_pools as a single-
    element list, [majority_class_pool]. With only one eligible class, the
    log(N_i)-weighted multinomial always selects it regardless of the exact
    weighting formula (there is nothing to weight BETWEEN), so this
    function's class-selection step is a formal no-op under the current
    call site; it is implemented generically here both because it is what
    the paper specifies and to stay correct if the eligible set is ever
    widened to more than one class.
    """
    if num_samples <= 0 or minor_indices.numel() == 0 or not target_class_pools:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    pool_sizes = torch.tensor([pool.numel() for pool in target_class_pools], dtype=torch.float32)
    if float(pool_sizes.sum()) == 0.0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty

    minor_sel = minor_indices[torch.randint(0, minor_indices.shape[0], (num_samples,))]

    class_weights = torch.log(pool_sizes.clamp_min(1.0))
    class_weights[pool_sizes == 0] = 0.0  # empty pools are never eligible, regardless of log(1)=0
    class_choice = torch.multinomial(class_weights, num_samples, replacement=True)

    target_sel = torch.empty(num_samples, dtype=torch.long)
    for c, pool in enumerate(target_class_pools):
        idx = (class_choice == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        target_sel[idx] = pool[torch.randint(0, pool.numel(), (idx.numel(),))]
    return minor_sel, target_sel


def compute_mixing_ratio(
    minor_confidence: torch.Tensor,
    target_confidence: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """phi_hat = sigmoid(KL(minor_confidence || target_confidence))
    (Algorithm 1, items 3-4). KL divergence is non-negative by construction
    (Gibbs' inequality: sum p*log(p/q) >= 0 for any two distributions p, q),
    so phi_hat is PROVABLY >= 0.5 always -- the blended feature/neighbor
    distributions built from it never weight the target ego network more
    heavily than the source's own.
    """
    p = minor_confidence.clamp_min(eps)
    q = target_confidence.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    kl = (p * (p.log() - q.log())).sum(dim=-1)
    return torch.sigmoid(kl)


def _sample_beta22(n: int, device: torch.device) -> torch.Tensor:
    if n == 0:
        return torch.empty(0, device=device)
    beta = torch.distributions.Beta(2.0, 2.0)
    return beta.sample((n,)).to(device)


def saliency_masked_mixup(
    minor_features: torch.Tensor,
    target_features: torch.Tensor,
    target_saliency: torch.Tensor,
    phi_hat: torch.Tensor,
    k: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 2 / Algorithm 1 lines 20-23, verified against the paper PDF
    (Section 4.2) and the reference implementation (models/gens.py:
    saliency_mixup): v_mixed = Lambda*v_minor + (1-Lambda)*v_target, where
    Lambda starts as a per-node Beta(2,2) draw broadcast across all D
    features, then gets OVERWRITTEN to exactly 1.0 at the K masked
    positions.

    - lambda ~ Beta(2, 2), one draw per synthetic node (item 2), broadcast
      across all D features as the baseline blend.
    - K = round(k * phi_hat): a per-node COUNT of features to mask (item 5).
      CONFIRMED DECISION: this implements the paper's literal formula
      (Section 4.2: "K = kphi_hat where k is a hyperparameter"), a single
      direct step with no intermediate variable. The reference
      implementation (models/gens.py:94,98, saliency_mixup) instead computes
      a FIXED, dimension-relative K = int(node_dim * keep_prob) and applies
      phi_hat only as a separate post-hoc truncation,
      kl_mask = round(sigmoid(dist_kl / 3.) * K) -- an extra keep_prob
      variable and an undocumented /3 temperature term that appear nowhere
      in the paper's text. Deliberately not implemented here; this follows
      the paper over the reference code, per the same policy used for the
      confidence-aggregation formula.
    - The K masked positions are the top-K under a categorical draw WITHOUT
      replacement, weighted by softmax(target_saliency) (item 6) -- i.e.
      the K features that mattered most to correctly classifying v_target
      last epoch. CONFIRMED DECISION: softmax is the paper's literal
      wording (Section 4.2: "sampled from multinomial distribution by
      applying softmax to S_vtarget"). The reference implementation
      (models/gens.py:90-92) instead L1-normalizes the absolute value
      (saliency_dst.abs(); saliency_dst /= sum(saliency_dst)) -- not
      softmax. Deliberately not implemented here, same policy as above.
    - Per the paper's own wording, M_K "masks the K% of node attributes to
      0": the K masked positions get Lambda forced to 1.0 (PURE v_minor,
      blocking target-specific attributes from being injected there), while
      every other position keeps the normal lambda blend. This is the
      opposite assignment from an earlier version of this function, which
      blended at the masked positions and forced pure v_minor everywhere
      else -- confirmed backwards against both the paper text and the
      reference code, fixed here.

    Sampling the K positions is done via the Gumbel-top-k trick
    (Efraimidis-Spirakis), vectorized across the whole batch instead of a
    per-row torch.multinomial loop, since batch sizes here can reach into
    the thousands at aggressive oversampling ratios.

    minor_features, target_features, target_saliency: (n, D) tensors, already
    gathered to the n sampled pairs for this epoch. phi_hat: (n,) tensor.
    target_saliency is expected to already be |gradient| (paper Section 4.2:
    s_(v,i) = |[dL/dX]_(v,i)|) -- callers must take the absolute value
    before passing it in; this function does not do so itself.
    Returns (mixed_features, lambda_used).
    """
    device = minor_features.device
    n, d = minor_features.shape
    lam = _sample_beta22(n, device)

    k_counts = torch.round(k * phi_hat).clamp(min=0, max=d).long()

    saliency_probs = torch.softmax(target_saliency, dim=-1)
    gumbel = -torch.log(-torch.log(torch.rand_like(saliency_probs).clamp_min(1e-20)))
    scores = torch.log(saliency_probs.clamp_min(1e-20)) + gumbel
    order = torch.argsort(scores, dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(d, device=device).unsqueeze(0).expand(n, d))
    masked = rank < k_counts.unsqueeze(1)  # True at the K masked (top target-saliency) positions

    lam_matrix = lam.unsqueeze(1).expand(n, d).clone()
    lam_matrix = lam_matrix.masked_fill(masked, 1.0)

    mixed = lam_matrix * minor_features + (1 - lam_matrix) * target_features
    return mixed, lam


def blended_neighbor_sampling(
    minor_neighbors: torch.Tensor,
    target_neighbors: torch.Tensor,
    phi_hat: float,
    num_samples: int,
) -> torch.Tensor:
    """Eq. 1: p(u | v_mixed) = phi_hat * p(u | v_minor) + (1 - phi_hat) * p(u | v_target),
    with p(u | v) uniform over v's real neighbor set. Draws up to
    `num_samples` DISTINCT neighbors WITHOUT replacement from that blended
    distribution, matching the paper (Section 4.1: "we sample neighbors
    from this distribution without replacement") and the reference
    implementation (models/gens.py: neighbor_sampling's
    torch.multinomial(..., replacement default False)). `num_samples` is
    degree-matched -- see graphens.sample_augmented_degree, which the
    caller uses to draw it from the graph's real degree distribution
    (item 9), not simply deg(v_minor).

    A node that is a neighbor of BOTH v_minor and v_target has its two
    contributions summed into a single sampling slot (matching the
    reference's dense per-node probability vector, where both ego networks'
    mass lands on the same index) rather than appearing as two separate,
    independently-drawable pool entries -- so it cannot be double-counted
    into the returned set under without-replacement sampling.

    Returns a 1D LongTensor of sampled real-node ids (length <=
    num_samples, since without-replacement sampling cannot exceed the
    number of distinct candidate neighbors), meant to be wired as directed,
    INCOMING-only edges (neighbor -> synthetic node) -- see the caller in
    experiments_supervised.py for why these are never symmetrized (paper
    footnote: message passing runs only on incoming edges to oversampled
    nodes).
    """
    if num_samples <= 0:
        return torch.empty(0, dtype=torch.long)
    n_minor = minor_neighbors.numel()
    n_target = target_neighbors.numel()
    if n_minor == 0 and n_target == 0:
        return torch.empty(0, dtype=torch.long)

    combined = torch.cat([minor_neighbors, target_neighbors])
    unique_nodes, inverse = torch.unique(combined, return_inverse=True)
    weights = torch.zeros(unique_nodes.numel(), dtype=torch.float32)
    minor_mass = (phi_hat / n_minor) if n_minor > 0 else 0.0
    target_mass = ((1.0 - phi_hat) / n_target) if n_target > 0 else 0.0
    if n_minor > 0:
        weights.index_add_(0, inverse[:n_minor], torch.full((n_minor,), float(minor_mass)))
    if n_target > 0:
        weights.index_add_(0, inverse[n_minor:], torch.full((n_target,), float(target_mass)))

    if not torch.isfinite(weights).all() or (weights < 0).any():
        # phi_hat (and hence these weights) is derived from the live model's
        # confidence for this epoch; if upstream logits blow up (e.g. large-
        # magnitude unnormalized input features on some networks) the softmax
        # in aggregate_confidence can collapse to NaN, which torch.multinomial
        # refuses outright. Fall back to a uniform draw over the candidate
        # neighbors rather than hard-crashing the whole training run.
        weights = torch.ones_like(weights)
    if weights.sum() <= 0:
        return torch.empty(0, dtype=torch.long)
    num_samples = min(num_samples, unique_nodes.numel())  # without replacement cannot exceed the candidate pool
    chosen = torch.multinomial(weights, num_samples, replacement=False)
    return unique_nodes[chosen]


def sample_augmented_degree(
    degrees: torch.Tensor,
    cap: torch.Tensor,
) -> torch.Tensor:
    """Algorithm 1 line 27: r ~ p_degree(D), the graph's own empirical
    degree distribution, then capped at the source node's own degree
    (matches the reference implementation, models/gens.py:204-208:
    aug_degree = min(multinomial_draw_from_degree_histogram, deg(v_minor));
    the paper's pseudocode doesn't show this cap explicitly but the
    reference code applies it).

    `degrees`: 1D LongTensor, every real node's degree -- used to build the
    histogram p_degree(D) once (static, since the real graph doesn't change
    across epochs). `cap`: 1D LongTensor of length n (one cap per synthetic
    node this epoch, typically deg(v_minor) for each). Returns an (n,)
    LongTensor of degree-matched neighbor counts, one per synthetic node.
    """
    n = cap.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long)
    max_degree = int(degrees.max().item())
    hist = torch.bincount(degrees, minlength=max_degree + 1).float()
    drawn = torch.multinomial(hist, n, replacement=True)
    return torch.minimum(drawn, cap)


def aggregate_confidence(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-node ego-network confidence, o_hat (Algorithm 1, item 10; feeds
    next epoch's KL in compute_mixing_ratio).

    CONFIRMED DESIGN DECISION: implements the REFERENCE CODE's formula --
    mean-pool this epoch's raw logits over each node's ego network (itself +
    real-graph neighbors), THEN apply softmax once to the pooled result.
    Algorithm 1's written formula instead softmaxes each neighbor's logits
    individually first and averages the resulting probability vectors --
    mean-then-softmax and softmax-then-mean are NOT equivalent (softmax is
    nonlinear), and the reference implementation at
    github.com/JoonHyung-Park/GraphENS demonstrably uses mean-then-softmax.
    This is a deliberate choice to match the authors' validated
    implementation over their paper's prose, confirmed explicitly before
    implementation.
    """
    num_nodes = logits.shape[0]
    if edge_index.numel() == 0:
        return torch.softmax(logits / temperature, dim=-1)
    src, dst = edge_index[0], edge_index[1]
    neighbor_sum = torch.zeros_like(logits)
    neighbor_sum.index_add_(0, dst, logits[src])
    neighbor_count = torch.zeros(num_nodes, device=logits.device)
    neighbor_count.index_add_(0, dst, torch.ones(dst.shape[0], device=logits.device))
    total_sum = neighbor_sum + logits  # ego network includes the node itself
    total_count = neighbor_count + 1.0
    mean_logits = total_sum / total_count.unsqueeze(1)
    return torch.softmax(mean_logits / temperature, dim=-1)


def warmup_mixup(
    minor_features: torch.Tensor,
    target_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warmup-phase simple path (Algorithm 1, item 12): plain mixup, no
    saliency masking. v_mixed = lambda * v_minor + (1 - lambda) * v_target,
    lambda ~ Beta(2, 2). New edges for these synthetic nodes are formed by
    duplicating v_minor's real neighbors directly (see caller) rather than
    the KL/confidence-blended sampling used post-warmup, since o_hat and
    saliency history aren't meaningful yet this early in training.

    Algorithm 1 line 12 writes this as v_mixed = (1-lambda)*v_minor +
    lambda*v_target (lambda multiplying v_target, not v_minor). This
    function's lambda plays the complementary role instead, matching
    saliency_masked_mixup's convention (lambda=1 -> pure v_minor) for
    internal consistency between the two mixup paths. The two formulas are
    NOT just relabeled by convention -- they are genuinely, exactly
    distributionally identical, because Beta(2,2) is symmetric about 0.5
    (1 - Beta(alpha,alpha) is itself Beta(alpha,alpha)-distributed for any
    alpha), so which symbol is called "lambda" does not change the
    distribution of v_mixed either way.
    """
    device = minor_features.device
    n = minor_features.shape[0]
    lam = _sample_beta22(n, device)
    mixed = lam.unsqueeze(1) * minor_features + (1 - lam).unsqueeze(1) * target_features
    return mixed, lam
