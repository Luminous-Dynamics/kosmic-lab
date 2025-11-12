# Reciprocal Policy Coupling (RPC) Specification

This document specifies the Phase 2.5 upgrade that replaces the static TE bridge
with an adaptive, reciprocity-driven coupling layer operating on *controller
dynamics* rather than raw simulator parameters. It is intended to guide the
implementation of `core/reciprocity_coupling.py` and associated tests.

---

## 1. Motivation

Static transfer-entropy (TE) computed over scalar harmonies (`H1`, `H4`, etc.)
failed to expose inter-universe information flow (β ≈ −0.012). Track B/C success
shows that coherence emerges when agents learn to align *policy gradients* with
bioelectric feedback. The RPC upgrade couples universes at the level of their
controller updates, enabling genuine co-learning.

---

## 2. Conceptual Model

For each universe \(U_i\) we maintain a controller (policy parameters \(\theta_i\)).
After each local update (e.g., SAC step), we derive a *policy change vector*
\(\Delta \theta_i\). Coupling acts by exchanging these updates:

\[
\theta^{\text{new}}_j \leftarrow \theta_j + \eta_{ij} \, \mathbf{W}_{ij} \big(\Delta\theta_i - \Delta\theta_j\big)
\]

Where:

- \(\eta_{ij}\) is a coupling gain derived from reciprocity metrics
  (information flow, similarity, energy budget).
- \(\mathbf{W}_{ij}\) is a projection (identity, low-rank map, or attention
  weights) ensuring dimensions align and respecting parameter groupings
  (actor vs. critic).

Energy gating multiplies the update by
\(g_j = 1 - \frac{E^j_t}{E_{\max}}\), preventing runaway excitation.

---

## 3. Data Flow & Interfaces

### 3.1 Core Classes

```python
@dataclass
class PolicySnapshot:
    params: Dict[str, np.ndarray]       # e.g., {"actor": θ, "critic": φ}
    grads: Dict[str, np.ndarray]        # accumulated gradient/Δθ per group
    energy: float                       # current energetic load (0–1)
    metadata: Dict[str, Any]            # optional (entropy, reward, etc.)

class ReciprocityCoupler:
    def __init__(self, config: CouplerConfig): ...

    def register_universe(self, uid: str, snapshot: PolicySnapshot) -> None:
        """Initialise state buffers for universe uid."""

    def update_snapshot(self, uid: str, snapshot: PolicySnapshot) -> None:
        """Store latest policy/grads for uid prior to coupling."""

    def compute_coupling(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Return Δparams per uid/group to be applied by caller."""

    def step(self, uid: str, **kwargs) -> None:
        """Optional hook for online updates (e.g., smoothing, TE caches)."""
```

### 3.2 CouplerConfig

```python
@dataclass
class CouplerConfig:
    eta_min: float = 0.0                 # lower bound on coupling gain
    eta_max: float = 0.1                 # upper bound on coupling gain
    window: int = 32                     # samples for reciprocity estimator
    symmetry_lambda: float = 0.1         # bidirectional smoothing
    projection: str = "identity"        # or "actor_only", "attention"
    energy_cap: float = 1.0              # max energy for gating
    use_knn_te: bool = True              # switch between TE estimators
    k_neighbors: int = 5                 # KSG parameter
```

Configurations can be loaded from YAML (e.g., `configs/reciprocity.yaml`).

---

## 4. Coupling Pipeline

1. **Snapshot ingestion** — every simulation tick, each universe submits current
   policy parameters and the latest gradient/parameter delta (e.g., from SAC).
2. **Reciprocity estimation** — compute metric \(R_{ij}\) between universes \(i\)
   and \(j\).
   - Options:
     - kNN TE: `knn_transfer_entropy(Δθ_i, Δθ_j)`
     - Cosine similarity: \(\cos(Δθ_i, Δθ_j)\)
     - Hybrid: average of both.
3. **Symmetry blending** — enforce reciprocity fairness:
   \[ \tilde{R}_{ij} = (1-\lambda) R_{ij} + \lambda R_{ji} \]
4. **Gain mapping** — map to coupling strength:
   \[ \eta_{ij} = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \sigma(\alpha (\tilde{R}_{ij} - \tau)) \]
   where \(\sigma\) is sigmoid, \(\tau\) a baseline threshold.
5. **Energy gating** — compute \(g_j = \max(0, 1 - E_j/E_{\max})\).
6. **Parameter update** — apply component-wise differences per parameter group.

Pseudo-implementation:

```python
def compute_coupling(self):
    deltas = {uid: {name: np.zeros_like(arr)
                    for name, arr in snap.params.items()}
              for uid, snap in self.snapshots.items()}

    for j, (uid_j, snap_j) in enumerate(self.snapshots.items()):
        energy_gate = max(0.0, 1.0 - snap_j.energy / self.cfg.energy_cap)
        for i, (uid_i, snap_i) in enumerate(self.snapshots.items()):
            if uid_i == uid_j:
                continue
            reciprocity = self._estimate_reciprocity(uid_i, uid_j)
            eta = self._map_to_gain(reciprocity) * energy_gate
            for group in self._parameter_groups(snap_i, snap_j):
                proj = self._project(group, snap_i, snap_j)
                diff = proj @ (snap_i.grads[group] - snap_j.grads[group])
                deltas[uid_j][group] += eta * diff
    return deltas
```

---

## 5. Reciprocity Estimators

### 5.1 kNN Transfer Entropy (default)

Use `core.te_bridge.knn_transfer_entropy` on flattened gradient windows:

```python
grad_i = rolling_buffer[uid_i]  # shape (window, dim)
grad_j = rolling_buffer[uid_j]
te_ij = knn_transfer_entropy(grad_i[:-1], grad_j[:-1], grad_j[1:], k=k)
```

### 5.2 Cosine Similarity (fallback)

\[ \text{cos}(Δθ_i, Δθ_j) = \frac{Δθ_i \cdot Δθ_j}{\|Δθ_i\| \|Δθ_j\| + \varepsilon} \]

### 5.3 Hybrid Metric

\[ R_{ij} = w_{\text{TE}} \cdot \text{TE}_{ij} + (1 - w_{\text{TE}}) \cdot \text{cos}_{ij} \]

Parameters \(w_{\text{TE}}\) exposed in config.

---

## 6. Projection Strategies (\(\mathbf{W}_{ij}\))

1. **Identity** (default): gradients must be same shape.
2. **Actor-only**: ignore critic updates if policies diverge in dimension.
3. **Attention-based**: compute attention weights via dot-product similarity
   across gradient components.
4. **Low-rank map**: learn a small linear projector via running least-squares.

Implementation should start with Identity / Actor-only for simplicity, exposing
a hook for future extensions.

---

## 7. Energy Gating

Each universe reports an energy measure \(E_j \in [0, E_{\max}]\). The coupling
gain is scaled by \(g_j = \max(0, 1 - E_j / E_{\max})\).

Optional damping: multiply by entropy regulariser \(\exp(-H_j)\) to reduce
coupling when the policy is already highly entropic.

---

## 8. API Usage Example

```python
coupler = ReciprocityCoupler(CouplerConfig(eta_min=0.0, eta_max=0.05))

for uid in universe_ids:
    coupler.register_universe(uid, initial_snapshot(uid))

while running:
    for uid in universe_ids:
        snapshot = collect_policy_snapshot(uid)
        coupler.update_snapshot(uid, snapshot)

    delta_params = coupler.compute_coupling()

    for uid, updates in delta_params.items():
        apply_policy_updates(uid, updates)
```

---

## 9. Unit & Integration Tests

1. **Unit: reciprocity estimator ratio** — with synthetic AR gradients where
   `grad_j[t] = 0.4 * grad_i[t-1] + noise`, verify TE(i→j) > TE(j→i).
2. **Unit: energy gating** — when `energy = energy_cap`, coupling gains drop to
   zero.
3. **Unit: projection identity** — gradients of mismatched shapes raise
   informative errors unless projection strategy is provided.
4. **Integration: two-universe coupling** — feed deterministic gradient series
   and confirm parameters converge (difference decreases) when coupling enabled.
5. **Integration: symmetry λ** — ensure that setting `symmetry_lambda = 0.5`
   yields symmetric gains within tolerance.

Testing harness should live in `tests/test_reciprocity_coupling.py`.

---

## 10. Logging & Diagnostics

Coupler should emit:
- Coupling gains \(\eta_{ij}\)
- Energy gates \(g_j\)
- Reciprocity metrics per pair
- Norm of applied parameter deltas per group

These metrics feed into the Stage 2 lab log and enable quick diagnosis of
co-learning behaviour.

---

## 11. Future Extensions

- **Mediator node**: central policy that aggregates gradients, applies
  attention, and redistributes updates (Graph Neural Network style).
- **Stochastic resonance**: inject noise proportional to reciprocity score to
  discover resonance bands.
- **Curriculum**: schedule coupling gains based on corridor rate progression.

---

With this specification in place, implementation can proceed by creating
`core/reciprocity_coupling.py`, wiring it into `fre/multi_universe.py`, and
extending the scaling experiment runner to choose between `static_te` and
`rpc` modes via configuration.
