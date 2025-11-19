# Future Experiments Design

Building on the temporal scaling law discovery and developmental dynamics findings.

---

## Track F: Causal Intervention

**Question**: Does *imposing* flexibility improve coordination, or is it just correlational?

### Experiment F1: Flexibility Regularization
Add entropy bonus to encourage action diversity:
```python
loss = -reward + λ * entropy_bonus  # λ controls flexibility
```
Test λ ∈ [0, 0.01, 0.05, 0.1, 0.5]

**Prediction**: Moderate λ improves coordination; too high hurts (exploration without exploitation)

### Experiment F2: Flexibility Curriculum
Train with decreasing exploration over time:
```python
noise_scale = 1.0 * (1 - episode / max_episodes)  # Anneal from 1 to 0
```
Compare to constant noise.

**Prediction**: Curriculum beats constant (flexible early, exploit late)

### Experiment F3: Constrained Policies
Train agents with artificially reduced flexibility (low-rank policies, limited action space).

**Prediction**: Constrained agents perform worse, confirming causal link

---

## Track G: Transfer and Generalization

**Question**: Does flexibility learned in one task transfer?

### Experiment G1: Domain Transfer
Train on Task A (target-reaching), test on Task B (formation control).
- Measure flexibility in both domains
- Compare transfer of flexible vs rigid teams

**Prediction**: Flexible teams transfer better (adaptation capacity generalizes)

### Experiment G2: Team Composition Transfer
Train 4-agent team, test with:
- 3 original + 1 new agent
- 2 original + 2 new agents
- 1 original + 3 new agents

**Prediction**: Flexible teams integrate new members faster

### Experiment G3: Environment Perturbation
After training, add:
- Observation noise (2x, 5x, 10x baseline)
- Action noise
- Communication dropout (10%, 30%, 50%)

**Prediction**: Flexible teams degrade gracefully; rigid teams fail catastrophically

---

## Track H: Scaling Laws Deep Dive

**Question**: Can we refine the scaling formula?

### Experiment H1: Fine-Grained Team Sizes
Test: 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20 agents
At: 100, 150, 200, 250, 300, 400, 500 steps

**Goal**: Fit more precise formula, find asymptotes

### Experiment H2: Power Law vs Linear
Current formula: Steps = 150 + 25*(n-4)
Alternative: Steps = a * n^b

Test which fits better across full range.

### Experiment H3: Topology × Team Size
Does optimal episode length vary by network structure?
- Fully connected
- Ring
- Small world
- Hierarchical
- Random sparse

**Prediction**: Sparse topologies need more time (slower information propagation)

---

## Track I: Specialization vs Flexibility Trade-off

**Question**: Do flexible teams specialize less?

### Experiment I1: Role Emergence
Measure action variance per agent position:
- Do some agents become "leaders"?
- Does flexibility correlate with role differentiation?

### Experiment I2: Skill Diversity
Train on multi-skill task (navigate + manipulate + communicate).
Measure per-agent skill profiles.

**Prediction**: Flexible teams have more generalist agents; rigid teams have specialists

### Experiment I3: Optimal Mix
What if we force some agents to be flexible, others rigid?
Test: 0%, 25%, 50%, 75%, 100% flexible agents

**Prediction**: Optimal mix depends on task; pure flexibility not always best

---

## Track J: Real-World Validation

**Question**: Does the effect hold in established benchmarks?

### Experiment J1: Multi-Agent Particle Environments (MPE)
Test on:
- Simple Spread
- Cooperative Navigation
- Predator-Prey

Measure flexibility-reward correlation in each.

### Experiment J2: StarCraft Micromanagement
SMAC scenarios with varying unit counts.
Test if flexibility predicts win rate.

### Experiment J3: Traffic Control
Multi-intersection traffic light coordination.
Test if flexibility predicts throughput.

---

## Priority Ranking

### High Priority (Paper 4 Material)
1. **F1: Flexibility Regularization** - Establishes causality
2. **G3: Environment Perturbation** - Practical implications
3. **J1: MPE Validation** - Benchmark credibility

### Medium Priority (Deep Understanding)
4. **H1: Fine-Grained Scaling** - Refine the formula
5. **I1: Role Emergence** - Specialization trade-off
6. **G1: Domain Transfer** - Generalization test

### Lower Priority (Exploratory)
7. **F3: Constrained Policies** - Confirms causal direction
8. **I3: Optimal Mix** - Practical design question
9. **J2/J3: Complex Benchmarks** - Validation extension

---

## Suggested Next Session

**Focus**: Track F (Causal Intervention)

**Why**:
- Current work is correlational
- Causal evidence is stronger for publication
- F1 (regularization) directly tests if flexibility is causal

**Experiments to run**:
1. F1: Flexibility regularization sweep
2. F2: Flexibility curriculum vs constant
3. G3: Environment perturbation (quick robustness test)

**Expected output**: Paper 4 draft on causal mechanisms

---

## Implementation Notes

### Flexibility Regularization (F1)
```python
def compute_loss(agent, actions, rewards):
    policy_loss = -rewards.mean()

    # Entropy bonus encourages flexibility
    action_std = actions.std(dim=0).mean()
    entropy_bonus = action_std.log()  # Higher std = more flexible

    return policy_loss - lambda_flex * entropy_bonus
```

### Environment Perturbation (G3)
```python
def perturb_observation(obs, noise_level):
    return obs + np.random.randn(*obs.shape) * noise_level * obs.std()

def communication_dropout(messages, dropout_rate):
    mask = np.random.rand(len(messages)) > dropout_rate
    return [m if keep else np.zeros_like(m) for m, keep in zip(messages, mask)]
```

### Role Emergence Metric (I1)
```python
def compute_role_specialization(agent_actions):
    # High variance across agents = specialization
    per_agent_means = [np.mean(a, axis=0) for a in agent_actions]
    return np.std(per_agent_means, axis=0).mean()
```

---

*Designed with emphasis on causal evidence and practical implications.*
