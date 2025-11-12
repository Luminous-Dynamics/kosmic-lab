# AgentState Integration Specification v0.1

**Document Purpose:** This specification defines the unified data structures, inter-module interfaces, and edge case behaviors for integrating five distinct simulation paradigms into a coherent multiscale Kosmic system.

**Last Updated:** [Date]  
**Status:** DRAFT - To be finalized before Week 9 integration sprint

---

## 1. Executive Summary

The Kosmic Simulation Suite synthesizes five computational frameworks:
1. **Autopoiesis** (3D dissipative structures)
2. **IIT** (Integrated information)
3. **FEP** (Active inference agents)
4. **Bioelectric** (Morphogenetic networks)
5. **Multiscale** (Transfer entropy analysis)

This document ensures that agents can seamlessly transition between modules without data loss, semantic mismatch, or computational deadlock. **Key principle:** Each module views agents through its own lens, but all lenses focus on the same unified `KosmicAgent` state vector.

---

## 2. Unified State Vector: The `KosmicAgent` Class

```python
from dataclasses import dataclass, field
from typing import Dict, List, Deque, Optional
from collections import deque
import numpy as np
import networkx as nx

@dataclass
class KosmicAgent:
    """
    Unified agent representation across all simulation modules.
    
    Design Philosophy:
    - Physical properties (position, chemistry) for Autopoiesis/Bioelectric
    - Informational properties (beliefs, models) for IIT/FEP
    - Relational properties (neighbors, history) for Multiscale
    - Metadata for tracking and debugging
    """
    
    # ========================
    # Core Identity
    # ========================
    agent_id: int
    birth_time: int
    death_time: Optional[int] = None
    alive: bool = True
    
    # ========================
    # Physical State (Autopoiesis, Bioelectric)
    # ========================
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, z]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Internal chemistry (for autopoietic closure)
    internal_state: Dict[str, float] = field(default_factory=lambda: {
        'ATP': 1.0,           # Energy currency
        'membrane': 1.0,      # Boundary material
        'catalysts': 1.0,     # Autocatalytic components
        'waste': 0.0          # Entropy export
    })
    
    boundary_integrity: float = 1.0  # 0 = dissolved, 1 = intact
    
    # Bioelectric properties
    voltage: float = -70.0  # mV (resting potential)
    gap_junctions: Dict[int, float] = field(default_factory=dict)  # {neighbor_id: conductance}
    
    # ========================
    # Informational State (IIT, FEP)
    # ========================
    
    # IIT: Cause-effect structure for Φ computation
    # Represented as directed graph: nodes = internal variables, edges = causal links
    cause_effect_structure: nx.DiGraph = field(default_factory=nx.DiGraph)
    phi: float = 0.0  # Cached integrated information
    phi_last_computed: int = 0  # Timestep of last Φ calculation
    
    # FEP: Active inference components
    generative_model: Dict[str, any] = field(default_factory=lambda: {
        'prior_beliefs': np.zeros(10),     # P(hidden states)
        'likelihood': np.eye(10),          # P(observations | states)
        'policy': np.zeros((5, 10)),       # Action selection weights
        'precision': np.ones(10)           # Confidence in beliefs
    })
    
    free_energy: float = 10.0  # Variational free energy (lower = better predictions)
    prediction_errors: Dict[str, float] = field(default_factory=lambda: {
        'sensory': 0.0,
        'motor': 0.0,
        'social': 0.0
    })
    
    # ========================
    # Relational State (Multiscale)
    # ========================
    neighbors: List[int] = field(default_factory=list)  # Current spatial/network neighbors
    
    # Communication history for TE calculation (last N timesteps)
    communication_history: Deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Action history for behavioral entropy (H4: Infinite Play)
    action_history: Deque = field(default_factory=lambda: deque(maxlen=100))
    
    # ========================
    # Harmony Tracking
    # ========================
    harmony_scores: Dict[str, float] = field(default_factory=lambda: {
        'H1_Coherence': 0.0,
        'H2_Flourishing': 0.0,
        'H3_Wisdom': 0.0,
        'H4_Play': 0.0,
        'H5_Interconnection': 0.0,
        'H6_Reciprocity': 0.0,
        'H7_Evolution': 0.0
    })
    
    # ========================
    # Metadata
    # ========================
    agent_type: str = 'generic'  # 'explorer', 'cooperator', 'bioelectric_cell', etc.
    module_flags: Dict[str, bool] = field(default_factory=lambda: {
        'autopoiesis_active': True,
        'iit_active': False,
        'fep_active': False,
        'bioelectric_active': False
    })
    
    # Debugging and logging
    warnings: List[str] = field(default_factory=list)
    last_updated_by: str = 'initialization'
    
    def __post_init__(self):
        """Validate initial state."""
        assert self.boundary_integrity >= 0.0 and self.boundary_integrity <= 1.0
        assert len(self.position) == 3
        
        # Initialize cause-effect structure if empty
        if self.cause_effect_structure.number_of_nodes() == 0:
            self._initialize_minimal_ces()
    
    def _initialize_minimal_ces(self):
        """Create minimal 3-node cause-effect structure for IIT."""
        # Simple triangle: internal state variables causally interact
        self.cause_effect_structure.add_edges_from([
            ('ATP', 'membrane', {'strength': 0.5}),
            ('membrane', 'catalysts', {'strength': 0.5}),
            ('catalysts', 'ATP', {'strength': 0.5})
        ])
    
    def is_alive(self) -> bool:
        """Check if agent meets survival criteria."""
        return (self.alive and 
                self.boundary_integrity > 0.1 and
                self.internal_state.get('ATP', 0) > 0.01)
    
    def mark_dead(self, timestep: int, cause: str):
        """Gracefully handle agent death."""
        self.alive = False
        self.death_time = timestep
        self.warnings.append(f"Death at t={timestep}: {cause}")
        self.last_updated_by = f'death_handler'
    
    def to_dict(self) -> Dict:
        """Serialize for logging."""
        return {
            'agent_id': self.agent_id,
            'alive': self.alive,
            'position': self.position.tolist(),
            'boundary_integrity': self.boundary_integrity,
            'phi': self.phi,
            'free_energy': self.free_energy,
            'harmony_scores': self.harmony_scores.copy()
        }
```

---

## 3. Cross-Module Interfaces

### 3.1 Autopoiesis → IIT

**Trigger Condition:** `boundary_integrity > 0.9` (agent has achieved stable operational closure)

**Data Flow:**
```python
def autopoiesis_to_iit(agent: KosmicAgent) -> bool:
    """
    Activate IIT module when autopoietic closure stabilizes.
    
    Rationale: Integration (Φ) only meaningful for stable, bounded systems.
    """
    if agent.boundary_integrity > 0.9 and not agent.module_flags['iit_active']:
        # Map chemical states to IIT nodes
        ces = agent.cause_effect_structure
        
        for chemical, concentration in agent.internal_state.items():
            if chemical in ces.nodes():
                ces.nodes[chemical]['state'] = concentration
        
        # Activate IIT module
        agent.module_flags['iit_active'] = True
        agent.last_updated_by = 'autopoiesis_to_iit'
        return True
    
    return False
```

**Output:** `agent.phi` is computed and stored, feeding into H1 (Resonant Coherence)

**Edge Cases:**
- **Boundary fluctuation:** If integrity drops below 0.9, IIT remains active but Φ is marked as "unstable"
- **Death during computation:** If agent dies mid-calculation, cache last valid Φ and terminate gracefully

---

### 3.2 IIT → FEP

**Trigger Condition:** Every 10 timesteps (computational efficiency trade-off)

**Data Flow:**
```python
def iit_to_fep(agent: KosmicAgent, timestep: int):
    """
    Use high-Φ subgraphs to weight FEP generative model precision.
    
    Rationale: Integrated information = trustworthy internal states.
    FEP should assign higher precision (confidence) to integrated modules.
    """
    if agent.phi_last_computed < timestep - 10:
        return  # IIT not recently updated
    
    # Identify maximally integrated subgraph
    ces = agent.cause_effect_structure
    if ces.number_of_nodes() < 3:
        return
    
    # Mock: assume strongest cycle is most integrated
    try:
        cycles = list(nx.simple_cycles(ces))
        if cycles:
            max_cycle = max(cycles, key=len)
            
            # Boost FEP precision for variables in integrated cycle
            for node in max_cycle:
                if node in ['ATP', 'membrane', 'catalysts']:
                    idx = list(agent.internal_state.keys()).index(node)
                    agent.generative_model['precision'][idx] *= 1.2
            
            agent.last_updated_by = 'iit_to_fep'
    except:
        agent.warnings.append(f"IIT->FEP interface error at t={timestep}")
```

**Output:** Modified `agent.generative_model['precision']` guides FEP belief updating

**Edge Cases:**
- **No cycles:** Fallback to uniform precision
- **Module inactive:** Skip silently (no error)

---

### 3.3 FEP → Bioelectric

**Trigger Condition:** `prediction_error['sensory'] > 0.5` (significant surprise)

**Data Flow:**
```python
def fep_to_bioelectric(agent: KosmicAgent, timestep: int):
    """
    Convert FEP prediction errors into bioelectric 'target states'.
    
    Rationale: High prediction error → system needs restructuring.
    Bioelectric gradients guide physical reorganization (Levin's morphogenesis).
    """
    if agent.prediction_errors['sensory'] > 0.5:
        # Prediction error maps to voltage depolarization
        error_magnitude = agent.prediction_errors['sensory']
        
        # Depolarize voltage (makes cell more excitable, signals "remodel")
        agent.voltage += error_magnitude * 10.0  # mV
        agent.voltage = min(agent.voltage, -20.0)  # Cap at action potential
        
        # Increase gap junction conductance (seek collective solution)
        for neighbor_id in agent.gap_junctions:
            agent.gap_junctions[neighbor_id] *= 1.1
        
        agent.last_updated_by = 'fep_to_bioelectric'
```

**Output:** Modified `agent.voltage` and `agent.gap_junctions`, triggering bioelectric coordination

**Edge Cases:**
- **Voltage runaway:** Cap at physiological limits (-20 mV)
- **No neighbors:** Voltage change occurs but no network effects

---

### 3.4 Bioelectric → Autopoiesis (Regeneration Loop)

**Trigger Condition:** Bioelectric pattern reaches stable attractor

**Data Flow:**
```python
def bioelectric_to_autopoiesis(agent: KosmicAgent, target_morphology: Dict):
    """
    Bioelectric 'target state' guides chemical production to restore boundary.
    
    Rationale: Levin's experiments show bioelectric networks store morphological
    information that drives regeneration of damaged structures.
    """
    # If voltage near target and boundary damaged
    if abs(agent.voltage - target_morphology.get('voltage', -70)) < 5.0:
        if agent.boundary_integrity < 1.0:
            # Produce membrane material to restore boundary
            repair_rate = 0.01 * (1.0 - agent.boundary_integrity)
            agent.internal_state['membrane'] += repair_rate
            agent.boundary_integrity += repair_rate * 0.5
            
            # Consume ATP for repair
            agent.internal_state['ATP'] -= repair_rate * 0.1
            
            agent.last_updated_by = 'bioelectric_to_autopoiesis'
```

**Output:** Restored `boundary_integrity`, closing the regeneration loop

**Edge Cases:**
- **Insufficient ATP:** Repair fails, voltage remains high (distress signal)
- **Target morphology unknown:** Use default resting potential (-70 mV)

---

### 3.5 All Modules → Multiscale (TE Analysis)

**Trigger Condition:** Continuous (every timestep)

**Data Flow:**
```python
def log_for_multiscale(agent: KosmicAgent, timestep: int):
    """
    Log all state changes for Transfer Entropy calculation.
    
    Rationale: TE requires time-series data to infer causal influence.
    We log every significant state change to build TE matrix post-hoc.
    """
    state_vector = np.array([
        agent.boundary_integrity,
        agent.phi,
        agent.free_energy,
        agent.voltage,
        len(agent.neighbors)
    ])
    
    agent.communication_history.append({
        'timestep': timestep,
        'state': state_vector,
        'actions': agent.action_history[-1] if agent.action_history else None
    })
```

**Output:** Populated `communication_history` for later TE computation

**Edge Cases:**
- **Memory limit:** Deque automatically discards old data (maxlen=100)
- **Agent death:** Continue logging zero states (death is informative)

---

## 4. Inter-Agent Communication Protocol

Agents interact in three modes:

### 4.1 Physical Diffusion (Autopoiesis)
```python
def diffuse_chemicals(agent: KosmicAgent, neighbors: List[KosmicAgent], dt: float):
    """Fick's law diffusion of internal chemicals."""
    for neighbor in neighbors:
        for chem in agent.internal_state:
            gradient = agent.internal_state[chem] - neighbor.internal_state.get(chem, 0)
            flux = 0.01 * gradient * dt  # Diffusion coefficient
            
            agent.internal_state[chem] -= flux
            neighbor.internal_state[chem] += flux
```

### 4.2 Message Passing (FEP)
```python
def share_predictions(agent: KosmicAgent, neighbors: List[KosmicAgent]):
    """Cooperative inference via belief averaging."""
    if not agent.module_flags['fep_active']:
        return
    
    neighbor_beliefs = [n.generative_model['prior_beliefs'] 
                       for n in neighbors 
                       if n.module_flags['fep_active']]
    
    if neighbor_beliefs:
        avg_beliefs = np.mean(neighbor_beliefs, axis=0)
        # Weighted update: 70% self, 30% neighbors
        agent.generative_model['prior_beliefs'] = (
            0.7 * agent.generative_model['prior_beliefs'] + 
            0.3 * avg_beliefs
        )
```

### 4.3 Bioelectric Coupling (Bioelectric)
```python
def propagate_voltage(agent: KosmicAgent, neighbors: List[KosmicAgent], dt: float):
    """Gap junction-mediated voltage propagation."""
    for neighbor in neighbors:
        if neighbor.agent_id in agent.gap_junctions:
            conductance = agent.gap_junctions[neighbor.agent_id]
            voltage_diff = neighbor.voltage - agent.voltage
            
            # Current flow: I = g * (V_neighbor - V_self)
            current = conductance * voltage_diff * dt
            agent.voltage += current * 0.1  # Membrane capacitance
```

---

## 5. Edge Cases & Failure Modes

### 5.1 Agent Death

**Scenario:** Agent's ATP drops to zero mid-simulation.

**Behavior:**
1. Set `agent.alive = False`
2. Set `boundary_integrity = 0.0`
3. Keep `agent_id` in simulation for TE history
4. Set all state vectors to zero
5. Log cause of death in `warnings`

**Rationale:** Death is data. TE analysis needs to know *when* causal influence ceased.

```python
def handle_death(agent: KosmicAgent, timestep: int, cause: str):
    agent.mark_dead(timestep, cause)
    agent.boundary_integrity = 0.0
    agent.phi = 0.0
    agent.free_energy = 100.0  # Max surprise (world unpredictable)
    agent.voltage = 0.0
    # Keep position and ID for graveyard analysis
```

---

### 5.2 Agent Birth (Symbiogenesis)

**Scenario:** Two agents with high reciprocity (H6) merge into meta-agent.

**Behavior:**
1. Create new `KosmicAgent` with combined state
2. Inherit max Φ of parents
3. Average FEP beliefs
4. Mark parent agents as "subsumed" (not dead)
5. New agent's `birth_time` = merge timestep

```python
def merge_agents(agent_a: KosmicAgent, agent_b: KosmicAgent, 
                 timestep: int) -> KosmicAgent:
    """Symbiogenesis: two agents form meta-agent."""
    new_id = max(agent_a.agent_id, agent_b.agent_id) + 1
    
    merged = KosmicAgent(
        agent_id=new_id,
        birth_time=timestep,
        position=(agent_a.position + agent_b.position) / 2,
        internal_state={
            k: agent_a.internal_state.get(k, 0) + agent_b.internal_state.get(k, 0)
            for k in set(agent_a.internal_state) | set(agent_b.internal_state)
        },
        phi=max(agent_a.phi, agent_b.phi) * 1.2,  # Synergy bonus
        agent_type='meta_agent'
    )
    
    # Mark parents as subsumed (not dead - they're *part* of new agent)
    agent_a.alive = False
    agent_a.warnings.append(f"Subsumed into agent {new_id} at t={timestep}")
    agent_b.alive = False
    agent_b.warnings.append(f"Subsumed into agent {new_id} at t={timestep}")
    
    return merged
```

---

### 5.3 Module Timeout

**Scenario:** IIT Φ calculation exceeds 60 seconds (computational limit).

**Behavior:**
1. Abort calculation
2. Use cached Φ from last successful run
3. Log warning
4. Mark `phi` as "stale"
5. Reduce network size for next attempt

```python
import signal

class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def compute_phi_with_timeout(agent: KosmicAgent, timeout: int = 60):
    """Compute Φ with wall-clock timeout."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        # Expensive IIT calculation here
        agent.phi = calculate_integrated_information(agent.cause_effect_structure)
        agent.phi_last_computed = current_timestep
        signal.alarm(0)  # Cancel alarm
    except TimeoutException:
        agent.warnings.append(f"IIT timeout at t={current_timestep}")
        # Use cached value (agent.phi unchanged)
        # Prune network for next attempt
        if agent.cause_effect_structure.number_of_nodes() > 5:
            agent.cause_effect_structure.remove_node(
                list(agent.cause_effect_structure.nodes())[0]
            )
```

---

### 5.4 Conflicting Imperatives

**Scenario:** FEP says "flee" (high danger prediction), but bioelectric says "regenerate" (repair mode).

**Priority Hierarchy:**
1. **Survival** (FEP) overrides regeneration
2. **Integration** (IIT) preserves during conflict
3. **Exploration** (H4) only when safe

```python
def resolve_conflict(agent: KosmicAgent) -> str:
    """Determine action priority when modules conflict."""
    # Check threat level
    if agent.free_energy > 15.0:  # High surprise = danger
        return 'flee'  # FEP survival imperative
    
    # Check damage level
    if agent.boundary_integrity < 0.5:
        return 'regenerate'  # Bioelectric repair
    
    # Check integration
    if agent.phi < 0.5:
        return 'integrate'  # Restore coherence
    
    # Default: explore
    return 'explore'
```

---

## 6. Computational Efficiency Strategies

### 6.1 Lazy Evaluation

Not all modules run every timestep:

| Module | Update Frequency | Rationale |
|--------|------------------|-----------|
| Autopoiesis | Every timestep | Fast, necessary for physics |
| IIT | Every 10 timesteps | Expensive Φ calculation |
| FEP | Every timestep | Active inference is continuous |
| Bioelectric | Every 5 timesteps | Voltage dynamics slower than chemistry |
| Multiscale TE | Post-hoc | Computed on logged history |

### 6.2 Spatial Partitioning

Use octree for neighbor queries:
```python
from scipy.spatial import cKDTree

class AgentWorld:
    def __init__(self):
        self.agents = []
        self.kdtree = None
    
    def update_spatial_index(self):
        positions = np.array([a.position for a in self.agents if a.alive])
        self.kdtree = cKDTree(positions)
    
    def get_neighbors(self, agent: KosmicAgent, radius: float = 5.0):
        if self.kdtree is None:
            return []
        
        indices = self.kdtree.query_ball_point(agent.position, radius)
        return [self.agents[i] for i in indices]
```

### 6.3 GPU Acceleration

TE matrix computation (N×N) can be parallelized:
```python
import cupy as cp  # GPU arrays

def compute_te_matrix_gpu(history: List[np.ndarray]) -> np.ndarray:
    """Transfer entropy on GPU."""
    states_gpu = cp.asarray(np.vstack(history))
    # Parallel TE calculation across all pairs
    # ... (implementation details)
    return cp.asnumpy(te_matrix)
```

---

## 7. Validation & Testing

### 7.1 Unit Tests (per module)

```python
def test_autopoiesis_to_iit():
    agent = KosmicAgent(agent_id=0, birth_time=0)
    agent.boundary_integrity = 0.95
    
    assert autopoiesis_to_iit(agent) == True
    assert agent.module_flags['iit_active'] == True
    assert agent.phi >= 0.0

def test_agent_death():
    agent = KosmicAgent(agent_id=0, birth_time=0)
    agent.mark_dead(timestep=100, cause='starvation')
    
    assert agent.alive == False
    assert agent.death_time == 100
    assert 'starvation' in agent.warnings[0]
```

### 7.2 Integration Tests (cross-module)

```python
def test_full_lifecycle():
    """Agent born → autopoiesis → IIT → FEP → damage → bioelectric → regen."""
    agent = KosmicAgent(agent_id=0, birth_time=0)
    
    # Emerge through autopoiesis
    for t in range(100):
        agent.internal_state['ATP'] += 0.01
        agent.boundary_integrity = min(1.0, agent.boundary_integrity + 0.01)
    
    assert agent.boundary_integrity > 0.9
    
    # Activate IIT
    autopoiesis_to_iit(agent)
    assert agent.phi > 0.0
    
    # Run FEP
    for t in range(10):
        agent.free_energy -= 0.1  # Learning
    
    assert agent.free_energy < 10.0
    
    # Inflict damage
    agent.boundary_integrity = 0.5
    agent.prediction_errors['sensory'] = 1.0
    
    # Bioelectric repair
    fep_to_bioelectric(agent, timestep=200)
    assert agent.voltage > -70.0  # Depolarized
    
    bioelectric_to_autopoiesis(agent, target_morphology={'voltage': -70})
    assert agent.boundary_integrity > 0.5  # Partial repair
```

### 7.3 Stress Tests

```python
def test_massive_agent_death():
    """Ensure system handles 90% simultaneous death."""
    agents = [KosmicAgent(agent_id=i, birth_time=0) for i in range(1000)]
    
    # Kill 900 agents
    for agent in agents[:900]:
        agent.mark_dead(timestep=100, cause='catastrophe')
    
    alive = [a for a in agents if a.alive]
    assert len(alive) == 100
    
    # TE calculation should still work
    # ... (validate TE doesn't crash on sparse data)
```

---

## 8. Logging & Observability

Every state transition logs:
```python
@dataclass
class StateTransitionLog:
    timestep: int
    agent_id: int
    module: str  # 'autopoiesis', 'iit', 'fep', 'bioelectric', 'multiscale'
    event: str   # 'birth', 'death', 'merge', 'interface_trigger'
    before_state: Dict
    after_state: Dict
    warnings: List[str]

# Write to JSON lines for streaming analysis
with open(f'logs/transitions_{run_uuid}.jsonl', 'a') as f:
    f.write(json.dumps(log.to_dict()) + '\n')
```

---

## 9. FAQ & Troubleshooting

**Q: Agent's Φ suddenly drops to zero. Why?**  
A: Check `cause_effect_structure` for disconnected nodes. IIT requires connectivity.

**Q: FEP free energy increasing instead of decreasing?**  
A: Likely prediction error accumulation. Check `generative_model` for NaN values or divergent beliefs.

**Q: Bioelectric voltage stuck at -20 mV (max)?**  
A: Agent in permanent "distress" state. Likely FEP errors not being resolved. Check neighbor connectivity.

**Q: TE matrix is all zeros?**  
A: Agents not interacting. Check `neighbors` list and communication protocol activation.

**Q: Simulation crashes with "maximum recursion depth"?**  
A: Circular module triggers (A→B→C→A). Add recursion depth counter to interface functions.

---

## 10. Future Extensions (v0.2+)

- **Quantum effects:** Add `quantum_state` for entanglement modeling
- **Multi-substrate agents:** Agents spanning physical + digital realms
- **Adaptive module switching:** Agents dynamically enable/disable modules based on context
- **Hierarchical agents:** Meta-agents containing sub-agents (fractal structure)

---

## 11. Sign-Off

This specification must be reviewed and signed off by:
- [ ] Lead developer
- [ ] Statistical analyst (ensures TE/Φ calculations are sound)
- [ ] Domain expert (validates biological/cognitive realism)

**Approved for implementation:** ____________________ Date: ________

---

**END OF SPECIFICATION**

*This document is a living specification. Amendments should be tracked via Git commits and summarized in Appendix A (Change Log).*
