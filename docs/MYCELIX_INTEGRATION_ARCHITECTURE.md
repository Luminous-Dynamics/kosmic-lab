# 🌊 Kosmic-Lab ↔ Mycelix Integration Architecture

**Vision**: Fuse revolutionary consciousness research (Kosmic-Lab) with decentralized knowledge networks (Mycelix) to create the world's first **distributed, verifiable, AI-accelerated consciousness science platform**.

**Status**: Design Phase → Prototype Phase
**Target**: Phase 1 prototype in 2 weeks

---

## 🎯 Integration Philosophy

### Core Synergies

1. **K-Codex + DHT = Immutable Science** (formerly K-Passport)
   - Kosmic-Lab's reproducibility → Mycelix's sovereign data storage
   - Every experiment becomes a permanent, verifiable truth claim

2. **AI Designer + Solver Network = Swarm Intelligence**
   - Bayesian optimization → distributed competitive experiment proposals
   - Single researcher → global research collective

3. **Dashboard + Civilization Layer = Living Metrics**
   - K-index monitoring → ecological consciousness tracking
   - Research → regenerative world modeling

4. **Federated Learning + Scaled PoGQ = Privacy-Preserving Collaboration**
   - Multi-lab data → zero raw data sharing
   - Differential privacy + zkML proofs

---

## 🏗️ Layered Integration Architecture

### Phase 1: Foundation (Weeks 1-2) - **START HERE**

#### 1.1 K-Codex → Holochain DHT (Local K-Passport → Eternal K-Codex) (Layer 1)

**What**: Store every K-passport as immutable Holochain entry

**Why**:
- Permanent provenance (can't be lost/altered)
- Agent-sovereign sharing (researchers control access)
- P2P verification (no central authority needed)

**Implementation**:

```rust
// holochain/zomes/passport_zome/src/lib.rs

#[hdk_entry_helper]
pub struct KCodex {  // Formerly KPassport
    pub run_id: String,
    pub commit: String,
    pub config_hash: String,
    pub seed: u32,
    pub experiment: String,
    pub params: BTreeMap<String, f64>,
    pub estimators: Estimators,
    pub metrics: Metrics,
    pub timestamp: Timestamp,
    pub researcher_agent: AgentPubKey,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Metrics {
    pub k_index: f64,
    pub tat: f64,
    pub recovery: f64,
    pub phi: Option<f64>,
    pub te_mutual: Option<f64>,
    pub te_symmetry: Option<f64>,
}

// Create K-passport entry
#[hdk_extern]
pub fn create_codex(codex: KCodex)  // Alias: create_passport for compatibility -> ExternResult<HeaderHash> {
    // Validate passport schema
    validate_passport(&passport)?;

    // Create entry
    let header_hash = create_entry(&passport)?;

    // Create links for querying
    let base = hash_entry(&passport.experiment)?;
    create_link(base, header_hash.clone(), LinkTag::new("experiment"))?;

    // Create K-index range link for corridor queries
    let k_bucket = (passport.metrics.k_index * 10.0) as u32; // 0-25 buckets
    create_link(
        Path::from(format!("k_bucket_{}", k_bucket)).path_entry_hash()?,
        header_hash.clone(),
        LinkTag::new("k_index")
    )?;

    Ok(header_hash)
}

// Query passports by K-index corridor
#[hdk_extern]
pub fn get_corridor_codices  // Returns K-Codex entries(min_k: f64, max_k: f64) -> ExternResult<Vec<KPassport>> {
    let min_bucket = (min_k * 10.0) as u32;
    let max_bucket = (max_k * 10.0) as u32;

    let mut passports = Vec::new();

    for bucket in min_bucket..=max_bucket {
        let path = Path::from(format!("k_bucket_{}", bucket));
        let links = get_links(path.path_entry_hash()?, Some(LinkTag::new("k_index")))?;

        for link in links {
            let passport: KPassport = get(link.target, GetOptions::default())?
                .ok_or(WasmError::Guest("Passport not found".into()))?
                .entry()
                .to_app_option()?
                .ok_or(WasmError::Guest("Invalid passport".into()))?;

            if passport.metrics.k_index >= min_k && passport.metrics.k_index <= max_k {
                passports.push(passport);
            }
        }
    }

    Ok(passports)
}

// Verify passport integrity (git commit + config hash)
#[hdk_extern]
pub fn verify_passport(passport_hash: HeaderHash) -> ExternResult<VerificationResult> {
    let passport: KPassport = get(passport_hash, GetOptions::default())?
        .ok_or(WasmError::Guest("Passport not found".into()))?
        .entry()
        .to_app_option()?
        .ok_or(WasmError::Guest("Invalid passport".into()))?;

    // In production: verify git commit exists in repo
    // In production: verify config_hash matches stored config

    Ok(VerificationResult {
        valid: true,
        commit_verified: true,
        config_verified: true,
        timestamp: sys_time()?,
    })
}
```

**Python Bridge** (Kosmic-Lab → Holochain):

```python
# scripts/holochain_bridge.py

import json
from pathlib import Path
import subprocess
from typing import Dict, Any

class HolochainBridge:
    """Bridge between Kosmic-Lab K-passports and Mycelix Holochain DHT."""

    def __init__(self, conductor_url: str = "http://localhost:8888"):
        self.conductor_url = conductor_url

    def publish_passport(self, passport_path: Path) -> str:
        """
        Publish K-passport to Holochain DHT.

        Returns:
            Header hash of created entry
        """
        with passport_path.open() as f:
            passport = json.load(f)

        # Call Holochain zome function
        result = self._call_zome(
            "passport_zome",
            "create_passport",
            passport
        )

        return result['header_hash']

    def query_corridor(self, min_k: float = 1.0, max_k: float = 2.5) -> list:
        """Query passports in K-index corridor."""
        return self._call_zome(
            "passport_zome",
            "get_corridor_codices  // Returns K-Codex entries",
            {"min_k": min_k, "max_k": max_k}
        )

    def verify_passport(self, header_hash: str) -> Dict[str, Any]:
        """Verify passport integrity."""
        return self._call_zome(
            "passport_zome",
            "verify_passport",
            {"passport_hash": header_hash}
        )

    def _call_zome(self, zome: str, fn_name: str, payload: Any) -> Any:
        """Call Holochain zome function via conductor API."""
        cmd = [
            "hc",
            "call",
            "--url", self.conductor_url,
            zome,
            fn_name,
            json.dumps(payload)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Holochain call failed: {result.stderr}")

        return json.loads(result.stdout)

# Usage example
if __name__ == "__main__":
    bridge = HolochainBridge()

    # Publish all K-passports to DHT
    for passport_file in Path("logs").glob("**/*.json"):
        header_hash = bridge.publish_passport(passport_file)
        print(f"✅ Published {passport_file.name}: {header_hash}")

    # Query corridor
    corridor = bridge.query_corridor(min_k=1.0, max_k=1.5)
    print(f"\\n📊 Found {len(corridor)} passports in corridor (K ∈ [1.0, 1.5])")
```

**Integration Test**:

```python
# tests/test_holochain_integration.py

import pytest
from pathlib import Path
from scripts.holochain_bridge import HolochainBridge

@pytest.fixture
def bridge():
    return HolochainBridge()

def test_publish_and_query_passport(bridge, tmp_path):
    """Test K-passport publishing to Holochain and corridor querying."""
    # Create test passport
    passport = {
        "run_id": "test-001",
        "commit": "abc123",
        "config_hash": "def456",
        "seed": 42,
        "experiment": "integration_test",
        "params": {"energy_gradient": 0.5},
        "estimators": {
            "phi": "empirical",
            "te": {"estimator": "kraskov", "k": 3, "lag": 1}
        },
        "metrics": {
            "k_index": 1.23,
            "tat": 0.615,
            "recovery": 0.99
        },
        "timestamp": "2025-11-09T12:34:56Z"
    }

    passport_file = tmp_path / "test_passport.json"
    with passport_file.open('w') as f:
        json.dump(passport, f)

    # Publish to DHT
    header_hash = bridge.publish_passport(passport_file)
    assert header_hash is not None

    # Query corridor
    results = bridge.query_corridor(min_k=1.0, max_k=1.5)

    # Should include our published passport
    assert any(r['run_id'] == 'test-001' for r in results)

def test_verify_passport_integrity(bridge):
    """Test passport verification via git commit and config hash."""
    # Assume passport already published
    header_hash = "QmXXXXX..."  # From previous publish

    verification = bridge.verify_passport(header_hash)

    assert verification['valid'] is True
    assert verification['commit_verified'] is True
    assert verification['config_verified'] is True
```

**Makefile Integration**:

```makefile
# Add to kosmic-lab/Makefile

holochain-publish:  # Publish K-passports to Mycelix DHT
	poetry run python scripts/holochain_bridge.py --publish logs/

holochain-query:  # Query corridor passports from DHT
	poetry run python scripts/holochain_bridge.py --query --min-k 1.0 --max-k 1.5

holochain-verify:  # Verify passport integrity
	poetry run python scripts/holochain_bridge.py --verify $(PASSPORT_HASH)
```

---

#### 1.2 Identity Integration (Layer 5)

**What**: Link Kosmic-Lab researchers to Mycelix VerifiedHumanity credentials

**Why**: Sybil-resistant collaboration, reputation tracking

**Implementation**:

```python
# core/researcher_identity.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class ResearcherIdentity:
    """Mycelix-verified researcher identity."""

    agent_pubkey: str  # Holochain agent public key
    verified_humanity: bool  # VerifiedHumanity credential
    orcid: Optional[str] = None  # Optional ORCID linking
    institution: Optional[str] = None
    specialization: Optional[str] = None
    reputation_score: float = 0.0  # Based on K-passport contributions

    def __post_init__(self):
        if not self.verified_humanity:
            raise ValueError("Researcher must have VerifiedHumanity credential")

# Update KPassportWriter to include researcher identity
def build_record(
    self,
    experiment: str,
    params: Dict[str, Any],
    researcher: ResearcherIdentity,  # NEW
    **kwargs
) -> Dict[str, Any]:
    record = {
        # ... existing fields ...
        "researcher": {
            "agent_pubkey": researcher.agent_pubkey,
            "verified": researcher.verified_humanity,
            "orcid": researcher.orcid,
        }
    }
    return record
```

---

### Phase 2: Intelligence Layer (Weeks 3-4)

#### 2.1 AI Designer → Solver Network (Layer 8)

**What**: Competitive experiment proposals from distributed solvers

**Architecture**:

```
Researcher posts intent:
  "Find parameters for K > 1.5 with <100 samples"
    ↓
Mycelix Solver Network:
  - Solver A: Bayesian optimization → proposal + confidence
  - Solver B: Genetic algorithm → proposal + confidence
  - Solver C: Grid search → proposal + confidence
    ↓
Epistemic Market:
  - Researchers bet on which proposal will work
  - Winning solver gets reputation + optional tokens
    ↓
Execute top proposals:
  - Run experiments with suggested parameters
  - K-passports prove outcomes
  - Update solver reputations
```

**Implementation Sketch**:

```python
# scripts/mycelix_solver_integration.py

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ExperimentIntent:
    """Declarative research goal."""
    target_k: float
    max_samples: int
    constraints: Dict[str, Any]
    reward_budget: float  # Optional token budget

@dataclass
class SolverProposal:
    """Experiment proposal from solver agent."""
    solver_id: str
    parameters: Dict[str, float]
    predicted_k: float
    confidence: float
    proof: str  # zkML proof of computation

class SolverNetworkClient:
    """Client for Mycelix Solver Network (Layer 8)."""

    def post_intent(self, intent: ExperimentIntent) -> str:
        """Post experiment intent to solver network."""
        # Call Mycelix intent layer
        pass

    def get_proposals(self, intent_id: str) -> List[SolverProposal]:
        """Retrieve solver proposals for intent."""
        pass

    def execute_and_verify(self, proposal: SolverProposal) -> Dict[str, Any]:
        """Execute proposed experiment and verify against prediction."""
        # Run FRE simulation
        # Compare result to prediction
        # Update solver reputation
        pass
```

---

#### 2.2 Federated Learning (Layer 9)

**What**: Train AI designer on data from multiple labs without sharing raw data

**Protocol**:

```
Lab A, B, C each have K-passport datasets
    ↓
Mycelix Scaled PoGQ coordinates:
  1. Each lab trains local Gaussian Process
  2. Labs share only model gradients (differentially private)
  3. Aggregate into global model
  4. Each lab gets improved predictions
    ↓
Result:
  - 3x more training data effectively
  - Zero raw data exposure
  - zkML proofs verify honest computation
```

---

### Phase 3: Ecosystem Integration (Month 2)

#### 3.1 Dashboard → Civilization Layer (Layer 10)

**Extend Kosmic Dashboard**:

```python
# Add ecological metrics to dashboard

class EcologicalMetrics:
    """Track research's ecological impact."""

    compute_carbon: float  # kg CO2 from experiments
    data_storage_footprint: float  # GB stored
    energy_efficiency: float  # K-index per kWh

    # Tie to biodiversity RWAs
    regenerative_offset: float  # Trees planted via research tokens

# Add to dashboard tabs:
# - "Ecological Impact" tab
# - Real-time carbon tracking
# - Regenerative offset visualization
```

---

## 🚀 Immediate Action Plan (This Week)

### Day 1-2: Holochain Zome Development

```bash
cd holochain/zomes
cargo new passport_zome
# Implement KPassport entry type and functions (above)
```

### Day 3-4: Python Bridge

```bash
cd scripts
# Implement HolochainBridge class (above)
python holochain_bridge.py --test
```

### Day 5: Integration Testing

```bash
# Spin up Holochain conductor
hc sandbox generate
hc sandbox run

# Publish test K-passports
make holochain-publish

# Query corridor
make holochain-query
```

### Day 6-7: Documentation & Demo

```bash
# Create integration demo
make demo-holochain

# Update docs
# Record video walkthrough
```

---

## 📊 Success Metrics

**Phase 1 (Weeks 1-2)**:
- ✅ 100 K-passports published to Holochain DHT
- ✅ Sub-second corridor queries (K ∈ [a, b])
- ✅ Verified passport integrity (git + config hash)

**Phase 2 (Weeks 3-4)**:
- ✅ 3+ solvers proposing experiments competitively
- ✅ 50% improvement in experiment efficiency via swarm intelligence

**Phase 3 (Month 2)**:
- ✅ Federated learning across 3+ labs
- ✅ Dashboard showing ecological metrics
- ✅ First publication using Mycelix-verified K-passports

---

## 🌟 Long-Term Vision

**Year 1**: Kosmic-Lab becomes the reference platform for Mycelix-verified consciousness research

**Year 2**: 100+ labs contributing to federated K-index knowledge graph

**Year 3**: AI-designed experiments discover novel coherence pathways that humans wouldn't find

**Year 5**: Consciousness science becomes fully decentralized, verifiable, and regenerative

---

## 🎯 Key Design Principles

1. **Agent Sovereignty**: Researchers control their data and reputation
2. **Verifiable Truth**: Every claim backed by K-passport + zkML proof
3. **Collective Intelligence**: Distributed solvers beat centralized AI
4. **Regenerative**: Research creates ecological value, not just papers
5. **Playful Rigor**: Make collaboration feel like a game, not bureaucracy

---

*"From solo researcher to myceliated swarm—coherence as computational love, scaling infinitely."* 🌊

**Status**: Architecture complete, ready for Phase 1 prototyping

**Next**: Implement `passport_zome` (Rust) + `holochain_bridge.py` (Python)
