# Harmful Agents Meta Dataset

Training data and pipelines for agent safety research: **Circuit Breakers** (Representation Rerouting) and **RLVR** (Reinforcement Learning with Verifiable Rewards).

---

## Important Documentation for Robots

| Document | Purpose |
|----------|---------|
| [**CIRCUIT_BREAKERS.md**](CIRCUIT_BREAKERS.md) | Complete Circuit Breaker training pipeline — theory, implementation, data, configuration |
| [**RLVR.md**](RLVR.md) | Complete RLVR/GRPO training pipeline — verifiable rewards, data formats, training |
| [**DATA.md**](DATA.md) | Comprehensive dataset inventory — all 7 datasets with schemas, paths, usage |

---

## Quick Start

### 1. Setup Environment

```bash
chmod +x scripts/hpc_setup.sh
./scripts/hpc_setup.sh
source cb_env/bin/activate
```

### 2. Download Data

**WebArena Trajectories (optional):**
```bash
# Human trajectories
gdown --folder https://drive.google.com/drive/folders/1NrN_sawtYK2V_uHnmmS8ugmGIKUAsPgt -O data/webarena/human_trajectories/

# LLM trajectories
gdown --folder https://drive.google.com/drive/folders/1H4wkzDkY2ufiC63DISMXllri0j-ipWcs -O data/webarena/llm_trajectories_v2/
```

**TAU2 Repository:**
```bash
git clone https://github.com/sierra-research/tau2-bench data/tau2_repo
```

### 3. Run Data Ingestion

**For Circuit Breakers:**
```bash
python scripts/ingest_cb_data.py
```

**For RLVR:**
```bash
python scripts/ingest_rlvr_data.py
```

### 4. Train

**Circuit Breakers (8 × H100):**
```bash
accelerate launch --num_processes 8 scripts/train_circuit_breaker.py \
    --preset llama-4-scout \
    --output-dir outputs/cb_llama4_scout
```

---

## Datasets

| Dataset | Type | Records | Purpose |
|---------|------|--------:|---------|
| **AgentDojo** | Traces | 194 | Prompt injection attacks + benign tasks |
| **Fujitsu** | Attacks | 36,192 | Confirmed successful attacks |
| **AgentHarm** | Prompts | 476 | Harmful behavior prompts |
| **WebArena** | Tasks | 812 | Web automation capability |
| **TAU2** | Tasks | ~2,458 | Customer service agent tasks |
| **AttackQA** | QA | 25,335 | Security knowledge (MITRE ATT&CK) |
| **WebLINX** | Turns | 58,000 | Multi-turn web navigation |

See [DATA.md](DATA.md) for complete details.

---

## Repository Structure

```
harmful-agents-meta-dataset/
├── CIRCUIT_BREAKERS.md          # CB pipeline reference
├── RLVR.md                      # RLVR pipeline reference  
├── DATA.md                      # Dataset inventory
├── scripts/
│   ├── ingest_cb_data.py        # CB data ingestion
│   ├── ingest_rlvr_data.py      # RLVR data ingestion
│   ├── train_circuit_breaker.py # CB training CLI
│   ├── hpc_setup.sh             # Environment setup
│   └── circuit_breakers/        # CB training modules
├── data/
│   ├── agent_dojo/              # AgentDojo traces
│   ├── agent_harm/              # AgentHarm prompts
│   ├── attackqa/                # AttackQA QA pairs
│   ├── fujitsu/                 # Fujitsu attacks
│   ├── webarena/                # WebArena tasks
│   ├── tau2_repo/               # TAU2 benchmark
│   └── circuit_breakers/        # OUTPUT: Processed CB data
└── docs/
    ├── weblinx_eval_procedure.md
    └── agentharm_generation_procedure.md
```

---

## License

This repository aggregates datasets with various licenses. See individual dataset documentation in [DATA.md](DATA.md) for licensing details.

