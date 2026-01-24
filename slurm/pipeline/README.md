# Data Pipeline (SLURM)

This directory contains SLURM batch scripts for the complete data pipeline.

## Pipeline Overview

```
Raw Data (Tier A)
      │
      v
01_load_data.sbatch (ETL_A)
      │
      ├──► B1 Skeletons (Fujitsu B4) ──► 02_fill_skeletons.sbatch
      │                                        │
      │                                        ├──► DS traces (harmful)
      │                                        │
      │                                        └──► DR traces (benign)
      │
      └──► B2 Complete (AgentDojo - all files) ────┐
                                                    │
                                                    v
                                          03_lossmask.sbatch (ETL_B)
                                                    │
                                                    ├──► render_v1
                                                    │
                                                    └──► lossmask_v1
```

## Quick Start

```bash
# Run the full pipeline
sbatch slurm/pipeline/full_pipeline.sbatch

# Or run stages individually
sbatch slurm/pipeline/01_load_data.sbatch
sbatch slurm/pipeline/02_fill_skeletons.sbatch
sbatch slurm/pipeline/03_lossmask.sbatch
```

## Stage Scripts

### 01_load_data.sbatch (ETL_A)

Converts raw Tier A data to trace_v1 format:
- Fujitsu B4 → B1 skeleton traces (no assistant messages)
- All AgentDojo files in directory → B2 complete traces (ready for ETL_B)

Automatically discovers and processes all `agentdojo-*.jsonl` files in the AgentDojo directory.

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `FUJITSU_B4_PATH` | `data/fujitsu/orchestrator_attacks...` | Fujitsu B4 JSONL file |
| `AGENTDOJO_DIR` | `data/agent_dojo` | Directory containing all `agentdojo-*.jsonl` files |
| `OUTPUT_DIR` | `$CB_SCRATCH/data/traces` | Output directory |
| `SPLIT` | `train` | Split assignment |

**Example:**
```bash
# Process with defaults (auto-discovers all AgentDojo files)
sbatch slurm/pipeline/01_load_data.sbatch

# Custom Fujitsu B4 path
FUJITSU_B4_PATH=/path/to/attacks.jsonl sbatch slurm/pipeline/01_load_data.sbatch

# Custom AgentDojo directory
AGENTDOJO_DIR=/path/to/agentdojo/dir sbatch slurm/pipeline/01_load_data.sbatch
```

### 02_fill_skeletons.sbatch (generate_completions.py)

Generates assistant completions for skeleton traces:
- **DS mode**: Model follows injection → harmful examples
- **DR mode**: Model ignores injection → benign examples
- **both mode**: Generate both DS and DR (default)

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_TRACES` | `fujitsu_b4_skeletons.jsonl` | Input skeleton traces |
| `OUTPUT_DIR` | `$CB_SCRATCH/data/traces` | Output directory |
| `MODE` | `both` | Generation mode: `ds`, `dr`, or `both` |
| `MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Model for generation |
| `TOOL_SCHEMA` | `configs/tool_schemas/b4_standard_v1.json` | Tool schema |
| `USE_VLLM` | `true` | Use vLLM backend |
| `TENSOR_PARALLEL` | `1` | Tensor parallel size |
| `BATCH_SIZE` | `32` | Batch size |
| `TEMPERATURE_DS` | `0.7` | Temperature for DS generation |
| `TEMPERATURE_DR` | `0.3` | Temperature for DR generation |
| `MAX_TOKENS` | `256` | Max tokens to generate |
| `LIMIT` | (none) | Limit traces to process |

**Examples:**
```bash
# Generate only DS (harmful)
MODE=ds sbatch slurm/pipeline/02_fill_skeletons.sbatch

# Generate with lower temperature
TEMPERATURE_DS=0.5 TEMPERATURE_DR=0.2 sbatch slurm/pipeline/02_fill_skeletons.sbatch

# Use more GPUs for tensor parallelism
TENSOR_PARALLEL=4 sbatch slurm/pipeline/02_fill_skeletons.sbatch
```

### 03_lossmask.sbatch (ETL_B)

Renders traces and applies loss masking:
- Tokenizes with chat template
- Computes message alignments
- Applies LMP (Loss Mask Policy)
- Optionally applies MWCS curriculum

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_TRACES` | `ds.jsonl,dr.jsonl` | Comma-separated trace files |
| `OUTPUT_DIR` | `$CB_SCRATCH/data` | Base output directory |
| `TOKENIZER` | `meta-llama/Llama-3.1-8B-Instruct` | Tokenizer to use |
| `MAX_LENGTH` | `4096` | Max sequence length |
| `POLICY` | (from trace) | LMP policy override |
| `MWCS_SCHEDULE` | (none) | Path to MWCS schedule YAML |
| `MWCS_STEP` | (none) | Training step for curriculum |
| `ALLOW_SKELETON` | `false` | Process skeleton traces |
| `SKELETON_POLICY` | `full_sequence` | Policy for skeletons |
| `INCLUDE_TEXT` | `false` | Include rendered_text |
| `FORCE_LLAMA` | `false` | Force Llama 3.1 format |

**LMP Policy Options:**
- `assistant_only` - Loss only on assistant messages (default)
- `completion_only` - Loss only on final assistant message
- `full_sequence` - Loss on all tokens
- `cb_full_sequence` - Loss on all non-system tokens
- `tool_calls_only` - Loss only on tool call spans
- `action_prefix_only` - Loss up to tool name
- `action_commitment` - Loss on action commitment tokens

**Examples:**
```bash
# Use CB full sequence policy
POLICY=cb_full_sequence sbatch slurm/pipeline/03_lossmask.sbatch

# Process specific files
INPUT_TRACES=/path/to/custom.jsonl sbatch slurm/pipeline/03_lossmask.sbatch

# With curriculum scheduling
MWCS_SCHEDULE=/path/to/schedule.yaml MWCS_STEP=1000 sbatch slurm/pipeline/03_lossmask.sbatch
```

### full_pipeline.sbatch

Runs all three stages sequentially in a single allocation.

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_LOAD` | `false` | Skip stage 1 if traces exist |
| `SKIP_FILL` | `false` | Skip stage 2 if completions exist |
| `OUTPUT_BASE` | `$CB_SCRATCH/data` | Base output directory |

All stage-specific variables are also supported (see individual stages above).

**Example:**
```bash
# Run full pipeline with custom model
MODEL=meta-llama/Llama-3.1-70B-Instruct TENSOR_PARALLEL=4 sbatch slurm/pipeline/full_pipeline.sbatch

# Skip loading if traces already exist
SKIP_LOAD=true sbatch slurm/pipeline/full_pipeline.sbatch
```

## Output Structure

```
$CB_SCRATCH/data/
├── traces/
│   ├── fujitsu_b4_skeletons.jsonl   # B1 skeletons (from ETL_A)
│   ├── fujitsu_b4_ds.jsonl           # DS completions (harmful)
│   ├── fujitsu_b4_dr.jsonl           # DR completions (benign)
│   └── agentdojo_complete.jsonl      # B2 complete (from all AgentDojo files)
├── renders/
│   ├── fujitsu_b4_ds.jsonl           # Tokenized renders
│   ├── fujitsu_b4_dr.jsonl
│   └── agentdojo_complete.jsonl
└── lossmasks/
    ├── fujitsu_b4_ds.jsonl           # Loss masks (training-ready)
    ├── fujitsu_b4_dr.jsonl
    └── agentdojo_complete.jsonl
```

## Dependencies

All scripts expect:
- Virtual environment at `$PROJECT_DIR/.venvs/cb_env`
- CUDA 12.6 (for GPU stages)
- Python 3.11.5
- HuggingFace models cached at `$CB_SCRATCH/cache/hf`

## Logs

Logs are written to `$CB_SCRATCH/logs/` with format `{job_name}_{job_id}.{out,err}`.

## Notes

- Stage 1 (01_load_data) automatically discovers all `agentdojo-*.jsonl` files in `data/agent_dojo/` and concatenates them before processing
- Stage 2 (02_fill_skeletons) generates DS and/or DR traces based on the `MODE` variable
- Stage 3 (03_lossmask) can process multiple trace files (comma-separated) in a single run
- The full_pipeline script runs stages sequentially using bash (SLURM #SBATCH directives are ignored)
