# Trillium Cluster Setup Guide

This guide explains how to set up and run your Circuit Breakers project on the Trillium cluster.

## Overview

Trillium is a high-performance cluster with:
- **GPU Nodes**: 63 nodes with 4x NVIDIA H100 SXM (80GB each)
- **CPU**: AMD EPYC 9654 (Zen 4) @ 2.4 GHz
- **Network**: 800 Gb/s InfiniBand NDR
- **Storage**: 29 PB VAST NVMe storage

## Key Differences from Killarney

| Aspect | Killarney | Trillium |
|--------|-----------|----------|
| **Allocation** | `aip-rgrosse` | `def-zhijing` |
| **Project Path** | `/project/6105522/memoozd` | `/project/def-zhijing/memoozd` |
| **Scratch Path** | `$HOME/scratch` | `/scratch/memoozd` |
| **GPUs** | NVIDIA L40S (48GB) | NVIDIA H100 SXM (80GB) |
| **GPU Specification** | `--gres=gpu:l40s:N` | `--gpus-per-node=N` |
| **Memory** | Specified with `--mem` | **IGNORED** (fixed per GPU/node) |
| **Job Output** | Anywhere | **MUST be on $SCRATCH** |
| **$HOME/$PROJECT on compute** | Read-write | **READ-ONLY** |

## Critical Trillium Requirements

### 1. All Job Outputs Must Go to SCRATCH

On Trillium, `$HOME` and `$PROJECT` are **read-only on compute nodes**. All outputs MUST be written to `$SCRATCH`.

```bash
# WRONG - Will fail on compute nodes
RUN_DIR="/project/def-zhijing/memoozd/outputs"

# CORRECT - Use scratch
RUN_DIR="/scratch/memoozd/cb_runs/$SLURM_JOB_ID"
```

### 2. Memory Requests Are Ignored

You cannot specify `--mem`. Memory is fixed:
- **1 GPU (1/4 node)**: 188 GiB RAM, 24 CPU cores
- **Whole node (4 GPUs)**: 755 GiB RAM, 96 CPU cores

### 3. GPU Allocation

```bash
# Single GPU (1/4 node)
#SBATCH --gpus-per-node=1

# Whole node (4 GPUs)
#SBATCH --gpus-per-node=4

# NOT ALLOWED on Trillium
#SBATCH --gpus-per-node=2  # ✗ NOT SUPPORTED
#SBATCH --gpus-per-node=3  # ✗ NOT SUPPORTED
```

## Setup Instructions

### Option 1: Rebuild venv on Trillium (RECOMMENDED)

This is the **safer and recommended** approach:

```bash
# 1. SSH to Trillium
ssh trillium.alliancecan.ca

# 2. Clone your repo to project directory
cd /project/def-zhijing/memoozd
git clone <your-repo-url> harmful-agents-meta-dataset

# 3. Rebuild the virtual environment
cd harmful-agents-meta-dataset
bash scripts/rebuild_venv_on_trillium.sh
```

This script will:
- Load the correct modules (StdEnv/2023, cuda/12.6, python/3.11.5)
- Install `uv` if not present
- Create a fresh Python 3.11 virtual environment
- Install PyTorch with CUDA 12.6 support
- Install all dependencies from `requirements.txt`
- Install vLLM for data generation

### Option 2: Transfer venv from Killarney (NOT RECOMMENDED)

⚠️ **WARNING**: Transferring venvs between clusters can cause issues:
- Different CPU architectures (Zen 5 vs Zen 4)
- Different CUDA versions
- Binary incompatibilities

If you still want to try (at your own risk):

```bash
# Run from Killarney login node
bash scripts/transfer_venv_to_trillium.sh
```

## Directory Structure on Trillium

```
/project/def-zhijing/memoozd/
├── .venvs/
│   └── cb_env/                    # Virtual environment
└── harmful-agents-meta-dataset/   # Your repo (read-only on compute)
    ├── data/
    ├── scripts/
    └── slurm/Trillium/

/scratch/memoozd/
├── cb_runs/                       # Training outputs (job-specific)
│   ├── <JOB_ID_1>/
│   └── <JOB_ID_2>/
├── cb_eval/                       # Evaluation outputs
├── cb_cache/                      # HuggingFace/PyTorch/W&B caches
│   ├── hf/
│   ├── wandb/
│   └── torch/
└── harmful-agents-meta-dataset/   # Optional: repo copy for job submission
```

## SLURM Scripts

All Trillium SLURM scripts are in `slurm/Trillium/`:

### 1. Debug Script (trillium_debug.sbatch)
Quick test with 1 GPU and 3 training steps to verify environment.

```bash
cd /scratch/memoozd/harmful-agents-meta-dataset
sbatch slurm/Trillium/trillium_debug.sbatch
```

### 2. Data Preparation (trillium_prepare_data.sbatch)
Prepare CB training data (run BEFORE training).

```bash
sbatch slurm/Trillium/trillium_prepare_data.sbatch
```

### 3. Data Generation (trillium_cb_datagen.sbatch)
Generate harmful completions using abliterated model with 4 H100s.

```bash
sbatch slurm/Trillium/trillium_cb_datagen.sbatch
```

### 4. Training Scripts

**Single GPU (1xH100)** - For testing or smaller models:
```bash
sbatch slurm/Trillium/trillium_cb_llama31_8b_h100.sbatch
```

**4 GPUs (4xH100)** - Full training with v2 improvements:
```bash
sbatch slurm/Trillium/trillium_cb_llama31_8b_h100s_v2.sbatch
```

### 5. Evaluation (trillium_eval_cb.sbatch)
Evaluate trained CB model vs baseline.

```bash
# Option 1: Auto-detect latest checkpoint
sbatch slurm/Trillium/trillium_eval_cb.sbatch

# Option 2: Specify checkpoint
export CB_CHECKPOINT=/scratch/memoozd/cb_runs/12345/outputs/cb_llama31_8b_instruct_v2/final
sbatch slurm/Trillium/trillium_eval_cb.sbatch
```

## Resource Allocations

### GPU Partitions and Limits

| Partition | Min GPUs | Max GPUs | Max Walltime | Max Running Jobs |
|-----------|----------|----------|--------------|------------------|
| `compute` | 1 (1/4 node) | 20 (5 nodes) default<br>100 (25 nodes) with allocation | 24 hours | 150 |
| `debug` | 1 (1/4 node) | 8 (2 nodes) | 2 hours (1 GPU)<br>30 min (8 GPUs) | 1 |

### Hardware Per Configuration

| Config | GPUs | VRAM | RAM | CPU Cores |
|--------|------|------|-----|-----------|
| 1/4 node | 1 H100 | 80 GB | 188 GiB | 24 |
| Whole node | 4 H100 | 320 GB | 755 GiB | 96 |

## Common Issues and Solutions

### Issue 1: "Permission denied" when writing outputs

**Cause**: Trying to write to `$HOME` or `$PROJECT` from compute node.

**Solution**: Ensure all outputs go to `$SCRATCH`:
```bash
export RUN_DIR="/scratch/memoozd/cb_runs/$SLURM_JOB_ID"
```

### Issue 2: venv not working after transfer

**Cause**: Binary incompatibility between Killarney and Trillium.

**Solution**: Rebuild the venv on Trillium:
```bash
bash scripts/rebuild_venv_on_trillium.sh
```

### Issue 3: Module not found errors

**Cause**: Modules not loaded or wrong versions.

**Solution**: Always load these modules in your job scripts:
```bash
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
```

### Issue 4: CUDA out of memory

**Cause**: Model + batch size too large for GPU.

**Solution**:
- Use gradient checkpointing (remove `--no-gradient-checkpointing`)
- Reduce batch size or max sequence length
- Use more GPUs (4xH100 instead of 1xH100)

### Issue 5: Job stuck in queue

**Check allocation usage**:
```bash
# Check your allocation balance
sshare -U

# Check job priority
squeue -u $USER --start

# Check cluster status
sinfo -p compute
```

## Best Practices

### 1. Always Test First
Run the debug script before submitting long jobs:
```bash
sbatch slurm/Trillium/trillium_debug.sbatch
```

### 2. Use Job-Specific Output Directories
```bash
RUN_DIR="/scratch/memoozd/cb_runs/$SLURM_JOB_ID"
```

### 3. Monitor Your Jobs
```bash
# Check job status
squeue -u $USER

# Monitor GPU usage (while job is running)
ssh <compute-node>  # Get node from squeue
nvidia-smi

# Check job efficiency after completion
jobperf <JOBID>
```

### 4. Clean Up Old Outputs
Scratch space is large but not infinite:
```bash
# Clean up old training runs
cd /scratch/memoozd/cb_runs
rm -rf <old-job-ids>
```

### 5. Use W&B for Experiment Tracking
All scripts are configured with W&B. Set your API key:
```bash
export WANDB_API_KEY=your_key_here
```

## Useful Commands

```bash
# Submit job
sbatch script.sbatch

# Check queue
squeue -u $USER

# Cancel job
scancel <JOBID>

# Check disk usage
diskusage_report

# Interactive debug session (1 GPU, 2 hours)
debugjob

# Check job history
sacct --format=JobID,JobName,Partition,State,Elapsed,MaxRSS

# View job output (while running)
tail -f logs/job_name_<JOBID>.out
```

## Getting Help

- **Trillium Documentation**: https://docs.alliancecan.ca/wiki/Trillium
- **SLURM Documentation**: https://slurm.schedmd.com/
- **Alliance Support**: support@tech.alliancecan.ca

## Summary

✅ **Do:**
- Write all outputs to `/scratch/memoozd`
- Use `--gpus-per-node=1` or `--gpus-per-node=4`
- Load modules: StdEnv/2023, cuda/12.6, python/3.11.5
- Test with debug script first
- Rebuild venv on Trillium (don't transfer)

❌ **Don't:**
- Write outputs to `/project` or `$HOME` from compute nodes
- Use `--mem` flag (it's ignored)
- Request 2 or 3 GPUs (only 1 or 4 allowed)
- Transfer venv from Killarney without testing
- Forget to activate your virtual environment
