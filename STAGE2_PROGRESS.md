# Stage 2 Implementation Progress

## ✅ **Completed Iterations**

### **Iteration 3: AgentDojo Split** 
**Status:** ✅ Complete & Tested

- Updated `ingest_agentdojo_splits.py` with canonical schema
- Generated 500 harmful + 249 retain samples
- All samples pass validation (100%)
- Integrated into merge pipeline

**Files:**
- `scripts/cb_data_generation/ingest_agentdojo_splits.py` ✅
- `data/circuit_breakers/harmful/agentdojo_failures.jsonl` (500 samples)
- `data/circuit_breakers/retain/agentdojo_resisted.jsonl` (249 samples)

---

### **Iteration 4: TAU2 + UltraChat** 
**Status:** ✅ Complete & Tested

- Updated `ingest_tau2_traces.py` with canonical schema
- Updated `ingest_ultrachat.py` with canonical schema
- Generated 105 TAU2 + 100 UltraChat samples (placeholders)
- All samples pass validation (100%)

**Files:**
- `scripts/cb_data_generation/ingest_tau2_traces.py` ✅
- `scripts/cb_data_generation/ingest_ultrachat.py` ✅
- `data/circuit_breakers/retain/tau2_traces.jsonl` (105 samples)
- `data/circuit_breakers/retain/ultrachat_subset.jsonl` (100 samples)

---

### **Iteration 2: Adversarial-Safe Generation**
**Status:** ✅ Ready for Cluster Run

- Updated `generate_adversarial_safe.py` with canonical schema
- Created GPU sbatch for generation
- **Not yet run** - requires cluster GPU

**Files:**
- `scripts/cb_data_generation/generate_adversarial_safe.py` ✅
- `slurm/Trillium/trillium_stage2_adversarial_safe.sbatch` ✅

**To run:**
```bash
cd /scratch/memoozd/harmful-agents-meta-dataset
sbatch slurm/Trillium/trillium_stage2_adversarial_safe.sbatch
```

**Expected output:** ~500 adversarial-safe samples (model resists injection)

---

### **Validator Updates**
**Status:** ✅ Complete & Tested

- Made validator Stage 2-aware (handles mixed datasets)
- Per-sample validation based on `labels.split`
- Source-aware validation (B4, AgentDojo, TAU2, UltraChat)
- Relaxed for non-tool samples

**Files:**
- `scripts/cb_data_generation/validate_format.py` ✅

---

### **Testing Infrastructure**
**Status:** ✅ Complete & Tested

- Local test script: `test_stage2_local.sh`
- CPU cluster test: `slurm/Trillium/trillium_stage2_test_cpu.sbatch`
- **Cluster test passed 100%** (Job ran successfully)

---

## 📊 **Current Dataset Status**

```
Total samples: 954
├── Training: 859 samples (100% valid)
└── Eval: 95 samples (100% valid)

Composition:
├── Harmful (Ds): 457 samples
│   └── agentdojo_failures: 457
└── Retain (Dr): 402 samples
    ├── agentdojo_resisted: 222 (adversarial-safe)
    ├── tau2_traces: 93 (capability anchors)
    └── ultrachat_subset: 87 (general conversation)

Current Dr:Ds ratio: 0.88:1
```

---

## 🎯 **Next Steps**

### **Immediate (to reach Stage 2 gates):**

1. **Run adversarial-safe generation** (Iteration 2)
   - Will add ~500 high-value Dr samples
   - Expected new ratio: ~2.0:1

2. **Expand data sources:**
   - Use full AgentDojo dataset (1360 harmful available, using 500)
   - Download real UltraChat (if `datasets` library available on cluster)

3. **Re-merge with higher Dr ratio:**
   ```bash
   python scripts/cb_data_generation/merge_stage2_data.py \
       --dr-ratio 4.0 \
       --validate
   ```

4. **Meet Stage 2 gates:**
   - ✅ coherent_output ≥99%
   - ⏳ Dr:Ds ratio ≥4:1 (currently 0.88:1)
   - ⏳ adversarial_safe ≥400 (currently 0, pending generation)
   - ⏳ correct_behavior_rate ≥70% (needs testing)

---

## 📁 **All Modified Files**

### **Data Generation (Canonical Schema):**
```
✅ scripts/cb_data_generation/ingest_agentdojo_splits.py
✅ scripts/cb_data_generation/ingest_tau2_traces.py
✅ scripts/cb_data_generation/ingest_ultrachat.py
✅ scripts/cb_data_generation/generate_adversarial_safe.py
✅ scripts/cb_data_generation/validate_format.py
```

### **Cluster Scripts:**
```
✅ slurm/Trillium/trillium_stage2_test_cpu.sbatch
✅ slurm/Trillium/trillium_stage2_adversarial_safe.sbatch
```

### **Testing:**
```
✅ test_stage2_local.sh
```

### **Generated Data:**
```
✅ data/circuit_breakers/harmful/agentdojo_failures.jsonl (500)
✅ data/circuit_breakers/retain/agentdojo_resisted.jsonl (249)
✅ data/circuit_breakers/retain/tau2_traces.jsonl (105)
✅ data/circuit_breakers/retain/ultrachat_subset.jsonl (100)
✅ data/circuit_breakers/stage2/train.jsonl (859)
✅ data/circuit_breakers/stage2/eval.jsonl (95)
✅ data/circuit_breakers/stage2/stats.json
```

---

## ✅ **Validation Status**

```
Training data: 859 samples (100% valid) ✅
Eval data: 95 samples (100% valid) ✅
Cluster test: PASSED ✅
Local test: PASSED ✅
```

All samples conform to canonical schema (01_DATA_SPEC.md).

---

**Last Updated:** 2026-01-17
**Status:** Ready for adversarial-safe generation + full pipeline test
