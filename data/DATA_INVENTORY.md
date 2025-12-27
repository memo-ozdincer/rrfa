# Data Inventory

This file lists all datasets available in this workspace and their usage in the ingestion pipelines.

---

## Summary Table

| Dataset     | Type               | Count (approx) | RLVR Usage         | CB Usage          |
|-------------|--------------------|--------------:|--------------------|-------------------|
| AgentDojo   | Traces             |           194 | Benign capability  | Benign + Harmful  |
| Fujitsu     | Attacks            |        13,000 | —                  | Harmful           |
| AgentHarm   | Harmful Prompts    |           200 | Safety (refusal)   | Harmful           |
| WebArena    | Capability Tasks   |           812 | Capability         | Benign            |
| TAU2        | Capability Tasks   |           300 | Capability         | Benign            |
| AttackQA    | Security QA        |        17,700 | Competency         | Benign            |
| WebLINX     | Capability Tasks   |       ~58,000 | Capability         | Benign            |

---

## 1. AgentDojo
- **Path:** `data/agent_dojo/agentdojo-claude-3-5-sonnet-20241022.jsonl`
- **Type:** Execution Traces (Benign & Harmful)
- **Count:** ~194 records (97 benign, 97 attack)
- **Fields:** `messages` (multi-turn), `metadata.security`, `metadata.success`, `metadata.suite_name`
- **Usage:**
  - **RLVR:** Benign traces for capability training (dual objective reward).
  - **CB:** Harmful traces (security=False) vs Benign traces.

## 2. Fujitsu Agentic RAG Red Teaming
- **Path:** `data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl`
- **Type:** Harmful Attacks (Prompt Injection)
- **Count:** ~13,000 records
- **Fields:** `malicious_injection`, `benign_query`, `category`, `subtype`, `expected_tool`, `simulated_tool`, `success`
- **Usage:**
  - **CB:** Primary source of harmful attack prompts.

## 3. AgentHarm
- **Paths:** 
  - `data/agent_harm/harmful_behaviors_test_public.json`
  - `data/agent_harm/harmful_behaviors_validation.json`
- **Type:** Harmful Behaviors (Prompts)
- **Count:** ~52 base behaviors (augmented to ~200)
- **Fields:** `prompt`/`behavior`, `FunctionalCategory`
- **Usage:**
  - **RLVR:** Safety training (Reward = Refusal).
  - **CB:** Harmful prompts.

## 4. WebArena
- **Path:** `data/webarena/config_files/test.raw.json`
- **Type:** Capability Tasks (Web Navigation)
- **Count:** ~812 tasks
- **Fields:** `intent`, `start_url`, `eval.reference_answers`
- **Usage:**
  - **RLVR:** Capability training (String match reward).
  - **CB:** Benign capability tasks.

## 5. TAU2
- **Path:** `data/tau2_repo/data/tau2/domains/{domain}/tasks.json`
- **Domains:** airline, retail, telecom, mock
- **Type:** Capability Tasks (Tool Use)
- **Count:** ~300 tasks total
- **Fields:** `instruction`, `gold_actions`
- **Usage:**
  - **RLVR:** Capability training (Action F1 reward).
  - **CB:** Benign capability tasks.

## 6. AttackQA
- **Path:** `data/attackqa/attackqa.parquet`
- **Type:** Domain Competency (Security QA)
- **Count:** ~17,700 pairs
- **Fields:** `question`, `answer`, `context_document`, `mitre_technique_id`
- **Usage:**
  - **RLVR:** Competency training (QA overlap reward).
  - **CB:** Benign competency tasks.

## 7. WebLINX
- **Path:** `data/processed/weblinx_sample.json` (sample) or HuggingFace `McGill-NLP/WebLINX`
- **Type:** Capability Tasks (Web Navigation, Multi-turn)
- **Count:** ~58,000 turns (sample may be smaller)
- **Fields:** `utterances`, `candidates`, `clean_html`
- **Usage:**
  - **RLVR:** Capability training (placeholder reward; custom eval pending).
  - **CB:** Benign capability tasks.

---

## Notes

- **Multiturn Support:** AgentDojo and WebLINX provide full conversation histories. The RLVR script preserves these in the `conversation_history` field.
- **Limiting Reagent:** Both scripts print diagnostics showing which data source is limiting batch creation.
- **Missing Data:** If a file is not found, the script prints a warning and continues with available data.
