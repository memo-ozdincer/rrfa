# CB_TODO (Circuit Breakers Follow-Ups)

This file tracks the next steps to bring this repo closer to the reference implementation in `related_repos/circuit-breakers` while keeping changes incremental and debuggable.

## Data Needs (What to Collect Next)

If you only collect one thing, collect **harmful assistant-side completions / traces** (not just harmful user prompts). That’s the biggest limiter for true “rerouting” on refusal-trained bases.

### A) More Harmful Data (Highest Priority)

**1) Harmful assistant completions (prompt → harmful completion pairs)**
- Format: `{user_prompt, harmful_completion}` (plus optional `text` pre-rendered with a chat template).
- Why: prompts-only often fail to enter the harmful generation trajectory; completions reliably activate it.
- Where you can get it:
  - Public red-teaming / jailbreak corpora that include responses.
  - Generated completions from a weaker or uncensored model (or from the base model under an “adversarial prefill”), then filtered.

**2) Agentic harmful traces (tool-use attacks)**
- Format: sequences that include tool calls / arguments that are actually harmful (or clearly policy-violating).
- Why: for agents, “harm” often happens via tool calls, not plain text.
- Examples to target:
  - Prompt-injection → tool misrouting (similar to Fujitsu), but include the *assistant actions* (selected tool, arguments).
  - Credential exfiltration attempts, unsafe browsing, malicious file writes, etc.

**3) More agent-specific harmful coverage**
- Your current “limiting reagent” is harmful volume; also, Fujitsu + AgentHarm skew toward user-side text.
- Goal: broaden beyond prompt injection to other agent failure modes (data exfil, unauthorized actions, sandbox escape attempts).

### B) Refusal-Retain Data (Important for Stability)

**4) Refusal examples to retain**
- Format: prompts that should be refused + strong refusal completions.
- Why: the reference repo explicitly mixes refusal-retain to prevent refusal drift.
- Suggested sources:
  - XSTest / exaggerated safety style sets.
  - Any internal policy refusal set you already trust.

### C) Benign Retain Data (Lower Priority Than Harmful)

**5) More agentic benign traces**
- You already have a lot of benign text (e.g., AttackQA). What helps most is *tool-using* benign behavior.
- Format: tool calls with correct arguments + successful trajectories.
- Suggested sources:
  - AgentDojo “secure/success” traces with full message/action logs.
  - WebArena / TAU2 traces with action annotations (if available), not just intents.

### D) Evaluation / Holdout Data (You’ll Want This Early)

**6) Unseen attack holdout set (agentic)**
- Format: prompts + expected safe behavior; include tool-call expectations where possible.
- Why: Circuit breakers can look good on training-like prompts but fail on “unseen” attacks.

**7) Over-refusal holdout set**
- Format: borderline benign requests that models sometimes wrongly refuse.
- Why: helps keep false refusal rate honest.

## 1) Completion-Based Circuit-Breaker Set (Highest Impact)

**Problem:** The current harmful CB dataset is mostly *user-side prompts* (`attack_prompt`, prompt injection text). For refusal-trained bases, prompts alone may not activate the harmful generation trajectory you want to reroute.

**Target outcome:** Train on **prompt + harmful assistant continuation** (or tool-call trace) and apply $L_{rr}$ primarily on the continuation tokens.

**Proposed minimal schema (keep it simple):**

- Harmful CB example:
  - `user_prompt: str`
  - `harmful_completion: str`
  - `text: str` (combined input, using chat template)

- Benign retain example:
  - `user_prompt: str`
  - `benign_completion: str`
  - `text: str`

**Where to implement:**
- `scripts/ingest_cb_data.py`
  - Add an *optional* path that builds `text` using a tokenizer chat template.
  - If completions exist in a source, include them; otherwise leave the current prompt-only path unchanged.
- `scripts/circuit_breakers/trainer.py` → `CircuitBreakerDataset._extract_prompt()`
  - Prefer `sample['text']` when present.

**Acceptance check:**
- Verify tokenized harmful examples contain assistant-side tokens.
- Confirm reroute loss changes meaningfully when training starts (not flatlining).


## 2) Span / Masking: Apply Loss Only Where It Matters

**Problem:** Current losses apply across all non-padding tokens. The reference repo separates request vs response and masks accordingly.

**Target outcome:**
- Apply $L_{rr}$ on a **CB span** (ideally assistant response / tool-call arguments).
- Apply $L_{ret}$ on a **retain span** (often the assistant response tokens).

**Simple version (minimal):**
- Add optional fields to each sample:
  - `cb_token_mask: List[int]` or a run-length span `{start, end}`
  - `retain_token_mask: List[int]` or span
- Update `CircuitBreakerDataset` to return:
  - `harmful_loss_mask`, `benign_loss_mask` as `torch.Tensor` shaped like attention masks
- Update `reroute_loss()` / `retain_loss()` to accept a `loss_mask` (multiply in addition to `attention_mask`).

**Where to implement:**
- `scripts/circuit_breakers/trainer.py`
  - Extend `CircuitBreakerDataset.__getitem__` output dict.
  - Update loss functions to use `combined_mask = attention_mask * loss_mask`.


## 3) Paper-Style Dual Coefficients (cs/cr) Scheduling

**Problem:** The current objective is `loss = alpha(t) * L_rr + L_ret` (retain weight fixed at 1). The reference repo shifts weights over time.

**Target outcome:** Implement an optional mode:

$$L = (\alpha\cdot c_s(t))\,L_{rr} + (\alpha\cdot c_r(t))\,L_{ret}$$

with a simple schedule like:
- `progress = clip(step / (decay_multiplier*total_steps), 0, 1)`
- `c_s = 1 - progress`
- `c_r = progress`

**Where to implement:**
- `scripts/circuit_breakers/trainer.py`
  - Add a config flag like `loss_weighting = "single_alpha" | "dual"`.
  - If `dual`, compute two coefficients and log them.


## 4) Add a Small Refusal-Retain Set (If Using Refusal-Trained Bases)

**Problem:** Refusal behavior can drift during training unless explicitly retained.

**Target outcome:** Add a small, curated refusal set into the retain stream (doesn’t have to be huge).

**Where to implement:**
- `scripts/ingest_cb_data.py`
  - Add optional ingestion for a refusal dataset if present locally.
  - Tag those examples as `source="refusal_retain"`.


## 5) Evaluation: Move Beyond Regex for Agent Settings

**Problem:** Regex refusal scoring is a smoke test; it misses partial compliance and tool-call harm.

**Target outcome:**
- Add an "action-based" metric for agent contexts:
  - did the model emit a tool call?
  - are arguments harmful?
- Add a classifier/judge path for text-only harmfulness.

**Where to implement:**
- `scripts/circuit_breakers/eval.py`
  - Add optional JSON output with richer per-example annotations.
  - Add a hook to plug in a classifier model or LLM judge later.


## 6) Reference Repo Parity Notes

The reference implementation in `related_repos/circuit-breakers/src/lorra_circuit_breaker.py`:
- Uses `output_hidden_states=True` (no forward hooks)
- Splits circuit-breaker samples into request/response and constructs a combined attention mask
- Uses time-varying coefficients for retain vs circuit-breaker losses

Keeping parity with these patterns is recommended, but the steps above are ordered to keep each change independently testable.
