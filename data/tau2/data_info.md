# TAU2-Bench: Dataset & Environment Overview

This document summarizes what TAU2-bench provides, how much data and evaluation artifacts are available, the formats you can work with, how to run it locally, and how it fits into reinforcement learning workflows for conversational agents.

## Overview

TAU2-bench is a dual-control simulation environment to evaluate customer service agents in realistic, tool-using, policy-constrained conversations. It includes:
- Gymnasium-compatible RL interface for training and evaluation.
- Standardized task splits: `base` (use for benchmark comparability) and train/test splits for RL.
- Domains with policies, tools, tasks, and optional user tools.
- CLI commands for running, viewing, and submitting results to the live leaderboard.

See the upstream README at [data/tau2_repo/README.md](data/tau2_repo/README.md) for details.

## Size & Scope

- Domains: `mock`, `airline`, `retail`, `telecom`.
- Each domain defines a policy for the agent, a toolset the agent can call, a task set, and sometimes user tools (e.g., telecom).
- RL-oriented splits exist per domain; the `base` split matches original evaluation for comparability.

Domain directory listing: [data/tau2_repo/data/tau2/domains](data/tau2_repo/data/tau2/domains)

## Data & Formats

TAU2 primarily produces simulation trajectories and evaluation artifacts:
- Baseline results (JSON trajectories and summaries): [data/tau2_repo/data/tau2/results/final](data/tau2_repo/data/tau2/results/final)
- Local runs save simulations to `data/tau2/simulations/` (created when you run `tau2 run`).

A typical final results file (e.g., telecom default, 4 trials) includes top-level keys like:
- `timestamp`: ISO timestamp of run
- `info`: run configuration and environment context:
  - `git_commit`, `num_trials`, `max_steps`, `max_errors`
  - `user_info`: `implementation`, `llm`, `llm_args`, global simulation guidelines
  - `agent_info`: `implementation`, `llm`, `llm_args`
  - `environment_info`: `domain_name`, `policy` (full text), optional `tool_defs`
  - `seed`
- `tasks`: list of task instances with:
  - `id`, `description`, `user_scenario` (persona, instructions)
  - `ticket` text
  - `initial_state`: `initialization_actions` for user/assistant envs
  - `evaluation_criteria`: required `actions` and `env_assertions`; reward basis (e.g., `ENV_ASSERTION`)

Example file path: [data/tau2_repo/data/tau2/results/final/gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json](data/tau2_repo/data/tau2/results/final/gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json)

## Environment Interface (Gym)

TAU2 exposes Gymnasium environments for both roles:
- `AgentGymEnv`: play as the agent against a user simulator
- `UserGymEnv`: play as the user against an automated agent

Key parameters:
- `solo_mode` (agent works independently on tickets when true)
- `user_llm` / `user_llm_args`
- `agent_llm` / `agent_llm_args` (when using `UserGymEnv`)

See examples and API in [data/tau2_repo/src/tau2/gym/README.md](data/tau2_repo/src/tau2/gym/README.md).

## CLI Usage

Core commands:
- `tau2 run`: execute benchmark runs over a domain, configure trials, task filters, concurrency.
- `tau2 play`: interactive mode; play as Agent or User.
- `tau2 view`: browse simulation files and metrics.
- `tau2 domain <domain>`: serve domain policy + API docs at `http://127.0.0.1:8004/redoc`.
- `tau2 check-data`: verify data directory setup.
- `tau2 submit prepare|validate|verify-trajs`: build and validate leaderboard submission packages.

Reference: [data/tau2_repo/README.md](data/tau2_repo/README.md)

## Task Splits

- `base`: full task set consistent with original τ²-bench; use this for comparable evaluations and leaderboard.
- Train/Test: standardized splits per domain to support RL training and generalization testing.

If you omit `--task-split`, it defaults to `base` for backward-compatible evaluation.

## User Simulator

TAU2 includes user simulation with strict grounding requirements and special termination tokens:
- Tokens: `###STOP###` (goal satisfied), `###TRANSFER###` (transfer), `###OUT-OF-SCOPE###` (insufficient info).
- Tool-call constraints: one action or message per turn; tool calls only when requested or necessary; messages must be grounded in tool results.

Guidelines:
- General: [data/tau2_repo/data/tau2/user_simulator/simulation_guidelines.md](data/tau2_repo/data/tau2/user_simulator/simulation_guidelines.md)
- Tools & turn-taking: [data/tau2_repo/data/tau2/user_simulator/simulation_guidelines_tools.md](data/tau2_repo/data/tau2/user_simulator/simulation_guidelines_tools.md)

## Results & Metrics

- Leaderboard displays Pass^k success rates (k = 1,2,3,4) across domains.
- Submission tooling computes metrics from your trajectory files and verifies consistency.
- Prepare submissions with `tau2 submit prepare`, then `tau2 submit validate` before PR submission.

See submission instructions in [data/tau2_repo/README.md](data/tau2_repo/README.md) and the web submission README referenced therein.

## Installation & Data Directory

- Install editable for CLI: `pip install -e .` in the TAU2 repo.
- If not editable install, set `TAU2_DATA_DIR` to the repo’s `data/` path so CLI can find domain/task files.
- Verify setup with `tau2 check-data`.

Repo README: [data/tau2_repo/README.md](data/tau2_repo/README.md)

## Strengths & Limitations

- Strengths: dual-control realism, explicit domain policies, tool-based actions, gym interface for RL, standardized splits + leaderboard.
- Limitations: domain-specific simulated environments (no general web browsing), API keys and model configuration required, evaluation hinges on domain assertions and tool semantics.

## Integration Notes (This Project)

- TAU2 is available under [data/tau2_repo](data/tau2_repo) with baseline results under [data/tau2_repo/data/tau2/results/final](data/tau2_repo/data/tau2/results/final).
- Local runs will produce new trajectories under `data/tau2/simulations/`.
- TAU2 is not currently ingested into the unified SQLite database in this workspace. If needed, we can add an ingestion step to store trajectories and summary metrics.

## Related Files

- Upstream README: [data/tau2_repo/README.md](data/tau2_repo/README.md)
- Gym docs: [data/tau2_repo/src/tau2/gym/README.md](data/tau2_repo/src/tau2/gym/README.md)
- Domains directory: [data/tau2_repo/data/tau2/domains](data/tau2_repo/data/tau2/domains)
- Baseline results: [data/tau2_repo/data/tau2/results/final](data/tau2_repo/data/tau2/results/final)
- User simulator guidelines: [data/tau2_repo/data/tau2/user_simulator/simulation_guidelines.md](data/tau2_repo/data/tau2/user_simulator/simulation_guidelines.md), [data/tau2_repo/data/tau2/user_simulator/simulation_guidelines_tools.md](data/tau2_repo/data/tau2/user_simulator/simulation_guidelines_tools.md)
