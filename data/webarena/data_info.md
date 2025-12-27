# WebArena Dataset Info

## Overview

**WebArena** is a standalone, self-hostable web environment for building and evaluating autonomous agents. It provides realistic web tasks across multiple domains (e-commerce, forums, project management, maps, etc.) with automated evaluation metrics.

- **Paper**: [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854)
- **Repository**: https://github.com/web-arena-x/webarena
- **Website**: https://webarena.dev
- **Leaderboard**: Available on project website

## Dataset Statistics

### Size & Scope
- **Total Tasks**: 812
- **Unique Intent Templates**: 241
- **Multi-Turn Support**: ❌ No (single-turn task completion)
- **Trajectory Data**: ✅ Available separately (see Trajectory Resources section)
  - **Human Trajectories**: 179 tasks (78.24% success rate)
  - **LLM Trajectories v2**: ~810 tasks across 6 model configurations
  - **Format**: Playwright traces + HTML renders + logs

### Site Distribution
```
gitlab:           204 tasks (25.1%)
shopping:         192 tasks (23.6%)
shopping_admin:   184 tasks (22.7%)
reddit:           129 tasks (15.9%)
map:              128 tasks (15.8%)
wikipedia:         23 tasks (2.8%)
```

### Authentication
- **All tasks require login**: 812/812 (100%)
- **Auth storage**: `.auth/{site}_state.json` files contain cookies
- **Auto-login script**: `browser_env/auto_login.py`

### Evaluation Types
WebArena uses 3 distinct evaluation methodologies:

1. **`string_match`**: 335 tasks (41.3%)
   - Exact string matching
   - Multiple answer matching (`must_include`)
   - Used for factual queries (e.g., "What is the price of X?")

2. **`url_match`**: 205 tasks (25.2%)
   - Validates navigation to correct page
   - Common for browsing/exploration tasks

3. **`program_html`**: 411 tasks (50.6%)
   - Executes Python code to validate DOM state
   - Most sophisticated evaluation (checks element properties, form states, etc.)
   - Can verify complex multi-step tasks

*Note: Tasks may use multiple evaluation types (total > 812)*

## Data Format

### File Structure
```
webarena/
├── config_files/
│   └── test.raw.json          # 812 task configurations
└── [trajectories not included in this repo, available separately]
```

### Task Schema

Each task in `test.raw.json` follows this structure:

```json
{
  "sites": ["shopping_admin"],
  "task_id": 0,
  "require_login": true,
  "storage_state": "./.auth/shopping_admin_state.json",
  "start_url": "__SHOPPING_ADMIN__",
  "geolocation": null,
  
  "intent_template": "What is the top-{{n}} best-selling product in {{year}}",
  "instantiation_dict": {
    "n": 1,
    "year": 2022
  },
  "intent": "What is the top-1 best-selling product in 2022",
  "intent_template_id": 279,
  
  "require_reset": false,
  
  "eval": {
    "eval_types": ["string_match"],
    "reference_answers": {
      "exact_match": "Quest Lumaflex™ Band"
    },
    "reference_url": "",
    "program_html": [],
    "string_note": "",
    "reference_answer_raw_annotation": "Quest Lumaflex™ Band"
  }
}
```

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | int | Unique identifier (0-811) |
| `sites` | list[str] | Target website(s) - usually single site |
| `intent` | str | The natural language task description |
| `intent_template` | str | Template with `{{variables}}` for task generation |
| `instantiation_dict` | dict | Values for template variables |
| `start_url` | str | Initial page (uses placeholders like `__SHOPPING__`) |
| `require_login` | bool | Always true in this dataset |
| `storage_state` | str/null | Path to auth cookies (null for sites without login) |
| `require_reset` | bool | Whether env must be reset after task (all false in v0.2.0) |
| `eval` | object | Evaluation configuration (see below) |

### Evaluation Schema

```json
{
  "eval_types": ["string_match", "url_match", "program_html"],
  "reference_answers": {
    "exact_match": "single expected answer",
    "must_include": ["answer1", "answer2"],  // For multi-answer tasks
    "fuzzy_match": ["variant1", "variant2"]  // Allows typos/variations
  },
  "reference_url": "expected URL pattern",
  "program_html": [
    "assert query_selector('#element').text == 'expected'"
  ],
  "string_note": "human-readable explanation of expected answer",
  "reference_answer_raw_annotation": "original annotator's answer"
}
```

## Task Categories

### Top Intent Templates (by frequency)

1. **E-commerce queries** (7x): `"Get the {{attribute}} of the {{status}} order"`
2. **Map navigation** (7x): `"Find the page of {{description}} on the map"`
3. **Review analysis** (6x): `"List out reviewers who mention about {{description}}"`
4. **Repository operations** (6x): `"Fork {{repo}}"`, `"Create a repo named {{name}}"`
5. **Social actions** (6x): `"Like/DisLike all submissions by {{user}} in {{subreddit}}"`
6. **Admin operations** (6x): `"{{action}} the price of {{product}} by {{amount}}"`

*(241 unique templates total - diverse task distribution)*

## Action Space

WebArena agents interact through browser automation with these action types:

### Page Operations
- `click [id]` - Click element by accessibility tree ID
- `type [id] [content] [press_enter_after=0|1]` - Type into form field
- `hover [id]` - Hover over element
- `press [key_comb]` - Keyboard shortcuts (e.g., `Ctrl+v`)
- `scroll [direction=down|up]` - Page scrolling

### Tab Management
- `new_tab` - Open new browser tab
- `tab_focus [tab_index]` - Switch active tab
- `close_tab` - Close current tab

### Navigation
- `goto [url]` - Navigate to URL
- `go_back` / `go_forward` - Browser history navigation

### Completion
- `stop [answer]` - Signal task completion with answer

## Observation Space

Agents receive observations in these formats:

1. **Accessibility Tree** (default)
   - Structured representation of DOM
   - Each element has unique `[id]` for action targeting
   - Simplified, agent-friendly format

2. **Raw HTML**
   - Full page source
   - More information but harder to parse

3. **Screenshot** (optional)
   - Visual observations for multimodal agents

**Key Feature**: The accessibility tree includes `current_viewport_only` mode to limit observation size (critical for LLM context windows).

## Prompt Templates

WebArena includes agent prompt templates in `agent/prompts/`:

### Naming Convention
```
{description}.{action_space}.{observation_space}.json
```

Example: `p_cot_id_actree_2s.json`
- `p` = Prompt-based
- `cot` = Chain of Thought reasoning
- `id` = ID-based actions (use element IDs)
- `actree` = Accessibility tree observations
- `2s` = 2-shot examples

### Prompt Structure
Each prompt file contains:
```python
{
  "intro": "System instructions and task description",
  "examples": [
    (observation, response),  # Few-shot examples
    ...
  ],
  "template": "Format string with placeholders for {observation}, {url}, {objective}, {previous_action}",
  "meta_data": {
    "observation": "accessibility_tree",
    "action_type": "id_accessibility_tree",
    "keywords": ["url", "objective", "observation", "previous_action"],
    "prompt_constructor": "CoTPromptConstructor",
    "answer_phrase": "In summary, the next action I will perform is",
    "action_splitter": "```"
  }
}
```

**Available Prompts** (examples):
- `p_cot_id_actree_2s.json` - CoT with 2 examples
- `p_cot_id_actree_2s_no_na.json` - CoT without "Not Applicable" option
- Various reasoning styles (CoT, ReAct, etc.)

## What WebArena Has

✅ **Strengths:**
1. **Diverse, realistic tasks** across 6 distinct web environments
2. **Automated evaluation** with multiple validation methods
3. **Extensive trajectory data** - 179 human + ~4,860 LLM executions
4. **Rich execution traces** - Playwright recordings with DOM, network, screenshots
5. **Self-hostable** - full control over test environment
6. **Well-structured action/observation spaces** for agent development
7. **Intent templates** - enables task generation and augmentation
8. **Professional-grade benchmark** - widely cited, active leaderboard
9. **Multiple model baselines** - GPT-4, GPT-3.5, PaLM with different prompting strategies

## What WebArena Lacks

❌ **Limitations:**
   - Each trajectory is one continuous execution toward a goal

2. **Trajectory data not in base distribution**
   - Must download separately from Google Drive (~12.5GB)
   - Only 2 samples per type included in this repository
   - Hundreds of files to manage for full dataset

3. **No adversarial/safety evaluation** 
   - Pure capability benchmark
   - No prompt injection tests
   - No malicious task detection
   - No jailbreak scenarios

4. **Large storage requirements for complete data**
   - Full trajectory data: ~12.5GB
   - Playwright traces are verbose (include full DOM snapshots)
   - This repo contains <1% of available trajectory data

5. **Single-agent only** - no multi-agent scenarios

6. **Static evaluation** - no dynamic user simulation

7. **Requires self-hosting for execution** 
   - 5 Docker containers (shopping, reddit, gitlab, map, wikipedia)
   - Complex initialization scripts
   - Not suitable for quick prototyping
   - Only task configs and trajectories available without self-hostficant infrastructure setup
   - 5 Docker containers (shopping, reddit, gitlab, map, wikipedia)
   - Complex initialization scripts
   - Not suitable for quick prototyping

## Trajectory Resources


**With Trajectory Data** ✅:
- **State-action pairs available** - Observations + actions at each step
- **Sequential decision-making captured** - Full execution traces
- **Success/failure outcomes** - Can derive rewards
- **Human demonstrations** - 179 high-quality trajectories (78.24% success)
- **LLM baselines** - ~4,860 trajectories for comparison

**Trajectory Format Enables**:
1. **Imitation Learning** - Learn from human demonstrations
2. **Behavioral Cloning** - Train on LLM trajectories
3. **Inverse RL** - Infer reward from human actions
4. **Failure Analysis** - Study where agents get stuck
5. **Reward Shaping** - Derive intermediate rewards from successful trajectories

**Without Trajectory Data** ⚠️:
- Only task specifications available
- Must generate your own trajectories
- Can use as goal specification datasetction recordings
- **Network traffic** - HTTP requests/responses
- **Screenshots** - Visual state at each step
- **DOM snapshots** - Complete HTML at each step

#### Task Sampling Strategy
- One task per intent template (or similar semantic groups)
- Covers diverse task types across all 6 sites
- Template IDs available in task config file

#### Viewing Human Trajectories
```bash
# Install Playwright (if not already installed)
pip install playwright
playwright install

# View a specific trajectory
playwright show-trace <task_id>.zip
```

This opens an interactive viewer showing:
- Timeline of actions
- Screenshots at each step
- Network activity
- Console logs
- DOM snapshots

### LLM Trajectories v2 (810 tasks)

**Release Date**: November 3, 2023  
**Coverage**: Nearly complete test set (810/812 tasks)  
**Source**: [Google Drive](https://drive.google.com/drive/folders/1Hj4hELlsXXGdTX1Y3F5Y8kqYWO3S0Gq_)

#### Available Model Configurations

1. **text-bison-001** (Google PaLM)
   - Prompt: CoT + UA Hint
   - ~810 tasks

2. **GPT-3.5-turbo-16k** (4 variants)
   - Direct + UA Hint
   - Direct (no hint)
   - CoT + UA Hint
   - CoT (no hint)
   - ~810 tasks each

3. **GPT-4** 
   - Prompt: CoT
   - ~810 tasks

**Total**: ~4,860 LLM execution traces across models

#### Trajectory Format

Each model folder contains:

##### 1. `render_*.html` (One per task)
HTML visualization with:
- **Accessibility tree observations** (`<div class="state_obv">`)
- **URLs visited** (`<h3 class="url">`)
- **Raw LLM predictions** (`<div class="raw_parsed_prediction">`)
  - Includes CoT reasoning (e.g., "Let's think step-by-step...")
- **Parsed actions** (`<div class="action_object">`)
- **Screenshots** embedded inline

##### 2. `merge_log.txt`
Task results in format:
```
2023-09-24 16:32:42,509 - INFO - [Intent]: What is the top-1 best-selling product in 2022
2023-09-24 16:33:07,065 - INFO - [Result] (PASS) /path/to/0.json
```

Contains:
- Task intent (objective)
- Pass/Fail outcome
- Task ID and config path

##### 3. `trace/*.zip`
Playwright recordings (same format as human trajectories):
- Full browser interaction
- Network traffic
- DOM snapshots
- Screenshots

#### Parsing LLM Trajectories

**Extract from HTML** (see notebook code):
```python
from bs4 import BeautifulSoup

with open("render_<id>.html", 'r') as f:
    content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    
    # Get observations (what agent sees)
    observations = [div.get_text() for div in soup.find_all("div", {"class": "state_obv"})]
    
    # Get URLs visited
    urls = [h3.get_text() for h3 in soup.find_all("h3", {"class": "url"})]
    
    # Get raw LLM outputs (includes reasoning)
    raw_predictions = [div.get_text() for div in soup.find_all("div", {"class": "raw_parsed_prediction"})]
    
    # Get parsed actions
    actions = [div.get_text() for div in soup.find_all("div", {"class": "action_object"})]
```

**Parse merge_log.txt**:
```python
import re

def parse_merged_log(log_path):
    results = {}
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    current_intent = None
    for line in lines:
        if "[Intent]:" in line:
            current_intent = re.search(r"\[Intent\]:\s*(.+)", line).group(1).strip()
        
        if "[Result]" in line:
            result = re.search(r"\((\w+)\)", line).group(1)  # PASS or FAIL
            task_id = re.search(r"/(\d+)\.json", line).group(1)
            results[task_id] = {
                "intent": current_intent,
                "passed": result.lower() == "pass"
            }
    return results
```

### LLM Trajectories v1 (Deprecated)

**Release Date**: August 7, 2023  
**Models**: GPT-4, GPT-3.5-turbo (Direct and CoT)  
**Status**: Superseded by v2

Use v2 for all new work (better prompts, more models, same format).

## Data Availability

### In This Repository
- ✅ `config_files/test.raw.json` - All 812 task configurations (30KB)
- ⚠️ **2 sample human trajectories** - Representative samples only
- ⚠️ **2 sample LLM v2 trajectories** - Representative samples only

**Note**: Only samples included due to size constraints. Full trajectory data is **hundreds of files**.

### Available Online (Google Drive)
- ✅ **Human trajectories** - 179 `.zip` files (~1.5GB total)
- ✅ **LLM trajectories v2** - 6 model folders with ~810 tasks each (~8GB total)
- ✅ **LLM trajectories v1** - 3 model folders (deprecated, ~3GB)

**Total Available**: ~12.5GB of trajectory data

### What This Repository Has vs. What Exists

| Data Type | In Repo | Available Online | Notes |
|-----------|---------|------------------|-------|
| Task configs | ✅ 812 tasks | ✅ Same | Complete |
| Human traces | ⚠️ 2 samples | ✅ 179 tasks | 0.01% sampled |
| LLM traces (v2) | ⚠️ 2 samples | ✅ ~4,860 traces | 0.04% sampled |
| Human success rate | - | 78.24% | Benchmark |
| LLM best (GPT-4) | - | ~35% | Benchmark |

### How to Get Full Trajectory Data

1. **Access Google Drive**: https://drive.google.com/drive/folders/1Hj4hELlsXXGdTX1Y3F5Y8kqYWO3S0Gq_ (from WebArena resources page)

2. **Download specific folders**:
   ```bash
   # Human trajectories
   wget <drive_link> -O human_trajectories.zip
   
   # LLM v2 trajectories (choose model)
   wget <drive_link> -O v2_919_gpt4_8k_cot.zip
   ```

3. **Unzip and organize**:
   ```bash
   unzip human_trajectories.zip -d data/webarena/human_trajectories/
   unzip v2_919_gpt4_8k_cot.zip -d data/webarena/llm_trajectories_v2/
   ```

4. **Ingest using notebook code** (see `notebooks/ingest_data.ipynb` cells 4-5)

### Not Available Anywhere
- ❌ Live environment access - must self-host
- ❌ Training data - test-only benchmark
- ❌ Multi-turn extensions
Extracting Trajectories for RL

**From HTML Renders** (see notebook code in `ingest_data.ipynb`):
```python
# Parse a single LLM trajectory
parsed = parse_webarena_html("render_0.html")

trajectory = {
    "id": "webarena_llm_gpt4_task_0",
    "goal": "What is the top-1 best-selling product in 2022",
    "steps": [
        {
            "state": parsed["observations"][i],
            "url": parsed["urls"][i],
            "action": parsed["actions"][i],
            "reasoning": parsed["predictions"][i],  # Includes CoT
            "reward": 1.0 if i == len(parsed["actions"])-1 else 0.0
        }
        for i in range(parsed["num_steps"])
    ],
    "success": parsed["passed"],
    "num_steps": parsed["num_steps"]
}
```

**From Playwright Traces**:
```bash
# Extract detailed interaction data
playwright show-trace task_0.zip --save-json trace.json

# Parse the JSON to get:
# - DOM snapshots at each step
# - Network requests/responses
# - Console logs
# - Screenshots
```

**Recommended Format** (see `RL_TRAINING_GUIDE.md`):
```json
{
  "id": "webarena_human_task_0",
  "source": "webarena_human",
  "goal": "What is the top-1 best-selling product in 2022",
  "trajectory": [
    {
      "step": 0,
      "state": {"accessibility_tree": "...", "url": "http://shopping-admin/"},
      "action": {"type": "click", "target_id": 123},
      "reward": 0.0
    },
    {
      "step": 1,
      "state": {"accessibility_tree": "...", "url": "http://shopping-admin/reports"},
      "action": {"type": "type", "target_id": 456, "content": "product sales"},
      "reward": 0.0
    },
    {
      "step": 2,
      "state": {"accessibility_tree": "...", "url": "http://shopping-admin/reports/best-sellers"},
      "action": {"type": "stop", "answer": "Quest Lumaflex Band"},
      "reward": 1.0
    }
  ],
  "success": true,
  "num_steps": 3,
  "source_trace": "4.zip"
- Use separately available trajectory data
- Treat as "goal specification" dataset
- Combine with execution traces from other sources

### As Evaluation Benchmark
**Ideal Use Case**: Test agent capability on realistic web tasks

**Metrics**:
- Task success rate (pass/fail)
- Per-site performance
- Evaluation type breakdown

### Recommended Restructuring for RL
See `RL_TRAINING_GUIDE.md` for conversion to multi-turn format:
```json
{
  "id": "webarena_task_0",
  "goal": "What is the top-1 best-selling product in 2022",
  "trajectory": [
    {"state": "...", "action": "click [123]", "reward": 0},
    {"state": "...", "action": "type [456] [product sales]", "reward": 0},
    {"state": "...", "action": "stop [Quest Lumaflex Band]", "reward": 1}
  ],
  "success": true
}
```

## Online Data Sources

### Official Resources
- **GitHub**: https://github.com/web-arena-x/webarena
- **HuggingFace**: Dataset not on HF (self-hosted only)
- **Leaderboard**: https://webarena.dev (submission required for eval)

### Extended Versions
- **VisualWebArena**: Multimodal extension with image-based tasks
- **AgentLab/BrowserGym**: Enhanced infrastructure with parallel execution
- **TheAgentCompany**: Newer benchmark with more complex scenarios

### Related Benchmarks
- **MiniWoB**: Simpler web tasks (predecessor)
- **WebShop**: E-commerce focused
- **Mind2Web**: Real-world website tasks (static crawls)

## Integration Notes

### For Your Project
Based on `RL_TRAINING_GUIDE.md`, WebArena serves as:
1. **Capability baseline** - tests if agent can perform non-adversarial tasks
2. **Task diversity** - 241 intent templates provide varied scenarios
3. **Evaluation infrastructure** - reusable validation logic

### Combining with Other Datasets
- **WebArena** (capability) + **AgentHarm** (safety) = Complete agent evaluation
- **WebArena tasks** + **AttackQA traces** = Adversarial robustness testing
- **WebArena intents** + **Fujitsu attacks** = Jailbreak resistance in web agents

## Citation

```bibtex
@article{zhou2023webarena,
  title={WebArena: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and others},
  journal={arXiv preprint arXiv:2307.13854},
  year={2023}
}
```

## Version History

- **v0.2.0** (Oct 2023): Current stable version
  - Fixed annotation bugs
  - All `require_reset=false` (stable evaluation)
  - 812 finalized tasks
  
- **v0.1.0** (Jul 2023): Initial release

## Additional Notes

### Environment Setup Requirements
If using full WebArena (not just config files):
1. Docker containers for each website
2. Environment variable configuration for URLs
3. Browser automation (Playwright)
4. Python 3.10+
5. Auto-login cookies generation

### Recommendation from Authors
As of Dec 2024, authors recommend using **AgentLab framework** for experiments:
- Better parallelization (BrowserGym)
- Unified benchmark interface
- Improved error handling
- Integration with multiple benchmarks

---

**Last Updated**: December 22, 2025
**Dataset Version**: v0.2.0 (Oct 21, 2023)
