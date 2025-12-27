# WebLINX Dataset Information

## Overview
**WebLINX** is a large-scale benchmark for real-world website navigation with multi-turn dialogue. It captures natural human-computer interactions where an instructor verbally guides a navigator through complex web tasks, creating a dataset of conversational web agents.

**Paper**: [WebLINX: Real-World Website Navigation with Multi-Turn Dialogue](https://arxiv.org/abs/2402.05930) (Xing Han Lù, Zdeněk Kasner, Siva Reddy)  
**HuggingFace**: [McGill-NLP/WebLINX](https://huggingface.co/datasets/McGill-NLP/WebLINX)  
**Website**: https://mcgill-nlp.github.io/weblinx  
**Code**: https://github.com/McGill-NLP/weblinx  

---

## Size & Scope

### Dataset Statistics
```
Total Demonstrations:  155 recorded sessions
Total Turns:           ~58,000 interaction turns
Unique Websites:       150+ real-world websites
Average Demo Length:   21.3 turns per demonstration
Multi-turn Dialogue:   ✅ Natural instructor-navigator conversations
Action Types:          click, textInput, submit, load, scroll, say, change
```

### Split Breakdown
| Split        | Turns   | Demos | Purpose                          |
|--------------|---------|-------|----------------------------------|
| `train`      | 24,418  | ~115  | Primary training data            |
| `validation` | 2,126   | 10    | Model selection & hyperparameters|
| `test_iid`   | 2,308   | 10    | In-distribution test (IID)       |
| `test_cat`   | 5,971   | 28    | Category generalization test     |
| `test_geo`   | 7,672   | 36    | Geographic generalization test   |
| `test_vis`   | 9,023   | 42    | Visual variation test            |
| `test_web`   | 4,856   | 23    | Website generalization test      |

**Total**: ~58,000 turns across 155 demonstrations

### Evaluation Splits Explained
- **IID (test_iid)**: Same distribution as training (standard benchmark)
- **Category (test_cat)**: Websites from unseen categories (e.g., if training has e-commerce, test has education sites)
- **Geographic (test_geo)**: Websites from different countries/languages
- **Visual (test_vis)**: Websites with different visual layouts and styles
- **Website (test_web)**: Completely unseen websites (zero-shot generalization)

---

## Data Format

### Chat Format (Primary)
Each turn in WebLINX contains:

```python
{
  "demo": "apfyesq",                          # Unique demonstration ID
  "turn": 6,                                   # Turn number in sequence
  "action": "click(uid=\"67e2a5fb-8b1d-41a0\")",  # Ground truth action
  "action_history": "say(...)</s><s>[INST] say(...)",  # Previous actions (LLaMA format)
  "utterances": "[00:05] Hello ;",            # Timestamped dialogue history
  "candidates": "(uid = 67e2a5fb-8b1d...) [[tag]] input...",  # Candidate HTML elements
  "clean_html": "(html(body(div class=\"container\"...",  # Simplified HTML tree
  "viewport": "746h x 1536w"                  # Browser viewport dimensions
}
```

### Field Descriptions

#### Core Identification
- **`demo`**: Unique identifier for the demonstration session (e.g., `"apfyesq"`)
- **`turn`**: Turn index within the demonstration (0-indexed)

#### Action Data
- **`action`**: Ground truth action in structured format:
  - `click(uid="...")`: Click on element with unique ID
  - `textInput(text="...", uid="...")`: Type text into element
  - `submit(uid="...")`: Submit form
  - `load(url="...")`: Navigate to URL
  - `scroll(x=int, y=int)`: Scroll page
  - `say(speaker="...", utterance="...")`: Dialogue turn
  - `change(value="...", uid="...")`: Change dropdown/select value
  - `tabcreate()`, `tabswitch()`: Tab management

- **`action_history`**: Sequence of previous actions formatted for LLaMA-style models with special tokens (`</s><s>[INST]`)

#### Dialogue Context
- **`utterances`**: Timestamped conversation history between instructor and navigator
  - Format: `"[MM:SS] Speaker message ; [MM:SS] Speaker message ;"`
  - Speakers: `instructor` (gives commands), `navigator` (executes/responds)

#### HTML & Page State
- **`candidates`**: Pre-filtered candidate elements from the DOM that could be actionable
  - Format: `(uid = UNIQUE_ID) [[tag]] TAG_NAME [[xpath]] XPATH [[text]] TEXT [[bbox]] x=X y=Y width=W height=H [[attributes]] ATTRS`
  - Includes: tag name, XPath, text content, bounding box, attributes
  - Pre-filtered to ~10-50 most relevant elements per turn

- **`clean_html`**: Simplified HTML representation
  - Stripped of scripts, styles, and non-essential attributes
  - Parenthesized tree format: `(html(body(div class="..."(...))))`
  - Truncated to first 5000 characters in database

- **`viewport`**: Browser viewport size (e.g., `"746h x 1536w"`)

---

## Accessing the Data

### From HuggingFace (Preprocessed)
```python
from datasets import load_dataset

# Load validation split
val = load_dataset("McGill-NLP/WebLINX", split="validation")

# Load all test splits
test_iid = load_dataset("McGill-NLP/WebLINX", split="test_iid")
test_cat = load_dataset("McGill-NLP/WebLINX", split="test_cat")
test_geo = load_dataset("McGill-NLP/WebLINX", split="test_geo")
test_vis = load_dataset("McGill-NLP/WebLINX", split="test_vis")
test_web = load_dataset("McGill-NLP/WebLINX", split="test_web")

print(f"Validation turns: {len(val)}")
print(f"Sample: {val[0]}")
```

### Using Input Templates
WebLINX provides model-specific input templates:

```python
from huggingface_hub import snapshot_download

# Download templates (LLaMA, Flan-T5, etc.)
snapshot_download(
    "McGill-NLP/WebLINX",
    repo_type="dataset",
    allow_patterns="templates/*",
    local_dir="./data/weblinx"
)

# Use LLaMA template
with open('data/weblinx/templates/llama.txt') as f:
    template = f.read()

turn = val[0]
input_text = template.format(**turn)
```

Available templates:
- `llama.txt`: For LLaMA-style models (2.7B, 7B, etc.)
- `flan-t5.txt`: For Flan-T5 models
- `sheared-llama.txt`: For Sheared-LLaMA models

### Raw Demonstration Data
```python
from huggingface_hub import snapshot_download

# Download specific demonstrations
demo_names = ['saabwsg', 'ygprzve', 'iqaazif']
patterns = [f"demonstrations/{name}/*" for name in demo_names]

snapshot_download(
    repo_id="McGill-NLP/WebLINX-full",
    repo_type="dataset",
    local_dir="./data/weblinx_raw",
    allow_patterns=patterns
)
```

Each raw demonstration folder contains:
- `replay.json`: Complete interaction trace with full HTML snapshots
- `metadata.json`: Demo-level metadata (start URL, task description, etc.)
- `screenshots/`: Screenshots at each turn (if available)

---

## What WebLINX Has ✅

### Multi-Turn Dialogue
- ✅ **Natural conversations** between instructor and navigator
- ✅ **Temporal context** with timestamped utterances
- ✅ **Turn-level granularity** for fine-grained action prediction
- ✅ **Dialogue history** maintained across turns

### Real-World Websites
- ✅ **150+ real websites** covering diverse domains
- ✅ **Live websites** (not simplified/mock environments)
- ✅ **Complex modern web** (JavaScript, dynamic content, forms)
- ✅ **Multiple languages** and geographic regions

### Rich Action Space
- ✅ **7 action types**: click, textInput, submit, load, scroll, say, change
- ✅ **Element-level grounding**: Actions tied to specific HTML elements via UIDs
- ✅ **Dialogue actions**: Captures navigator responses ("say" actions)
- ✅ **Navigation actions**: URL loads, scrolling, tab management

### Comprehensive HTML Context
- ✅ **Full DOM snapshots** in raw data
- ✅ **Cleaned HTML trees** for efficient processing
- ✅ **Pre-filtered candidates** reduce search space from 1000s to 10s of elements
- ✅ **Element attributes**: XPath, bounding box, text, tag, attributes

### Generalization Test Splits
- ✅ **5 test splits** for different generalization scenarios
- ✅ **Category transfer**: Unseen website categories
- ✅ **Geographic transfer**: Different languages/regions
- ✅ **Visual transfer**: Different layouts and styles
- ✅ **Zero-shot websites**: Completely unseen domains

### Training Support
- ✅ **24K training turns** across 115 demonstrations
- ✅ **Input templates** for popular model architectures
- ✅ **Preprocessed data** ready for training
- ✅ **BrowserGym integration** for environment interaction

---

## What WebLINX Lacks ⚠️

### Task Definition & Success Metrics
- ⚠️ **No explicit task descriptions** like WebArena (only conversational context)
- ⚠️ **No automated success evaluation** (no eval scripts like WebArena)
- ⚠️ **Subjective task completion** (depends on conversation flow)
- ⚠️ **No task templates** or intent specifications

### Environment & Execution
- ⚠️ **No live environment** (recordings only, not executable like WebArena/MiniWoB)
- ⚠️ **Websites may change** since recording (URLs, layouts, content)
- ⚠️ **No containerized environment** (websites are public, uncontrolled)
- ⚠️ **Cannot replay/verify** actions on current websites

### Action Space Limitations
- ⚠️ **Limited action types** (no drag-drop, hover, right-click, keyboard shortcuts)
- ⚠️ **No multi-element actions** (e.g., shift-click, ctrl-click)
- ⚠️ **No iframe interactions** explicitly captured
- ⚠️ **Tab actions** less common than single-page interactions

### Data Sparsity Issues
- ⚠️ **Small test sets** for some splits (10-42 demos per split)
- ⚠️ **Imbalanced action distribution** (many clicks, few other actions)
- ⚠️ **Short demonstrations** (avg 21.3 turns, some very short)
- ⚠️ **Limited error recovery** examples (most demos are successful paths)

### HTML & Visual Data
- ⚠️ **Truncated HTML** in processed data (5000 char limit in DB)
- ⚠️ **No visual features** in preprocessed data (screenshots separate)
- ⚠️ **Simplified HTML** may lose important structural information
- ⚠️ **Candidates pre-filtered** (may miss correct element if filtering wrong)

### Dialogue Limitations
- ⚠️ **Mostly English** conversations (limited language diversity)
- ⚠️ **Informal dialogue** may not transfer to formal agent instructions
- ⚠️ **Speaker disambiguation** can be ambiguous in some turns
- ⚠️ **No dialogue acts** or intent annotations

---

## Use Cases for RL Training

### ✅ Suitable For

#### 1. **Imitation Learning from Demonstrations**
- **Behavioral Cloning**: Learn to map (HTML + dialogue context) → action
- **Sequence Modeling**: LLaMA/T5 models can directly predict next action
- **Pre-training**: Use WebLINX as pre-training before fine-tuning on task-specific data

```python
# Example: Train action prediction model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Format input using template
with open('templates/llama.txt') as f:
    template = f.read()

turn = val[0]
input_text = template.format(**turn)
target_action = turn['action']

# Train with cross-entropy loss on action tokens
```

#### 2. **Dialogue-Grounded Web Agents**
- **Conversational Agents**: Learn to follow natural language instructions
- **Clarification Handling**: Train agents to ask questions when unclear
- **Context Tracking**: Model long-term dialogue dependencies

#### 3. **Element Ranking/Reranking**
WebLINX provides reranking data format:

```python
rerank_val = load_dataset("McGill-NLP/WebLINX", name="reranking", split="validation")

print(rerank_val[0]['query'])       # Input context
print(rerank_val[0]['positives'])   # Correct elements
print(rerank_val[0]['negatives'])   # Incorrect elements (hard negatives)
```

Use for:
- **Candidate ranking** models (e.g., Sentence-BERT)
- **Element selection** as a classification task
- **Hard negative mining** for contrastive learning

#### 4. **Multi-Task Learning**
- **Action prediction** + **element ranking** + **next utterance prediction**
- **Joint training** across action types
- **Transfer learning** between splits (train on IID, test on geo/vis/cat/web)

#### 5. **Offline Evaluation**
- **Metrics**: Exact match, element recall, action type accuracy
- **Splits**: Test generalization across 5 different axes
- **Baselines**: Compare against official Sheared-LLaMA models

### ⚠️ Not Suitable For

#### 1. **Online RL Training**
- ❌ **No environment to interact with** (recordings only)
- ❌ **Cannot explore alternative actions** or receive rewards
- ❌ **No A/B testing** of agent strategies
- 💡 **Alternative**: Use WebArena or MiniWoB for online RL

#### 2. **Reward Modeling**
- ❌ **No explicit rewards** per turn
- ❌ **No failure trajectories** for contrast
- ❌ **Success is implicit** (end of demonstration)
- 💡 **Alternative**: Define sparse reward at demo end (0/1)

#### 3. **Multi-Agent Learning**
- ❌ **Only 2 agents** (instructor + navigator) with fixed roles
- ❌ **No competitive settings**
- ❌ **No agent-agent negotiation** (instructor always leads)

#### 4. **Vision-Only Models**
- ❌ **Preprocessed data lacks screenshots** (need raw data)
- ❌ **No pixel-level annotations**
- ❌ **Candidates are text-based** (not visual regions)
- 💡 **Alternative**: Download raw demonstrations with screenshots

---

## Data Availability

### In Repository (Preprocessed)
| Component            | Location                    | Size    | Coverage |
|----------------------|-----------------------------|---------|----------|
| Chat Data (Train)    | HuggingFace Dataset         | ~50 MB  | 24,418 turns |
| Chat Data (Val)      | HuggingFace Dataset         | ~5 MB   | 2,126 turns  |
| Chat Data (Test IID) | HuggingFace Dataset         | ~5 MB   | 2,308 turns  |
| Chat Data (Test Cat) | HuggingFace Dataset         | ~12 MB  | 5,971 turns  |
| Chat Data (Test Geo) | HuggingFace Dataset         | ~15 MB  | 7,672 turns  |
| Chat Data (Test Vis) | HuggingFace Dataset         | ~18 MB  | 9,023 turns  |
| Chat Data (Test Web) | HuggingFace Dataset         | ~10 MB  | 4,856 turns  |
| Reranking Data       | HuggingFace Dataset (name="reranking") | ~30 MB | All splits |
| Input Templates      | HuggingFace Repo (templates/) | <1 MB | 5 templates |
| Sample JSON          | `data/processed/weblinx_sample.json` | ~500 KB | 100 turns |

**Total Preprocessed**: ~150 MB across all splits

### Online (Full Raw Data)
| Component            | Source                      | Size    | Coverage |
|----------------------|-----------------------------|---------|----------|
| Full Demonstrations  | `McGill-NLP/WebLINX-full`   | ~5-10 GB | 155 demos |
| Screenshots          | Included in raw demos       | ~3 GB    | Variable per demo |
| Replay JSON          | Included in raw demos       | ~2 GB    | Full HTML snapshots |

**Note**: Raw data is large (multiple GB). Download selectively by demo name if needed.

### Accessing Data

#### Preprocessed (Recommended)
```python
# Fast - downloads ~50MB for validation
val = load_dataset("McGill-NLP/WebLINX", split="validation")
```

#### Raw (For Full HTML/Screenshots)
```python
# Slow - downloads GBs depending on number of demos
snapshot_download(
    repo_id="McGill-NLP/WebLINX-full",
    repo_type="dataset",
    local_dir="./data/weblinx_raw",
    allow_patterns="demonstrations/demo_name/*"
)
```

---

## Integration with Project

### Database Schema (SQLite)
```sql
CREATE TABLE weblinx (
    id TEXT PRIMARY KEY,            -- "weblinx_{demo}_{turn}"
    demo_id TEXT,                    -- Demonstration identifier
    turn_id INT,                     -- Turn number in demo
    action TEXT,                     -- Ground truth action
    action_history TEXT,             -- Previous actions (LLaMA format)
    utterances TEXT,                 -- Timestamped dialogue
    candidates TEXT,                 -- Pre-filtered HTML elements
    clean_html TEXT,                 -- Simplified HTML (truncated to 5000 chars)
    viewport TEXT,                   -- Viewport dimensions
    source TEXT DEFAULT 'weblinx'
);
```

### Ingestion Status
- ✅ **1,000 samples** ingested (from validation split)
- ✅ **Templates downloaded** to `data/weblinx/templates/`
- ✅ **Sample JSON** saved to `data/processed/weblinx_sample.json`
- ⚠️ **Full dataset** not ingested (24K+ turns would be ~500MB in SQLite)

### Notebook Cells
- **Cell 1**: Setup & database initialization
- **Cell 2**: WebLINX ingestion (loads first 1000 from validation)
  - Connects to HuggingFace
  - Downloads templates
  - Creates SQLite table
  - Inserts data with HTML truncation

### Querying in Notebook
```python
# Use utility functions from Cell 8
query_weblinx(limit=5)  # Returns DataFrame with 5 samples

# Or query directly
conn = sqlite3.connect('data/db/unified.db')
df = pd.read_sql_query("""
    SELECT demo_id, turn_id, action, utterances 
    FROM weblinx 
    WHERE action LIKE 'click%'
    LIMIT 10
""", conn)
```

---

## Relevant Models & Baselines

### Official Models (McGill-NLP)
1. **Sheared-LLaMA-2.7B-weblinx**
   - Pruned from LLaMA-7B, fine-tuned on WebLINX
   - Efficient (2.7B params) with strong performance
   - HuggingFace: `McGill-NLP/Sheared-LLaMA-2.7B-weblinx`

2. **Llama-3-8B-Web**
   - LLaMA-3-8B fine-tuned on WebLINX
   - State-of-the-art on WebLINX benchmark
   - HuggingFace: `McGill-NLP/Llama-3-8B-Web`

3. **Flan-T5 Models**
   - Encoder-decoder baselines
   - Various sizes (base, large, XL)

### Using Official Models
```python
from transformers import pipeline

# Load Sheared-LLaMA
action_model = pipeline(
    model="McGill-NLP/Sheared-LLaMA-2.7B-weblinx",
    device=0,
    torch_dtype='auto'
)

# Format input with template
with open('data/weblinx/templates/llama.txt') as f:
    template = f.read()

turn = val[0]
input_text = template.format(**turn)

# Generate action
pred = action_model(
    input_text,
    return_full_text=False,
    max_new_tokens=64,
    truncation=True
)[0]['generated_text']

print("Ref: ", turn['action'])
print("Pred:", pred)
```

### Benchmark Leaderboard
Check latest results at: https://mcgill-nlp.github.io/weblinx

---

## Data Quality & Considerations

### Strengths
- ✅ **High-quality human demonstrations** (not synthetic)
- ✅ **Natural dialogue** between real users
- ✅ **Diverse websites** (150+ domains)
- ✅ **Multiple test axes** (generalization evaluation)
- ✅ **Preprocessed and ready** (minimal cleaning needed)

### Limitations
- ⚠️ **Websites may have changed** since recording (URLs, layouts)
- ⚠️ **No automated evaluation** (need to build your own metrics)
- ⚠️ **Dialogue can be ambiguous** (informal, context-dependent)
- ⚠️ **Small test sets** for some splits (10-42 demos)
- ⚠️ **Imbalanced actions** (many clicks, fewer other types)

### Data Freshness
- **Recording Period**: 2023 (check paper for exact dates)
- **Website Stability**: Some websites may have:
  - Changed layout/structure (HTML different)
  - Changed content (products, articles)
  - Changed URLs (pages moved/removed)
  - Implemented new authentication (access blocked)

⚠️ **Recommendation**: WebLINX is best used for **offline imitation learning** and **action prediction**, not for online web automation (use WebArena/MiniWoB for that).

---

## Citation

```bibtex
@misc{lu-2024-weblinx,
    title={WebLINX: Real-World Website Navigation with Multi-Turn Dialogue}, 
    author={Xing Han Lù and Zdeněk Kasner and Siva Reddy},
    year={2024},
    eprint={2402.05930},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

---

## License

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**

### Terms of Use
- ✅ **Research use** permitted
- ✅ **Attribution** required
- ✅ **Derivative works** allowed (must share under same license)
- ❌ **Commercial use** prohibited
- ⚠️ **Third-party data** may have additional restrictions

See HuggingFace repo for full license text and terms.

---

## Additional Resources

- **Website**: https://mcgill-nlp.github.io/weblinx
- **Paper**: https://arxiv.org/abs/2402.05930
- **Code**: https://github.com/McGill-NLP/weblinx
- **HuggingFace**: https://huggingface.co/datasets/McGill-NLP/WebLINX
- **Models**: https://huggingface.co/McGill-NLP
- **Colab Tutorial**: https://colab.research.google.com/... (check repo)
- **Leaderboard**: https://mcgill-nlp.github.io/weblinx (if available)
- **BrowserGym Integration**: https://github.com/McGill-NLP/weblinx-browsergym

---

**Last Updated**: December 2024  
**Dataset Version**: 1.1 (BrowserGym release)  
**Status**: ✅ Active benchmark, maintained by McGill NLP
