#!/usr/bin/env python3
"""
RLVR (GRPO) Data Ingestion Script

This script ingests data from multiple local sources to create a unified dataset for 
Reinforcement Learning with Verification Rewards (RLVR) using Group Relative Policy Optimization (GRPO).

It generates:
1. GRPO Training Data: JSONL files with prompts, multiple responses (groups), and rewards.
   Output: data/rl_training/grpo/*.jsonl
2. Multi-turn Conversation Data: JSONL files for supervised fine-tuning or general training.
   Output: data/rl_training/multiturn/*.jsonl

DATA DEPENDENCIES:
This script relies on the following local files. If a file is missing, that dataset will be skipped 
with a warning, but the script will continue.

1. WebArena
   - Path: data/webarena/config_files/test.raw.json
   - Description: Contains task definitions, intents, and evaluation criteria (exact_match, must_include).
   - Usage: Generates capability tasks with string-match rewards.

2. TAU2
   - Path: data/tau2_repo/data/tau2/domains/{telecom,airline,retail,mock}/tasks.json
   - Description: Task definitions for various domains with gold action sequences.
   - Usage: Generates capability tasks with action-sequence F1 rewards.

3. AgentDojo
   - Path: data/agent_dojo/agentdojo-claude-3-5-sonnet-20241022.jsonl
   - Description: Execution traces from Claude 3.5 Sonnet on AgentDojo tasks.
   - Usage: Extracts benign tasks (where security=True or success=True) and computes dual capability/safety rewards.

4. WebLINX
   - Path: data/processed/weblinx_sample.json
   - Description: A sample of WebLINX data exported by notebooks/notebook.py.
   - Usage: Generates tasks with a placeholder reward (custom evaluator pending).
   - Note: If this file is missing, run the ingestion logic in notebooks/notebook.py or provide raw WebLINX data.

OUTPUT FORMAT (GRPO):
{
    "id": "unique_id",
    "prompt": { ... },
    "group_responses": [
        {"response_id": 0, "content/actions": ..., "score": 1.0, "safety_score": 1.0},
        ...
    ],
    "group_average": 0.5,
    "advantages": [0.5, -0.5, ...],
    "metadata": { ... }
}
"""

import os
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import pandas as pd
except ImportError:
    pd = None
    print("⚠️  Pandas not found. AttackQA (parquet) will be skipped.")

# --- Configuration & Paths ---
# Assuming script is run from scripts/ or root. Adjust BASE_DIR accordingly.
# If run from scripts/, parent is root.
if Path.cwd().name == 'scripts':
    BASE_DIR = Path('..').resolve()
else:
    BASE_DIR = Path('.').resolve()

DATA_DIR = BASE_DIR / 'data'
OUT_GRPO_DIR = DATA_DIR / 'rl_training' / 'grpo'
OUT_MULTI_DIR = DATA_DIR / 'rl_training' / 'multiturn'

# Define expected input paths
PATH_WEBARENA = DATA_DIR / 'webarena' / 'config_files' / 'test.raw.json'
PATH_TAU2_REPO = DATA_DIR / 'tau2_repo' / 'data' / 'tau2' / 'domains'
PATH_AGENTDOJO = DATA_DIR / 'agent_dojo' / 'agentdojo-claude-3-5-sonnet-20241022.jsonl'
PATH_WEBLINX = DATA_DIR / 'processed' / 'weblinx_sample.json'
PATH_ATTACKQA = DATA_DIR / 'attackqa' / 'attackqa.parquet'
PATH_AGENTHARM_TEST = DATA_DIR / 'agent_harm' / 'harmful_behaviors_test_public.json'
PATH_AGENTHARM_VAL = DATA_DIR / 'agent_harm' / 'harmful_behaviors_validation.json'

# --- Utility Functions ---

def setup_directories():
    """Ensure output directories exist."""
    OUT_GRPO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MULTI_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directories ready:\n  {OUT_GRPO_DIR}\n  {OUT_MULTI_DIR}")

def read_json(path: Path) -> Any:
    """Read a JSON file with error handling."""
    if not path.exists():
        print(f"⚠️  File not found: {path}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON {path}: {e}")
        return None

def read_jsonl(path: Path, limit: int = None) -> List[Dict[str, Any]]:
    """Read a JSONL file with error handling and optional limit."""
    items = []
    if not path.exists():
        print(f"⚠️  File not found: {path}")
        return items
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue # Skip malformed lines
    except Exception as e:
        print(f"❌ Error reading JSONL {path}: {e}")
    return items

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    """Write a list of dictionaries to a JSONL file."""
    if not rows:
        print(f"⚠️  No data to write to {path}")
        return
    
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"✅ Wrote {len(rows)} rows to {path}")
    except Exception as e:
        print(f"❌ Error writing to {path}: {e}")

def simple_f1(pred: List[str], gold: List[str]) -> float:
    """Compute set-based F1 score for sequences of tokens/actions."""
    sp, sg = set(pred), set(gold)
    tp = len(sp & sg)
    if tp == 0 and (len(sp) == 0 or len(sg) == 0):
        return 0.0
    precision = tp / max(1, len(sp))
    recall = tp / max(1, len(sg))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def calculate_advantages(scores: List[float]) -> List[float]:
    """Calculate advantages (score - mean) for a group."""
    if not scores:
        return []
    avg = sum(scores) / len(scores)
    return [round(s - avg, 6) for s in scores]

def as_messages_from_tool_calls(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert AgentDojo tool call format to standard messages list."""
    messages = []
    for m in sample.get('messages', []):
        content = m.get('content', '')
        messages.append({'role': m.get('role', 'user'), 'content': str(content)})
    return messages

# --- Reward Functions ---

def reward_webarena(pred_answer: str, eval_spec: Dict[str, Any]) -> float:
    """
    Reward for WebArena: 1.0 if exact match or all 'must_include' phrases are present.
    """
    ref = eval_spec.get('reference_answers', {})
    exact = ref.get('exact_match')
    must_include = ref.get('must_include')
    
    if exact:
        return 1.0 if str(pred_answer).strip() == str(exact).strip() else 0.0
    
    if must_include:
        predlow = (pred_answer or '').lower()
        # Check if all required phrases are in the prediction
        ok = all(str(x).lower() in predlow for x in must_include)
        return 1.0 if ok else 0.0
    
    return 0.0

def reward_tau2(pred_actions: List[str], gold_actions: List[str]) -> float:
    """Reward for TAU2: F1 score of predicted actions vs gold actions."""
    return simple_f1(pred_actions, gold_actions)

def reward_agentdojo(response: Dict[str, Any]) -> Dict[str, float]:
    """
    Reward for AgentDojo:
    - Capability: 1.0 if tool calls are present (proxy for attempting task).
    - Safety: 1.0 if 'security_ok' is True.
    - Total: Capability * Safety.
    """
    task_success = 1.0 if response.get('tool_calls') else 0.0
    security_score = 1.0 if response.get('security_ok', True) else 0.0
    total = task_success * security_score
    return {'capability_score': task_success, 'safety_score': security_score, 'total': total}

def reward_weblinx_placeholder(has_ground_truth: bool) -> float:
    """Placeholder reward for WebLINX until custom evaluator is integrated."""
    return 1.0 if has_ground_truth else 0.0

# --- Data Loading & Processing ---

def process_webarena():
    """Load and process WebArena data."""
    print("\n--- Processing WebArena ---")
    tasks = read_json(PATH_WEBARENA)
    if not tasks:
        return [], []

    total_raw = len(tasks)
    skipped_no_ground_truth = 0
    grpo_data = []
    multiturn_data = []
    
    for t in tasks: 
        task_id = t.get('task_id')
        prompt = {'intent': t.get('intent'), 'start_url': t.get('start_url')}
        eval_spec = t.get('eval', {})
        
        # Determine Gold Answer
        ref = eval_spec.get('reference_answers') or {}
        gold_ans = ref.get('exact_match')
        if not gold_ans and ref.get('must_include'):
            gold_ans = ' '.join(ref.get('must_include'))
        
        if not gold_ans:
            skipped_no_ground_truth += 1
            continue

        # --- GRPO Group Generation (Synthetic) ---
        # 1. Gold Response
        r0 = {'response_id': 0, 'trajectory': [{'action': 'submit', 'observation': 'ok'}], 'pred_answer': gold_ans}
        s0 = reward_webarena(gold_ans, eval_spec)
        
        # 2. Partial/Degraded Response
        partial = str(gold_ans)[:len(str(gold_ans))//2] if gold_ans else ""
        r1 = {'response_id': 1, 'trajectory': [], 'pred_answer': partial}
        s1 = reward_webarena(partial, eval_spec)
        
        # 3. Wrong Response
        wrong = "Incorrect Answer"
        r2 = {'response_id': 2, 'trajectory': [], 'pred_answer': wrong}
        s2 = reward_webarena(wrong, eval_spec)
        
        scores = [s0, s1, s2]
        
        grpo_data.append({
            'id': f"webarena_task_{task_id}_grpo",
            'prompt': prompt,
            'group_responses': [
                {**r0, 'score': s0, 'safety_score': 1.0},
                {**r1, 'score': s1, 'safety_score': 1.0},
                {**r2, 'score': s2, 'safety_score': 1.0}
            ],
            'group_average': sum(scores)/3,
            'advantages': calculate_advantages(scores),
            'metadata': {'eval_type': 'string_match', 'domain': ','.join(t.get('sites', []))}
        })

        # --- Multiturn Data ---
        multiturn_data.append({
            'id': f"webarena_task_{task_id}_mt",
            'conversation': [
                {'role': 'system', 'content': 'You are a helpful agent.'},
                {'role': 'user', 'content': t.get('intent')},
                {'role': 'assistant', 'content': str(gold_ans)}
            ],
            'source_dataset': 'webarena'
        })

    print(f"  Total raw: {total_raw}, Skipped (no ground truth): {skipped_no_ground_truth}, Final: {len(grpo_data)}")
    if skipped_no_ground_truth > len(grpo_data) * 0.5:
        print(f"  ⚠️  LIMITING FACTOR: Many records lack 'reference_answers'. Consider enriching data.")
    return grpo_data, multiturn_data

def process_tau2():
    """Load and process TAU2 data."""
    print("\n--- Processing TAU2 ---")
    if not PATH_TAU2_REPO.exists():
        print(f"⚠️  TAU2 domains directory not found: {PATH_TAU2_REPO}")
        return [], []

    tasks = []
    for dom in ['telecom', 'airline', 'retail', 'mock']:
        p = PATH_TAU2_REPO / dom / 'tasks.json'
        d = read_json(p)
        if d:
            for t in d:
                t['domain'] = dom
            tasks.extend(d)
    
    if not tasks:
        return [], []

    total_raw = len(tasks)
    skipped_no_actions = 0
    grpo_data = []
    multiturn_data = []

    for i, t in enumerate(tasks):
        # TAU2 has nested structure: user_scenario.instructions.task_instructions
        instruction = None
        try:
            us = t.get('user_scenario')
            if us and isinstance(us, dict):
                instr = us.get('instructions')
                if instr and isinstance(instr, dict):
                    instruction = instr.get('task_instructions')
        except Exception:
            pass
        if not instruction:
            instruction = t.get('instruction') or t.get('intent') or t.get('title') or t.get('id') or 'Task'
        gold = t.get('gold_actions') or t.get('actions') or []
        if not isinstance(gold, list): 
            gold = []
            
        prompt = {'task_instructions': instruction, 'domain': t.get('domain', 'unknown')}

        # --- GRPO Group Generation ---
        # 1. Gold
        r0_actions = list(gold)
        s0 = reward_tau2(r0_actions, gold)
        
        # 2. Partial (drop last action)
        r1_actions = r0_actions[:-1] if len(r0_actions) > 0 else []
        s1 = reward_tau2(r1_actions, gold)
        
        # 3. Wrong
        r2_actions = ['noop']
        s2 = reward_tau2(r2_actions, gold)
        
        scores = [s0, s1, s2]

        grpo_data.append({
            'id': f"tau2_{i:04d}_grpo",
            'prompt': prompt,
            'group_responses': [
                {'response_id': 0, 'action_sequence': r0_actions, 'score': s0, 'safety_score': 1.0},
                {'response_id': 1, 'action_sequence': r1_actions, 'score': s1, 'safety_score': 1.0},
                {'response_id': 2, 'action_sequence': r2_actions, 'score': s2, 'safety_score': 1.0}
            ],
            'group_average': sum(scores)/3,
            'advantages': calculate_advantages(scores),
            'metadata': {'eval_metric': 'action_f1', 'gold_actions': gold}
        })

        # --- Multiturn Data ---
        multiturn_data.append({
            'id': f"tau2_{i:04d}_mt",
            'conversation': [
                {'role': 'system', 'content': 'You are a helpful agent.'},
                {'role': 'user', 'content': instruction},
                {'role': 'assistant', 'content': ' ; '.join(map(str, gold))}
            ],
            'source_dataset': 'tau2'
        })

    print(f"  Total raw: {total_raw}, Final: {len(grpo_data)}")
    return grpo_data, multiturn_data

def process_agentdojo():
    """Load and process AgentDojo data (Benign only)."""
    print("\n--- Processing AgentDojo ---")
    rows = read_jsonl(PATH_AGENTDOJO)  # No limit - use all data
    if not rows:
        return [], []

    total_raw = len(rows)
    # Filter for benign/successful tasks
    benign_rows = []
    harmful_count = 0
    for r in rows:
        md = r.get('metadata', {})
        if md.get('security') is True or md.get('success') is True:
            benign_rows.append(r)
        elif md.get('security') is False:
            harmful_count += 1
    
    print(f"  Total raw: {total_raw}, Benign: {len(benign_rows)}, Harmful (skipped for RLVR): {harmful_count}")

    grpo_data = []
    multiturn_data = []

    for i, r in enumerate(benign_rows):
        md = r.get('metadata', {})
        user_task_id = md.get('user_task_id', f'user_task_{i}')
        messages = as_messages_from_tool_calls(r)
        
        # Extract user instruction
        user_instr = "Task"
        if len(messages) > 1 and messages[1]['role'] == 'user':
            user_instr = messages[1]['content']

        # --- GRPO Group Generation ---
        # 1. Gold (Simulated success)
        r0 = {'response_id': 0, 'tool_calls': [{'tool': 'get_balance', 'args': {'account_id': 'user'}}], 'security_ok': True}
        rw0 = reward_agentdojo(r0)
        
        # 2. Partial
        r1 = {'response_id': 1, 'tool_calls': [{'tool': 'get_transactions', 'args': {'n': 5}}], 'security_ok': True}
        rw1 = reward_agentdojo(r1)
        
        # 3. Wrong/Unsafe
        r2 = {'response_id': 2, 'tool_calls': [], 'security_ok': False}
        rw2 = reward_agentdojo(r2)
        
        scores = [rw0['total'], rw1['total'], rw2['total']]

        grpo_data.append({
            'id': f"agentdojo_{user_task_id}_grpo",
            'prompt': {'user_instruction': user_instr, 'conversation_history': messages},  # Include full history
            'group_responses': [
                {**r0, 'score': rw0['total'], 'safety_score': rw0['safety_score']},
                {**r1, 'score': rw1['total'], 'safety_score': rw1['safety_score']},
                {**r2, 'score': rw2['total'], 'safety_score': rw2['safety_score']}
            ],
            'group_average': sum(scores)/3,
            'advantages': calculate_advantages(scores),
            'metadata': {'eval_type': 'dual_objective'}
        })

        # --- Multiturn Data (full conversation) ---
        if len(messages) >= 2:
            multiturn_data.append({
                'id': f"agentdojo_{i:04d}_mt",
                'conversation': messages,  # Full conversation for multiturn training
                'source_dataset': 'agentdojo_benign'
            })

    print(f"  Final GRPO: {len(grpo_data)}, Multiturn: {len(multiturn_data)}")
    return grpo_data, multiturn_data

def process_weblinx():
    """Load and process WebLINX data."""
    print("\n--- Processing WebLINX ---")
    rows = read_json(PATH_WEBLINX)
    if not rows:
        return [], []

    total_raw = len(rows)
    grpo_data = []
    multiturn_data = []

    for i, s in enumerate(rows):
        prompt = {
            'utterances': str(s.get('utterances', '')), 
            'candidates': str(s.get('candidates', '')), 
            'clean_html': str(s.get('clean_html', ''))[:1000]
        }

        # --- GRPO Group Generation ---
        # Placeholder logic
        r0 = {'response_id': 0, 'actions': [{'type': 'load', 'url': 'https://example.com'}]}
        s0 = reward_weblinx_placeholder(True)
        
        r1 = {'response_id': 1, 'actions': [{'type': 'click', 'uid': 'wrong-uid'}]}
        s1 = reward_weblinx_placeholder(False)
        
        r2 = {'response_id': 2, 'actions': []}
        s2 = reward_weblinx_placeholder(False)
        
        scores = [s0, s1, s2]

        grpo_data.append({
            'id': f"weblinx_{i:04d}_grpo",
            'prompt': prompt,
            'group_responses': [
                {**r0, 'score': s0, 'safety_score': 1.0},
                {**r1, 'score': s1, 'safety_score': 1.0},
                {**r2, 'score': s2, 'safety_score': 1.0}
            ],
            'group_average': sum(scores)/3,
            'advantages': calculate_advantages(scores),
            'metadata': {'eval_type': 'custom_weblinx'}
        })

        # --- Multiturn Data ---
        multiturn_data.append({
            'id': f"weblinx_{i:04d}_mt",
            'conversation': [
                {'role': 'system', 'content': 'You are a helpful agent.'},
                {'role': 'user', 'content': str(s.get('utterances', ''))},
                {'role': 'assistant', 'content': 'load(url=...) ; say("done")'}
            ],
            'source_dataset': 'weblinx'
        })

    print(f"  Total raw: {total_raw}, Final: {len(grpo_data)}")
    return grpo_data, multiturn_data


def process_attackqa():
    """Load and process AttackQA data (Security Competency QA)."""
    print("\n--- Processing AttackQA ---")
    if pd is None:
        print("  ⚠️  Skipping: pandas not installed.")
        return [], []
    if not PATH_ATTACKQA.exists():
        print(f"  ⚠️  File not found: {PATH_ATTACKQA}")
        return [], []

    try:
        df = pd.read_parquet(PATH_ATTACKQA)
    except Exception as e:
        print(f"  ❌ Error reading parquet: {e}")
        return [], []

    total_raw = len(df)
    skipped = 0
    grpo_data = []
    multiturn_data = []

    for i, row in df.iterrows():
        question = str(row.get('question', '')).strip()
        answer = str(row.get('answer', '')).strip()
        if not question or not answer:
            skipped += 1
            continue

        # Reward: overlap between predicted and gold answer
        def qa_overlap(pred: str, gold: str) -> float:
            pred_tokens = set(pred.lower().split())
            gold_tokens = set(gold.lower().split())
            if not gold_tokens: return 0.0
            return len(pred_tokens & gold_tokens) / len(gold_tokens)

        r0 = {'response_id': 0, 'content': answer}
        s0 = qa_overlap(answer, answer)
        r1 = {'response_id': 1, 'content': "I don't know."}
        s1 = qa_overlap("I don't know.", answer)
        r2 = {'response_id': 2, 'content': "Incorrect information."}
        s2 = qa_overlap("Incorrect information.", answer)
        scores = [s0, s1, s2]

        grpo_data.append({
            'id': f"attackqa_{i:06d}_grpo",
            'prompt': {'question': question, 'context': str(row.get('context_document', ''))[:500]},
            'group_responses': [
                {**r0, 'score': s0, 'safety_score': 1.0},
                {**r1, 'score': s1, 'safety_score': 1.0},
                {**r2, 'score': s2, 'safety_score': 1.0}
            ],
            'group_average': sum(scores)/3,
            'advantages': calculate_advantages(scores),
            'metadata': {'eval_type': 'qa_overlap', 'mitre_id': str(row.get('mitre_technique_id', ''))}
        })

        multiturn_data.append({
            'id': f"attackqa_{i:06d}_mt",
            'conversation': [
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': answer}
            ],
            'source_dataset': 'attackqa'
        })

    print(f"  Total raw: {total_raw}, Skipped (empty Q/A): {skipped}, Final: {len(grpo_data)}")
    return grpo_data, multiturn_data


def process_agentharm():
    """Load and process AgentHarm data (Safety/Refusal training)."""
    print("\n--- Processing AgentHarm ---")
    behaviors = []
    for p in [PATH_AGENTHARM_TEST, PATH_AGENTHARM_VAL]:
        d = read_json(p)
        if d:
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, list): behaviors.extend(v)
                    elif isinstance(v, dict): behaviors.append(v)
            elif isinstance(d, list):
                behaviors.extend(d)

    if not behaviors:
        print("  ⚠️  No behaviors found.")
        return [], []

    total_raw = len(behaviors)
    skipped = 0
    grpo_data = []
    multiturn_data = []

    refusal_keywords = ["i cannot", "i can't", "sorry", "illegal", "harmful", "unethical", "cannot assist"]
    def refusal_reward(text: str) -> float:
        return 1.0 if any(r in text.lower() for r in refusal_keywords) else 0.0

    for i, b in enumerate(behaviors):
        prompt_text = b.get('prompt') or b.get('behavior') or b.get('Behavior')
        if not prompt_text:
            skipped += 1
            continue

        r0 = {'response_id': 0, 'content': "I cannot assist with that request as it may be harmful."}
        s0 = refusal_reward(r0['content'])
        r1 = {'response_id': 1, 'content': "Sure, here's how to do it..."}
        s1 = refusal_reward(r1['content'])
        r2 = {'response_id': 2, 'content': "I can help with that."}
        s2 = refusal_reward(r2['content'])
        scores = [s0, s1, s2]

        grpo_data.append({
            'id': f"agentharm_{i:04d}_grpo",
            'prompt': {'text': prompt_text, 'category': b.get('FunctionalCategory', b.get('category', ''))},
            'group_responses': [
                {**r0, 'score': s0, 'safety_score': 1.0},
                {**r1, 'score': s1, 'safety_score': 0.0},
                {**r2, 'score': s2, 'safety_score': 0.0}
            ],
            'group_average': sum(scores)/3,
            'advantages': calculate_advantages(scores),
            'metadata': {'eval_type': 'safety_refusal'}
        })

        multiturn_data.append({
            'id': f"agentharm_{i:04d}_mt",
            'conversation': [
                {'role': 'user', 'content': prompt_text},
                {'role': 'assistant', 'content': "I cannot assist with that request."}
            ],
            'source_dataset': 'agentharm'
        })

    print(f"  Total raw: {total_raw}, Skipped (no prompt): {skipped}, Final: {len(grpo_data)}")
    return grpo_data, multiturn_data


# --- Main Execution ---

def main():
    setup_directories()
    
    # Process all datasets
    wa_grpo, wa_mt = process_webarena()
    tau_grpo, tau_mt = process_tau2()
    ad_grpo, ad_mt = process_agentdojo()
    wl_grpo, wl_mt = process_weblinx()
    aq_grpo, aq_mt = process_attackqa()
    ah_grpo, ah_mt = process_agentharm()
    
    # Write GRPO outputs
    write_jsonl(OUT_GRPO_DIR / 'webarena_grpo.jsonl', wa_grpo)
    write_jsonl(OUT_GRPO_DIR / 'tau2_grpo.jsonl', tau_grpo)
    write_jsonl(OUT_GRPO_DIR / 'agentdojo_grpo.jsonl', ad_grpo)
    write_jsonl(OUT_GRPO_DIR / 'weblinx_grpo.jsonl', wl_grpo)
    write_jsonl(OUT_GRPO_DIR / 'attackqa_grpo.jsonl', aq_grpo)
    write_jsonl(OUT_GRPO_DIR / 'agentharm_grpo.jsonl', ah_grpo)
    
    # Write Unified Multiturn output
    all_multiturn = wa_mt + tau_mt + ad_mt + wl_mt + aq_mt + ah_mt
    write_jsonl(OUT_MULTI_DIR / 'multiturn_unified.jsonl', all_multiturn)
    
    # --- Summary with Limiting Reagent Analysis ---
    counts = {
        'WebArena': len(wa_grpo),
        'TAU2': len(tau_grpo),
        'AgentDojo': len(ad_grpo),
        'WebLINX': len(wl_grpo),
        'AttackQA': len(aq_grpo),
        'AgentHarm': len(ah_grpo)
    }
    
    print("\n" + "="*60)
    print("                    RLVR INGESTION SUMMARY")
    print("="*60)
    for name, count in counts.items():
        print(f"  {name:12s}: {count:6d} GRPO groups")
    print("-"*60)
    print(f"  {'TOTAL':12s}: {sum(counts.values()):6d} GRPO groups")
    print(f"  Multiturn Conversations: {len(all_multiturn)}")
    
    # Identify limiting reagent
    if counts:
        min_name = min(counts, key=counts.get)
        min_val = counts[min_name]
        max_val = max(counts.values())
        if min_val < max_val * 0.1 and min_val > 0:
            print(f"\n  ⚠️  LIMITING REAGENT: '{min_name}' has only {min_val} samples.")
            print(f"      Consider adding more data to this dataset for balance.")
        elif min_val == 0:
            zero_datasets = [n for n, c in counts.items() if c == 0]
            print(f"\n  ⚠️  MISSING DATA: {', '.join(zero_datasets)}")
            print(f"      Check file paths or run prerequisite ingestion scripts.")
    print("="*60)

if __name__ == "__main__":
    main()
