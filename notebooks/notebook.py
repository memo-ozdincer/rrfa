!pip3 install datasets
# ============================================================================
# CELL 1: SETUP & ENVIRONMENT
# ============================================================================

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import zipfile
import shutil
from bs4 import BeautifulSoup
import re

# Dataset loading
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Create project structure
BASE_DIR = Path("../")
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
for directory in [DATA_DIR, DB_DIR, PROCESSED_DIR, 
                  DATA_DIR / "weblinx",
                  DATA_DIR / "webarena" / "config_files",
                  DATA_DIR / "webarena" / "human_trajectories",
                  DATA_DIR / "webarena" / "llm_trajectories_v2",
                  DATA_DIR / "tau2" / "domains",
                  DATA_DIR / "tau2" / "results",
                  DATA_DIR / "tau2" / "user_simulator"]:
    directory.mkdir(parents=True, exist_ok=True)

# Initialize SQLite database
DB_PATH = DB_DIR / "unified.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print(f"✅ Project structure created at {BASE_DIR}")
print(f"✅ Database path: {DB_PATH}")
print(f"✅ All directories ready")

# Add this right after loading weblinx_val
print("Available columns:", weblinx_val.column_names)
print("First sample keys:", weblinx_val[0].keys())

# ============================================================================
# CELL 2: INGEST WEBLINX DATASET (CORRECTED KEYS)
# ============================================================================

print("🔄 Loading WebLINX dataset from HuggingFace...")

# Load validation and test splits
weblinx_val = load_dataset("McGill-NLP/weblinx", split="validation")
weblinx_test_iid = load_dataset("McGill-NLP/weblinx", split="test_iid")

print(f"✅ Loaded {len(weblinx_val)} validation samples")
print(f"🔍 Columns found: {weblinx_val.column_names}")

# Download templates (for preprocessing)
template_dir = DATA_DIR / "weblinx" / "templates"
snapshot_download(
    "McGill-NLP/WebLINX",
    repo_type="dataset",
    allow_patterns="templates/*",
    local_dir=DATA_DIR / "weblinx"
)

# Create SQLite table for WebLINX
cursor.execute("""
    CREATE TABLE IF NOT EXISTS weblinx (
        id TEXT PRIMARY KEY,
        demo_id TEXT,
        turn_id INT,
        action TEXT,
        action_history TEXT,
        utterances TEXT,
        candidates TEXT,
        clean_html TEXT,
        viewport TEXT,
        source TEXT DEFAULT 'weblinx'
    )
""")
conn.commit()

# Insert WebLINX data (sample first 1000)
weblinx_combined = weblinx_val.select(range(min(1000, len(weblinx_val))))

for idx, sample in enumerate(weblinx_combined):
    # CORRECT KEYS: 'demo' and 'turn' (not demo_id/turn_id)
    demo_id = sample['demo']
    turn_id = sample['turn']
    
    doc_id = f"weblinx_{demo_id}_{turn_id}"
    
    cursor.execute("""
        INSERT OR REPLACE INTO weblinx VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        doc_id,
        demo_id,
        turn_id,
        str(sample.get('action', '')),
        str(sample.get('action_history', '')),
        str(sample.get('utterances', '')),
        str(sample.get('candidates', '')),
        str(sample.get('clean_html', ''))[:5000], # Truncate HTML
        str(sample.get('viewport', '')),
        'weblinx'
    ))
    
    if (idx + 1) % 100 == 0:
        print(f"  ✓ Inserted {idx + 1} WebLINX records")

conn.commit()
print(f"✅ WebLINX data inserted into database ({len(weblinx_combined)} samples)")

# Save WebLINX JSON for reference
# Save WebLINX JSON for reference
weblinx_json_path = PROCESSED_DIR / "weblinx_sample.json"

# FIX: dataset[:100] returns a dict of lists. 
# We need to iterate the dataset object to get rows.
json_safe_sample = []

# Iterate over the first 100 rows explicitly
for i in range(min(100, len(weblinx_combined))):
    row = weblinx_combined[i] # Accessing by index yields a dict (row)
    # Convert all values to string to be safe for JSON
    safe_row = {k: str(v) for k, v in row.items()}
    json_safe_sample.append(safe_row)

with open(weblinx_json_path, "w") as f:
    json.dump(json_safe_sample, f, indent=2)

print(f"✅ Sample saved to {weblinx_json_path}")

# ============================================================================
# CELL 3: INGEST WEBARENA CONFIG (test.raw.json)
# ============================================================================
# Assumes you've already downloaded test.raw.json manually

webarena_config_path = DATA_DIR / "webarena" / "config_files" / "test.raw.json"

if webarena_config_path.exists():
    print(f"✅ Found WebArena config at {webarena_config_path}")
    
    with open(webarena_config_path, 'r') as f:
        webarena_tasks = json.load(f)
    
    print(f"✅ Loaded {len(webarena_tasks)} WebArena tasks")
    
    # Create table for WebArena tasks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS webarena_tasks (
            task_id INTEGER PRIMARY KEY,
            sites TEXT,
            require_login BOOLEAN,
            start_url TEXT,
            intent TEXT,
            intent_template TEXT,
            eval_types TEXT,
            reference_answers TEXT,
            source TEXT DEFAULT 'webarena'
        )
    """)
    conn.commit()
    
    # Insert WebArena tasks
    for task in webarena_tasks:
        cursor.execute("""
            INSERT OR REPLACE INTO webarena_tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.get('task_id'),
            str(task.get('sites', [])),
            task.get('require_login', False),
            task.get('start_url', ''),
            task.get('intent', ''),
            task.get('intent_template', ''),
            str(task.get('eval', {}).get('eval_types', [])),
            json.dumps(task.get('eval', {}).get('reference_answers', {})),
            'webarena'
        ))
    
    conn.commit()
    print(f"✅ WebArena tasks inserted into database")
    
    # Save sample
    webarena_sample_path = PROCESSED_DIR / "webarena_tasks_sample.json"
    with open(webarena_sample_path, "w") as f:
        json.dump(webarena_tasks[:100], f, indent=2)
    print(f"✅ Sample saved to {webarena_sample_path}")
    
else:
    print(f"⚠️  WebArena config not found at {webarena_config_path}")
    print(f"   Expected location: {webarena_config_path}")

# ============================================================================
# CELL 4: PARSE WEBARENA HUMAN TRAJECTORIES (.trace format)
# ============================================================================

def parse_trace_directory(trace_dir: Path) -> Dict[str, Any]:
    """
    Parse .trace directory from human trajectories.
    Structure:
    - resources/
    - trace.network
    - trace.stacks
    - trace.trace
    """
    files_present = {
        "has_resources": (trace_dir / "resources").exists(),
        "has_network": (trace_dir / "trace.network").exists(),
        "has_stacks": (trace_dir / "trace.stacks").exists(),
        "has_trace": (trace_dir / "trace.trace").exists(),
    }
    
    result = {
        "trace_dir": trace_dir.name,
        "files": files_present,
        "resource_count": 0
    }
    
    # Count resources
    if files_present["has_resources"]:
        resource_files = list((trace_dir / "resources").glob("*"))
        result["resource_count"] = len(resource_files)
    
    return result

# Create table for WebArena human trajectories
cursor.execute("""
    CREATE TABLE IF NOT EXISTS webarena_human_traces (
        trace_id TEXT PRIMARY KEY,
        trace_dir_name TEXT,
        has_network BOOLEAN,
        has_stacks BOOLEAN,
        has_trace BOOLEAN,
        resource_count INT,
        source TEXT DEFAULT 'webarena_human_traces'
    )
""")
conn.commit()

human_traj_dir = DATA_DIR / "webarena" / "human_trajectories"

if human_traj_dir.exists():
    print(f"🔄 Processing WebArena human trajectories from {human_traj_dir}...")
    
    # Find all .trace directories (e.g., 4.trace, 7.trace, etc.)
    trace_dirs = [d for d in human_traj_dir.iterdir() if d.is_dir() and d.name.endswith('.trace')]
    
    print(f"   Found {len(trace_dirs)} .trace directories")
    
    for trace_dir in sorted(trace_dirs):
        try:
            parsed = parse_trace_directory(trace_dir)
            trace_id = f"webarena_human_{trace_dir.name}"
            
            cursor.execute("""
                INSERT OR REPLACE INTO webarena_human_traces VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                trace_id,
                parsed['trace_dir'],
                parsed['files']['has_network'],
                parsed['files']['has_stacks'],
                parsed['files']['has_trace'],
                parsed['resource_count'],
                'webarena_human_traces'
            ))
        except Exception as e:
            print(f"   ⚠️  Error processing {trace_dir.name}: {e}")
    
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM webarena_human_traces")
    count = cursor.fetchone()[0]
    print(f"✅ Inserted {count} human trace records")
else:
    print(f"⚠️  Human trajectories directory not found at {human_traj_dir}")
    print(f"   Expected structure: {human_traj_dir}/4.trace, 7.trace, etc.")

# ============================================================================
# CELL 5: PARSE WEBARENA LLM TRAJECTORIES v2 (HTML render format)
# ============================================================================

def parse_merged_log(log_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse merged_log.txt from LLM trajectories.
    Extract: Intent, Result (PASS/FAIL) for each render_*.html
    
    Format:
    2023-09-24 16:32:42,509 - INFO - [Intent]: What is the top-1 best-selling product in 2022
    2023-09-24 16:33:07,065 - INFO - [Result] (FAIL) /path/to/0.json
    """
    results = {}
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    current_task_id = None
    current_intent = None
    
    for line in lines:
        # Extract intent
        if "[Intent]:" in line:
            match = re.search(r"\[Intent\]:\s*(.+?)(?:\s|$)", line)
            if match:
                current_intent = match.group(1).strip()
        
        # Extract result (PASS/FAIL)
        if "[Result]" in line:
            # Extract task ID from path (e.g., /tmp/.../0.json → 0)
            match_result = re.search(r"\((\w+)\)", line)
            match_task = re.search(r"/(\d+)\.json", line)
            
            if match_result and match_task:
                task_id = match_task.group(1)
                result_status = match_result.group(1)
                
                results[task_id] = {
                    "intent": current_intent,
                    "result": result_status.lower() == "pass"
                }
    
    return results

def parse_webarena_html(html_path: Path) -> Dict[str, Any]:
    """
    Parse render_*.html files from WebArena LLM trajectories.
    Extract: observations, URLs, predictions, actions
    """
    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        observations = [div.get_text()[:200] for div in soup.find_all("div", {"class": "state_obv"})]
        urls = [h3.get_text() for h3 in soup.find_all("h3", {"class": "url"})]
        raw_predictions = [div.get_text()[:200] for div in soup.find_all("div", {"class": "raw_parsed_prediction"})]
        actions = [div.get_text()[:200] for div in soup.find_all("div", {"class": "action_object"})]
        
        return {
            "observations": observations,
            "urls": urls,
            "predictions": raw_predictions,
            "actions": actions,
            "num_steps": len(observations)
        }
    except Exception as e:
        return {"error": str(e), "num_steps": 0}

# Create table for WebArena LLM trajectories
cursor.execute("""
    CREATE TABLE IF NOT EXISTS webarena_llm_traces (
        trajectory_id TEXT PRIMARY KEY,
        model TEXT,
        config TEXT,
        task_id INT,
        intent TEXT,
        passed BOOLEAN,
        num_steps INT,
        observations TEXT,
        actions TEXT,
        urls TEXT,
        predictions TEXT,
        source TEXT DEFAULT 'webarena_llm_traces'
    )
""")
conn.commit()

llm_traj_v2_dir = DATA_DIR / "webarena" / "llm_trajectories_v2"

if llm_traj_v2_dir.exists():
    print(f"🔄 Processing WebArena LLM v2 trajectories from {llm_traj_v2_dir}...")
    
    # Find all unzipped model folders (e.g., v2_919_gpt4_8k_cot/)
    model_dirs = [d for d in llm_traj_v2_dir.iterdir() if d.is_dir()]
    
    print(f"   Found {len(model_dirs)} model directories")
    
    for model_dir in sorted(model_dirs):
        # Parse model name from directory
        # Expected format: v2_919_gpt4_8k_cot or similar
        model_name = model_dir.name
        
        # Parse merged_log.txt for pass/fail + intent
        log_file = model_dir / "merged_log.txt"
        if not log_file.exists():
            print(f"   ⚠️  No merged_log.txt in {model_name}")
            continue
        
        log_data = parse_merged_log(log_file)
        print(f"   Processing {model_name}: {len(log_data)} tasks")
        
        # Parse all render_*.html files
        for html_file in sorted(model_dir.glob("render_*.html")):
            try:
                task_id = int(html_file.stem.replace("render_", ""))
                parsed_html = parse_webarena_html(html_file)
                
                # Get intent and result from log
                log_info = log_data.get(str(task_id), {})
                intent = log_info.get("intent", "unknown")
                passed = log_info.get("result", False)
                
                trajectory_id = f"webarena_llm_{model_name}_task_{task_id}"
                
                cursor.execute("""
                    INSERT OR REPLACE INTO webarena_llm_traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trajectory_id,
                    model_name,
                    "",  # config can be parsed from model_name if needed
                    task_id,
                    intent,
                    passed,
                    parsed_html.get("num_steps", 0),
                    json.dumps(parsed_html.get("observations", [])),
                    json.dumps(parsed_html.get("actions", [])),
                    json.dumps(parsed_html.get("urls", [])),
                    json.dumps(parsed_html.get("predictions", [])),
                    'webarena_llm_traces'
                ))
            except Exception as e:
                print(f"     ⚠️  Error processing {html_file}: {e}")
        
        conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM webarena_llm_traces")
    count = cursor.fetchone()[0]
    print(f"✅ Inserted {count} LLM trajectory records")
else:
    print(f"⚠️  LLM trajectories v2 directory not found at {llm_traj_v2_dir}")
    print(f"   Expected structure: {llm_traj_v2_dir}/v2_919_gpt4_8k_cot/")

# ============================================================================
# CELL 6: INGEST TAU2-BENCH DATASETS (FIXED MOCK DOMAIN)
# ============================================================================

# Assume tau2-bench has been cloned or downloaded manually
tau2_repo_path = DATA_DIR / "tau2_repo"

if not tau2_repo_path.exists():
    print(f"⚠️  TAU2-BENCH not found at {tau2_repo_path}")
    print(f"   Please clone: git clone https://github.com/sierra-research/tau2-bench.git {tau2_repo_path}")
else:
    print(f"✅ TAU2-BENCH found at {tau2_repo_path}")
    
    # =========================================================================
    # Parse TAU2 Domains (airline, retail, telecom, mock)
    # =========================================================================
    
    tau2_domains_path = tau2_repo_path / "data" / "tau2" / "domains"
    
    if tau2_domains_path.exists():
        # UPDATED SCHEMA: Changed task_num from INT to TEXT
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tau2_domains (
                task_id TEXT PRIMARY KEY,
                domain TEXT,
                task_num TEXT, 
                reason_for_call TEXT,
                known_info TEXT,
                unknown_info TEXT,
                task_instructions TEXT,
                actions_required TEXT,
                source TEXT DEFAULT 'tau2'
            )
        """)
        conn.commit()
        
        domains = ['airline', 'retail', 'telecom', 'mock']
        
        for domain in domains:
            domain_path = tau2_domains_path / domain
            tasks_file = domain_path / "tasks.json"
            
            if tasks_file.exists():
                print(f"🔄 Processing TAU2 domain: {domain}")
                
                with open(tasks_file, 'r') as f:
                    tasks = json.load(f)
                
                for task in tasks:
                    task_id = f"tau2_{domain}_{task['id']}"
                    user_scenario = task.get('user_scenario', {})
                    
                    # FIX: 'instructions' can be a string (in mock) or dict (others)
                    instructions_raw = user_scenario.get('instructions', {})
                    
                    if isinstance(instructions_raw, dict):
                        # Standard format (Airline, Retail, Telecom)
                        instructions = instructions_raw
                        task_instr = instructions.get('task_instructions', '')
                    else:
                        # Mock format: 'instructions' is just the instruction text directly
                        instructions = {}
                        task_instr = str(instructions_raw) # Treat the whole string as the task instruction
                    
                    # FIX: Treat ID as string, do not int()
                    original_id = str(task['id'])
                    eval_criteria = task.get('evaluation_criteria', {})
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO tau2_domains VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task_id,
                        domain,
                        original_id,
                        instructions.get('reason_for_call', ''),
                        instructions.get('known_info', ''),
                        instructions.get('unknown_info', ''),
                        task_instr, # Uses the string directly if mock
                        json.dumps(eval_criteria.get('actions', []))[:1000],
                        'tau2'
                    ))
                
                conn.commit()
                print(f"  ✅ Inserted {len(tasks)} {domain} tasks")
            else:
                print(f"  ⚠️  {tasks_file} not found")
    else:
        print(f"⚠️  TAU2 domains path not found: {tau2_domains_path}")
    
    # =========================================================================
    # Parse TAU2 Results
    # =========================================================================
    
    tau2_results_path = tau2_repo_path / "data" / "tau2" / "results" / "final"
    
    if tau2_results_path.exists():
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tau2_results (
                result_id TEXT PRIMARY KEY,
                model TEXT,
                domain TEXT,
                num_trials INT,
                result_json TEXT,
                source TEXT DEFAULT 'tau2_results'
            )
        """)
        conn.commit()
        
        print(f"🔄 Processing TAU2 results from {tau2_results_path}")
        
        for result_file in tau2_results_path.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                result_id = result_file.stem
                parts = result_file.stem.split('_')
                model = parts[0] if parts else 'unknown'
                domain = parts[1] if len(parts) > 1 else 'unknown'
                
                cursor.execute("""
                    INSERT OR REPLACE INTO tau2_results VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result_id,
                    model,
                    domain,
                    len(result_data) if isinstance(result_data, list) else 1,
                    json.dumps(result_data)[:5000],
                    'tau2_results'
                ))
            except Exception as e:
                print(f"  ⚠️  Error processing {result_file}: {e}")
        
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM tau2_results")
        count = cursor.fetchone()[0]
        print(f"✅ Inserted {count} TAU2 result records")
    else:
        print(f"⚠️  TAU2 results path not found: {tau2_results_path}")

# ============================================================================
# CELL 7: CREATE UNIFIED DATABASE VIEWS & SUMMARY (FIXED)
# ============================================================================

# Re-connect to ensure we are reading the actual persistent database file
# This fixes "Cannot operate on a closed database" and ensures we see committed data
if 'conn' in locals():
    try:
        conn.close()
    except:
        pass

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

summary_stats = {}

# Get counts for all tables
tables = [
    'weblinx',
    'webarena_tasks',
    'webarena_human_traces',
    'webarena_llm_traces',
    'tau2_domains',
    'tau2_results'
]

print(f"Reading from database: {DB_PATH}")

for table in tables:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        summary_stats[f'{table}_count'] = count
    except sqlite3.OperationalError:
        # Table doesn't exist
        summary_stats[f'{table}_count'] = 0
    except Exception as e:
        print(f"⚠️ Error reading {table}: {e}")
        summary_stats[f'{table}_count'] = 0

# Print summary
print("╔═══════════════════════════════════════════════════════════════╗")
print("║              UNIFIED DATABASE SUMMARY                        ║")
print("╚═══════════════════════════════════════════════════════════════╝")
print(f"\n📊 Data Ingestion Complete:")
print(f"   WebLINX samples:              {summary_stats.get('weblinx_count', 0):,}")
print(f"   WebArena tasks:               {summary_stats.get('webarena_tasks_count', 0):,}")
print(f"   WebArena human traces:        {summary_stats.get('webarena_human_traces_count', 0):,}")
print(f"   WebArena LLM traces:          {summary_stats.get('webarena_llm_traces_count', 0):,}")
print(f"   TAU2 domain tasks:            {summary_stats.get('tau2_domains_count', 0):,}")
print(f"   TAU2 result logs:             {summary_stats.get('tau2_results_count', 0):,}")

total_samples = sum(summary_stats.values())
print(f"\n   TOTAL RECORDS:                {total_samples:,}")

# Save summary
summary_path = PROCESSED_DIR / "dataset_summary.json"
with open(summary_path, 'w') as f:
    json.dump({
        **summary_stats,
        "ingestion_date": datetime.now().isoformat(),
        "database_path": str(DB_PATH),
        "tables": tables
    }, f, indent=2)

print(f"\n✅ Summary saved to {summary_path}")

# List all tables in database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
all_tables = cursor.fetchall()
print(f"\n📋 Tables in database: {[t[0] for t in all_tables]}")

# Close database
conn.close()
print(f"✅ Database closed: {DB_PATH}")

# ============================================================================
# CELL 9: INGEST FUJITSU AGENTIC RAG RED TEAMING DATASET
# ============================================================================

fujitsu_dir = DATA_DIR / "fujitsu"
fujitsu_dir.mkdir(parents=True, exist_ok=True)

# Define the expected files based on the dataset readme
fujitsu_files = {
    "orchestrator": "orchestrator_attacks_combined_deduplicated.jsonl",
    "rag_poison": "rag_poisoning_benchmark_combined_deduplicated.jsonl",
    "direct_query": "safety_benchmark_direct_query_combined_deduplicated.jsonl",
    "image_poison": "image_poisoning_simulation_results_20250504_202954.jsonl"
}

# 1. Create Table
# We use a flexible schema because columns vary significantly between B1, B2, B3, B4
cursor.execute("""
    CREATE TABLE IF NOT EXISTS fujitsu_red_teaming (
        unique_id TEXT PRIMARY KEY,
        original_id TEXT,
        dataset_subset TEXT,  -- orchestrator, rag_poison, etc.
        adversarial_goal TEXT,
        user_query TEXT,      -- The actual input sent to the system
        injection_payload TEXT, -- Specific payload if separated (e.g., hidden JSON)
        target_output TEXT,   -- The LLM/System response
        success BOOLEAN,      -- Whether the attack succeeded
        full_json TEXT,       -- Store all specific metadata here
        source TEXT DEFAULT 'fujitsu'
    )
""")
conn.commit()

print(f"🔄 Processing Fujitsu Red Teaming data from {fujitsu_dir}...")

total_fujitsu_records = 0

for subset_name, filename in fujitsu_files.items():
    file_path = fujitsu_dir / filename
    
    if file_path.exists():
        print(f"   Processing {subset_name}...")
        records_added = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    
                    # --- NORMALIZE FIELDS BASED ON SUBSET ---
                    # Different files use different keys (id vs attack_id, etc.)
                    
                    # 1. ID Handling
                    orig_id = record.get('id') or record.get('attack_id') or record.get('record_id')
                    unique_id = f"fujitsu_{subset_name}_{orig_id}"
                    
                    # 2. Query/Payload Handling
                    user_query = ""
                    injection_payload = ""
                    goal = record.get('adversarial_goal', '')
                    output = ""
                    
                    if subset_name == "orchestrator":
                        # B4: Has benign_query + malicious_injection -> combined_query
                        user_query = record.get('combined_query', '')
                        injection_payload = record.get('malicious_injection', '')
                        output = record.get('simulated_tool', '') # The tool it flipped to
                        
                    elif subset_name == "rag_poison":
                        # B1: Poison content is retrieved, user_query triggers it
                        user_query = record.get('user_query', '')
                        injection_payload = record.get('poison_content', '')
                        output = record.get('target_llm_output', '')
                        
                    elif subset_name == "direct_query":
                        # B3: Direct prompt injection
                        user_query = record.get('user_query', '')
                        injection_payload = record.get('adversarial_suffix', '') # Sometimes suffix, sometimes embedded
                        output = record.get('target_llm_output', '')
                        
                    elif subset_name == "image_poison":
                        # B2: Payload is in the image
                        user_query = record.get('user_query', '')
                        injection_payload = record.get('poison_payload', '')
                        output = record.get('mta_output', '')

                    # 3. Success Handling (Usually boolean or string judgment)
                    success_raw = record.get('success') or record.get('mta_rag_success')
                    # Some files might not have explicit boolean success in all lines, default to True as this is a "successful attacks" corpus
                    is_success = True if success_raw is None else bool(success_raw)

                    # Insert
                    cursor.execute("""
                        INSERT OR REPLACE INTO fujitsu_red_teaming VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        unique_id,
                        str(orig_id),
                        subset_name,
                        goal,
                        user_query,
                        injection_payload,
                        output,
                        is_success,
                        json.dumps(record),
                        'fujitsu'
                    ))
                    records_added += 1
                    
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    # simplistic error skipping to keep ingestion moving
                    continue
        
        print(f"     ✓ Added {records_added} records from {subset_name}")
        total_fujitsu_records += records_added
    else:
        print(f"   ⚠️ File not found: {filename} (Skipping)")

conn.commit()
print(f"✅ Fujitsu ingestion complete. Total records: {total_fujitsu_records}")
# ============================================================================
# CELL 8: UTILITY FUNCTIONS FOR QUERYING & EXPORTING
# ============================================================================

def get_db_connection():
    """Get connection to unified database"""
    return sqlite3.connect(DB_PATH)

def query_weblinx(limit: int = 5) -> pd.DataFrame:
    """Query WebLINX samples"""
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM weblinx LIMIT {limit}", conn)
    conn.close()
    return df

def query_webarena_tasks(limit: int = 5) -> pd.DataFrame:
    """Query WebArena tasks"""
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM webarena_tasks LIMIT {limit}", conn)
    conn.close()
    return df

def query_webarena_llm(model: Optional[str] = None, limit: int = 5) -> pd.DataFrame:
    """Query WebArena LLM trajectories"""
    conn = get_db_connection()
    if model:
        query = f"SELECT * FROM webarena_llm_traces WHERE model LIKE '%{model}%' LIMIT {limit}"
    else:
        query = f"SELECT * FROM webarena_llm_traces LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def query_tau2(domain: str, limit: int = 5) -> pd.DataFrame:
    """Query TAU2 tasks by domain"""
    conn = get_db_connection()
    df = pd.read_sql_query(
        f"SELECT * FROM tau2_domains WHERE domain = '{domain}' LIMIT {limit}",
        conn
    )
    conn.close()
    return df

def export_to_json(table_name: str, output_path: Path) -> None:
    """Export entire table to JSON"""
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    df.to_json(output_path, orient='records', indent=2)
    print(f"✅ Exported {len(df)} records to {output_path}")

# Test queries
print("Testing database queries...\n")

print("1️⃣  WebLINX sample:")
print(query_weblinx(limit=1).to_string())

print("\n2️⃣  WebArena tasks sample:")
try:
    print(query_webarena_tasks(limit=1).to_string())
except Exception as e:
    print(f"   (No data: {e})")

print("\n3️⃣  WebArena LLM traces sample:")
try:
    print(query_webarena_llm(limit=1).to_string())
except Exception as e:
    print(f"   (No data: {e})")

print("\n4️⃣  TAU2 retail tasks sample:")
try:
    print(query_tau2(domain='retail', limit=1).to_string())
except Exception as e:
    print(f"   (No data: {e})")

print("\n✅ Utility functions ready for use")
def query_fujitsu(subset: Optional[str] = None, limit: int = 5) -> pd.DataFrame:
    """Query Fujitsu Red Teaming samples"""
    conn = get_db_connection()
    if subset:
        query = f"SELECT * FROM fujitsu_red_teaming WHERE dataset_subset = '{subset}' LIMIT {limit}"
    else:
        query = f"SELECT * FROM fujitsu_red_teaming LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Test the query
try:
    print("\n5️⃣  Fujitsu Orchestrator sample:")
    print(query_fujitsu(subset='orchestrator', limit=1)[['dataset_subset', 'adversarial_goal', 'success']].to_string())
except Exception as e:
    print(f"   (No data: {e})")
# ============================================================================
# CELL 10: INGEST AGENT DOJO DATASET
# ============================================================================

agent_dojo_dir = DATA_DIR / "agent_dojo"
agent_dojo_dir.mkdir(parents=True, exist_ok=True)

# Create Table for AgentDojo
# We store the full conversation structure as it is rich with tool calls and 'thinking' tags
cursor.execute("""
    CREATE TABLE IF NOT EXISTS agent_dojo (
        unique_id TEXT PRIMARY KEY,
        suite_name TEXT,
        user_task_id TEXT,
        injection_task_id TEXT, -- If NULL, this is a benign task
        is_attack BOOLEAN,      -- Derived: True if injection_task_id is not None
        success BOOLEAN,        -- Did the agent complete the user task?
        security BOOLEAN,       -- Did the agent prevent the attack? (True = Secure, False = Compromised)
        prompt TEXT,            -- The main user instruction
        thinking_trace TEXT,    -- Extracted <thinking> blocks (CoT)
        full_conversation TEXT, -- Full JSON dump of messages
        source TEXT DEFAULT 'agent_dojo'
    )
""")
conn.commit()

print(f"🔄 Processing AgentDojo data from {agent_dojo_dir}...")

dojo_files = list(agent_dojo_dir.glob("*.jsonl"))
total_dojo_records = 0

for file_path in dojo_files:
    print(f"   Processing {file_path.name}...")
    file_records = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                
                # Skip file headers if present (lines that are just metadata summaries)
                if "messages" not in record:
                    continue
                
                meta = record.get("metadata", {})
                
                # 1. IDs and Labels
                # Format: user_task_0 (benign) or user_task_0_injection_task_2 (attack)
                u_id = meta.get("user_task_id", "unknown")
                i_id = meta.get("injection_task_id") # None if benign
                
                # Unique DB ID
                suite = meta.get("suite_name", "general")
                unique_key = f"dojo_{suite}_{u_id}"
                if i_id:
                    unique_key += f"_{i_id}"
                
                is_attack = i_id is not None
                
                # 2. Extract Prompt (First User Message)
                messages = record.get("messages", [])
                prompt = ""
                for m in messages:
                    if m['role'] == 'user':
                        prompt = m['content']
                        break
                
                # 3. Extract Chain of Thought (<thinking>)
                # AgentDojo models often output <thinking> tags before tool calls
                thinking_steps = []
                for m in messages:
                    if m['role'] == 'assistant' and isinstance(m.get('content'), str):
                        content = m['content']
                        if "<thinking>" in content:
                            # Simple extraction between tags
                            start = content.find("<thinking>") + len("<thinking>")
                            end = content.find("</thinking>")
                            if start != -1 and end != -1:
                                thinking_steps.append(content[start:end].strip())
                
                thinking_str = "\n---\n".join(thinking_steps)

                # 4. Insert
                cursor.execute("""
                    INSERT OR REPLACE INTO agent_dojo VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_key,
                    suite,
                    u_id,
                    str(i_id) if i_id else None,
                    is_attack,
                    meta.get("success", False),
                    meta.get("security", True), # Default to secure if not specified
                    prompt,
                    thinking_str,
                    json.dumps(messages),
                    'agent_dojo'
                ))
                
                file_records += 1
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                # print(f"Error on line: {e}") 
                continue
                
    print(f"     ✓ Added {file_records} records")
    total_dojo_records += file_records

conn.commit()
print(f"✅ AgentDojo ingestion complete. Total records: {total_dojo_records}")
# ============================================================================
# CELL 11: INGEST ATTACKQA DATASET (LOCAL PARQUET)
# ============================================================================

import pandas as pd # Ensure pandas is imported

# Define the local path (adjusted relative to your notebook location)
attackqa_path = DATA_DIR / "attackqa" / "attackqa.parquet"

cursor.execute("""
    CREATE TABLE IF NOT EXISTS attack_qa (
        unique_id TEXT PRIMARY KEY,
        mitre_technique_id TEXT,
        question TEXT,
        thought_trace TEXT,   -- The CoT reasoning
        answer TEXT,          -- The final answer
        context_document TEXT, -- The retrieval snippet (ground truth)
        is_human_question BOOLEAN,
        is_human_answer BOOLEAN,
        source_relation TEXT, -- e.g. relationships_detects
        full_json TEXT,
        source TEXT DEFAULT 'attack_qa'
    )
""")
conn.commit()

print(f"🔄 Loading AttackQA dataset from {attackqa_path}...")

if attackqa_path.exists():
    try:
        # Load directly from local parquet file
        df = pd.read_parquet(attackqa_path)
        
        print(f"   Found {len(df)} records in local parquet file")
        
        records_added = 0
        
        # Iterate over DataFrame rows
        for idx, row in df.iterrows():
            try:
                # Create unique ID
                tech_id = row.get('subject_id') or 'unknown'
                unique_id = f"attackqa_{tech_id}_{idx}"
                
                # Extract fields with safe defaults
                q = row.get('question', '')
                a = row.get('answer', '')
                t = row.get('thought', '') # Use empty string if missing
                doc = row.get('document', '')
                
                # Metadata (handle potential numpy bools by casting)
                human_q = bool(row.get('human_question', False))
                human_a = bool(row.get('human_answer', False))
                
                cursor.execute("""
                    INSERT OR REPLACE INTO attack_qa VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_id,
                    str(tech_id),
                    str(q),
                    str(t) if t else "",
                    str(a),
                    str(doc),
                    human_q,
                    human_a,
                    str(row.get('source', '')),
                    row.to_json(), # Store full row as JSON
                    'attack_qa'
                ))
                
                records_added += 1
                if (idx + 1) % 5000 == 0:
                    print(f"     Processed {idx + 1} rows...")
                    
            except Exception as e:
                # print(f"Error on row {idx}: {e}")
                continue

        conn.commit()
        print(f"✅ AttackQA ingestion complete. Total records: {records_added}")

    except Exception as e:
        print(f"❌ Error loading local parquet file: {e}")
else:
    print(f"❌ File not found: {attackqa_path}")
    print("   Please ensure the parquet file is at ../data/attackqa/attackqa.parquet")
# ============================================================================
# CELL 13: INGEST AGENTHARM BENCHMARK
# ============================================================================

from huggingface_hub import hf_hub_download

agent_harm_dir = DATA_DIR / "agent_harm"
agent_harm_dir.mkdir(parents=True, exist_ok=True)

# Define the files from the 'benchmark' folder in the repo
agent_harm_files = [
    "benign_behaviors_test_public.json",
    "benign_behaviors_validation.json",
    "harmful_behaviors_test_public.json",
    "harmful_behaviors_validation.json",
    # Chat files are smaller aux files, but we include them
    "chat_public_test.json",
    "chat_validation.json"
]

print(f"⬇️ Downloading AgentHarm benchmark files to {agent_harm_dir}...")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS agent_harm (
        unique_id TEXT PRIMARY KEY,
        subset TEXT,          -- benign or harmful
        behavior_id TEXT,
        prompt TEXT,          -- The user instruction
        description TEXT,     -- Detailed task description
        target_tools TEXT,    -- Tools available/required
        source TEXT DEFAULT 'agent_harm'
    )
""")
conn.commit()

# Download and Ingest
total_harm_records = 0

for filename in agent_harm_files:
    try:
        # Download
        local_path = hf_hub_download(
            repo_id="ai-safety-institute/AgentHarm",
            filename=f"benchmark/{filename}",
            repo_type="dataset",
            local_dir=agent_harm_dir,
            local_dir_use_symlinks=False
        )
        
        # Move out of nested 'benchmark' folder if created
        downloaded = Path(local_path)
        final_path = agent_harm_dir / filename
        if downloaded.name == filename and downloaded.parent.name == "benchmark":
            downloaded.rename(final_path)
            # Try to remove empty benchmark dir
            try: downloaded.parent.rmdir() 
            except: pass
        
        # Load JSON
        # Structure: {"behaviors": [ ... ]}
        if final_path.exists():
            with open(final_path, 'r') as f:
                data = json.load(f)
            
            # Determine subset from filename
            subset = "benign" if "benign" in filename else "harmful"
            if "chat" in filename: subset = "chat_" + subset
            
            # Extract list
            items = data.get('behaviors', []) if 'behaviors' in data else data
            
            for item in items:
                # Handle different schemas if chat files differ
                b_id = item.get('id', 'unknown')
                unique_id = f"agentharm_{subset}_{b_id}"
                prompt = item.get('prompt', '') or item.get('behavior', '')
                
                cursor.execute("""
                    INSERT OR REPLACE INTO agent_harm VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_id,
                    subset,
                    str(b_id),
                    prompt,
                    item.get('description', ''),
                    json.dumps(item.get('tools', [])),
                    'agent_harm'
                ))
                total_harm_records += 1
            
            print(f"   ✓ Processed {filename} ({len(items)} records)")
            
    except Exception as e:
        print(f"   ❌ Error processing {filename}: {e}")

conn.commit()
print(f"✅ AgentHarm ingestion complete. Total records: {total_harm_records}")
# ============================================================================
# BONUS: QUICK FILE VALIDATION
# ============================================================================

print("📁 Checking downloaded file structure...\n")

checks = {
    "WebLinX config": (DATA_DIR / "weblinx" / "templates").exists(),
    "WebArena config": (DATA_DIR / "webarena" / "config_files" / "test.raw.json").exists(),
    "WebArena human traces": (DATA_DIR / "webarena" / "human_trajectories").exists() and len(list((DATA_DIR / "webarena" / "human_trajectories").glob("*.trace"))) > 0,
    "WebArena LLM v2": (DATA_DIR / "webarena" / "llm_trajectories_v2").exists() and len(list((DATA_DIR / "webarena" / "llm_trajectories_v2").glob("*/"))) > 0,
    "TAU2 repo": (DATA_DIR / "tau2_repo" / "data" / "tau2" / "domains").exists(),
}

for check_name, exists in checks.items():
    status = "✅" if exists else "⚠️ "
    print(f"{status} {check_name}")

# Count files
human_traces = list((DATA_DIR / "webarena" / "human_trajectories").glob("*.trace"))
llm_models = list((DATA_DIR / "webarena" / "llm_trajectories_v2").glob("*/"))

print(f"\n📊 File Counts:")
print(f"   Human trace directories: {len(human_traces)}")
print(f"   LLM model directories: {len(llm_models)}")

# ============================================================================
# HARMONIZATION: EXPORT TO TRAINING FORMAT
# ============================================================================

TRAINING_DATA_PATH = PROCESSED_DIR / "agent_training_data_unified.json"

def normalize_weblinx(row):
    """Convert WebLINX row to standard format"""
    return {
        "id": row['id'],
        "source": "weblinx",
        "prompt": row['utterances'], # You might need to parse this to get just the instruction
        "trace": row['action_history'], # Already sequential
        "label": 0 # Assume benign for now
    }

def normalize_tau2(row):
    """Convert TAU2 row to standard format"""
    return {
        "id": row['task_id'],
        "source": "tau2",
        "prompt": row['task_instructions'],
        "trace": row['actions_required'], # These are gold-standard actions
        "label": 0 # Assume benign (gold standard)
    }

def normalize_webarena_llm(row):
    """Convert WebArena LLM Trace to standard format"""
    # row keys: trajectory_id, intent, actions, passed
    return {
        "id": row['trajectory_id'],
        "source": "webarena_llm",
        "prompt": row['intent'],
        "trace": row['actions'],
        "label": 1 if row['passed'] == 0 else 0 # FAIL might imply vulnerability or just incompetence. 
        # For adversarial training, you might care specifically about *successful* attacks. 
        # Adjust logic based on whether you want 'failed tasks' or 'successful attacks'.
    }

def normalize_fujitsu(row):
    """
    Convert Fujitsu Red Teaming row to standard format.
    
    Mapping Strategy:
    - prompt: The 'user_query' (this is the input vector containing the attack).
    - trace: The 'target_output' (this shows the system failing/leaking).
    - label: 1 (Adversarial/Unsafe).
    - meta: Store the adversarial goal to understand intent.
    """
    return {
        "id": row['unique_id'],
        "source": f"fujitsu_{row['dataset_subset']}",
        "prompt": row['user_query'],
        "trace": row['target_output'], # The system's response to the attack
        "label": 1, # These are explicitly adversarial/red-teaming examples
        "meta_intent": row['adversarial_goal']
    }
def query_agent_dojo(limit: int = 5) -> pd.DataFrame:
    """Query AgentDojo samples"""
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM agent_dojo LIMIT {limit}", conn)
    conn.close()
    return df
def normalize_agent_dojo(row):
    """
    Convert AgentDojo row to standard format.
    
    Mapping Strategy:
    - prompt: The user's request.
    - trace: The extracted internal monologue (<thinking>) + tool usage.
      (We use the thinking trace here as it provides unique insight into *why* the agent failed)
    - label: 1 if it contains an injection attack, 0 otherwise.
    """
    return {
        "id": row['unique_id'],
        "source": "agent_dojo",
        "prompt": row['prompt'],
        "trace": row['thinking_trace'], # Utilizing the CoT data specifically
        "label": 1 if row['is_attack'] else 0,
        "meta_security": row['security'] # Did the agent survive the attack?
    }
# ============================================================================
# HARMONIZATION: EXPORT TO TRAINING FORMAT
# ============================================================================

def query_attackqa(limit: int = 5) -> pd.DataFrame:
    """Query AttackQA samples"""
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM attack_qa LIMIT {limit}", conn)
    conn.close()
    return df

def normalize_attackqa(row):
    """
    Convert AttackQA row to standard format.
    Label 0: This is factual security knowledge, not an attack.
    """
    trace_combined = f"<thinking>{row['thought_trace']}</thinking>\n{row['answer']}" if row['thought_trace'] else row['answer']
    
    return {
        "id": row['unique_id'],
        "source": "attack_qa",
        "prompt": row['question'],
        "trace": trace_combined,
        "label": 0, 
        "meta_technique": row['mitre_technique_id']
    }
def query_agent_harm(limit: int = 5) -> pd.DataFrame:
    """Query AgentHarm samples"""
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM agent_harm LIMIT {limit}", conn)
    conn.close()
    return df

def normalize_agent_harm(row):
    """
    Convert AgentHarm row to standard format.
    Label 1 for 'harmful' subset, 0 for 'benign'.
    """
    is_harmful = "harmful" in row['subset']
    
    # We use the detailed task description as the 'trace' context
    # This helps the model understand what the prompt actually entails.
    context_trace = f"Task Description: {row['description']}\nTools: {row['target_tools']}"
    
    return {
        "id": row['unique_id'],
        "source": f"agent_harm_{row['subset']}",
        "prompt": row['prompt'],
        "trace": context_trace,
        "label": 1 if is_harmful else 0,
        "meta_subset": row['subset']
    }

# --- EXECUTE EXPORT ---

all_training_data = []

print("🚀 Starting Final Harmonization...")

# 1. WebLINX (Benign UI)
try: all_training_data.extend([normalize_weblinx(r) for _, r in query_weblinx(1000).iterrows()])
except: pass

# 2. TAU2 (Benign Domain)
try: all_training_data.extend([normalize_tau2(r) for _, r in query_tau2('retail', limit=1000).iterrows()])
except: pass

# 3. WebArena (Benign/Mixed)
try: all_training_data.extend([normalize_webarena_llm(r) for _, r in query_webarena_llm(1000).iterrows()])
except: pass

# 4. Fujitsu (Red Teaming / Adversarial)
try: all_training_data.extend([normalize_fujitsu(r) for _, r in query_fujitsu(2000).iterrows()])
except: pass

# 5. AgentDojo (Red Teaming / Injection)
try: all_training_data.extend([normalize_agent_dojo(r) for _, r in query_agent_dojo(2000).iterrows()])
except: pass

# 6. AttackQA (Security Knowledge)
try: all_training_data.extend([normalize_attackqa(r) for _, r in query_attackqa(2000).iterrows()])
except: pass

# 7. AgentHarm (Red Teaming / Intent)
try:
    df_harm = query_agent_harm(limit=2000)
    all_training_data.extend([normalize_agent_harm(r) for _, r in df_harm.iterrows()])
    print(f"   + Added {len(df_harm)} AgentHarm records")
except Exception as e:
    print(f"   ⚠️ Could not export AgentHarm: {e}")

# Save
with open(TRAINING_DATA_PATH, 'w') as f:
    json.dump(all_training_data, f, indent=2)

print(f"\n🎉 SUCCESS: Harmonized {len(all_training_data)} samples into:")
print(f"   {TRAINING_DATA_PATH}")