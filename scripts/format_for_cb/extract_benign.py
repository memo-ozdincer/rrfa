#!/usr/bin/env python3
"""
Extract benign pairs from capability datasets for Circuit Breaker retain set.

Usage:
    python scripts/format_for_cb/extract_benign.py \
        --sources weblinx,webarena,tau2 \
        --output-agent data/circuit_breakers/benign/agent_tools.jsonl \
        --output-general data/circuit_breakers/benign/general_capability.jsonl \
        --target-size 15000
"""

import argparse
import json
import sqlite3
import sys
import random
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.db_utils import get_db_connection


def extract_weblinx_benign(conn: sqlite3.Connection, limit: int = None) -> List[Dict]:
    """Extract benign pairs from WebLINX."""
    print("Extracting WebLINX benign pairs...")

    cursor = conn.cursor()
    query = """
        SELECT id, demo_id, turn_id, utterances, action
        FROM weblinx
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)

    pairs = []
    for row in cursor.fetchall():
        doc_id, demo_id, turn_id, utterances, action = row

        pair = {
            "id": doc_id,
            "source": "weblinx",
            "benign_pair": {
                "user_prompt": utterances,
                "assistant_response": action,
                "task_type": "web_navigation"
            },
            "metadata": {
                "demo_id": demo_id,
                "turn_id": turn_id,
                "category": "general_capability"
            }
        }
        pairs.append(pair)

    print(f"  ✓ Extracted {len(pairs)} WebLINX pairs")
    return pairs


def extract_webarena_benign(conn: sqlite3.Connection, limit: int = None) -> List[Dict]:
    """Extract benign pairs from WebArena (successful trajectories only)."""
    print("Extracting WebArena benign pairs...")

    cursor = conn.cursor()
    query = """
        SELECT trajectory_id, model, task_id, intent,
               observations, actions, passed
        FROM webarena_llm_traces
        WHERE passed = 1
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)

    pairs = []
    for row in cursor.fetchall():
        traj_id, model, task_id, intent, observations, actions, passed = row

        try:
            obs_list = json.loads(observations) if observations else []
            act_list = json.loads(actions) if actions else []

            # Take first observation + first action as a single turn
            if obs_list and act_list:
                user_prompt = f"Task: {intent}\n\nObservation: {obs_list[0]}"
                assistant_response = act_list[0]

                pair = {
                    "id": traj_id,
                    "source": "webarena_llm",
                    "benign_pair": {
                        "user_prompt": user_prompt,
                        "assistant_response": assistant_response,
                        "task_type": "agent_tool_use"
                    },
                    "metadata": {
                        "task_id": task_id,
                        "model": model,
                        "task_success": bool(passed),
                        "category": "agent_capability"
                    }
                }
                pairs.append(pair)
        except json.JSONDecodeError:
            continue

    print(f"  ✓ Extracted {len(pairs)} WebArena pairs")
    return pairs


def extract_tau2_benign(conn: sqlite3.Connection, limit: int = None) -> List[Dict]:
    """Extract benign pairs from TAU2."""
    print("Extracting TAU2 benign pairs...")

    cursor = conn.cursor()
    query = """
        SELECT task_id, domain, task_instructions, actions_required
        FROM tau2_domains
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)

    pairs = []
    for row in cursor.fetchall():
        task_id, domain, instructions, actions = row

        pair = {
            "id": task_id,
            "source": "tau2",
            "benign_pair": {
                "user_prompt": instructions,
                "assistant_response": actions,
                "task_type": "agent_tool_use"
            },
            "metadata": {
                "domain": domain,
                "category": "agent_capability"
            }
        }
        pairs.append(pair)

    print(f"  ✓ Extracted {len(pairs)} TAU2 pairs")
    return pairs


def extract_agentdojo_benign(conn: sqlite3.Connection, limit: int = None) -> List[Dict]:
    """Extract benign pairs from AgentDojo (non-attack samples)."""
    print("Extracting AgentDojo benign pairs...")

    cursor = conn.cursor()
    query = """
        SELECT unique_id, suite_name, prompt, full_conversation, success
        FROM agent_dojo
        WHERE is_attack = 0
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)

    pairs = []
    for row in cursor.fetchall():
        unique_id, suite, prompt, full_convo, success = row

        try:
            messages = json.loads(full_convo)
            # Find first assistant response (tool call or message)
            assistant_response = None

            for msg in messages:
                if msg.get('role') == 'assistant':
                    if msg.get('tool_calls'):
                        assistant_response = json.dumps(msg['tool_calls'][0])
                    elif msg.get('content'):
                        assistant_response = msg['content']
                    if assistant_response:
                        break

            if assistant_response:
                pair = {
                    "id": unique_id,
                    "source": f"agentdojo_{suite}",
                    "benign_pair": {
                        "user_prompt": prompt,
                        "assistant_response": assistant_response,
                        "task_type": "agent_tool_use"
                    },
                    "metadata": {
                        "suite_name": suite,
                        "task_success": bool(success),
                        "category": "agent_capability"
                    }
                }
                pairs.append(pair)
        except json.JSONDecodeError:
            continue

    print(f"  ✓ Extracted {len(pairs)} AgentDojo benign pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Extract benign pairs from capability datasets"
    )
    parser.add_argument(
        '--sources',
        type=str,
        required=True,
        help='Comma-separated list: weblinx,webarena,tau2,agentdojo'
    )
    parser.add_argument(
        '--output-agent',
        type=Path,
        required=True,
        help='Output path for agent-specific benign pairs'
    )
    parser.add_argument(
        '--output-general',
        type=Path,
        required=True,
        help='Output path for general capability pairs'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=15000,
        help='Target total size (will upsample if needed)'
    )
    parser.add_argument(
        '--db',
        type=Path,
        default=Path('data/db/unified.db'),
        help='Path to SQLite database'
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database not found at {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    sources = [s.strip() for s in args.sources.split(',')]

    all_pairs = []

    # Extract from each source
    for source in sources:
        if source == 'weblinx':
            all_pairs.extend(extract_weblinx_benign(conn))
        elif source == 'webarena':
            all_pairs.extend(extract_webarena_benign(conn))
        elif source == 'tau2':
            all_pairs.extend(extract_tau2_benign(conn))
        elif source == 'agentdojo':
            all_pairs.extend(extract_agentdojo_benign(conn))
        else:
            print(f"Warning: Unknown source '{source}', skipping")

    conn.close()

    # Separate agent vs general
    agent_pairs = [p for p in all_pairs if p['metadata']['category'] == 'agent_capability']
    general_pairs = [p for p in all_pairs if p['metadata']['category'] == 'general_capability']

    print(f"\n📊 Total extracted: {len(all_pairs)} pairs")
    print(f"   Agent-specific: {len(agent_pairs)}")
    print(f"   General capability: {len(general_pairs)}")

    # Upsample if needed
    target_agent = args.target_size // 2
    target_general = args.target_size // 2

    if len(agent_pairs) < target_agent:
        print(f"\n⚠️  Upsampling agent pairs from {len(agent_pairs)} to {target_agent}")
        while len(agent_pairs) < target_agent:
            agent_pairs.extend(random.sample(agent_pairs, min(len(agent_pairs), target_agent - len(agent_pairs))))

    if len(general_pairs) < target_general:
        print(f"⚠️  Upsampling general pairs from {len(general_pairs)} to {target_general}")
        while len(general_pairs) < target_general:
            general_pairs.extend(random.sample(general_pairs, min(len(general_pairs), target_general - len(general_pairs))))

    # Write outputs
    with open(args.output_agent, 'w') as f:
        for pair in agent_pairs[:target_agent]:
            f.write(json.dumps(pair) + '\n')

    with open(args.output_general, 'w') as f:
        for pair in general_pairs[:target_general]:
            f.write(json.dumps(pair) + '\n')

    print(f"\n✅ Saved {target_agent} agent pairs to {args.output_agent}")
    print(f"✅ Saved {target_general} general pairs to {args.output_general}")
    print(f"📊 Total benign pairs: {target_agent + target_general}")


if __name__ == '__main__':
    main()
