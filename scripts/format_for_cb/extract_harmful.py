#!/usr/bin/env python3
"""
Extract harmful pairs from attack datasets for Circuit Breaker training.

Usage:
    python scripts/format_for_cb/extract_harmful.py \
        --sources fujitsu,agentdojo \
        --output-agent data/circuit_breakers/harmful/fujitsu_agent.jsonl \
        --output-general data/circuit_breakers/harmful/fujitsu_general.jsonl
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.db_utils import get_db_connection


def extract_fujitsu_harmful(
    conn: sqlite3.Connection,
    output_agent: Optional[Path] = None,
    output_general: Optional[Path] = None,
    agent_categories: List[str] = None,
    general_categories: List[str] = None,
    sample_size: Optional[int] = None
) -> Dict[str, int]:
    """
    Extract harmful pairs from Fujitsu dataset.

    Args:
        conn: Database connection
        output_agent: Path for agent-specific harmful pairs (orchestrator)
        output_general: Path for general harmful pairs (RAG poison, direct query)
        agent_categories: List of agent-specific categories (default: ['orchestrator'])
        general_categories: List of general categories (default: ['rag_poison', 'direct_query'])
        sample_size: Max samples to extract per category (None for all)

    Returns:
        Dict with counts: {'agent': N, 'general': M}
    """
    if agent_categories is None:
        agent_categories = ['orchestrator']
    if general_categories is None:
        general_categories = ['rag_poison', 'direct_query']

    counts = {'agent': 0, 'general': 0}

    # Extract agent-specific harmful
    if output_agent:
        print(f"Extracting agent-specific harmful pairs from Fujitsu...")
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(agent_categories))
        query = f"""
            SELECT unique_id, dataset_subset, adversarial_goal,
                   user_query, injection_payload, target_output, success
            FROM fujitsu_red_teaming
            WHERE dataset_subset IN ({placeholders})
              AND success = 1
        """
        if sample_size:
            query += f" LIMIT {sample_size}"

        cursor.execute(query, agent_categories)

        with open(output_agent, 'w') as f:
            for row in cursor.fetchall():
                unique_id, subset, goal, user_query, injection, target_output, success = row

                pair = {
                    "id": unique_id,
                    "source": f"fujitsu_{subset}",
                    "harmful_pair": {
                        "user_prompt": user_query,
                        "assistant_response": target_output,
                        "attack_type": subset,
                        "harm_category": "agent_tool_misuse"
                    },
                    "metadata": {
                        "adversarial_goal": goal,
                        "injection_payload": injection if injection else None,
                        "attack_success": bool(success)
                    }
                }
                f.write(json.dumps(pair) + '\n')
                counts['agent'] += 1

        print(f"  ✓ Extracted {counts['agent']} agent-specific pairs to {output_agent}")

    # Extract general harmful
    if output_general:
        print(f"Extracting general harmful pairs from Fujitsu...")
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(general_categories))
        query = f"""
            SELECT unique_id, dataset_subset, adversarial_goal,
                   user_query, injection_payload, target_output, success
            FROM fujitsu_red_teaming
            WHERE dataset_subset IN ({placeholders})
              AND success = 1
        """
        if sample_size:
            query += f" LIMIT {sample_size}"

        cursor.execute(query, general_categories)

        with open(output_general, 'w') as f:
            for row in cursor.fetchall():
                unique_id, subset, goal, user_query, injection, target_output, success = row

                pair = {
                    "id": unique_id,
                    "source": f"fujitsu_{subset}",
                    "harmful_pair": {
                        "user_prompt": user_query,
                        "assistant_response": target_output,
                        "attack_type": subset,
                        "harm_category": "content_generation"
                    },
                    "metadata": {
                        "adversarial_goal": goal,
                        "injection_payload": injection if injection else None,
                        "attack_success": bool(success)
                    }
                }
                f.write(json.dumps(pair) + '\n')
                counts['general'] += 1

        print(f"  ✓ Extracted {counts['general']} general pairs to {output_general}")

    return counts


def extract_agentdojo_harmful(
    conn: sqlite3.Connection,
    output: Path,
    security_filter: int = 0
) -> int:
    """
    Extract harmful pairs from AgentDojo (attacks that succeeded).

    Args:
        conn: Database connection
        output: Path for output JSONL
        security_filter: Include only rows with this security value (0 = attack succeeded)

    Returns:
        Number of pairs extracted
    """
    print(f"Extracting AgentDojo harmful pairs (security={security_filter})...")

    cursor = conn.cursor()
    query = """
        SELECT unique_id, suite_name, user_task_id, injection_task_id,
               prompt, thinking_trace, full_conversation, success, security
        FROM agent_dojo
        WHERE is_attack = 1
          AND security = ?
    """
    cursor.execute(query, (security_filter,))

    count = 0
    with open(output, 'w') as f:
        for row in cursor.fetchall():
            (unique_id, suite, user_task_id, injection_task_id,
             prompt, thinking, full_convo, success, security) = row

            # Parse conversation to extract harmful tool call
            try:
                messages = json.loads(full_convo)
                harmful_response = None

                # Find first assistant tool call
                for msg in messages:
                    if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                        harmful_response = json.dumps(msg['tool_calls'][0])
                        break

                if not harmful_response:
                    # If no tool call, use the full assistant response
                    for msg in messages:
                        if msg.get('role') == 'assistant' and msg.get('content'):
                            harmful_response = msg['content']
                            break

                if harmful_response:
                    pair = {
                        "id": unique_id,
                        "source": f"agentdojo_{suite}",
                        "harmful_pair": {
                            "user_prompt": prompt,
                            "assistant_response": harmful_response,
                            "attack_type": "function_calling_injection",
                            "harm_category": "agent_tool_misuse"
                        },
                        "metadata": {
                            "user_task_id": user_task_id,
                            "injection_task_id": injection_task_id,
                            "thinking_trace": thinking,
                            "task_success": bool(success),
                            "security_success": bool(security)
                        }
                    }
                    f.write(json.dumps(pair) + '\n')
                    count += 1
            except json.JSONDecodeError:
                print(f"  Warning: Could not parse conversation for {unique_id}")
                continue

    print(f"  ✓ Extracted {count} pairs to {output}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Extract harmful pairs from attack datasets for Circuit Breaker training"
    )
    parser.add_argument(
        '--sources',
        type=str,
        required=True,
        help='Comma-separated list of sources: fujitsu,agentdojo'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for single-source extraction (agentdojo)'
    )
    parser.add_argument(
        '--output-agent',
        type=Path,
        help='Output path for agent-specific harmful pairs (fujitsu)'
    )
    parser.add_argument(
        '--output-general',
        type=Path,
        help='Output path for general harmful pairs (fujitsu)'
    )
    parser.add_argument(
        '--agent-categories',
        type=str,
        default='orchestrator',
        help='Comma-separated list of agent categories (default: orchestrator)'
    )
    parser.add_argument(
        '--general-categories',
        type=str,
        default='rag_poison,direct_query',
        help='Comma-separated list of general categories (default: rag_poison,direct_query)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Max samples per category (None for all)'
    )
    parser.add_argument(
        '--filter',
        type=str,
        help='Filter condition (e.g., security=0 for AgentDojo)'
    )
    parser.add_argument(
        '--db',
        type=Path,
        default=Path('data/db/unified.db'),
        help='Path to SQLite database'
    )

    args = parser.parse_args()

    # Validate database
    if not args.db.exists():
        print(f"Error: Database not found at {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    sources = [s.strip() for s in args.sources.split(',')]

    total_extracted = 0

    for source in sources:
        if source == 'fujitsu':
            if not (args.output_agent or args.output_general):
                print("Error: For fujitsu, provide --output-agent and/or --output-general")
                sys.exit(1)

            agent_cats = [c.strip() for c in args.agent_categories.split(',')]
            general_cats = [c.strip() for c in args.general_categories.split(',')]

            counts = extract_fujitsu_harmful(
                conn,
                output_agent=args.output_agent,
                output_general=args.output_general,
                agent_categories=agent_cats,
                general_categories=general_cats,
                sample_size=args.sample_size
            )
            total_extracted += counts['agent'] + counts['general']

        elif source == 'agentdojo':
            if not args.output:
                print("Error: For agentdojo, provide --output")
                sys.exit(1)

            # Parse filter (e.g., "security=0")
            security_val = 0
            if args.filter:
                parts = args.filter.split('=')
                if len(parts) == 2 and parts[0] == 'security':
                    security_val = int(parts[1])

            count = extract_agentdojo_harmful(conn, args.output, security_val)
            total_extracted += count

        else:
            print(f"Warning: Unknown source '{source}', skipping")

    conn.close()

    print(f"\n✅ Total extracted: {total_extracted} harmful pairs")
    print(f"📊 Ready for Circuit Breaker training")


if __name__ == '__main__':
    main()
