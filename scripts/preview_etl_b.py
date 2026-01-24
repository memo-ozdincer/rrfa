#!/usr/bin/env python3
"""
Preview ETL_B outputs with human-readable prints.

This script reads trace_v1 JSONL, runs the ETL_B rendering + masking logic
in-memory, and prints key parts of:
- trace_v1
- render_v1 (rendered view + alignment)
- lossmask_v1 (mask stats + spans)

Optionally, it can also load existing render_v1 / lossmask_v1 JSONL files
and print their summaries for eyeballing.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.schemas.trace import Trace
from src.schemas.lossmask import LossMask
from src.schemas.tools import ETL_B as etl_b


# =============================================================================
# IO helpers
# =============================================================================

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _truncate(text: str, max_chars: int = 220) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


# =============================================================================
# Pretty printers
# =============================================================================

def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_trace(trace: Trace, max_chars: int) -> None:
    _print_header(f"TRACE {trace.id}")
    print(f"source={trace.source.dataset}  tier={trace.tier}  completeness={trace.completeness}")
    if trace.labels:
        print(f"labels={trace.labels}")
    if trace.training:
        print(f"training.loss_mask_policy={trace.training.loss_mask_policy}")
        print(f"training.sample_weight={trace.training.sample_weight}")
        if trace.training.mixture:
            print(f"training.mixture.class_id={trace.training.mixture.class_id}")
    print("messages:")
    for i, msg in enumerate(trace.messages):
        content = _truncate(msg.content, max_chars)
        tool_calls = f" tool_calls={len(msg.tool_calls)}" if msg.tool_calls else ""
        name = f" name={msg.name}" if msg.name else ""
        print(f"  [{i}] role={msg.role}{name}{tool_calls} :: {content}")


def _print_render_summary(render, tokenizer, print_rendered_text: bool, max_tokens: int) -> None:
    _print_header("RENDER SUMMARY")
    print(f"render_id={render.render_id}")
    print(f"tokenizer_id={render.tokenizer_id}")
    print(f"input_ids_len={len(render.input_ids)} attention_mask_len={len(render.attention_mask)}")
    print(f"render_options={render.render_options}")

    if print_rendered_text and render.rendered_text:
        _print_header("RENDERED TEXT (TRUNCATED)")
        print(_truncate(render.rendered_text, max_chars=1200))

    if render.alignment:
        _print_header("ALIGNMENT")
        if render.alignment.message_spans:
            print("message_spans:")
            for span in render.alignment.message_spans:
                print(f"  role={span.role} idx={span.message_index} tokens=[{span.token_start}, {span.token_end})")
        if render.alignment.assistant_spans:
            print("assistant_spans:")
            for span in render.alignment.assistant_spans:
                print(f"  idx={span.message_index} tokens=[{span.token_start}, {span.token_end})")
        if render.alignment.tool_call_spans:
            print("tool_call_spans:")
            for span in render.alignment.tool_call_spans:
                print(
                    f"  msg_idx={span.message_index} call_idx={span.call_index} tool={span.tool_name} "
                    f"tokens=[{span.token_start}, {span.token_end}) name_token_end={span.name_token_end}"
                )

    if render.signals:
        _print_header("SIGNALS")
        if render.signals.injection_spans:
            print("injection_spans:")
            for span in render.signals.injection_spans:
                print(f"  tokens=[{span.token_start}, {span.token_end}) method={span.detection_method}")
        if render.signals.action_commitments:
            print("action_commitments:")
            for ac in render.signals.action_commitments:
                print(
                    f"  commit_idx={ac.commitment_token_idx} type={ac.commit_type} "
                    f"tool={ac.committed_tool} prefix_len={len(ac.prefix_token_indices) if ac.prefix_token_indices else 0}"
                )
        if render.signals.detector_metadata:
            dm = render.signals.detector_metadata
            print(f"detector_metadata: shock={dm.shock_detector_id} commitment={dm.commitment_detector_id}")

    _print_header("TOKEN PREVIEW")
    preview_ids = render.input_ids[:max_tokens]
    preview_text = tokenizer.decode(preview_ids, skip_special_tokens=False)
    print(f"input_ids[:{max_tokens}]={preview_ids}")
    print(f"decoded[:{max_tokens}]={_truncate(preview_text, max_chars=800)}")


def _mask_spans(mask: List[float]) -> List[Tuple[int, int]]:
    spans = []
    in_span = False
    start = 0
    for i, m in enumerate(mask):
        if m > 0 and not in_span:
            start = i
            in_span = True
        elif m <= 0 and in_span:
            spans.append((start, i))
            in_span = False
    if in_span:
        spans.append((start, len(mask)))
    return spans


def _print_lossmask(lossmask: LossMask, max_tokens: int) -> None:
    _print_header("LOSSMASK SUMMARY")
    print(f"lossmask_id={lossmask.lossmask_id}")
    print(f"policy_id={lossmask.policy_id} policy_version={lossmask.policy_version}")
    print(f"sample_weight={lossmask.sample_weight}")
    if lossmask.stats:
        print(
            f"stats: total={lossmask.stats.total_tokens} "
            f"masked={lossmask.stats.masked_tokens} "
            f"unmasked={lossmask.stats.unmasked_tokens} "
            f"ratio={lossmask.stats.mask_ratio:.4f}"
        )

    spans = _mask_spans(lossmask.loss_mask)
    print(f"nonzero_mask_spans (count={len(spans)}): {spans[:50]}")

    _print_header("LOSSMASK PREVIEW")
    print(f"loss_mask[:{max_tokens}]={lossmask.loss_mask[:max_tokens]}")
    if lossmask.labels:
        print(f"labels[:{max_tokens}]={lossmask.labels[:max_tokens]}")


def _print_jsonl_summary(path: Path, kind: str, limit: int) -> None:
    _print_header(f"{kind.upper()} JSONL SUMMARY: {path}")
    for idx, row in enumerate(_iter_jsonl(path), start=1):
        if idx > limit:
            break
        if kind == "render":
            print(
                f"[{idx}] render_id={row.get('render_id')} trace_id={row.get('trace_id')} "
                f"input_len={len(row.get('input_ids', []))}"
            )
        elif kind == "lossmask":
            stats = row.get("stats", {})
            print(
                f"[{idx}] lossmask_id={row.get('lossmask_id')} trace_id={row.get('trace_id')} "
                f"policy_id={row.get('policy_id')} ratio={stats.get('mask_ratio')}"
            )
        else:
            print(f"[{idx}] keys={list(row.keys())}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Preview ETL_B rendering + lossmasks.")
    parser.add_argument("--traces", type=Path, required=True, help="Input trace_v1 JSONL")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Chat template string or path to template file",
    )
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--limit", type=int, default=3, help="How many traces to preview")
    parser.add_argument("--max-chars", type=int, default=220, help="Max chars per message")
    parser.add_argument("--max-tokens", type=int, default=120, help="Max tokens to preview")
    parser.add_argument("--add-generation-prompt", action="store_true", help="Append generation prompt")
    parser.add_argument("--print-rendered-text", action="store_true", help="Print rendered text preview")
    parser.add_argument("--lmp-registry", type=Path, default=None, help="Path to LMP registry JSON")
    parser.add_argument("--policy-override", type=str, default=None, help="Override policy ID")
    parser.add_argument("--mwcs-registry", type=Path, default=None, help="Path to MWCS registry JSON")
    parser.add_argument("--mwcs-schedule", type=str, default=None, help="MWCS schedule ID")
    parser.add_argument("--mwcs-step", type=int, default=None, help="Training step for curriculum")
    parser.add_argument("--lmp-schedule", type=Path, default=None, help="Path to MWCS schedule YAML")
    parser.add_argument("--step", type=int, default=None, help="Training step for YAML schedule")
    parser.add_argument("--allow-skeleton", action="store_true", help="Allow skeleton traces")
    parser.add_argument("--skeleton-policy", type=str, default="full_sequence", help="Policy for skeleton traces")
    parser.add_argument("--render-jsonl", type=Path, default=None, help="Existing render_v1 JSONL to inspect")
    parser.add_argument("--lossmask-jsonl", type=Path, default=None, help="Existing lossmask_v1 JSONL to inspect")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    chat_template = args.chat_template
    if chat_template and Path(chat_template).exists():
        chat_template = Path(chat_template).read_text()
    lmp_registry = etl_b._load_lmp_registry_safe(args.lmp_registry)

    mwcs_registry = None
    if args.mwcs_registry and args.mwcs_registry.exists():
        try:
            mwcs_registry = etl_b.load_mwcs_registry(args.mwcs_registry)
        except Exception:
            mwcs_registry = None
    else:
        try:
            mwcs_registry = etl_b.load_mwcs_registry()
        except Exception:
            mwcs_registry = None

    if args.render_jsonl:
        _print_jsonl_summary(args.render_jsonl, "render", args.limit)
    if args.lossmask_jsonl:
        _print_jsonl_summary(args.lossmask_jsonl, "lossmask", args.limit)

    for idx, row in enumerate(_iter_jsonl(args.traces), start=1):
        if args.limit is not None and idx > args.limit:
            break

        trace = Trace.from_dict(row)
        is_skeleton = (
            getattr(trace, "tier", None) == "B1" or
            getattr(trace, "completeness", None) == "skeleton"
        )

        if is_skeleton and not args.allow_skeleton:
            print(f"Skipping skeleton trace {trace.id}. Use --allow-skeleton to preview.")
            continue

        _print_trace(trace, args.max_chars)

        render = etl_b.render_trace(
            trace,
            tokenizer,
            max_length=args.max_length,
            add_generation_prompt=args.add_generation_prompt,
            include_rendered_text=args.print_rendered_text,
            chat_template=chat_template,
        )

        policy_override = args.policy_override
        if is_skeleton and args.allow_skeleton:
            policy_override = args.skeleton_policy

        schedule_step = args.step if args.step is not None else args.mwcs_step
        sample_weight, lmp_override = etl_b._apply_mwcs_weight_with_yaml(
            trace,
            args.lmp_schedule,
            schedule_step,
        )
        if args.lmp_schedule is None:
            sample_weight, lmp_override = etl_b._apply_mwcs_weight(
                trace,
                mwcs_registry,
                args.mwcs_schedule,
                schedule_step,
            )

        if lmp_override and policy_override is None:
            policy_override = lmp_override

        policy_id, policy = etl_b._resolve_policy(trace, lmp_registry, policy_override)
        mask_values = etl_b._apply_lmp_policy(render, policy)

        lossmask = LossMask.from_render(
            render,
            policy_id=policy_id,
            mask_fn=lambda _: mask_values,
            policy_version=lmp_registry.version,
            policy_params=policy.params,
            sample_weight=sample_weight,
        )

        _print_render_summary(render, tokenizer, args.print_rendered_text, args.max_tokens)
        _print_lossmask(lossmask, args.max_tokens)


if __name__ == "__main__":
    main()
