"""W&B logging helpers for Circuit Breakers.

Goals:
- Centralize W&B init kwargs and metadata capture (Slurm, git, host).
- Keep training code clean and rank-0 safe under Accelerate/DDP.
- Be robust on HPC where W&B may be offline or rate-limited.

This module is intentionally lightweight and only depends on stdlib by default.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_SLURM_KEYS = [
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_ACCOUNT",
    "SLURM_QOS",
    "SLURM_NODELIST",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE",
    "SLURM_SUBMIT_DIR",
]


def wandb_is_available() -> bool:
    try:
        import wandb  # noqa: F401

        return True
    except Exception:
        return False


def parse_tags(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    parts = [p.strip() for p in value.replace(";", ",").split(",")]
    tags = [p for p in parts if p]
    return tags or None


def get_slurm_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key in _SLURM_KEYS:
        if key in os.environ:
            meta[key] = os.environ.get(key)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        meta["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return meta


def _run_git(repo_dir: Path, args: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_git_metadata(repo_dir: Path) -> Dict[str, Any]:
    if not (repo_dir / ".git").exists():
        return {}

    sha = _run_git(repo_dir, ["rev-parse", "HEAD"])
    branch = _run_git(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(repo_dir, ["status", "--porcelain"])

    meta: Dict[str, Any] = {}
    if sha:
        meta["git_sha"] = sha
    if branch:
        meta["git_branch"] = branch
    if status is not None:
        meta["git_dirty"] = bool(status)

    return meta


def get_host_metadata() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def default_run_name(*, base_model: str, total_steps: int, preset: Optional[str] = None) -> str:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    model_short = base_model.split("/")[-1]
    preset_part = preset or "cb"
    name = f"{preset_part}_{model_short}_steps{total_steps}"
    if slurm_job_id:
        name += f"_job{slurm_job_id}"
    return name


def config_to_dict_for_wandb(config: Any) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return {"config": str(config)}


def build_wandb_init_kwargs(
    *,
    run_name: Optional[str],
    group: Optional[str],
    entity: Optional[str],
    tags: Optional[Iterable[str]],
    notes: Optional[str],
    dir_path: Optional[str],
    mode: Optional[str],
) -> Dict[str, Any]:
    """Build kwargs for Accelerate init_trackers(init_kwargs={"wandb": ...})."""
    init: Dict[str, Any] = {}

    if run_name:
        init["name"] = run_name
    if group:
        init["group"] = group
    if entity:
        init["entity"] = entity
    if notes:
        init["notes"] = notes
    if dir_path:
        init["dir"] = dir_path

    tags_list = list(tags) if tags else None
    if tags_list:
        init["tags"] = tags_list

    # Mode is typically controlled via WANDB_MODE env var, but wandb.init supports it.
    if mode:
        init["mode"] = mode

    return {"wandb": init}


def write_wandb_run_ref(output_dir: Path) -> Optional[Path]:
    """Write a small run reference file (main process only)."""
    try:
        import wandb

        run = getattr(wandb, "run", None)
        if run is None:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "wandb_run.json"
        payload = {
            "run_id": run.id,
            "run_name": run.name,
            "project": run.project,
            "entity": run.entity,
            "url": getattr(run, "url", None),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path
    except Exception:
        return None


def log_dir_as_artifact(
    *,
    artifact_name: str,
    artifact_type: str,
    dir_path: Path,
    aliases: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Log a directory (e.g., LoRA checkpoint) as a W&B artifact."""
    try:
        import wandb

        if getattr(wandb, "run", None) is None:
            return False

        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata or {})
        artifact.add_dir(str(dir_path))
        wandb.log_artifact(artifact, aliases=aliases or [])
        return True
    except Exception:
        return False
