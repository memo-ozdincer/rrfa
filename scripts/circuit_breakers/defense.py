"""
Agentic Defense Module - Runtime Protection for AI Agents

Provides defense mechanisms that can be integrated into agent frameworks:
1. RepresentationMonitor: Real-time hidden state monitoring during inference
2. ToolCallGuard: Validates tool calls before execution
3. TrajectoryAnalyzer: Analyzes multi-step agent traces for harm
4. CircuitBreakerInference: CB-enhanced inference wrapper

These defenses complement the Circuit Breaker training by providing
runtime protection even for models that haven't been CB-trained.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum

import torch
import torch.nn.functional as F


# =============================================================================
# Defense Configuration
# =============================================================================

class DefenseAction(Enum):
    """Actions the defense system can take."""
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    MODIFY = "modify"
    LOG = "log"


@dataclass
class DefenseConfig:
    """Configuration for runtime defense mechanisms."""

    # Representation monitoring
    enable_rep_monitoring: bool = True
    rep_similarity_threshold: float = 0.7  # Block if cosine sim > threshold
    rep_monitor_layers: List[int] = field(default_factory=lambda: [12, 24])

    # Tool call validation
    enable_tool_guard: bool = True
    blocked_tools: List[str] = field(default_factory=list)
    tool_argument_rules: Dict[str, List[str]] = field(default_factory=dict)

    # Trajectory analysis
    enable_trajectory_analysis: bool = True
    max_trajectory_length: int = 20
    trajectory_risk_threshold: float = 0.5

    # Response validation
    enable_response_validation: bool = True
    max_response_length: int = 4096

    # Logging
    log_all_decisions: bool = True
    log_path: Optional[str] = None


# =============================================================================
# Harmful Pattern Database
# =============================================================================

HARMFUL_TOOL_PATTERNS = {
    'data_exfiltration': [
        (r'send_email.*password', 0.9),
        (r'upload.*credentials', 0.9),
        (r'post.*(secret|key|token)', 0.8),
        (r'transfer.*sensitive', 0.7),
        (r'curl.*\|.*sh', 0.95),  # Piped curl execution
    ],
    'unauthorized_access': [
        (r'sudo\s', 0.6),
        (r'chmod\s+[0-7]*7[0-7]*', 0.7),  # World-writable permissions
        (r'--no-sandbox', 0.8),
        (r'disable.*security', 0.9),
        (r'bypass.*auth', 0.9),
    ],
    'malicious_execution': [
        (r'rm\s+-rf\s+/', 0.95),
        (r'format\s+c:', 0.95),
        (r':()\{\s*:\|:&\s*\};:', 0.99),  # Fork bomb
        (r'eval\([^)]*\)', 0.6),
        (r'exec\([^)]*\)', 0.6),
        (r'__import__', 0.7),
    ],
    'prompt_injection': [
        (r'ignore\s+(all\s+)?previous\s+instructions', 0.95),
        (r'disregard.*above', 0.8),
        (r'new\s+system\s+prompt', 0.9),
        (r'you\s+are\s+now', 0.7),
        (r'jailbreak', 0.9),
    ],
    'network_attacks': [
        (r'nmap\s', 0.7),
        (r'sqlmap', 0.9),
        (r'metasploit', 0.95),
        (r'reverse.*shell', 0.95),
        (r'bind.*shell', 0.95),
    ],
}


# =============================================================================
# Representation Monitor
# =============================================================================

class RepresentationMonitor:
    """
    Monitor hidden state representations during inference.

    Compares the model's current representations against a reference
    (frozen) model to detect if the model is entering a "harmful" state.
    """

    def __init__(
        self,
        reference_reps: Optional[Dict[int, torch.Tensor]] = None,
        threshold: float = 0.7,
        target_layers: List[int] = None,
    ):
        """
        Args:
            reference_reps: Pre-computed reference representations (optional)
            threshold: Similarity threshold for triggering defense
            target_layers: Layers to monitor
        """
        self.reference_reps = reference_reps or {}
        self.threshold = threshold
        self.target_layers = target_layers or [12, 24]
        self.alerts: List[Dict[str, Any]] = []

    def check_representations(
        self,
        current_reps: Dict[int, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[DefenseAction, float, Dict[str, Any]]:
        """
        Check if current representations indicate harmful intent.

        Args:
            current_reps: Current hidden states from model
            attention_mask: Optional attention mask

        Returns:
            (action, max_similarity, details) tuple
        """
        if not self.reference_reps:
            return DefenseAction.ALLOW, 0.0, {"reason": "No reference reps"}

        max_sim = 0.0
        layer_sims = {}

        for layer_idx in self.target_layers:
            if layer_idx not in current_reps or layer_idx not in self.reference_reps:
                continue

            h_current = current_reps[layer_idx]
            h_reference = self.reference_reps[layer_idx]

            # Normalize and compute cosine similarity
            h_current_norm = F.normalize(h_current, p=2, dim=-1)
            h_reference_norm = F.normalize(h_reference, p=2, dim=-1)

            # Mean similarity across sequence
            cos_sim = (h_current_norm * h_reference_norm).sum(dim=-1)

            if attention_mask is not None:
                cos_sim = cos_sim * attention_mask.float()
                layer_sim = cos_sim.sum() / (attention_mask.sum() + 1e-8)
            else:
                layer_sim = cos_sim.mean()

            layer_sim = layer_sim.item()
            layer_sims[layer_idx] = layer_sim
            max_sim = max(max_sim, layer_sim)

        details = {
            "layer_similarities": layer_sims,
            "max_similarity": max_sim,
            "threshold": self.threshold,
        }

        if max_sim > self.threshold:
            self.alerts.append({
                "type": "high_similarity",
                "similarity": max_sim,
                **details
            })
            return DefenseAction.BLOCK, max_sim, details

        return DefenseAction.ALLOW, max_sim, details

    def update_reference(
        self,
        harmful_reps: Dict[int, torch.Tensor],
        weight: float = 0.1,
    ):
        """
        Update reference representations with new harmful examples.

        Uses exponential moving average to incorporate new examples.
        """
        for layer_idx, reps in harmful_reps.items():
            if layer_idx in self.reference_reps:
                self.reference_reps[layer_idx] = (
                    (1 - weight) * self.reference_reps[layer_idx] +
                    weight * reps.detach()
                )
            else:
                self.reference_reps[layer_idx] = reps.detach()


# =============================================================================
# Tool Call Guard
# =============================================================================

class ToolCallGuard:
    """
    Validate tool calls before execution.

    Checks tool names, arguments, and patterns against known harmful patterns.
    """

    def __init__(
        self,
        config: Optional[DefenseConfig] = None,
        custom_patterns: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    ):
        self.config = config or DefenseConfig()
        self.patterns = {**HARMFUL_TOOL_PATTERNS}
        if custom_patterns:
            self.patterns.update(custom_patterns)

        self.blocked_tools = set(self.config.blocked_tools)
        self.decision_log: List[Dict[str, Any]] = []

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[DefenseAction, float, Dict[str, Any]]:
        """
        Validate a tool call before execution.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            (action, risk_score, details) tuple
        """
        # Check blocked tools
        if tool_name.lower() in self.blocked_tools:
            return DefenseAction.BLOCK, 1.0, {"reason": "Tool is blocked"}

        # Serialize for pattern matching
        tool_str = json.dumps({
            "tool": tool_name,
            "arguments": arguments
        }, default=str).lower()

        max_risk = 0.0
        matched_patterns = []

        for category, patterns in self.patterns.items():
            for pattern, risk in patterns:
                if re.search(pattern, tool_str, re.IGNORECASE):
                    max_risk = max(max_risk, risk)
                    matched_patterns.append({
                        "category": category,
                        "pattern": pattern,
                        "risk": risk,
                    })

        details = {
            "tool": tool_name,
            "matched_patterns": matched_patterns,
            "risk_score": max_risk,
        }

        # Log decision
        self.decision_log.append({
            "tool": tool_name,
            "risk_score": max_risk,
            "action": "block" if max_risk > 0.8 else "allow",
            **details
        })

        if max_risk > 0.8:
            return DefenseAction.BLOCK, max_risk, details
        elif max_risk > 0.5:
            return DefenseAction.WARN, max_risk, details
        else:
            return DefenseAction.ALLOW, max_risk, details

    def add_blocked_tool(self, tool_name: str):
        """Add a tool to the blocked list."""
        self.blocked_tools.add(tool_name.lower())

    def add_pattern(self, category: str, pattern: str, risk: float):
        """Add a custom harmful pattern."""
        if category not in self.patterns:
            self.patterns[category] = []
        self.patterns[category].append((pattern, risk))


# =============================================================================
# Trajectory Analyzer
# =============================================================================

@dataclass
class TrajectoryStep:
    """A single step in an agent trajectory."""
    step_idx: int
    action_type: str  # "tool_call", "response", "observation"
    content: Any
    risk_score: float = 0.0
    flags: List[str] = field(default_factory=list)


class TrajectoryAnalyzer:
    """
    Analyze multi-step agent trajectories for harmful patterns.

    Looks for patterns across steps that might indicate:
    - Gradual escalation toward harm
    - Multi-step attacks
    - Reconnaissance before attack
    """

    def __init__(self, config: Optional[DefenseConfig] = None):
        self.config = config or DefenseConfig()
        self.tool_guard = ToolCallGuard(config)
        self.trajectories: Dict[str, List[TrajectoryStep]] = {}

    def add_step(
        self,
        trajectory_id: str,
        step: TrajectoryStep,
    ) -> Tuple[DefenseAction, float, Dict[str, Any]]:
        """
        Add a step to a trajectory and analyze.

        Args:
            trajectory_id: Unique ID for this trajectory
            step: The step to add

        Returns:
            (action, cumulative_risk, details) tuple
        """
        if trajectory_id not in self.trajectories:
            self.trajectories[trajectory_id] = []

        trajectory = self.trajectories[trajectory_id]

        # Check trajectory length limit
        if len(trajectory) >= self.config.max_trajectory_length:
            return DefenseAction.BLOCK, 1.0, {
                "reason": "Trajectory too long",
                "length": len(trajectory)
            }

        # Analyze the step
        if step.action_type == "tool_call" and isinstance(step.content, dict):
            action, risk, details = self.tool_guard.validate_tool_call(
                step.content.get("tool", ""),
                step.content.get("arguments", {})
            )
            step.risk_score = risk
            if action == DefenseAction.BLOCK:
                step.flags.append("blocked_tool_call")

        trajectory.append(step)

        # Analyze trajectory patterns
        return self._analyze_trajectory(trajectory_id)

    def _analyze_trajectory(
        self,
        trajectory_id: str,
    ) -> Tuple[DefenseAction, float, Dict[str, Any]]:
        """Analyze a trajectory for harmful patterns."""
        trajectory = self.trajectories.get(trajectory_id, [])

        if not trajectory:
            return DefenseAction.ALLOW, 0.0, {}

        # Compute cumulative risk
        risks = [s.risk_score for s in trajectory]
        cumulative_risk = sum(risks) / len(risks) if risks else 0.0

        # Check for escalation pattern
        escalation = self._check_escalation(trajectory)

        # Check for reconnaissance pattern
        recon = self._check_reconnaissance(trajectory)

        details = {
            "trajectory_length": len(trajectory),
            "cumulative_risk": cumulative_risk,
            "escalation_detected": escalation,
            "reconnaissance_detected": recon,
            "step_risks": risks,
        }

        # Combine risk factors
        total_risk = cumulative_risk
        if escalation:
            total_risk += 0.2
        if recon:
            total_risk += 0.1

        total_risk = min(1.0, total_risk)

        if total_risk > self.config.trajectory_risk_threshold:
            return DefenseAction.BLOCK, total_risk, details
        elif total_risk > self.config.trajectory_risk_threshold * 0.7:
            return DefenseAction.WARN, total_risk, details

        return DefenseAction.ALLOW, total_risk, details

    def _check_escalation(self, trajectory: List[TrajectoryStep]) -> bool:
        """Check if risk is escalating over time."""
        if len(trajectory) < 3:
            return False

        risks = [s.risk_score for s in trajectory[-5:]]
        # Check if last 3 risks are increasing
        if len(risks) >= 3:
            return risks[-1] > risks[-2] > risks[-3]
        return False

    def _check_reconnaissance(self, trajectory: List[TrajectoryStep]) -> bool:
        """Check for reconnaissance patterns (info gathering before attack)."""
        recon_tools = {"list", "get", "read", "search", "find", "query", "ls", "cat"}
        attack_tools = {"delete", "remove", "execute", "run", "send", "post", "rm"}

        tool_sequence = []
        for step in trajectory:
            if step.action_type == "tool_call" and isinstance(step.content, dict):
                tool = step.content.get("tool", "").lower()
                if any(r in tool for r in recon_tools):
                    tool_sequence.append("recon")
                elif any(a in tool for a in attack_tools):
                    tool_sequence.append("attack")

        # Pattern: recon followed by attack
        for i in range(len(tool_sequence) - 1):
            if tool_sequence[i] == "recon" and tool_sequence[i + 1] == "attack":
                return True

        return False

    def clear_trajectory(self, trajectory_id: str):
        """Clear a trajectory."""
        if trajectory_id in self.trajectories:
            del self.trajectories[trajectory_id]


# =============================================================================
# Circuit Breaker Inference Wrapper
# =============================================================================

class CircuitBreakerInference:
    """
    Wrapper for CB-enhanced inference with runtime defenses.

    Combines:
    1. Representation monitoring during generation
    2. Tool call validation before execution
    3. Response validation after generation
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[DefenseConfig] = None,
        reference_model=None,
    ):
        """
        Args:
            model: The Circuit Breaker trained model
            tokenizer: The tokenizer
            config: Defense configuration
            reference_model: Optional frozen reference for rep monitoring
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DefenseConfig()
        self.reference_model = reference_model

        # Initialize defense components
        self.rep_monitor = RepresentationMonitor(
            threshold=self.config.rep_similarity_threshold,
            target_layers=self.config.rep_monitor_layers,
        )
        self.tool_guard = ToolCallGuard(self.config)
        self.trajectory_analyzer = TrajectoryAnalyzer(self.config)

        self.defense_log: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        trajectory_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate with runtime defense checks.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            trajectory_id: Optional ID for trajectory tracking
            **kwargs: Additional generation arguments

        Returns:
            Dict with 'response', 'blocked', 'defense_actions', etc.
        """
        result = {
            "prompt": prompt,
            "response": "",
            "blocked": False,
            "defense_actions": [],
            "tool_calls": [],
        }

        # Tokenize
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt

        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate with representation extraction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                output_hidden_states=self.config.enable_rep_monitoring,
                return_dict_in_generate=True,
                **kwargs,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        result["response"] = response

        # Validate response length
        if len(response) > self.config.max_response_length:
            result["blocked"] = True
            result["defense_actions"].append({
                "action": "BLOCK",
                "reason": "Response too long",
                "length": len(response),
            })
            result["response"] = response[:self.config.max_response_length] + "..."

        # Extract and validate tool calls
        from .eval import extract_tool_calls
        tool_calls = extract_tool_calls(response)
        result["tool_calls"] = tool_calls

        for tc in tool_calls:
            action, risk, details = self.tool_guard.validate_tool_call(
                tc.get("tool", ""),
                tc.get("arguments", {})
            )
            if action in [DefenseAction.BLOCK, DefenseAction.WARN]:
                result["defense_actions"].append({
                    "action": action.value.upper(),
                    "tool": tc.get("tool"),
                    "risk": risk,
                    **details,
                })
                if action == DefenseAction.BLOCK:
                    result["blocked"] = True

        # Add to trajectory if tracking
        if trajectory_id:
            step = TrajectoryStep(
                step_idx=len(self.trajectory_analyzer.trajectories.get(trajectory_id, [])),
                action_type="response",
                content={"response": response, "tool_calls": tool_calls},
            )
            action, risk, details = self.trajectory_analyzer.add_step(
                trajectory_id, step
            )
            if action == DefenseAction.BLOCK:
                result["blocked"] = True
                result["defense_actions"].append({
                    "action": "BLOCK",
                    "reason": "Trajectory risk threshold exceeded",
                    **details,
                })

        # Log decision
        if self.config.log_all_decisions:
            self.defense_log.append(result)

        return result

    def validate_before_execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a tool call before execution.

        Call this before actually executing any tool call.

        Returns:
            (allowed, details) tuple
        """
        action, risk, details = self.tool_guard.validate_tool_call(
            tool_name, arguments
        )
        allowed = action not in [DefenseAction.BLOCK]
        return allowed, {
            "action": action.value,
            "risk": risk,
            **details,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_defense_system(
    model,
    tokenizer,
    preset: str = "balanced",
) -> CircuitBreakerInference:
    """
    Create a defense system with preset configuration.

    Args:
        model: The language model
        tokenizer: The tokenizer
        preset: One of "strict", "balanced", "permissive"

    Returns:
        CircuitBreakerInference instance
    """
    presets = {
        "strict": DefenseConfig(
            rep_similarity_threshold=0.5,
            trajectory_risk_threshold=0.3,
            max_response_length=2048,
        ),
        "balanced": DefenseConfig(
            rep_similarity_threshold=0.7,
            trajectory_risk_threshold=0.5,
            max_response_length=4096,
        ),
        "permissive": DefenseConfig(
            rep_similarity_threshold=0.9,
            trajectory_risk_threshold=0.7,
            max_response_length=8192,
        ),
    }

    config = presets.get(preset, presets["balanced"])
    return CircuitBreakerInference(model, tokenizer, config)
