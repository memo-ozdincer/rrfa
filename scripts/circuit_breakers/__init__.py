"""
Circuit Breakers Training Module - Agentic Enhanced Version

Implements Representation Rerouting (RR) for training models to refuse harmful
requests while preserving benign capabilities.

AGENTIC ENHANCEMENTS:
1. Completion-based training with loss masking
2. Dual coefficient scheduling (cs/cr)
3. Action-based evaluation for agent tool calls
4. Runtime defense module for agent protection

Usage:
    from scripts.circuit_breakers import CircuitBreakerTrainer, get_config

    config = get_config("llama-4-scout")
    trainer = CircuitBreakerTrainer(config)
    trainer.train()

For runtime defense:
    from scripts.circuit_breakers import create_defense_system

    defense = create_defense_system(model, tokenizer, preset="balanced")
    result = defense.generate(prompt)
"""

from .config import (
    CircuitBreakerConfig,
    CircuitBreakerConfigLlama4Scout,
    CircuitBreakerConfigLlama3_8B,
    CircuitBreakerConfigMistral_7B,
    LoRAConfig,
    get_config,
    config_to_dict,
    CONFIG_PRESETS,
)

from .trainer import (
    CircuitBreakerTrainer,
    RepresentationExtractor,
    reroute_loss,
    retain_loss,
    get_alpha,
    get_dual_coefficients,
    CircuitBreakerDataset,
    create_completion_mask,
    find_assistant_start_position,
)

from .eval import (
    is_refusal,
    refusal_score,
    load_cb_model,
    generate_response,
    evaluate_refusal_rate,
    evaluate_capability,
    evaluate_attack_success_rate,
    run_full_evaluation,
    # Agentic evaluation
    extract_tool_calls,
    is_harmful_tool_call,
    evaluate_agent_safety,
    evaluate_tool_use_capability,
)

from .defense import (
    DefenseConfig,
    DefenseAction,
    RepresentationMonitor,
    ToolCallGuard,
    TrajectoryAnalyzer,
    CircuitBreakerInference,
    create_defense_system,
)

from .unsloth_bridge import (
    UnslothBridgeConfig,
    load_model_for_cb,
    load_with_unsloth,
    load_with_hf,
    prepare_for_cb_training,
    check_hidden_states_support,
    UNSLOTH_AVAILABLE,
)

__all__ = [
    # Config
    'CircuitBreakerConfig',
    'CircuitBreakerConfigLlama4Scout',
    'CircuitBreakerConfigLlama3_8B',
    'CircuitBreakerConfigMistral_7B',
    'LoRAConfig',
    'get_config',
    'config_to_dict',
    'CONFIG_PRESETS',

    # Trainer
    'CircuitBreakerTrainer',
    'RepresentationExtractor',
    'reroute_loss',
    'retain_loss',
    'get_alpha',
    'get_dual_coefficients',
    'CircuitBreakerDataset',
    'create_completion_mask',
    'find_assistant_start_position',

    # Evaluation
    'is_refusal',
    'refusal_score',
    'load_cb_model',
    'generate_response',
    'evaluate_refusal_rate',
    'evaluate_capability',
    'evaluate_attack_success_rate',
    'run_full_evaluation',
    # Agentic evaluation
    'extract_tool_calls',
    'is_harmful_tool_call',
    'evaluate_agent_safety',
    'evaluate_tool_use_capability',

    # Runtime Defense
    'DefenseConfig',
    'DefenseAction',
    'RepresentationMonitor',
    'ToolCallGuard',
    'TrajectoryAnalyzer',
    'CircuitBreakerInference',
    'create_defense_system',

    # Unsloth Bridge
    'UnslothBridgeConfig',
    'load_model_for_cb',
    'load_with_unsloth',
    'load_with_hf',
    'prepare_for_cb_training',
    'check_hidden_states_support',
    'UNSLOTH_AVAILABLE',
]
