"""
Utilities for Neural Algorithmic Reasoning implementations.

Provides common training patterns, analysis tools, and dataset utilities.
"""

from .common import (
    standard_arg_parser,
    cross_entropy_loss_fn,
    accuracy_fn,
    TrainingLoop,
    print_model_summary,
    validate_data_shapes,
    setup_reproducibility
)

from .analysis import (
    compute_activation_stats,
    compute_gradient_stats,
    compute_param_stats,
    compute_update_ratios,
    TrainingLogger,
    generate_training_report
)

__all__ = [
    # Common utilities
    'standard_arg_parser',
    'cross_entropy_loss_fn',
    'accuracy_fn',
    'TrainingLoop',
    'print_model_summary',
    'validate_data_shapes',
    'setup_reproducibility',
    
    # Analysis utilities
    'compute_activation_stats',
    'compute_gradient_stats',
    'compute_param_stats',
    'compute_update_ratios',
    'TrainingLogger',
    'generate_training_report',
]
