"""SkillVIS v2 deterministic M1–M3 implementation.

This package is intentionally isolated from :mod:`skillvis`. Importing it does
not alter SkillVIS v1 defaults. Use :func:`build_visualization_intent_ir` with
an explicitly enabled v2 configuration.
"""

from .config import SkillVISV2Config, load_config
from .contracts import (
    AnalyticalTaskResult,
    FieldResolutionResult,
    M1M3PipelineResult,
    QueryNormalizationResult,
    SchemaProfile,
    VisualizationIntentIR,
)
from .pipeline import build_visualization_intent_ir

__all__ = [
    "AnalyticalTaskResult",
    "FieldResolutionResult",
    "M1M3PipelineResult",
    "QueryNormalizationResult",
    "SchemaProfile",
    "SkillVISV2Config",
    "VisualizationIntentIR",
    "build_visualization_intent_ir",
    "load_config",
]
