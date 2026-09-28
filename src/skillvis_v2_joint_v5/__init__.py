"""Joint v1.4: auditable mention ownership plus isolated M4 construction."""

from .pipeline import (
    CONFIG_VERSION,
    plan_joint_v5_candidates,
    run_joint_v5_m1_m3_pipeline,
)

__all__ = ["CONFIG_VERSION", "plan_joint_v5_candidates", "run_joint_v5_m1_m3_pipeline"]
