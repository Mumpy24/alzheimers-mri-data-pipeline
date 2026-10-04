"""admri: an auditable, leakage-safe longitudinal MRI data pipeline."""

from admri.config import PipelineConfig
from admri.pipeline import PipelineResult, run_pipeline

__version__ = "0.2.0"

__all__ = ["PipelineConfig", "PipelineResult", "__version__", "run_pipeline"]
