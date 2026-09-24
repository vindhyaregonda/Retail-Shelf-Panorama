"""shelfpano - panorama stitching for retail shelf photography."""

from .pipeline import PipelineConfig, PipelineReport, run_and_save, stitch_store

__all__ = ["PipelineConfig", "PipelineReport", "stitch_store", "run_and_save"]
__version__ = "1.0.0"
