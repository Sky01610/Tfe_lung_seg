"""肺部多任务分割统一流水线（DICOM → NIfTI → 推理 → 结构化输出）。"""

from pipeline.runner import run_pipeline

__all__ = ["run_pipeline"]
