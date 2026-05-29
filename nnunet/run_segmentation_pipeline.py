#!/usr/bin/env python
"""
一键肺部分割流水线入口。

示例（DICOM 文件夹）:
  python run_segmentation_pipeline.py \\
    --input D:/data/dicom/case001 \\
    --output_dir D:/output \\
    --config pipeline_config.yaml

示例（已是 NIfTI）:
  python run_segmentation_pipeline.py \\
    --input D:/data/ct.nii.gz \\
    --output_dir D:/output \\
    --case_name case001
"""
from pipeline.runner import main

if __name__ == "__main__":
    main()
