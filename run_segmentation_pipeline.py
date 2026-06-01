#!/usr/bin/env python
"""
一键肺部分割流水线入口。

示例（仅需指定输入；配置默认 pipeline_config.yaml）:
  python run_segmentation_pipeline.py --input D:/data/case001/dicom
  python run_segmentation_pipeline.py --input /tmp/pycharm_project_ae6f5a2d/data/262321_30000023090901260965600005223_data/262321_30000023090901260965600005223

  # preprocessed/、final/ 等落在 case001/ 下，与 dicom/ 同级

示例（批量处理同一目录下所有病例子文件夹）:
  python run_segmentation_pipeline.py --input D:/data --batch

示例（已是 NIfTI）:
  python run_segmentation_pipeline.py --input D:/data/ct.nii.gz --case_name case001
"""
from pipeline.runner import main

if __name__ == "__main__":
    main()
