from __future__ import annotations

import os
import shutil
import sys

from pipeline.config import NnUnetTaskConfig, PipelineConfig
from pipeline.io_utils import list_nifti_files, nifti_stem

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_nnunet_task(
    *,
    cfg: PipelineConfig,
    task_cfg: NnUnetTaskConfig,
    input_dir: str,
    output_mask_path: str,
    case_name: str,
    weights_subdir: str | None = None,
) -> str:
    """对 input_dir（含 {case}_0000.nii.gz）推理，写出 output_mask_path。"""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    from nnunet.nnunet_predict_3d_fullres import resolve_model_folder, run_predict

    model_folder = cfg.resolve_nnunet_model_folder(
        task_cfg, weights_subdir=weights_subdir
    )
    tmp_out = output_mask_path + ".__nnunet_tmp__"
    if os.path.isdir(tmp_out):
        shutil.rmtree(tmp_out)
    os.makedirs(tmp_out, exist_ok=True)

    device = cfg.device if cfg.uses_cuda() else "cpu"
    print(f"  nnU-Net 设备: {device}, cuda_id={cfg.cuda_id}")

    run_predict(
        input_folder=input_dir,
        output_folder=tmp_out,
        model_folder=model_folder,
        folds=task_cfg.folds,
        device=device,
        cuda_id=cfg.cuda_id,
    )

    niftis = list_nifti_files(tmp_out)
    if not niftis:
        raise FileNotFoundError(f"nnU-Net 未在 {tmp_out} 产生输出")

    # 匹配病例名（兼容 case / case_0000）
    chosen = None
    for p in niftis:
        stem = nifti_stem(os.path.basename(p))
        if stem == case_name or stem == f"{case_name}_0000":
            chosen = p
            break
    if chosen is None:
        chosen = niftis[0]

    arr_path = _maybe_binarize(chosen, task_cfg)
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    shutil.copy2(arr_path, output_mask_path)
    shutil.rmtree(tmp_out, ignore_errors=True)
    return output_mask_path


def _maybe_binarize(path: str, task_cfg: NnUnetTaskConfig) -> str:
    if not task_cfg.foreground_labels:
        return path

    import SimpleITK as sitk
    import numpy as np

    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    mask = np.zeros_like(arr, dtype=np.uint8)
    for lab in task_cfg.foreground_labels:
        mask[arr == lab] = 1
    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(img)
    tmp = path + ".bin.nii.gz"
    sitk.WriteImage(out, tmp)
    return tmp
