from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import SimpleITK as sitk

from pipeline.io_utils import write_image


def load_crop_meta(crop_indices_path: str) -> dict[str, Any]:
    with open(crop_indices_path, encoding="utf-8") as f:
        return json.load(f)


def paste_mask_to_full_size(
    cropped_mask_path: str,
    reference_ct_path: str,
    crop_meta: dict[str, Any],
    output_path: str,
) -> None:
    """将裁剪区域的分割贴回原始 CT 尺寸。"""
    ref = sitk.ReadImage(reference_ct_path)
    ref_arr = sitk.GetArrayFromImage(ref)
    full = np.zeros(ref_arr.shape, dtype=np.uint8)

    pred = sitk.ReadImage(cropped_mask_path)
    pred_arr = sitk.GetArrayFromImage(pred)
    pred_bin = (pred_arr > 0).astype(np.uint8)

    box = crop_meta["crop_box_zyx"]
    zmin, zmax = box["z"]
    ymin, ymax = box["y"]
    xmin, xmax = box["x"]

    cz, cy, cx = zmax - zmin, ymax - ymin, xmax - xmin
    if pred_bin.shape != (cz, cy, cx):
        raise ValueError(
            f"裁剪 mask 形状 {pred_bin.shape} 与 crop_box {(cz, cy, cx)} 不一致"
        )

    full[zmin:zmax, ymin:ymax, xmin:xmax] = pred_bin

    out = sitk.GetImageFromArray(full)
    out.CopyInformation(ref)
    write_image(out, output_path)


def restore_task_mask(
    cropped_mask_path: str,
    crop_indices_path: str,
    reference_ct_path: str,
    final_mask_path: str,
) -> None:
    if not os.path.isfile(cropped_mask_path):
        raise FileNotFoundError(cropped_mask_path)
    meta = load_crop_meta(crop_indices_path)
    paste_mask_to_full_size(
        cropped_mask_path,
        reference_ct_path,
        meta,
        final_mask_path,
    )
