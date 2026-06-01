from __future__ import annotations

import os
from typing import Any

import numpy as np
import SimpleITK as sitk
from skimage import measure

from pipeline.io_utils import save_json


def split_lesions(
    lesion_mask_path: str,
    output_lesions_dir: str,
    stats_path: str,
    *,
    min_voxels: int = 10,
) -> dict[str, Any]:
    """
    将病灶 mask 拆分为独立连通域，写入 lesions/lesion_XXX.nii.gz 与 stats.json。
    """
    os.makedirs(output_lesions_dir, exist_ok=True)
    img = sitk.ReadImage(lesion_mask_path)
    arr = sitk.GetArrayFromImage(img)
    binary = (arr > 0).astype(np.uint8)

    labeled, num = measure.label(binary, return_num=True, connectivity=1)
    lesions: list[dict[str, Any]] = []

    spacing = img.GetSpacing()
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])

    idx = 0
    for lab_id in range(1, num + 1):
        comp = (labeled == lab_id).astype(np.uint8)
        voxels = int(comp.sum())
        if voxels < min_voxels:
            continue
        idx += 1
        name = f"lesion_{idx:03d}.nii.gz"
        out_path = os.path.join(output_lesions_dir, name)
        out_img = sitk.GetImageFromArray(comp)
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, out_path)

        coords = np.argwhere(comp > 0)
        centroid_zyx = coords.mean(axis=0).tolist()
        lesions.append(
            {
                "id": idx,
                "label_id": int(lab_id),
                "filename": name,
                "voxel_count": voxels,
                "volume_mm3": round(voxels * voxel_volume_mm3, 2),
                "centroid_zyx": [round(c, 2) for c in centroid_zyx],
                "bbox_zyx": {
                    "z": [int(coords[:, 0].min()), int(coords[:, 0].max()) + 1],
                    "y": [int(coords[:, 1].min()), int(coords[:, 1].max()) + 1],
                    "x": [int(coords[:, 2].min()), int(coords[:, 2].max()) + 1],
                },
            }
        )

    stats = {
        "source_mask": lesion_mask_path,
        "num_lesions": len(lesions),
        "min_voxels": min_voxels,
        "spacing_xyz": list(spacing),
        "lesions": lesions,
    }
    save_json(stats_path, stats)
    return stats
