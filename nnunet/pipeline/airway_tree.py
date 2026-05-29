from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from pipeline.io_utils import save_json

_TFE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TfeNet")


def export_airway_tree_json(
    airway_mask_path: str,
    output_json_path: str,
    *,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """从气道 mask 导出树形结构 JSON（分支统计，非 LSD 细粒度）。"""
    if _TFE_ROOT not in sys.path:
        sys.path.insert(0, _TFE_ROOT)

    from gt_skeleton_tree_parsing import compute_gt_skeleton_tree
    from utils import load_itk_image

    vol, origin, spacing = load_itk_image(airway_mask_path)
    out = compute_gt_skeleton_tree(vol, threshold=threshold)
    tree = out["tree_parsing"]
    skeleton_parse = out["skeleton_parse"]
    num_branches = int(out["num_branches"])

    branches: list[dict[str, Any]] = []
    for bid in range(1, num_branches + 1):
        branch_mask = (tree == bid).astype(np.uint8)
        skel_mask = (skeleton_parse == bid).astype(np.uint8)
        branches.append(
            {
                "branch_id": bid,
                "voxel_count": int(branch_mask.sum()),
                "skeleton_voxels": int(skel_mask.sum()),
            }
        )

    payload = {
        "source_mask": airway_mask_path,
        "num_branches": num_branches,
        "foreground_voxels": int(out["mask_clean"].sum()),
        "skeleton_voxels": int(out["skeleton"].sum()),
        "origin": origin.tolist() if hasattr(origin, "tolist") else list(origin),
        "spacing": spacing.tolist() if hasattr(spacing, "tolist") else list(spacing),
        "branches": branches,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    save_json(output_json_path, payload)
    return payload
