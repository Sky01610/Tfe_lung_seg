from __future__ import annotations

import json
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from pipeline.io_utils import image_meta_dict, write_image


def _lung_mask_hu(arr: np.ndarray) -> np.ndarray:
    """无 lungmask 时的简易肺野估计。"""
    lung = (arr > -1000) & (arr < -200)
    lung = ndimage.binary_closing(lung, iterations=2)
    lung = ndimage.binary_fill_holes(lung)
    labeled, n = ndimage.label(lung)
    if n == 0:
        return lung.astype(np.uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    return (labeled == keep).astype(np.uint8)


def _lung_mask_lungmask(arr: np.ndarray) -> np.ndarray:
    from lungmask import LMInferer

    inferer = LMInferer()
    mask = inferer.apply(arr)
    return (mask > 0).astype(np.uint8)


def compute_crop_box(mask: np.ndarray, margin: int) -> tuple[int, int, int, int, int, int]:
    coords = np.nonzero(mask)
    if coords[0].size == 0:
        z, y, x = mask.shape
        return 0, z, 0, y, 0, x
    zmin = max(int(coords[0].min()) - margin, 0)
    zmax = min(int(coords[0].max()) + margin + 1, mask.shape[0])
    ymin = max(int(coords[1].min()) - margin, 0)
    ymax = min(int(coords[1].max()) + margin + 1, mask.shape[1])
    xmin = max(int(coords[2].min()) - margin, 0)
    xmax = min(int(coords[2].max()) + margin + 1, mask.shape[2])
    return zmin, zmax, ymin, ymax, xmin, xmax


def crop_lung_ct(
    ct_path: str,
    preprocessed_ct_path: str,
    crop_indices_path: str,
    *,
    case_name: str,
    margin: int = 5,
    use_lungmask: bool = True,
) -> dict[str, Any]:
    """
    裁剪肺野并保存 preprocessed CT 与 crop_indices.json。
    crop_box 顺序为 z,y,x（与 SimpleITK GetArray 一致）。
    """
    image = sitk.ReadImage(ct_path)
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    try:
        if use_lungmask:
            lung = _lung_mask_lungmask(arr)
        else:
            lung = _lung_mask_hu(arr)
    except ImportError:
        lung = _lung_mask_hu(arr)

    zmin, zmax, ymin, ymax, xmin, xmax = compute_crop_box(lung, margin)
    size_xyz = [xmax - xmin, ymax - ymin, zmax - zmin]
    index_xyz = [xmin, ymin, zmin]
    out_img = sitk.RegionOfInterest(image, size=size_xyz, index=index_xyz)

    write_image(out_img, preprocessed_ct_path)

    meta = {
        "case_name": case_name,
        "original_shape_zyx": list(arr.shape),
        "crop_box_zyx": {
            "z": [zmin, zmax],
            "y": [ymin, ymax],
            "x": [xmin, xmax],
        },
        "reference_ct": ct_path,
        "preprocessed_ct": preprocessed_ct_path,
        **image_meta_dict(image),
    }
    with open(crop_indices_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def copy_without_crop(
    ct_path: str,
    preprocessed_ct_path: str,
    crop_indices_path: str,
    *,
    case_name: str,
) -> dict[str, Any]:
    """不裁剪时，preprocessed 与原始一致，crop_box 为全图。"""
    image = sitk.ReadImage(ct_path)
    write_image(image, preprocessed_ct_path)
    arr = sitk.GetArrayFromImage(image)
    z, y, x = arr.shape
    meta = {
        "case_name": case_name,
        "original_shape_zyx": [z, y, x],
        "crop_box_zyx": {
            "z": [0, z],
            "y": [0, y],
            "x": [0, x],
        },
        "reference_ct": ct_path,
        "preprocessed_ct": preprocessed_ct_path,
        **image_meta_dict(image),
    }
    with open(crop_indices_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta
