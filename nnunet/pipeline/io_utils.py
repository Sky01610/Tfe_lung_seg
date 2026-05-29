from __future__ import annotations

import json
import os
import shutil
from typing import Any

import numpy as np
import SimpleITK as sitk


def nifti_stem(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return filename[:-7]
    if lower.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def list_nifti_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    out: list[str] = []
    for fn in sorted(os.listdir(folder)):
        if fn.startswith("._"):
            continue
        low = fn.lower()
        if low.endswith(".nii.gz") or low.endswith(".nii"):
            out.append(os.path.join(folder, fn))
    return out


def pick_primary_ct(nifti_paths: list[str]) -> str:
    """多序列时选体积最大的 NIfTI 作为主体 CT。"""
    if not nifti_paths:
        raise FileNotFoundError("未找到 NIfTI 文件")
    if len(nifti_paths) == 1:
        return nifti_paths[0]

    def score(path: str) -> tuple[int, int]:
        name = os.path.basename(path).lower()
        penalty = 0
        for bad in ("localizer", "scout", "survey", "derived", "secondary"):
            if bad in name:
                penalty -= 10_000_000
        img = sitk.ReadImage(path)
        voxels = int(np.prod(img.GetSize()))
        return penalty + voxels, voxels

    return max(nifti_paths, key=score)


def read_image(path: str) -> sitk.Image:
    return sitk.ReadImage(path)


def write_image(image: sitk.Image, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sitk.WriteImage(image, path)


def to_binary_array(image: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(image)
    return (arr > 0).astype(np.uint8)


def save_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def copy_as(path_src: str, path_dst: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path_dst)), exist_ok=True)
    shutil.copy2(path_src, path_dst)


def image_meta_dict(image: sitk.Image) -> dict[str, Any]:
    return {
        "origin": list(image.GetOrigin()),
        "spacing": list(image.GetSpacing()),
        "direction": list(image.GetDirection()),
        "size_xyz": list(image.GetSize()),
        "size_zyx": list(reversed(image.GetSize())),
    }
