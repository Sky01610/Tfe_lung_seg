from __future__ import annotations

import os
import sys

import SimpleITK as sitk

from pipeline.io_utils import copy_as, list_nifti_files, pick_primary_ct

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_dicom_series_dir(root: str, *, recursive: bool, max_depth: int = 5) -> str:
    """定位含 DICOM 序列的目录；recursive 时在子目录中搜索层数最多的序列。"""
    root = os.path.abspath(root)
    reader = sitk.ImageSeriesReader()

    def best_in_dir(directory: str) -> tuple[str, int]:
        best_dir = directory
        best_count = 0
        for series_id in reader.GetGDCMSeriesIDs(directory) or []:
            count = len(reader.GetGDCMSeriesFileNames(directory, series_id))
            if count > best_count:
                best_count = count
                best_dir = directory
        return best_dir, best_count

    best_dir, best_count = best_in_dir(root)
    if best_count > 0 or not recursive:
        return best_dir if best_count > 0 else root

    for dirpath, dirnames, _ in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames.clear()
            continue
        candidate_dir, count = best_in_dir(dirpath)
        if count > best_count:
            best_dir, best_count = candidate_dir, count

    return best_dir


def convert_dicom_to_nifti(
    dicom_folder: str,
    temp_nii_dir: str,
    *,
    case_name: str,
    recursive: bool = True,
) -> str:
    """DICOM → temp_nii/{case_name}_0000.nii.gz，返回该路径（SimpleITK / dicom_to_nii）。"""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from dicom_to_nii import convert_dicom_to_nifti as sitk_convert_dicom

    os.makedirs(temp_nii_dir, exist_ok=True)
    dicom_dir = _find_dicom_series_dir(dicom_folder, recursive=recursive)
    target = os.path.join(temp_nii_dir, f"{case_name}_0000.nii.gz")

    if dicom_dir != os.path.abspath(dicom_folder):
        print(f"  DICOM 序列目录: {dicom_dir}")

    ok = sitk_convert_dicom(dicom_dir, target)
    if not ok:
        raise RuntimeError(f"DICOM 转 NIfTI 失败: {dicom_dir}")

    return target


def standardize_existing_nifti(
    input_path: str,
    temp_nii_dir: str,
    case_name: str,
) -> str:
    """已有 NIfTI（文件或目录）→ temp_nii/{case_name}_0000.nii.gz。"""
    os.makedirs(temp_nii_dir, exist_ok=True)
    target = os.path.join(temp_nii_dir, f"{case_name}_0000.nii.gz")

    if os.path.isfile(input_path):
        copy_as(input_path, target)
        return target

    if os.path.isdir(input_path):
        niftis = list_nifti_files(input_path)
        primary = pick_primary_ct(niftis)
        copy_as(primary, target)
        return target

    raise FileNotFoundError(f"输入不存在: {input_path}")
