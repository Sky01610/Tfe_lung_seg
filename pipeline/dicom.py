from __future__ import annotations

import os
import sys

from pipeline.io_utils import copy_as, list_nifti_files, pick_primary_ct

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def convert_dicom_to_nifti(
    dicom_folder: str,
    temp_nii_dir: str,
    *,
    case_name: str,
    recursive: bool = True,
) -> str:
    """DICOM → temp_nii/{case_name}_0000.nii.gz，返回该路径。"""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from dicom_to_nifti_dcm2niix import convert_single

    os.makedirs(temp_nii_dir, exist_ok=True)
    code = convert_single(
        os.path.abspath(dicom_folder),
        temp_nii_dir,
        compress=True,
        search_depth=5 if recursive else 0,
        filename_format="%f_%p_%t_%s",
        bids_sidecar=False,
        gzip_level=None,
        verbose=1,
        dry_run=False,
    )
    if code not in (0, 11):
        raise RuntimeError(f"dcm2niix 转换失败，退出码 {code}")

    niftis = list_nifti_files(temp_nii_dir)
    if not niftis:
        raise FileNotFoundError(f"转换后未在 {temp_nii_dir} 找到 NIfTI")

    primary = pick_primary_ct(niftis)
    target = os.path.join(temp_nii_dir, f"{case_name}_0000.nii.gz")
    if os.path.abspath(primary) != os.path.abspath(target):
        copy_as(primary, target)
        # 清理其它序列，避免混淆
        for p in niftis:
            if os.path.abspath(p) != os.path.abspath(target):
                try:
                    os.remove(p)
                except OSError:
                    pass
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
