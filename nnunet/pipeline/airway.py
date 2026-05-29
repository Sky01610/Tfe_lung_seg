from __future__ import annotations

import os
import sys

import numpy as np
import SimpleITK as sitk

from pipeline.config import PipelineConfig
from pipeline.paths import CasePaths

_TFE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TfeNet")


def _ensure_tfenet_path() -> None:
    if _TFE_ROOT not in sys.path:
        sys.path.insert(0, _TFE_ROOT)
    os.chdir(_TFE_ROOT)


def run_nnunet_airway(
    paths: CasePaths,
    cfg: PipelineConfig,
    *,
    input_ct_dir: str,
) -> str:
    """nnU-Net 气管分割 → predictions/airway_nnunet/。"""
    if cfg.airway_nnunet is None:
        raise ValueError("未配置 airway_nnunet，无法运行 nnU-Net 气管分割")

    from pipeline.nnunet_infer import run_nnunet_task

    os.makedirs(paths.pred_task_dir("airway_nnunet"), exist_ok=True)
    out_path = paths.pred_mask_path("airway_nnunet")
    run_nnunet_task(
        cfg=cfg,
        task_cfg=cfg.airway_nnunet,
        input_dir=input_ct_dir,
        output_mask_path=out_path,
        case_name=paths.case_name,
        weights_subdir=cfg.airway_nnunet.weights_subdir or "airway",
    )
    return out_path


def run_airway_segmentation(
    paths: CasePaths,
    cfg: PipelineConfig,
    *,
    input_ct_dir: str,
    nnunet_mask_path: str | None = None,
) -> str:
    """
    TfeNet（全气道 + 细气道）→ 与 nnU-Net 气管并集 → 后处理 → predictions/airway/。

    后处理 IoU 过滤：pred = 融合 mask，reference = nnU-Net 气管（predictions/airway_nnunet/）。
    若已在外部运行 nnU-Net，传入 nnunet_mask_path；否则在 use_nnunet_fusion 时内部调用。
    中间结果：
      predictions/airway_tfenet/TfeNet/{case}.nii.gz
      predictions/airway_tfenet/TfeNetSmall/{case}.nii.gz（若 use_small）
      predictions/airway_tfenet/{case}.nii.gz（两路并集）
    """
    _ensure_tfenet_path()
    from win_dll_fix import prepare_torch_dll_path

    prepare_torch_dll_path()

    from evaluation import network_prediction
    from postprocessing import (
        filter_connected_components_by_iou,
        large_connected_domain,
    )

    tfecfg = cfg.airway
    ckpt_root = tfecfg.checkpoint_root or os.path.join(_TFE_ROOT, "checkpoint")
    ckpt_kw = dict(
        checkpoint_root=ckpt_root,
        checkpoint_dataset=tfecfg.checkpoint_dataset,
    )

    device = cfg.device
    if cfg.device == "cuda":
        device = f"cuda:{cfg.cuda_id}"

    use_fusion = tfecfg.use_nnunet_fusion and cfg.airway_nnunet is not None

    os.makedirs(paths.pred_airway_tfenet_dir, exist_ok=True)
    pred_norm_dir = paths.pred_tfenet_branch_dir("TfeNet")
    pred_small_dir = paths.pred_tfenet_branch_dir("TfeNetSmall")
    os.makedirs(pred_norm_dir, exist_ok=True)

    if tfecfg.use_small:
        os.makedirs(pred_small_dir, exist_ok=True)
        print(f"  TfeNetSmall → {pred_small_dir}")
        network_prediction(
            input_ct_dir,
            pred_small_dir,
            ifsmall=True,
            device=device,
            **ckpt_kw,
        )
    print(f"  TfeNet → {pred_norm_dir}")
    network_prediction(
        input_ct_dir,
        pred_norm_dir,
        ifsmall=False,
        device=device,
        **ckpt_kw,
    )

    tfenet_union = _union_masks(
        pred_norm_dir,
        pred_small_dir,
        paths.case_name,
        tfecfg.use_small,
    )

    tfenet_out = paths.pred_tfenet_union_path()
    sitk.WriteImage(sitk.ReadImage(tfenet_union), tfenet_out)
    print(f"  TfeNet 并集 → {tfenet_out}")
    union = tfenet_out

    if use_fusion:
        if nnunet_mask_path is None:
            nnunet_mask_path = run_nnunet_airway(
                paths, cfg, input_ct_dir=input_ct_dir
            )
        elif not os.path.isfile(nnunet_mask_path):
            raise FileNotFoundError(
                f"nnU-Net 气管 mask 不存在: {nnunet_mask_path}"
            )
        print(f"  融合: TfeNet ∪ nnU-Net → {paths.pred_mask_path('airway')}")
        union = _union_two_files(union, nnunet_mask_path)

    pred_img = sitk.ReadImage(union)
    pred_arr = sitk.GetArrayFromImage(pred_img)

    if (
        cfg.postprocess_airway_iou_with_nnunet
        and use_fusion
        and nnunet_mask_path
        and os.path.isfile(nnunet_mask_path)
    ):
        ref_arr = sitk.GetArrayFromImage(sitk.ReadImage(nnunet_mask_path))
        print(
            "  IoU 过滤: pred=融合 mask (TfeNet∪nnU-Net), "
            f"reference=nnU-Net 气管 ({nnunet_mask_path})"
        )
        pred_arr = filter_connected_components_by_iou(
            pred_arr,
            ref_arr,
            iou_threshold=cfg.postprocess_airway_iou_threshold,
        )
    elif cfg.postprocess_airway_iou_with_nnunet and use_fusion:
        print("  警告: 已开启 IoU 过滤但缺少 nnU-Net 气管 mask，跳过 IoU 过滤")

    if cfg.postprocess_airway_lcc:
        pred_arr = large_connected_domain(pred_arr)

    pred_arr = (pred_arr > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(pred_arr)
    out.CopyInformation(pred_img)
    os.makedirs(paths.pred_task_dir("airway"), exist_ok=True)
    sitk.WriteImage(out, paths.pred_mask_path("airway"))
    return paths.pred_mask_path("airway")


def _find_case_mask(folder: str, case_name: str) -> str | None:
    if not os.path.isdir(folder):
        return None
    for fn in os.listdir(folder):
        if fn.startswith("._"):
            continue
        low = fn.lower()
        if not (low.endswith(".nii.gz") or low.endswith(".nii")):
            continue
        stem = fn.split(".nii")[0]
        if stem == case_name or stem == f"{case_name}_0000":
            return os.path.join(folder, fn)
    return None


def _union_masks(norm_dir: str, small_dir: str, case_name: str, use_small: bool) -> str:
    norm = _find_case_mask(norm_dir, case_name)
    if not norm:
        raise FileNotFoundError(f"TfeNet 输出未找到: {norm_dir}")
    if not use_small:
        return norm
    small = _find_case_mask(small_dir, case_name)
    if not small:
        return norm
    return _union_two_files(norm, small)


def _union_two_files(path_a: str, path_b: str) -> str:
    a = sitk.GetArrayFromImage(sitk.ReadImage(path_a))
    b = sitk.GetArrayFromImage(sitk.ReadImage(path_b))
    if a.shape != b.shape:
        raise ValueError(f"合并 mask 尺寸不一致: {a.shape} vs {b.shape}")
    u = ((a > 0) | (b > 0)).astype(np.uint8)
    out = sitk.GetImageFromArray(u)
    out.CopyInformation(sitk.ReadImage(path_a))
    tmp = path_a + ".union.nii.gz"
    sitk.WriteImage(out, tmp)
    return tmp
