from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from pipeline.config import PipelineConfig
from pipeline.dicom import convert_dicom_to_nifti, standardize_existing_nifti
from pipeline.paths import CasePaths
from pipeline.preprocess import copy_without_crop, crop_lung_ct
from pipeline.restore import restore_task_mask

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_dicom_input(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for root, _dirs, files in os.walk(path):
        for fn in files:
            low = fn.lower()
            if low.endswith(".dcm") or low.endswith(".dicom"):
                return True
            if "." not in fn and not fn.startswith("."):
                return True
        break
    return False


def run_pipeline(
    *,
    input_path: str,
    output_dir: str,
    case_name: str | None = None,
    config: PipelineConfig | None = None,
    config_path: str | None = None,
) -> CasePaths:
    """执行完整分割流水线，返回病例路径对象。"""
    if config is None:
        if config_path:
            config = PipelineConfig.from_yaml(config_path)
        else:
            config = PipelineConfig()

    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)

    if not case_name:
        case_name = os.path.basename(input_path.rstrip(os.sep))
        if case_name.lower().endswith((".nii", ".nii.gz")):
            case_name = os.path.splitext(os.path.splitext(case_name)[0])[0]

    paths = CasePaths(case_name=case_name, output_dir=output_dir)
    paths.ensure_dirs()

    print(f"[1/6] 准备 NIfTI: {case_name}")
    if os.path.isfile(input_path) or (
        os.path.isdir(input_path)
        and not _is_dicom_input(input_path)
        and any(
            f.lower().endswith((".nii", ".nii.gz"))
            for f in os.listdir(input_path)
        )
    ):
        ct_full = standardize_existing_nifti(
            input_path, paths.temp_nii, case_name
        )
    else:
        ct_full = convert_dicom_to_nifti(
            input_path,
            paths.temp_nii,
            case_name=case_name,
            recursive=True,
        )

    print("[2/6] 肺野预处理")
    if config.crop_lung:
        crop_lung_ct(
            ct_full,
            paths.preprocessed_ct_path(),
            paths.crop_indices_path(),
            case_name=case_name,
            margin=config.crop_margin,
            use_lungmask=config.use_lungmask,
        )
    else:
        copy_without_crop(
            ct_full,
            paths.preprocessed_ct_path(),
            paths.crop_indices_path(),
            case_name=case_name,
        )

    infer_dir = paths.prepare_infer_input()

    tasks = {t.lower() for t in config.tasks}

    if "vessel" in tasks:
        print("[3/6] 血管分割 (nnU-Net)")
        from pipeline.nnunet_infer import run_nnunet_task

        run_nnunet_task(
            cfg=config,
            task_cfg=config.vessel,
            input_dir=infer_dir,
            output_mask_path=paths.pred_mask_path("vessel"),
            case_name=case_name,
            weights_subdir="vessel",
        )
        restore_task_mask(
            paths.pred_mask_path("vessel"),
            paths.crop_indices_path(),
            ct_full,
            paths.final_mask_path("vessel"),
        )
    if "lung_lesion" in tasks:
        print("[3/6] 病灶分割 (nnU-Net)")
        from pipeline.nnunet_infer import run_nnunet_task

        run_nnunet_task(
            cfg=config,
            task_cfg=config.lung_lesion,
            input_dir=infer_dir,
            output_mask_path=paths.pred_mask_path("lung_lesion"),
            case_name=case_name,
            weights_subdir="lung_lesion",
        )
        restore_task_mask(
            paths.pred_mask_path("lung_lesion"),
            paths.crop_indices_path(),
            ct_full,
            paths.final_mask_path("lung_lesion"),
        )
        if config.split_lesions:
            from pipeline.lesion_split import split_lesions

            split_lesions(
                paths.final_mask_path("lung_lesion"),
                paths.lesion_lesions_dir(),
                paths.lesion_stats_path(),
                min_voxels=config.min_lesion_voxels,
            )

    if "airway" in tasks:
        from pipeline.airway import run_airway_segmentation, run_nnunet_airway

        nnunet_airway_path: str | None = None
        if config.airway.use_nnunet_fusion and config.airway_nnunet is not None:
            print("[4a/6] 气管分割 (nnU-Net)")
            nnunet_airway_path = run_nnunet_airway(
                paths, config, input_ct_dir=infer_dir
            )
            print(f"  → {nnunet_airway_path}")

        print("[4b/6] 气管分割 (TfeNet) + 与 nnU-Net 并集")
        run_airway_segmentation(
            paths,
            config,
            input_ct_dir=infer_dir,
            nnunet_mask_path=nnunet_airway_path,
        )
        restore_task_mask(
            paths.pred_mask_path("airway"),
            paths.crop_indices_path(),
            ct_full,
            paths.final_mask_path("airway"),
        )
        if config.export_airway_tree:
            from pipeline.airway_tree import export_airway_tree_json

            export_airway_tree_json(
                paths.final_mask_path("airway"),
                paths.airway_tree_path(),
            )

    print(f"[完成] 输出目录: {paths.case_root}")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="肺部 DICOM/NIfTI 一键分割：气道 + 血管 + 病灶",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="DICOM 文件夹、NIfTI 文件或含 NIfTI 的目录",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="输出根目录（其下创建 {case_name}/...）",
    )
    p.add_argument("--case_name", default=None, help="病例名（默认取输入文件夹名）")
    p.add_argument(
        "--config",
        default=None,
        help="YAML 配置文件（模型 dataset_id、任务开关等）",
    )
    p.add_argument(
        "--tasks",
        default=None,
        help="逗号分隔任务列表: airway,vessel,lung_lesion",
    )
    p.add_argument("--cuda_id", type=int, default=None)
    p.add_argument("--no_crop", action="store_true", help="禁用肺野裁剪")
    p.add_argument(
        "--no_airway_tree",
        action="store_true",
        help="不导出气道树 JSON",
    )
    p.add_argument(
        "--no_lesion_split",
        action="store_true",
        help="不拆分独立病灶",
    )
    p.add_argument(
        "--airway_tree",
        action="store_true",
        help="导出气道树 JSON（final/airway_tree/）",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    cfg = PipelineConfig.load(args.config)
    if args.tasks:
        cfg.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.cuda_id is not None:
        cfg.cuda_id = args.cuda_id
    if args.no_crop:
        cfg.crop_lung = False
    if args.no_airway_tree:
        cfg.export_airway_tree = False
    if args.no_lesion_split:
        cfg.split_lesions = False
    if args.airway_tree:
        cfg.export_airway_tree = True

    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        case_name=args.case_name,
        config=cfg,
    )


if __name__ == "__main__":
    main()
