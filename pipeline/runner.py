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
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "pipeline_config.yaml")
_KNOWN_INPUT_SUBDIRS = frozenset(
    {"dicom", "dicoms", "dcm", "images", "image", "nifti", "nii", "scans", "scan"}
)


def _is_known_input_subdir(name: str) -> bool:
    return name.lower() in _KNOWN_INPUT_SUBDIRS


def _find_case_folder(input_path: str) -> str:
    """
    从输入路径向上定位病例目录（preprocessed/、final/ 等应落在此目录）。

    例如 .../case001/dicom/dcm → .../case001
    """
    abs_input = os.path.abspath(input_path.rstrip(os.sep))
    if os.path.isfile(abs_input):
        return os.path.dirname(abs_input)

    current = abs_input
    while True:
        base = os.path.basename(current)
        parent = os.path.dirname(current)
        if not parent or parent == current:
            return current
        if _is_known_input_subdir(base):
            current = parent
            continue
        return current


def _infer_case_name(input_path: str) -> str:
    """推断病例名；输入位于 .../{case}/dicom/... 时取 {case}。"""
    abs_input = os.path.abspath(input_path.rstrip(os.sep))
    if os.path.isfile(abs_input):
        name = os.path.basename(abs_input)
        if name.lower().endswith(".nii.gz"):
            return name[:-7]
        if name.lower().endswith(".nii"):
            return name[:-4]
        return os.path.splitext(name)[0]

    return os.path.basename(_find_case_folder(abs_input))


def _default_output_dir(input_path: str) -> str:
    """输出根目录默认为病例目录的上级（case_root = output_dir / case_name）。"""
    case_folder = _find_case_folder(input_path)
    parent = os.path.dirname(case_folder)
    return parent if parent else case_folder


def _resolve_case_layout(
    input_path: str,
    *,
    output_dir: str | None = None,
    case_name: str | None = None,
) -> tuple[str, str]:
    abs_input = os.path.abspath(input_path.rstrip(os.sep))
    if os.path.isfile(abs_input):
        resolved_case = case_name or _infer_case_name(abs_input)
        resolved_output = output_dir or os.path.dirname(abs_input)
        return os.path.abspath(resolved_output), resolved_case

    case_folder = _find_case_folder(abs_input)
    resolved_case = case_name or os.path.basename(case_folder)
    resolved_output = output_dir or os.path.dirname(case_folder)
    if not resolved_output:
        resolved_output = case_folder
    return os.path.abspath(resolved_output), resolved_case


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


def _uses_dicom_conversion(input_path: str) -> bool:
    """判断是否会走 DICOM 转换分支（而非 NIfTI）。"""
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return False
    if not os.path.isdir(input_path):
        return False
    if _is_dicom_input(input_path):
        return True
    if any(
        f.lower().endswith((".nii", ".nii.gz"))
        for f in os.listdir(input_path)
    ):
        return False
    return True


def _wrap_single_case(input_path: str) -> None:
    """为单个病例目录增加一层同名内层目录（DICOM 输入预处理）。"""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from wrap_subfolders import wrap_one

    case_folder = _find_case_folder(input_path)
    name = os.path.basename(case_folder)
    print(f"[0/6] 整理 DICOM 目录结构: {case_folder}")
    err = wrap_one(case_folder, name, dry_run=False)
    if err and not err.startswith("已是目标结构") and err != "子文件夹为空，跳过":
        raise RuntimeError(f"wrap_subfolders: {err}")


def _wrap_batch_root(batch_root: str) -> None:
    """为批量根目录下每个病例子文件夹增加一层同名内层目录。"""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from wrap_subfolders import wrap_subfolders

    print(f"[0/N] 整理 DICOM 目录结构: {batch_root}")
    code = wrap_subfolders(batch_root)
    if code != 0:
        raise RuntimeError(f"wrap_subfolders 失败，退出码 {code}")


def _batch_should_wrap(batch_root: str) -> bool:
    return any(
        _uses_dicom_conversion(_resolve_batch_case_input(case_dir))
        for case_dir in _list_batch_inputs(batch_root)
    )


def _list_batch_inputs(batch_root: str) -> list[str]:
    """列出批量根目录下的所有病例子文件夹（已排序）。"""
    batch_root = os.path.abspath(batch_root.rstrip(os.sep))
    if not os.path.isdir(batch_root):
        raise NotADirectoryError(f"批量输入目录不存在: {batch_root}")

    inputs: list[str] = []
    for name in sorted(os.listdir(batch_root)):
        if name.startswith("."):
            continue
        path = os.path.join(batch_root, name)
        if os.path.isdir(path):
            inputs.append(path)

    if not inputs:
        raise ValueError(f"未在 {batch_root} 下找到可处理的子文件夹")
    return inputs


def _resolve_batch_case_input(case_dir: str) -> str:
    """若病例目录下存在 dicom 等数据子目录，优先以其作为输入。"""
    for name in sorted(os.listdir(case_dir)):
        if name.startswith("."):
            continue
        if _is_known_input_subdir(name):
            candidate = os.path.join(case_dir, name)
            if os.path.isdir(candidate):
                return candidate
    return case_dir


def run_batch_pipeline(
    *,
    batch_root: str,
    output_dir: str | None = None,
    config: PipelineConfig | None = None,
    config_path: str | None = None,
) -> list[CasePaths]:
    """批量处理 batch_root 下每个子文件夹。"""
    batch_root = os.path.abspath(batch_root.rstrip(os.sep))
    if _batch_should_wrap(batch_root):
        _wrap_batch_root(batch_root)

    case_dirs = _list_batch_inputs(batch_root)
    total = len(case_dirs)
    completed: list[CasePaths] = []
    failures: list[tuple[str, BaseException]] = []

    print(f"批量处理: {batch_root}（共 {total} 个病例）")

    for index, case_dir in enumerate(case_dirs, start=1):
        case_name = os.path.basename(case_dir.rstrip(os.sep))
        input_path = _resolve_batch_case_input(case_dir)
        print(f"\n{'=' * 60}")
        print(f"[批次 {index}/{total}] {case_name}")
        print(f"输入: {input_path}")
        print(f"{'=' * 60}")
        try:
            paths = run_pipeline(
                input_path=input_path,
                output_dir=output_dir,
                case_name=None,
                config=config,
                config_path=config_path,
                skip_dicom_wrap=True,
            )
            completed.append(paths)
        except Exception as exc:
            failures.append((case_name, exc))
            print(f"[失败] {case_name}: {exc}", file=sys.stderr)

    print(f"\n{'=' * 60}")
    print(f"批量完成: 成功 {len(completed)}/{total}")
    for paths in completed:
        print(f"  ✓ {paths.case_name}: {paths.case_root}")
    if failures:
        print(f"失败 {len(failures)}/{total}:", file=sys.stderr)
        for case_name, exc in failures:
            print(f"  ✗ {case_name}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    return completed


def run_pipeline(
    *,
    input_path: str,
    output_dir: str | None = None,
    case_name: str | None = None,
    config: PipelineConfig | None = None,
    config_path: str | None = None,
    skip_dicom_wrap: bool = False,
) -> CasePaths:
    """执行完整分割流水线，返回病例路径对象。"""
    if config is None:
        if config_path:
            config = PipelineConfig.from_yaml(config_path)
        else:
            config = PipelineConfig()

    input_path = os.path.abspath(input_path)
    if not skip_dicom_wrap and _uses_dicom_conversion(input_path):
        _wrap_single_case(input_path)

    output_dir, case_name = _resolve_case_layout(
        input_path, output_dir=output_dir, case_name=case_name
    )

    paths = CasePaths(case_name=case_name, output_dir=output_dir)
    paths.ensure_dirs()

    print(f"病例输出目录: {paths.case_root}")
    if config.uses_cuda():
        print(f"推理设备: {config.device}, cuda_id={config.cuda_id} (PyTorch: {config.torch_device()})")
    else:
        print(f"推理设备: {config.device}")
    print(f"[1/6] 准备 NIfTI (SimpleITK / dicom_to_nii): {case_name}")
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
        print("[4/6] 病灶分割 (nnU-Net)")
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
            print("[5a/6] 气管分割 (nnU-Net)")
            nnunet_airway_path = run_nnunet_airway(
                paths, config, input_ct_dir=infer_dir
            )
            print(f"  → {nnunet_airway_path}")

        print("[5b/6] 气管分割 (TfeNet) + 与 nnU-Net 并集")
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
        help="DICOM 文件夹、NIfTI 文件或含 NIfTI 的目录；配合 --batch 时为含多个病例子文件夹的根目录",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="批量模式：处理 --input 下每个子文件夹为一个病例",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="输出根目录（其下创建 {case_name}/...；默认按输入路径自动推断）",
    )
    p.add_argument(
        "--case_name",
        default=None,
        help="病例名（默认取输入文件夹名；.../{case}/dicom 时取 {case}）",
    )
    p.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
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

    if args.batch:
        if args.case_name:
            print("错误: --batch 模式下不能指定 --case_name", file=sys.stderr)
            raise SystemExit(2)
        run_batch_pipeline(
            batch_root=args.input,
            output_dir=args.output_dir,
            config=cfg,
        )
        return

    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        case_name=args.case_name,
        config=cfg,
    )


if __name__ == "__main__":
    main()
