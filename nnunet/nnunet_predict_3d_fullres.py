"""
nnU-Net v2 3d_fullres 推理封装。

依赖: pip install nnunetv2
环境变量: nnUNet_results（或通过 --model_folder 指定）
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
from typing import Sequence


def resolve_model_folder(
    *,
    nnunet_results: str | None,
    dataset_id: int | None,
    model_folder: str | None,
    configuration: str = "3d_fullres",
) -> str:
    if model_folder:
        path = os.path.abspath(os.path.expanduser(model_folder))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"model_folder 不存在: {path}")
        return path

    if dataset_id is None:
        raise ValueError("请指定 dataset_id 或 model_folder")

    results = nnunet_results or os.environ.get("nnUNet_results")
    if not results:
        raise ValueError(
            "未设置 nnUNet_results，请 export nnUNet_results=... 或使用 --model_folder"
        )

    pattern = os.path.join(
        results,
        f"Dataset{dataset_id:03d}_*",
        f"nnUNetTrainer__nnUNetPlans__{configuration}",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"未找到模型: {pattern}")
    return matches[-1]


def _parse_configuration(model_folder: str) -> str:
    marker = "nnUNetTrainer__nnUNetPlans__"
    if marker in model_folder:
        tail = model_folder.split(marker, 1)[1]
        return tail.split(os.sep)[0].split("/")[0] or "3d_fullres"
    return "3d_fullres"


def _parse_dataset_id(model_folder: str) -> int | None:
    """从官方路径 .../Dataset502_xxx/... 解析 dataset id；自定义 weight/ 布局返回 None。"""
    for part in model_folder.replace("\\", "/").split("/"):
        if part.startswith("Dataset"):
            try:
                return int(part[4:].split("_")[0])
            except ValueError:
                continue
    return None


def run_predict(
    *,
    input_folder: str,
    output_folder: str,
    model_folder: str,
    folds: str = "all",
    device: str = "cuda",
    cuda_id: int = 0,
    checkpoint: str = "checkpoint_best.pth",
    prefer_cli: bool = True,
) -> None:
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    model_folder = os.path.abspath(model_folder)
    os.makedirs(output_folder, exist_ok=True)

    ds_id = _parse_dataset_id(model_folder)
    config = _parse_configuration(model_folder)

    env = os.environ.copy()
    if device.startswith("cuda"):
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)

    # 自定义 weight/vessel/... 等路径无 Dataset 前缀，必须用 Python API 加载 model_folder
    if prefer_cli and _has_cli() and ds_id is not None:
        cmd = [
            "nnUNetv2_predict",
            "-i",
            input_folder,
            "-o",
            output_folder,
            "-d",
            str(ds_id),
            "-c",
            config,
            "-f",
            folds,
            "-chk",
            checkpoint,
            "-device",
            device,
        ]
        print("执行 (CLI):", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
        return

    if ds_id is None:
        print(
            f"自定义模型路径（无 Dataset 前缀），使用 nnUNetv2 Python API:\n  {model_folder}"
        )
    _run_python_api(
        input_folder=input_folder,
        output_folder=output_folder,
        model_folder=model_folder,
        folds=folds,
        device=device,
        checkpoint=checkpoint,
    )


def _has_cli() -> bool:
    from shutil import which

    return which("nnUNetv2_predict") is not None


def _run_python_api(
    *,
    input_folder: str,
    output_folder: str,
    model_folder: str,
    folds: str,
    device: str,
    checkpoint: str,
) -> None:
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as exc:
        raise ImportError("未找到 nnunetv2，请执行: pip install nnunetv2") from exc

    import torch

    use_folds: str | list[int] = (
        "all" if folds == "all" else [int(f) for f in folds.split(",")]
    )
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=use_folds,
        checkpoint_name=checkpoint,
    )
    inputs = sorted(
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    )
    predictor.predict_from_files(
        [[p] for p in inputs],
        output_folder,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nnU-Net v2 3d_fullres 推理")
    p.add_argument("--dataset_id", type=int, default=None)
    p.add_argument("--model_folder", default=None)
    p.add_argument("--nnunet_results", default=None)
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--input_folder", required=True)
    p.add_argument("--output_folder", required=True)
    p.add_argument("--folds", default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cuda_id", type=int, default=0)
    p.add_argument("--checkpoint", default="checkpoint_final.pth")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_folder = resolve_model_folder(
        nnunet_results=args.nnunet_results,
        dataset_id=args.dataset_id,
        model_folder=args.model_folder,
        configuration=args.configuration,
    )
    print(f"使用模型: {model_folder}")
    run_predict(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_folder=model_folder,
        folds=args.folds,
        device=args.device,
        cuda_id=args.cuda_id,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
