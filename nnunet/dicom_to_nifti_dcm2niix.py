"""
使用 PyPI 包 dcm2niix 将 DICOM 转为 NIfTI（.nii / .nii.gz）。

依赖：pip install dcm2niix

示例：
  # 单个 DICOM 目录（含子目录搜索）
  python dicom_to_nifti_dcm2niix.py --input_folder "D:/data/089200/089200_dicom" --output_folder "D:/data/output/089200"

  # 批量：input 下每个子文件夹为一个病例
  python dicom_to_nifti_dcm2niix.py --input_folder "D:/data/dicom" --output_folder "D:/data/nifti" --batch_cases

  # 仅预览将转换的 DICOM（不写出 NIfTI）
  python dicom_to_nifti_dcm2niix.py --input_folder "D:/data/089200/089200_dicom" --dry_run
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import dcm2niix


# dcm2niix 退出码含义（便于日志）
_EXIT_CODE_HINTS: dict[int, str] = {
    0: "成功",
    1: "一般失败（I/O 或解析错误）",
    2: "未找到有效 DICOM",
    4: "发现损坏的 DICOM",
    5: "输入目录不存在或不可访问",
    6: "输出目录不存在或无法创建",
    7: "输出目录只读",
    8: "部分序列成功、部分失败",
    9: "重命名失败",
    10: "体数据不完整（缺层警告）",
    11: "无文件需要转换（可视为正常）",
}


def _is_dicom_dir(path: str) -> bool:
    """目录内是否含有 .dcm 或常见无扩展名 DICOM 文件。"""
    if not os.path.isdir(path):
        return False
    for name in os.listdir(path):
        if name.startswith("."):
            continue
        full = os.path.join(path, name)
        if os.path.isdir(full):
            continue
        lower = name.lower()
        if lower.endswith(".dcm") or lower.endswith(".dicom"):
            return True
        # 许多设备导出的 DICOM 无扩展名
        if "." not in name:
            return True
    return False


def _list_case_dirs(input_root: str) -> list[str]:
    """batch_cases 模式下，列出 input_root 下的一级子目录。"""
    cases: list[str] = []
    for name in sorted(os.listdir(input_root)):
        if name.startswith("."):
            continue
        full = os.path.join(input_root, name)
        if os.path.isdir(full):
            cases.append(full)
    return cases


def build_dcm2niix_args(
    input_dir: str,
    output_dir: str | None,
    *,
    compress: bool,
    search_depth: int,
    filename_format: str,
    bids_sidecar: bool,
    gzip_level: int | None,
    verbose: int,
    dry_run: bool,
) -> list[str]:
    args: list[str] = []

    if compress:
        args.extend(["-z", "y"])
        if gzip_level is not None:
            args.extend(["-" + str(gzip_level)])

    args.extend(["-d", str(search_depth)])
    args.extend(["-f", filename_format])
    args.extend(["-b", "y" if bids_sidecar else "n"])
    if verbose > 0:
        args.extend(["-v", str(verbose)])

    if dry_run:
        args.extend(["-q", "l"])
    else:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            args.extend(["-o", output_dir])

    args.append(input_dir)
    return args


def run_conversion(
    input_dir: str,
    output_dir: str | None,
    *,
    compress: bool,
    search_depth: int,
    filename_format: str,
    bids_sidecar: bool,
    gzip_level: int | None,
    verbose: int,
    dry_run: bool,
) -> int:
    if not os.path.isdir(input_dir):
        print(f"[错误] 输入目录不存在: {input_dir}", file=sys.stderr)
        return 5

    cli_args = build_dcm2niix_args(
        os.path.abspath(input_dir),
        os.path.abspath(output_dir) if output_dir else None,
        compress=compress,
        search_depth=search_depth,
        filename_format=filename_format,
        bids_sidecar=bids_sidecar,
        gzip_level=gzip_level,
        verbose=verbose,
        dry_run=dry_run,
    )

    label = os.path.basename(input_dir.rstrip(os.sep))
    action = "预览" if dry_run else "转换"
    out_msg = output_dir if output_dir else "(与输入同目录)"
    print(f"\n[{action}] {label}")
    print(f"  输入: {input_dir}")
    if not dry_run:
        print(f"  输出: {out_msg}")
    print(f"  命令: dcm2niix {' '.join(cli_args)}")

    result = dcm2niix.main(cli_args, capture_output=True, text=True)
    code = result.returncode

    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)

    hint = _EXIT_CODE_HINTS.get(code, "未知状态")
    if code not in (0, 11):
        print(f"  退出码 {code}: {hint}", file=sys.stderr)
    elif dry_run:
        print(f"  预览完成 (退出码 {code})")
    else:
        print(f"  完成 (退出码 {code}: {hint})")

    return code


def convert_single(
    input_folder: str,
    output_folder: str | None,
    **kwargs,
) -> int:
    return run_conversion(input_folder, output_folder, **kwargs)


def convert_batch_cases(
    input_root: str,
    output_root: str,
    **kwargs,
) -> tuple[int, int, int]:
    """每个一级子目录为一个病例，输出到 output_root/<case_name>/。"""
    case_dirs = _list_case_dirs(input_root)
    if not case_dirs:
        print(f"[错误] 未在 {input_root} 下找到子文件夹", file=sys.stderr)
        return 0, 0, 1

    os.makedirs(output_root, exist_ok=True)
    ok = fail = skip = 0

    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir.rstrip(os.sep))
        if not _is_dicom_dir(case_dir) and kwargs.get("search_depth", 0) == 0:
            print(f"[跳过] 非 DICOM 目录: {case_name}", file=sys.stderr)
            skip += 1
            continue

        out_dir = os.path.join(output_root, case_name)
        code = run_conversion(case_dir, out_dir, **kwargs)
        if code in (0, 11):
            ok += 1
        else:
            fail += 1

    return ok, skip, fail


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 dcm2niix 将 DICOM 转为 NIfTI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        help="DICOM 输入目录；配合 --batch_cases 时为包含多个病例子文件夹的根目录",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        help="NIfTI 输出目录；省略时由 dcm2niix 写入输入目录旁",
    )
    parser.add_argument(
        "--batch_cases",
        action="store_true",
        help="将 input_folder 下每个一级子文件夹作为独立病例转换到 output_folder/<病例名>/",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="在输入目录的子文件夹中搜索 DICOM（对应 dcm2niix -d，默认深度 5）",
    )
    parser.add_argument(
        "--search_depth",
        type=int,
        default=None,
        metavar="N",
        help="DICOM 子目录搜索深度（dcm2niix -d）；默认：--recursive 时为 5，否则为 0",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="输出 .nii.gz（dcm2niix -z y）",
    )
    parser.add_argument(
        "--gzip_level",
        type=int,
        choices=range(1, 10),
        default=None,
        metavar="1-9",
        help="gzip 压缩级别（dcm2niix -1..-9），需配合 --compress",
    )
    parser.add_argument(
        "--filename_format",
        default="%f_%p_%t_%s",
        help="输出文件名模板（dcm2niix -f），%%f=文件夹名, %%p=协议, %%s=序列号等",
    )
    parser.add_argument(
        "--bids",
        action="store_true",
        help="生成 BIDS JSON 侧车文件（dcm2niix -b y，默认不生成）",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="详细日志级别（dcm2niix -v）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅列出将处理的 DICOM，不生成 NIfTI（dcm2niix -q l）",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_folder = os.path.abspath(args.input_folder)
    output_folder = (
        os.path.abspath(args.output_folder) if args.output_folder else None
    )

    if args.batch_cases and not output_folder:
        print("[错误] --batch_cases 必须指定 --output_folder", file=sys.stderr)
        sys.exit(2)

    search_depth = args.search_depth
    if search_depth is None:
        search_depth = 5 if args.recursive else 0

    common = dict(
        compress=args.compress,
        search_depth=search_depth,
        filename_format=args.filename_format,
        bids_sidecar=args.bids,
        gzip_level=args.gzip_level if args.compress else None,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    print(f"dcm2niix 版本: {dcm2niix.__version__}")
    print(f"二进制路径: {dcm2niix.bin_path}")

    if args.batch_cases:
        ok, skip, fail = convert_batch_cases(
            input_folder, output_folder, **common
        )
        action = "预览" if args.dry_run else "转换"
        print(
            f"\n批量{action}完成：成功 {ok}，跳过 {skip}，失败 {fail}。"
        )
        if fail > 0:
            sys.exit(1)
        return

    code = convert_single(input_folder, output_folder, **common)
    if code not in (0, 11):
        sys.exit(1)


if __name__ == "__main__":
    main()
