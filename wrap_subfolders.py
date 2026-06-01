"""
为指定目录下的每个一级子文件夹，在其内部增加一层同名目录，并将原有内容移入其中。

变换示例（子文件夹名为 case001）：
  之前: root/case001/file.nii
  之后: root/case001/case001/file.nii

用法：
  python wrap_subfolders.py "D:/data/CT_5_20"
  python wrap_subfolders.py "D:/data/CT_5_20" --dry_run
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def _already_wrapped(sub_path: str, name: str) -> bool:
    """子目录是否已是「仅含同名内层目录」的结构。"""
    inner = os.path.join(sub_path, name)
    if not os.path.isdir(inner):
        return False
    entries = [e for e in os.listdir(sub_path) if e not in (".", "..")]
    return entries == [name]


def wrap_one(sub_path: str, name: str, *, dry_run: bool) -> str | None:
    """
    在 sub_path 下创建 name/name，并把 sub_path 内除该内层目录外的条目移入。
    返回 None 表示成功或无需处理；否则为跳过/失败原因。
    """
    if _already_wrapped(sub_path, name):
        return "已是目标结构，跳过"

    inner = os.path.join(sub_path, name)
    if os.path.exists(inner):
        return f"已存在 {inner}，且外层还有其他内容，请手动处理"

    to_move = [
        e
        for e in os.listdir(sub_path)
        if e not in (".", "..", name)
    ]
    if not to_move:
        return "子文件夹为空，跳过"

    if dry_run:
        print(f"  [dry-run] 将创建 {inner}")
        for e in to_move:
            print(f"  [dry-run] 移动 {os.path.join(sub_path, e)} -> {inner}/")
        return None

    os.makedirs(inner, exist_ok=True)
    for entry in to_move:
        src = os.path.join(sub_path, entry)
        dst = os.path.join(inner, entry)
        if os.path.exists(dst):
            return f"目标已存在，中止: {dst}"
        shutil.move(src, dst)
    return None


def wrap_subfolders(root: str, *, dry_run: bool = False) -> int:
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        print(f"错误: 不是目录: {root}", file=sys.stderr)
        return 1

    subdirs = sorted(
        e
        for e in os.listdir(root)
        if os.path.isdir(os.path.join(root, e)) and not e.startswith(".")
    )
    if not subdirs:
        print(f"未在 {root} 下找到子文件夹。")
        return 0

    failed = 0
    for name in subdirs:
        sub_path = os.path.join(root, name)
        print(f"处理: {name}")
        err = wrap_one(sub_path, name, dry_run=dry_run)
        if err:
            print(f"  跳过: {err}")
            if err.startswith("目标已存在") or err.startswith("已存在"):
                failed += 1
        else:
            print("  完成" if not dry_run else "  将执行（dry-run）")

    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="为每个一级子文件夹在其内部增加一层同名目录，并移入原有内容。"
    )
    parser.add_argument(
        "root",
        help="包含多个子文件夹的根目录",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印将要执行的操作，不实际移动文件",
    )
    args = parser.parse_args(argv)
    return wrap_subfolders(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
