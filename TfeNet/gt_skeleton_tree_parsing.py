"""
从 GT 二值气道 mask 生成 3D 骨架与树解析（branch ID）体数据。

流程（与 evaluation_metrics.get_parsing 一致）：
  1. 二值化 → 最大连通域 + 填洞
  2. 3D 骨架化
  3. skeleton_parsing：去掉过密邻域点、剔除 <5 体素的小分支
  4. tree_parsing：距离变换，将 mask 内每个体素归到最近骨架分支 ID

输出（默认）：
  output_folder/{原文件名}              — tree_parsing (uint16, 0=背景, 1..N=分支)
  output_folder/skeleton/{原文件名}     — 可选，二值骨架
  output_folder/skeleton_parse/{原文件名} — 可选，骨架分支标签 (uint16)
  output_folder/tree_parsing_summary.csv — 每例统计 + mean/std 汇总
  output_folder/pred_post/{原文件名}     — 可选，预测最大连通域后处理结果

若指定预测目录 folder_pred（与 GT 同名文件），额外计算（与 evaluation_metrics 一致）：
  DLR — GT 骨架被预测覆盖的比例
  DBR — GT 树分支中骨架覆盖率 ≥80% 的分支占比

在 TfeNet 目录下运行:
  python gt_skeleton_tree_parsing.py

或:
  python gt_skeleton_tree_parsing.py --input_folder ... --output_folder ...
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import skimage.measure as measure
from scipy import ndimage

try:
    from skimage.morphology import skeletonize as skeletonize_func
except ImportError:
    from skimage.morphology import skeletonize_3d as skeletonize_func

from utils import load_itk_image, save_itk

EPSILON = 1e-32
BRANCH_DETECT_RATIO = 0.8  # DBR：分支骨架覆盖率阈值（与 evaluation_metrics 一致）
MIN_IOU_FOR_LARGEST_CC = 0.1  # 预测最大连通域 IoU 过低时尝试次大连通域

# =============================================================================
# 直接改路径（无命令行参数时生效）
# =============================================================================
USE_PATHS_BELOW = True
INPUT_FOLDER = r"/tmp/pycharm_project_ae6f5a2d/case_correct/labelsTr"
OUTPUT_FOLDER = r"/tmp/pycharm_project_ae6f5a2d/case_correct/tree_parsing"
FOLDER_PRED = r"/tmp/pycharm_project_ae6f5a2d/predict_result_ATM22/output"  # 留空 "" 则不计算 DLR/DBR
THRESHOLD = 0.0
PRED_THRESHOLD = 0.0
SAVE_SKELETON = False
SAVE_SKELETON_PARSE = False
SAVE_PRED_POST = True
CSV_FILENAME = "tree_parsing_summary.csv"


def large_connected_domain(label: np.ndarray) -> np.ndarray:
    cd, num = measure.label(label, return_num=True, connectivity=1)
    if num == 0:
        return label.astype(np.uint8)
    volume = np.zeros(num)
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    out = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    out = ndimage.binary_fill_holes(out)
    return out.astype(np.uint8)


def skeleton_parsing(skeleton: np.ndarray):
    neighbor_filter = ndimage.generate_binary_structure(3, 3)
    skeleton_filtered = ndimage.convolve(skeleton, neighbor_filter) * skeleton
    skeleton_parse = skeleton.copy()
    skeleton_parse[skeleton_filtered > 3] = 0
    con_filter = ndimage.generate_binary_structure(3, 3)
    cd, num = ndimage.label(skeleton_parse, structure=con_filter)
    for i in range(num):
        a = cd[cd == (i + 1)]
        if a.shape[0] < 5:
            skeleton_parse[cd == (i + 1)] = 0
    cd, num = ndimage.label(skeleton_parse, structure=con_filter)
    return skeleton_parse, cd, num


def tree_parsing_func(skeleton_parse: np.ndarray, label: np.ndarray, cd: np.ndarray) -> np.ndarray:
    _, inds = ndimage.distance_transform_edt(1 - skeleton_parse, return_indices=True)
    tree_parsing = cd[inds[0, ...], inds[1, ...], inds[2, ...]] * label
    return tree_parsing.astype(np.uint16)


def compute_binary_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    intersection = np.sum(y_true * y_pred) + EPSILON
    union = np.sum(y_true) + np.sum(y_pred) - intersection + EPSILON
    return float(intersection / union)


def _squeeze_vol(vol: np.ndarray) -> np.ndarray:
    if len(vol.shape) > 3:
        return vol[0]
    return vol


def postprocess_prediction_largest_cc(
    pred: np.ndarray,
    label: np.ndarray,
    threshold: float = 0.0,
    min_iou: float = MIN_IOU_FOR_LARGEST_CC,
) -> tuple[np.ndarray, float]:
    """预测：最大连通域 + 填洞；IoU 过低时依次尝试次大连通域。"""
    pred = _squeeze_vol(pred)
    label = _squeeze_vol(label)
    pred_bin = (pred > threshold).astype(np.uint8)
    label_bin = (label > threshold).astype(np.uint8)

    cd, num = measure.label(pred_bin, return_num=True, connectivity=2)
    if num == 0:
        return pred_bin, compute_binary_iou(label_bin, pred_bin)

    volume = np.zeros(num)
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    large_cd = ndimage.binary_fill_holes(large_cd).astype(np.uint8)

    iou = compute_binary_iou(label_bin, large_cd)
    flag = -1
    while iou < min_iou:
        if abs(flag) > num:
            break
        large_cd = (cd == (volume_sort[flag - 1] + 1)).astype(np.uint8)
        large_cd = ndimage.binary_fill_holes(large_cd).astype(np.uint8)
        flag -= 1
        iou = compute_binary_iou(label_bin, large_cd)

    return large_cd, float(iou)


def compute_dlr_dbr(
    label: np.ndarray,
    tree_parsing: np.ndarray,
    pred: np.ndarray,
    *,
    gt_threshold: float = 0.0,
    pred_threshold: float = 0.0,
    branch_ratio: float = BRANCH_DETECT_RATIO,
) -> tuple[float, float, float, int, int, np.ndarray]:
    """
  返回: DLR, DBR, IoU(后处理预测 vs GT), detected_branches, num_branches, large_cd
    """
    label = _squeeze_vol(label)
    pred = _squeeze_vol(pred)
    label_bin = (label > gt_threshold).astype(np.uint8)

    skeleton = skeletonize_func(label_bin.astype(bool)).astype(np.uint8)
    skeleton = (skeleton > 0).astype(np.uint8)

    large_cd, iou = postprocess_prediction_largest_cc(
        pred, label, threshold=pred_threshold, min_iou=MIN_IOU_FOR_LARGEST_CC
    )

    sk_sum = int(skeleton.sum())
    dlr = float((large_cd * skeleton).sum() / (sk_sum + EPSILON)) if sk_sum > 0 else float("nan")

    num_branch = int(tree_parsing.max())
    detected_num = 0
    if num_branch > 0:
        for j in range(num_branch):
            branch_label = ((tree_parsing == (j + 1)).astype(np.uint8)) * skeleton
            br_sum = int(branch_label.sum())
            if br_sum > 0 and (large_cd * branch_label).sum() / br_sum >= branch_ratio:
                detected_num += 1

    dbr = float(detected_num / num_branch) if num_branch > 0 else float("nan")
    return dlr, dbr, iou, detected_num, num_branch, large_cd


def compute_gt_skeleton_tree(mask: np.ndarray, threshold: float = 0.0) -> dict[str, np.ndarray | int]:
    """
    输入 mask 数组，返回骨架与树解析相关体数据。
    """
    binary = (mask > threshold).astype(np.uint8)
    mask_clean = large_connected_domain(binary)
    skeleton = skeletonize_func(mask_clean.astype(bool)).astype(np.uint8)
    skeleton_parse, cd, num_branches = skeleton_parsing(skeleton)
    tree_parsing = tree_parsing_func(skeleton_parse, mask_clean, cd)
    return {
        "mask_clean": mask_clean,
        "skeleton": skeleton,
        "skeleton_parse": skeleton_parse.astype(np.uint16),
        "tree_parsing": tree_parsing,
        "num_branches": int(num_branches),
    }


def _list_nifti(folder: str) -> list[str]:
    names = []
    for fn in sorted(os.listdir(folder)):
        if fn.startswith("._"):
            continue
        low = fn.lower()
        if low.endswith(".nii.gz") or low.endswith(".nii"):
            names.append(fn)
    return names


def process_one(
    input_path: str,
    output_folder: str,
    *,
    threshold: float,
    save_skeleton: bool,
    save_skeleton_parse: bool,
    pred_path: str | None = None,
    pred_threshold: float = 0.0,
    save_pred_post: bool = False,
) -> dict[str, object]:
    vol, origin, spacing = load_itk_image(input_path)
    out = compute_gt_skeleton_tree(vol, threshold=threshold)
    fname = os.path.basename(input_path)
    mask_clean = out["mask_clean"]
    skeleton = out["skeleton"]
    tree_parsing = out["tree_parsing"]
    num_branches = int(out["num_branches"])

    os.makedirs(output_folder, exist_ok=True)
    tree_path = os.path.join(output_folder, fname)
    save_itk(tree_parsing, origin, spacing, tree_path)

    skeleton_path = ""
    skeleton_parse_path = ""
    if save_skeleton:
        skel_dir = os.path.join(output_folder, "skeleton")
        os.makedirs(skel_dir, exist_ok=True)
        skeleton_path = os.path.join(skel_dir, fname)
        save_itk(skeleton, origin, spacing, skeleton_path)

    if save_skeleton_parse:
        sp_dir = os.path.join(output_folder, "skeleton_parse")
        os.makedirs(sp_dir, exist_ok=True)
        skeleton_parse_path = os.path.join(sp_dir, fname)
        save_itk(out["skeleton_parse"], origin, spacing, skeleton_parse_path)

    row: dict[str, object] = {
        "filename": fname,
        "status": "ok",
        "error": "",
        "num_branches": num_branches,
        "foreground_voxels": int(mask_clean.sum()),
        "skeleton_voxels": int(skeleton.sum()),
        "tree_label_max": int(tree_parsing.max()),
        "DLR": "",
        "DBR": "",
        "iou": "",
        "detected_branches": "",
        "tree_output_path": tree_path,
        "skeleton_output_path": skeleton_path,
        "skeleton_parse_output_path": skeleton_parse_path,
        "pred_post_path": "",
    }

    if pred_path and os.path.isfile(pred_path):
        pred_vol, _, _ = load_itk_image(pred_path)
        dlr, dbr, iou, detected_num, num_branch_dbr, large_cd = compute_dlr_dbr(
            vol,
            tree_parsing,
            pred_vol,
            gt_threshold=threshold,
            pred_threshold=pred_threshold,
        )
        row["DLR"] = dlr
        row["DBR"] = dbr
        row["iou"] = iou
        row["detected_branches"] = detected_num
        if save_pred_post:
            post_dir = os.path.join(output_folder, "pred_post")
            os.makedirs(post_dir, exist_ok=True)
            post_path = os.path.join(post_dir, fname)
            save_itk(large_cd, origin, spacing, post_path)
            row["pred_post_path"] = post_path
    elif pred_path:
        row["status"] = "ok_no_pred"
        row["error"] = f"pred not found: {pred_path}"

    return row


def write_summary_csv(rows: list[dict[str, object]], csv_path: str) -> None:
    fieldnames = [
        "filename",
        "status",
        "num_branches",
        "foreground_voxels",
        "skeleton_voxels",
        "tree_label_max",
        "DLR",
        "DBR",
        "iou",
        "detected_branches",
        "tree_output_path",
        "skeleton_output_path",
        "skeleton_parse_output_path",
        "pred_post_path",
        "error",
    ]
    ok_rows = [r for r in rows if r.get("status") in ("ok", "ok_no_pred")]
    metric_cols = [
        "num_branches",
        "foreground_voxels",
        "skeleton_voxels",
        "tree_label_max",
        "DLR",
        "DBR",
        "iou",
        "detected_branches",
    ]

    def _mean(col: str) -> float | str:
        if not ok_rows:
            return ""
        vals = []
        for r in ok_rows:
            v = r.get(col, "")
            if v == "" or v is None:
                continue
            vals.append(float(v))
        return float(np.mean(vals)) if vals else ""

    def _std(col: str) -> float | str:
        if not ok_rows:
            return ""
        vals = []
        for r in ok_rows:
            v = r.get(col, "")
            if v == "" or v is None:
                continue
            vals.append(float(v))
        return float(np.std(vals)) if vals else ""

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        if ok_rows:
            mean_row = {"filename": "__MEAN__", "status": "", "error": ""}
            std_row = {"filename": "__STD__", "status": "", "error": ""}
            for col in metric_cols:
                mean_row[col] = _mean(col)
                std_row[col] = _std(col)
            for col in ("tree_output_path", "skeleton_output_path", "skeleton_parse_output_path", "pred_post_path"):
                mean_row[col] = ""
                std_row[col] = ""
            writer.writerow(mean_row)
            writer.writerow(std_row)


def run_batch(
    input_folder: str,
    output_folder: str,
    *,
    threshold: float,
    save_skeleton: bool,
    save_skeleton_parse: bool,
    csv_filename: str = CSV_FILENAME,
    folder_pred: str = "",
    pred_threshold: float = 0.0,
    save_pred_post: bool = False,
) -> int:
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    folder_pred = os.path.abspath(folder_pred) if folder_pred.strip() else ""
    if not os.path.isdir(input_folder):
        print(f"输入目录不存在: {input_folder}", file=sys.stderr)
        return 1
    if folder_pred and not os.path.isdir(folder_pred):
        print(f"预测目录不存在: {folder_pred}", file=sys.stderr)
        return 1

    files = _list_nifti(input_folder)
    if not files:
        print(f"未找到 NIfTI: {input_folder}", file=sys.stderr)
        return 1

    if folder_pred:
        print(f"DLR/DBR: 使用预测目录 {folder_pred}（同名文件配对）")
    else:
        print("未指定预测目录，仅生成 GT 骨架与树解析（不计算 DLR/DBR）")

    rows: list[dict[str, object]] = []
    ok, fail = 0, 0
    for fname in files:
        path = os.path.join(input_folder, fname)
        pred_path = os.path.join(folder_pred, fname) if folder_pred else None
        try:
            row = process_one(
                path,
                output_folder,
                threshold=threshold,
                save_skeleton=save_skeleton,
                save_skeleton_parse=save_skeleton_parse,
                pred_path=pred_path,
                pred_threshold=pred_threshold,
                save_pred_post=save_pred_post and bool(folder_pred),
            )
            rows.append(row)
            msg = f"OK {fname} | branches={row['num_branches']}"
            if row.get("DLR") != "":
                msg += f" | DLR={float(row['DLR']):.4f} DBR={float(row['DBR']):.4f} IoU={float(row['iou']):.4f}"
            print(msg)
            ok += 1
        except Exception as e:
            print(f"FAIL {fname}: {e}", file=sys.stderr)
            rows.append(
                {
                    "filename": fname,
                    "status": "fail",
                    "error": str(e),
                    "num_branches": "",
                    "foreground_voxels": "",
                    "skeleton_voxels": "",
                    "tree_label_max": "",
                    "DLR": "",
                    "DBR": "",
                    "iou": "",
                    "detected_branches": "",
                    "tree_output_path": "",
                    "skeleton_output_path": "",
                    "skeleton_parse_output_path": "",
                    "pred_post_path": "",
                }
            )
            fail += 1

    csv_path = os.path.join(output_folder, csv_filename)
    write_summary_csv(rows, csv_path)
    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    print(f"  NIfTI 输出: {output_folder}")
    print(f"  汇总 CSV:   {csv_path}")
    return 0 if fail == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_folder", required=True, help="GT mask 目录")
    ap.add_argument("--output_folder", required=True, help="树解析输出目录")
    ap.add_argument(
        "--folder_pred",
        default="",
        help="预测 mask 目录（与 GT 同名文件）；指定后计算 DLR/DBR",
    )
    ap.add_argument("--threshold", type=float, default=0.0, help="GT 前景阈值：值 > threshold")
    ap.add_argument("--pred_threshold", type=float, default=0.0, help="预测前景阈值")
    ap.add_argument("--save_skeleton", action="store_true", help="额外保存 skeleton/")
    ap.add_argument("--save_skeleton_parse", action="store_true", help="额外保存 skeleton_parse/")
    ap.add_argument("--save_pred_post", action="store_true", help="保存预测后处理 pred_post/")
    ap.add_argument(
        "--csv_filename",
        default=CSV_FILENAME,
        help=f"汇总 CSV 文件名（保存在 output_folder 下，默认 {CSV_FILENAME}）",
    )
    args = ap.parse_args()
    return run_batch(
        args.input_folder,
        args.output_folder,
        threshold=args.threshold,
        save_skeleton=args.save_skeleton,
        save_skeleton_parse=args.save_skeleton_parse,
        csv_filename=args.csv_filename,
        folder_pred=args.folder_pred,
        pred_threshold=args.pred_threshold,
        save_pred_post=args.save_pred_post,
    )


if __name__ == "__main__":
    if USE_PATHS_BELOW and len(sys.argv) <= 1:
        raise SystemExit(
            run_batch(
                INPUT_FOLDER,
                OUTPUT_FOLDER,
                threshold=THRESHOLD,
                save_skeleton=SAVE_SKELETON,
                save_skeleton_parse=SAVE_SKELETON_PARSE,
                folder_pred=FOLDER_PRED,
                pred_threshold=PRED_THRESHOLD,
                save_pred_post=SAVE_PRED_POST,
            )
        )
    raise SystemExit(main())
