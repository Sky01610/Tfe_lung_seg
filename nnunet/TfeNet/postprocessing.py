import SimpleITK as sitk
import numpy as np
import os
import skimage.measure as measure
from scipy import ndimage
import json
from typing import Optional

EPSILON = 1e-32
IOU_THRESHOLD = 0.3
# reference 膨胀迭代次数（体素），用于限定连通域分析范围
REF_DILATION_ITERS = 3
# ROI 包围盒在膨胀 mask 外的额外边距（体素）
ROI_MARGIN = 2


def _nifti_stem(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return filename[:-7]
    if lower.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def _output_exists(out_path: str) -> bool:
    """目标文件已存在则跳过，避免重复后处理。"""
    return os.path.isfile(out_path)


def _find_nifti_path(directory: str, stem: str) -> Optional[str]:
    """按病例名在目录中查找 .nii / .nii.gz。"""
    for ext in (".nii.gz", ".nii"):
        path = os.path.join(directory, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def _to_binary_mask(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(np.uint8)


def compute_binary_iou(y_true, y_pred):
    intersection     = np.sum(y_true * y_pred) + EPSILON
    union = np.sum(y_true) + np.sum(y_pred) - intersection + EPSILON
    iou = intersection / union
    return iou


def _build_reference_roi(ref_bin: np.ndarray, dilation_iters: int, conn: int) -> np.ndarray:
    """对 reference 做二值膨胀，得到允许分析 pred 连通域的 ROI。"""
    if dilation_iters <= 0:
        return ref_bin.astype(bool)
    structure = ndimage.generate_binary_structure(3, conn)
    return ndimage.binary_dilation(
        ref_bin.astype(bool), iterations=dilation_iters, structure=structure
    )


def _bbox_slices_from_mask(mask: np.ndarray, margin: int):
    """返回 mask 非零区域（加 margin）的 slice 元组；无前景时返回 None。"""
    coords = np.nonzero(mask)
    if coords[0].size == 0:
        return None
    slices = []
    for axis in range(mask.ndim):
        lo = max(int(coords[axis].min()) - margin, 0)
        hi = min(int(coords[axis].max()) + margin + 1, mask.shape[axis])
        slices.append(slice(lo, hi))
    return tuple(slices)


def _component_iou_vs_reference(
    cd: np.ndarray, ref_crop: np.ndarray, num: int
) -> np.ndarray:
    """向量化计算每个连通域标签与 reference 的 IoU，下标 0 为背景。"""
    labels_flat = cd.ravel()
    ref_flat = ref_crop.ravel().astype(np.float64)
    valid = labels_flat > 0
    if not np.any(valid):
        return np.zeros(num + 1, dtype=np.float64)

    lab = labels_flat[valid]
    inter = np.bincount(lab, weights=ref_flat[valid], minlength=num + 1)
    comp_sizes = np.bincount(lab, minlength=num + 1)
    ref_sum = ref_flat.sum()
    union = comp_sizes + ref_sum - inter + EPSILON
    iou = inter / union
    iou[0] = 0.0
    return iou


def filter_connected_components_by_iou(
    label,
    reference,
    iou_threshold: float = IOU_THRESHOLD,
    conn: int = 1,
    dilation_iters: int = REF_DILATION_ITERS,
    roi_margin: int = ROI_MARGIN,
):
    """
    仅保留与 reference IoU > iou_threshold 的连通域。
    先在 reference 膨胀 ROI 的包围盒内做连通域分析，避免全图逐域扫描。
    """
    label_bin = _to_binary_mask(label)
    ref_bin = _to_binary_mask(reference)
    if label_bin.shape != ref_bin.shape:
        raise ValueError(
            f"pred/reference shape mismatch: {label_bin.shape} vs {ref_bin.shape}"
        )

    roi = _build_reference_roi(ref_bin, dilation_iters, conn)
    if not roi.any():
        print("  warning: empty reference ROI, skip IoU filter")
        return label_bin.astype("float")

    bbox = _bbox_slices_from_mask(roi, roi_margin)
    if bbox is None:
        return label_bin.astype("float")

    label_crop = label_bin[bbox]
    ref_crop = ref_bin[bbox]
    roi_crop = roi[bbox]

    cd, num = measure.label(label_crop, return_num=True, connectivity=conn)
    if num == 0:
        return np.zeros_like(label_bin, dtype="float")

    iou_per_label = _component_iou_vs_reference(cd, ref_crop, num)

    labels_flat = cd.ravel()
    roi_flat = roi_crop.ravel()
    valid = labels_flat > 0
    touches_roi = np.bincount(
        labels_flat[valid],
        weights=roi_flat[valid].astype(np.float64),
        minlength=num + 1,
    ) > 0

    keep_ids = np.flatnonzero(touches_roi & (iou_per_label > iou_threshold))
    keep_ids = keep_ids[keep_ids > 0]
    if keep_ids.size == 0:
        print(
            f"  warning: no component with IoU > {iou_threshold} near reference ROI"
        )
        return np.zeros_like(label_bin, dtype="float")

    filtered_crop = np.isin(cd, keep_ids).astype(np.uint8)
    filtered = np.zeros_like(label_bin, dtype=np.uint8)
    filtered[bbox] = filtered_crop
    return filtered.astype("float")


def large_connected_domain(label, conn=1):
    label_bin = _to_binary_mask(label)
    cd, num = measure.label(label_bin, return_num=True, connectivity=conn)
    if num == 0:
        return label_bin.astype("float")
    counts = np.bincount(cd.ravel())
    largest_id = int(np.argmax(counts[1:]) + 1)
    large_cd = (cd == largest_id).astype(np.uint8)
    large_cd = ndimage.binary_fill_holes(large_cd)

    # iou = compute_binary_iou(label, large_cd)
    # print(iou)

    # flag=-1
    # while iou < 0.1:
    #     print(" failed cases, require find next large connected component")
    #     large_cd = (cd == (volume_sort[flag-1] + 1)).astype(np.uint8)
    #     flag -= 1
    #     iou = compute_binary_iou(label, large_cd)

    return large_cd.astype('float')

def postprocess(
    root,
    save_root,
    reference_root=None,
    iou_threshold: float = IOU_THRESHOLD,
    dilation_iters: int = REF_DILATION_ITERS,
    roi_margin: int = ROI_MARGIN,
):
    print("postprocess begin")
    if reference_root is not None:
        print(
            f"IoU filter enabled: reference={reference_root}, "
            f"threshold>{iou_threshold}, dilation_iters={dilation_iters}, "
            f"roi_margin={roi_margin}"
        )
    os.makedirs(save_root, exist_ok=True)
    name_list = os.listdir(root)
    labels_names = [file for file in name_list if '.nii' in file]
    name_list.sort()
    ok, skipped = 0, 0
    for label_name in labels_names:
        name = _nifti_stem(label_name)
        out_path = os.path.join(save_root, name + ".nii.gz")
        if _output_exists(out_path):
            print("skip (exists):", name)
            skipped += 1
            continue
        print('post processing on:', name)
        pred = sitk.ReadImage(os.path.join(root, label_name))
        pred_img = sitk.GetArrayFromImage(pred)
        if reference_root is not None:
            ref_path = _find_nifti_path(reference_root, name)
            if ref_path is None:
                raise FileNotFoundError(
                    f"reference mask not found for {name} under {reference_root}"
                )
            ref = sitk.ReadImage(ref_path)
            ref_img = sitk.GetArrayFromImage(ref)
            pred_img = filter_connected_components_by_iou(
                pred_img,
                ref_img,
                iou_threshold=iou_threshold,
                dilation_iters=dilation_iters,
                roi_margin=roi_margin,
            )
        pred_img = large_connected_domain(pred_img)

        pred_save = sitk.GetImageFromArray(pred_img)
        pred_save.SetOrigin(pred.GetOrigin())
        pred_save.SetDirection(pred.GetDirection())
        pred_save.SetSpacing(pred.GetSpacing())

    
        sitk.WriteImage(pred_save, out_path)
        ok += 1
    print(f"postprocess end: 新处理 {ok} 个, 跳过(已存在) {skipped} 个")

def back_original_size(result_root, save_root):
    ori_root = './data/imagesVal'
    # result_root = './result/test_post'
    # save_root = './result/test_orisize'
    file_path = './data/lung_bbox_val_dict.json'
    name_list = os.listdir(result_root)
    with open(file_path, 'r') as file:
        pos_dic = json.load(file)

    os.makedirs(save_root, exist_ok=True)
    name_list.sort()
    ok, skipped = 0, 0
    for name in name_list[:]:
        out_path = os.path.join(save_root, name)
        if _output_exists(out_path):
            print("skip (exists):", name)
            skipped += 1
            continue
        print(name)
        image = sitk.ReadImage(os.path.join(ori_root, name))
        array = sitk.GetArrayFromImage(image)
        result = np.zeros_like(array)
        pred = sitk.ReadImage(os.path.join(result_root, name))
        pred = sitk.GetArrayFromImage(pred)
        pos = pos_dic[name]
        zmin, zmax, ymin, ymax, xmin, xmax = pos
        shape = pred.shape
        result[zmin:zmin+shape[0], ymin:ymin+shape[1], xmin:xmin+shape[2]] = pred
        result_image = sitk.GetImageFromArray(result.astype(np.byte))
        result_image.SetOrigin(image.GetOrigin())
        result_image.SetDirection(image.GetDirection())
        result_image.SetSpacing(image.GetSpacing())
        sitk.WriteImage(result_image, out_path)
        ok += 1
    print(f"back_original_size end: 新处理 {ok} 个, 跳过(已存在) {skipped} 个")

def check_meta(name):
    my_root = './result/test_orisize'
    root = '/home/tangwen/Documents/2022_experiment_hospital/ATM2022/temp/'
    img1 = sitk.ReadImage(os.path.join(my_root, name))
    img2 = sitk.ReadImage(os.path.join(root, name))
    print(img1.GetOrigin(), img2.GetOrigin())
    print(img1.GetDirection(), img2.GetDirection())
    print(img1.GetSpacing(), img2.GetSpacing())
    arr1 = sitk.GetArrayFromImage(img1)
    arr2 = sitk.GetArrayFromImage(img2)
    print(arr1.shape, arr2.shape)
    print(arr1.dtype, arr2.dtype)
    cd1, num1 = measure.label(arr1, return_num=True, connectivity=1)
    cd2, num2 = measure.label(arr2, return_num=True, connectivity=1)
    print(num1, num2)

def merge_multi_result(folder_names, save_folder):
    root = './result'
    os.makedirs(save_folder, exist_ok=True)
    file_list = os.listdir(os.path.join(root, folder_names[0]))
    file_list = [f for f in file_list if f.endswith('.npy')]
    ok, skipped = 0, 0
    for file in file_list:
        out_path = os.path.join(save_folder, file.replace('.npy', '.nii.gz'))
        if _output_exists(out_path):
            print("skip (exists):", file)
            skipped += 1
            continue
        print(file)
        pred = np.load(os.path.join(root, folder_names[0], file))
        for name in folder_names[1:]:
            _pred = np.load(os.path.join(root, name, file))
            pred = pred + _pred
        pred = pred / (len(folder_names))
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        image = sitk.GetImageFromArray(pred.astype(np.byte))
        sitk.WriteImage(image, out_path)
        ok += 1
    print(f"merge_multi_result end: 新处理 {ok} 个, 跳过(已存在) {skipped} 个")


if __name__ == '__main__':
    # reference_root: 另一路分割目录（如 nnU-Net / TfeNet 另一分支），病例名与 concat 一致
    REFERENCE_ROOT = r'D:\pulmonary_vessel\nnunet\output'
    postprocess(
        r'C:\Users\mamah\PycharmProjects\nnunet\predict_result_AIIB23\concat',
        r'C:\Users\mamah\PycharmProjects\nnunet\predict_result_AIIB23\outputs',
        reference_root=REFERENCE_ROOT,
        iou_threshold=IOU_THRESHOLD,
    )
















