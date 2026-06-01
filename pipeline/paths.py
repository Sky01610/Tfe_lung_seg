from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CasePaths:
    """单病例输出目录布局。"""

    case_name: str
    output_dir: str

    @property
    def case_root(self) -> str:
        return os.path.join(self.output_dir, self.case_name)

    @property
    def temp_nii(self) -> str:
        return os.path.join(self.case_root, "temp_nii")

    @property
    def preprocessed(self) -> str:
        return os.path.join(self.case_root, "preprocessed")

    @property
    def infer_input(self) -> str:
        """仅含 CT NIfTI，供 TfeNet / nnU-Net 读取（避免读到 crop_indices.json）。"""
        return os.path.join(self.case_root, "_infer_input")

    @property
    def predictions(self) -> str:
        return os.path.join(self.case_root, "predictions")

    @property
    def final(self) -> str:
        return os.path.join(self.case_root, "final")

    def temp_ct_path(self) -> str:
        return os.path.join(self.temp_nii, f"{self.case_name}_0000.nii.gz")

    def preprocessed_ct_path(self) -> str:
        return os.path.join(self.preprocessed, f"{self.case_name}_0000.nii.gz")

    def crop_indices_path(self) -> str:
        return os.path.join(self.preprocessed, f"{self.case_name}_crop_indices.json")

    def pred_task_dir(self, task: str) -> str:
        return os.path.join(self.predictions, task)

    def pred_mask_path(self, task: str) -> str:
        return os.path.join(self.pred_task_dir(task), f"{self.case_name}.nii.gz")

    @property
    def pred_airway_tfenet_dir(self) -> str:
        """TfeNet 双分支输出根目录：{case}/predictions/airway_tfenet/。"""
        return self.pred_task_dir("airway_tfenet")

    def pred_tfenet_branch_dir(self, branch: str) -> str:
        """
        TfeNet 单模型输出目录。
        branch: 'TfeNet'（全气道）或 'TfeNetSmall'（细气道）
        """
        if branch not in ("TfeNet", "TfeNetSmall"):
            raise ValueError(f"未知 TfeNet 分支: {branch}")
        return os.path.join(self.pred_airway_tfenet_dir, branch)

    def pred_tfenet_union_path(self) -> str:
        """TfeNet ∪ TfeNetSmall 并集：{case}/predictions/airway_tfenet/{case}.nii.gz"""
        return self.pred_mask_path("airway_tfenet")

    def final_task_dir(self, task: str) -> str:
        return os.path.join(self.final, task)

    def final_mask_path(self, task: str) -> str:
        return os.path.join(self.final_task_dir(task), f"{self.case_name}.nii.gz")

    def lesion_split_dir(self) -> str:
        return os.path.join(self.final, "lesion_split", self.case_name)

    def lesion_lesions_dir(self) -> str:
        return os.path.join(self.lesion_split_dir(), "lesions")

    def lesion_stats_path(self) -> str:
        return os.path.join(self.lesion_split_dir(), "stats.json")

    def airway_tree_path(self) -> str:
        return os.path.join(self.final, "airway_tree", f"{self.case_name}_tree.json")

    def prepare_infer_input(self) -> str:
        """将 preprocessed CT 复制到仅含一个 NIfTI 的推理目录。"""
        import shutil

        os.makedirs(self.infer_input, exist_ok=True)
        for fn in os.listdir(self.infer_input):
            fp = os.path.join(self.infer_input, fn)
            if os.path.isfile(fp):
                os.remove(fp)
        dst = os.path.join(self.infer_input, f"{self.case_name}_0000.nii.gz")
        shutil.copy2(self.preprocessed_ct_path(), dst)
        return self.infer_input

    def ensure_dirs(self) -> None:
        for d in (
            self.temp_nii,
            self.preprocessed,
            self.infer_input,
            self.predictions,
            self.final,
            self.pred_task_dir("airway"),
            self.pred_task_dir("airway_nnunet"),
            self.pred_task_dir("airway_tfenet"),
            os.path.join(self.pred_task_dir("airway_tfenet"), "TfeNet"),
            os.path.join(self.pred_task_dir("airway_tfenet"), "TfeNetSmall"),
            self.pred_task_dir("vessel"),
            self.pred_task_dir("lung_lesion"),
            self.final_task_dir("airway"),
            self.final_task_dir("vessel"),
            self.final_task_dir("lung_lesion"),
            self.lesion_lesions_dir(),
            os.path.join(self.final, "airway_tree"),
        ):
            os.makedirs(d, exist_ok=True)
