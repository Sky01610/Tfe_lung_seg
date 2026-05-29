from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class NnUnetTaskConfig:
    dataset_id: int | None = None
    model_folder: str | None = None
    # 对应 weight/{weights_subdir}/nnUNetTrainer__...
    weights_subdir: str | None = None
    configuration: str = "3d_fullres"
    folds: str = "all"
    # 多类分割时保留的 label（None=非零为前景）
    foreground_labels: list[int] | None = None


@dataclass
class TfeNetConfig:
    checkpoint_dataset: str = "AIIB23"
    checkpoint_root: str | None = None
    use_small: bool = True
    # 是否与 nnU-Net 气管结果做体素级并集（需配置 airway_nnunet）
    use_nnunet_fusion: bool = True


NNUNET_TRAINER_DIR_TEMPLATE = "nnUNetTrainer__nnUNetPlans__{configuration}"


@dataclass
class PipelineConfig:
    nnunet_results: str | None = None
    # 自定义权重根目录，例如 weight/（其下 airway、vessel、lung_lesion 子文件夹）
    nnunet_weights_root: str | None = None
    device: str = "cuda"
    cuda_id: int = 0
    crop_lung: bool = True
    crop_margin: int = 5
    use_lungmask: bool = True
    tasks: list[str] = field(
        default_factory=lambda: ["airway", "vessel", "lung_lesion"]
    )
    airway: TfeNetConfig = field(default_factory=TfeNetConfig)
    vessel: NnUnetTaskConfig = field(default_factory=NnUnetTaskConfig)
    lung_lesion: NnUnetTaskConfig = field(default_factory=NnUnetTaskConfig)
    airway_nnunet: NnUnetTaskConfig | None = field(
        default_factory=lambda: NnUnetTaskConfig(
            weights_subdir="airway",
            configuration="3d_fullres",
            folds="all",
        )
    )
    postprocess_airway_lcc: bool = True
    # 融合后 mask 与 nnU-Net 气管参考做 IoU 连通域过滤（需 use_nnunet_fusion）
    postprocess_airway_iou_with_nnunet: bool = True
    postprocess_airway_iou_threshold: float = 0.3
    split_lesions: bool = True
    export_airway_tree: bool = False
    min_lesion_voxels: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> PipelineConfig:
        path = os.path.abspath(path)
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.from_dict(raw, config_dir=os.path.dirname(path))

    @classmethod
    def load(cls, config_path: str | None = None) -> PipelineConfig:
        """加载配置：显式路径 > 仓库根目录 pipeline_config.yaml > 默认。"""
        if config_path:
            return cls.from_yaml(config_path)
        default = os.path.join(_REPO_ROOT, "pipeline_config.yaml")
        if os.path.isfile(default):
            return cls.from_yaml(default)
        cfg = cls()
        cfg._finalize_paths(config_dir=_REPO_ROOT)
        return cfg

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, config_dir: str | None = None
    ) -> PipelineConfig:
        cfg = cls()
        base = config_dir or _REPO_ROOT
        if "nnunet_results" in raw:
            cfg.nnunet_results = _resolve_path(raw["nnunet_results"], base)
        if "nnunet_weights_root" in raw:
            cfg.nnunet_weights_root = _resolve_path(raw["nnunet_weights_root"], base)
        if "device" in raw:
            cfg.device = str(raw["device"])
        if "cuda_id" in raw:
            cfg.cuda_id = int(raw["cuda_id"])
        if "crop_lung" in raw:
            cfg.crop_lung = bool(raw["crop_lung"])
        if "crop_margin" in raw:
            cfg.crop_margin = int(raw["crop_margin"])
        if "use_lungmask" in raw:
            cfg.use_lungmask = bool(raw["use_lungmask"])
        if "tasks" in raw:
            cfg.tasks = list(raw["tasks"])
        if "postprocess_airway_lcc" in raw:
            cfg.postprocess_airway_lcc = bool(raw["postprocess_airway_lcc"])
        if "postprocess_airway_iou_with_nnunet" in raw:
            cfg.postprocess_airway_iou_with_nnunet = bool(
                raw["postprocess_airway_iou_with_nnunet"]
            )
        if "postprocess_airway_iou_threshold" in raw:
            cfg.postprocess_airway_iou_threshold = float(
                raw["postprocess_airway_iou_threshold"]
            )
        # 兼容旧配置项 postprocess_airway_iou_reference
        if "postprocess_airway_iou_reference" in raw:
            ref = raw["postprocess_airway_iou_reference"]
            if ref is None or str(ref).lower() in ("none", "null", ""):
                cfg.postprocess_airway_iou_with_nnunet = False
            elif str(ref).lower() not in ("airway_nnunet", "airway", "nnunet"):
                import warnings

                warnings.warn(
                    "postprocess_airway_iou_reference 已弃用；"
                    "IoU 参考现固定为 nnU-Net 气管输出（融合后 mask 为待过滤 pred）。"
                    f"忽略旧值: {ref!r}",
                    stacklevel=2,
                )
        if "split_lesions" in raw:
            cfg.split_lesions = bool(raw["split_lesions"])
        if "export_airway_tree" in raw:
            cfg.export_airway_tree = bool(raw["export_airway_tree"])
        if "min_lesion_voxels" in raw:
            cfg.min_lesion_voxels = int(raw["min_lesion_voxels"])

        if "airway" in raw:
            cfg.airway = _parse_tfenet(raw["airway"], cfg.airway)
        if "vessel" in raw:
            cfg.vessel = _parse_nnunet(raw["vessel"], cfg.vessel)
        if "lung_lesion" in raw:
            cfg.lung_lesion = _parse_nnunet(raw["lung_lesion"], cfg.lung_lesion)
        if "airway_nnunet" in raw and raw["airway_nnunet"] is not None:
            base_nn = cfg.airway_nnunet or NnUnetTaskConfig(weights_subdir="airway")
            cfg.airway_nnunet = _parse_nnunet(raw["airway_nnunet"], base_nn)
        elif raw.get("airway", {}).get("use_nnunet_fusion") is False:
            cfg.airway_nnunet = None

        if cfg.airway_nnunet is not None and "use_nnunet_fusion" not in raw.get(
            "airway", {}
        ):
            cfg.airway.use_nnunet_fusion = True

        env_results = os.environ.get("nnUNet_results") or os.environ.get(
            "NNUNET_RESULTS"
        )
        if not cfg.nnunet_results and env_results:
            cfg.nnunet_results = env_results
        cfg._finalize_paths(config_dir=base)
        return cfg

    def _finalize_paths(self, *, config_dir: str) -> None:
        """相对路径按配置文件/仓库根目录解析；自动探测 weight/。"""
        if self.nnunet_weights_root:
            self.nnunet_weights_root = _resolve_path(
                self.nnunet_weights_root, config_dir
            )
        elif os.path.isdir(os.path.join(config_dir, "weight")):
            self.nnunet_weights_root = os.path.join(config_dir, "weight")

        for task_cfg in (self.vessel, self.lung_lesion):
            if task_cfg.model_folder:
                task_cfg.model_folder = _resolve_path(task_cfg.model_folder, config_dir)
        if self.airway_nnunet is not None and self.airway_nnunet.model_folder:
            self.airway_nnunet.model_folder = _resolve_path(
                self.airway_nnunet.model_folder, config_dir
            )

    def resolve_nnunet_model_folder(
        self,
        task_cfg: NnUnetTaskConfig,
        *,
        weights_subdir: str | None = None,
    ) -> str:
        """
        解析 nnU-Net 模型目录，优先级：
        1. task_cfg.model_folder（显式路径）
        2. nnunet_weights_root/{weights_subdir}/nnUNetTrainer__nnUNetPlans__{configuration}
        3. nnunet_results + dataset_id（标准 nnU-Net 目录命名）
        """
        if task_cfg.model_folder:
            path = _expand(task_cfg.model_folder)
            if not os.path.isdir(path):
                raise FileNotFoundError(f"模型目录不存在: {path}")
            return path

        subdir = weights_subdir or task_cfg.weights_subdir
        if self.nnunet_weights_root and subdir:
            trainer_dir = NNUNET_TRAINER_DIR_TEMPLATE.format(
                configuration=task_cfg.configuration
            )
            path = os.path.join(self.nnunet_weights_root, subdir, trainer_dir)
            if os.path.isdir(path):
                return os.path.abspath(path)
            raise FileNotFoundError(
                f"在权重根目录下未找到模型: {path}\n"
                f"请确认存在 {subdir}/{trainer_dir}/\n"
                f"当前 nnunet_weights_root={self.nnunet_weights_root}"
            )

        if self.nnunet_weights_root and not subdir:
            raise ValueError(
                f"已设置 nnunet_weights_root={self.nnunet_weights_root}，"
                "但未指定 weights_subdir（调用参数或 yaml 中 vessel.weights_subdir 等）"
            )

        if task_cfg.dataset_id is None:
            raise ValueError(
                "须在配置中指定以下之一：\n"
                "  - model_folder（显式模型路径）\n"
                "  - nnunet_weights_root + weights_subdir（如 weight/vessel/...）\n"
                "  - dataset_id + nnunet_results（标准 nnU-Net 目录）\n"
                "提示：请使用 --config pipeline_config.yaml，或确保仓库根目录存在该文件。"
            )
        if not self.nnunet_results:
            raise ValueError(
                "未设置 nnUNet_results（配置文件或环境变量 nnUNet_results）"
            )
        import glob

        pattern = os.path.join(
            self.nnunet_results,
            f"Dataset{task_cfg.dataset_id:03d}_*",
            f"nnUNetTrainer__nnUNetPlans__{task_cfg.configuration}",
        )
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"在 {self.nnunet_results} 下未找到 Dataset{task_cfg.dataset_id:03d} "
                f"的 {task_cfg.configuration} 模型: {pattern}"
            )
        return matches[-1]


def _expand(value: str | None) -> str | None:
    if value is None:
        return None
    return os.path.expanduser(os.path.expandvars(str(value)))


def _resolve_path(value: str | None, base_dir: str) -> str | None:
    if value is None:
        return None
    text = _expand(str(value))
    if not text:
        return text
    if os.path.isabs(text):
        return text
    return os.path.abspath(os.path.join(base_dir, text))


def _parse_nnunet(raw: dict[str, Any], base: NnUnetTaskConfig) -> NnUnetTaskConfig:
    out = NnUnetTaskConfig(
        dataset_id=base.dataset_id,
        model_folder=base.model_folder,
        weights_subdir=base.weights_subdir,
        configuration=base.configuration,
        folds=base.folds,
        foreground_labels=base.foreground_labels,
    )
    if "dataset_id" in raw:
        out.dataset_id = int(raw["dataset_id"])
    if "model_folder" in raw:
        out.model_folder = _expand(raw["model_folder"])
    if "weights_subdir" in raw:
        out.weights_subdir = str(raw["weights_subdir"])
    if "configuration" in raw:
        out.configuration = str(raw["configuration"])
    if "folds" in raw:
        out.folds = str(raw["folds"])
    if "foreground_labels" in raw:
        val = raw["foreground_labels"]
        out.foreground_labels = None if val is None else list(val)
    return out


def _parse_tfenet(raw: dict[str, Any], base: TfeNetConfig) -> TfeNetConfig:
    out = TfeNetConfig(
        checkpoint_dataset=base.checkpoint_dataset,
        checkpoint_root=base.checkpoint_root,
        use_small=base.use_small,
        use_nnunet_fusion=base.use_nnunet_fusion,
    )
    if "checkpoint_dataset" in raw:
        out.checkpoint_dataset = str(raw["checkpoint_dataset"])
    if "checkpoint_root" in raw:
        out.checkpoint_root = _expand(raw["checkpoint_root"])
    if "use_small" in raw:
        out.use_small = bool(raw["use_small"])
    if "use_nnunet_fusion" in raw:
        out.use_nnunet_fusion = bool(raw["use_nnunet_fusion"])
    return out
