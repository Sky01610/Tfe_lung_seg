import argparse
import os
import pickle
import time
from pathlib import Path

from win_dll_fix import prepare_torch_dll_path

prepare_torch_dll_path()

import numpy as np
from importlib import import_module
import torch
from torch.utils.data import DataLoader
from data_ATM22 import SegValData
import skimage.measure as measure
from skimage.morphology import skeletonize
from utils import load_itk_image,save_itk
import tqdm

"""
Sample Usage
python3 evaluate.py --data_path 你的数据目录
if want to set GPU
python evaluation.py --data-path "D/pulmonary_vessel/case_correct/rerun" --device cuda
This device number refer to the number of the GPU, starets with 0.

只跑完整 TfeNet（不跑 Small）::
python evaluation.py --data-path 你的数据目录 --no-small
"""

_TFE_ROOT = Path(__file__).resolve().parent


def _resolve_checkpoint_path(
    ifsmall: bool,
    checkpoint_root: str | os.PathLike | None = None,
    checkpoint_dataset: str = "AIIB23",
) -> Path:
    root = Path(checkpoint_root) if checkpoint_root else _TFE_ROOT / "checkpoint"
    name = (
        "TfeNetSmall_checkpoint.ckpt"
        if ifsmall
        else "TfeNet_checkpoint.ckpt"
    )
    path = root / checkpoint_dataset / name
    if not path.is_file():
        raise FileNotFoundError(f"未找到权重文件: {path}")
    return path


def _torch_load_checkpoint(path):
    """加载检查点：优先 weights_only=True；ckpt 中含 argparse.Namespace 等时需回退（仅信任来源时使用）。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch 过旧，无 weights_only 参数
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        # 例如保存了 Namespace/hyperparams，非纯张量字典
        return torch.load(path, map_location="cpu", weights_only=False)


def network_prediction(
    data_path,
    save_path,
    ifsmall=False,
    device="cuda",
    *,
    checkpoint_root: str | os.PathLike | None = None,
    checkpoint_dataset: str = "AIIB23",
):
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已指定 CUDA，但 torch.cuda.is_available() 为 False，无法使用 GPU。")

    casemodel = import_module('TfeNet')
    config2, case_net = casemodel.get_model()
    val_path = os.path.abspath(os.path.expanduser(data_path))
    if not os.path.isdir(val_path):
        raise FileNotFoundError(
            "评估输入目录不存在或不是文件夹: {}\n"
            "请将要推理的体数据（如 .mhd/.raw 等，与 load_itk_image 一致）放在该目录下，\n"
            "或通过 --data-path / 环境变量 TFENET_EVAL_INPUT 指定正确路径。\n"
            "（报错里若出现 /home/wqb/... 说明仍在使用原作者机器上的路径，请改为你本机数据集路径。）"
            .format(val_path)
        )
    os.makedirs(save_path, exist_ok=True)
    ckpt_path = _resolve_checkpoint_path(
        ifsmall, checkpoint_root, checkpoint_dataset
    )
    checkpoint = _torch_load_checkpoint(str(ckpt_path))
    case_net.load_state_dict(checkpoint['state_dict'])
    dataset = SegValData(val_path)
    # 整例 3D 体数据体积大，pin_memory 在部分环境会触发 CUDA invalid argument；batch_size=1 时收益可忽略
    val_loader_case = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
    )
    case_net = case_net.to(dev)
    case_net.eval()
    # sliding window
    cube_size = 128
    step = 64
    for i, (x, origin, spacing, name) in enumerate(val_loader_case):
        case_name = name[0]
        print(case_name)
        pred = np.zeros(x.shape)
        pred_num = np.zeros(x.shape)
        # print(x.shape)
        x = x.to(dev)
        xnum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 else \
            (x.shape[2] - cube_size) // step + 2
        ynum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 else \
            (x.shape[3] - cube_size) // step + 2
        znum = (x.shape[4] - cube_size) // step + 1 if (x.shape[4] - cube_size) % step == 0 else \
            (x.shape[4] - cube_size) // step + 2
        for xx in range(xnum):
            xl = step * xx
            xr = step * xx + cube_size
            if xr > x.shape[2]:
                xr = x.shape[2]
                xl = x.shape[2] - cube_size
            for yy in range(ynum):
                yl = step * yy
                yr = step * yy + cube_size
                if yr > x.shape[3]:
                    yr = x.shape[3]
                    yl = x.shape[3] - cube_size
                for zz in range(znum):
                    zl = step * zz
                    zr = step * zz + cube_size
                    if zr > x.shape[4]:
                        zr = x.shape[4]
                        zl = x.shape[4] - cube_size

                    x_input = x[:, :, xl:xr, yl:yr, zl:zr]
                    p = case_net(x_input.contiguous())
                    p = p.cpu().detach().numpy()
                    pred[:, :, xl:xr, yl:yr, zl:zr] += p
                    pred_num[:, :, xl:xr, yl:yr, zl:zr] += 1

        pred = pred / pred_num
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = np.squeeze(pred)

        print(os.path.join(save_path,case_name))
        save_itk(pred,origin[0],spacing[0],os.path.join(save_path,case_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TfeNet ATM22 推理评估")
    default_data = os.environ.get("TFENET_EVAL_INPUT", "test_data")
    parser.add_argument(
        "--data-path",
        default=default_data,
        help="待推理影像所在目录（目录内为若干病例文件，与 SegValData 一致）。也可用环境变量 TFENET_EVAL_INPUT。",
    )
    parser.add_argument(
        "--save-small",
        default="C/Users/mamah/PycharmProjects/nnunet/predict_result_AIIB23/pred",
        help="TfeNetSmall 模型输出目录",
    )
    parser.add_argument(
        "--save-norm",
        default="C/Users/mamah/PycharmProjects/nnunet/predict_result_AIIB23/pred_small",
        help="完整 TfeNet 模型输出目录",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("TFENET_DEVICE", "cuda"),
        help="推理设备，例如 cuda、cuda:3。第 4 张卡为 cuda:3（编号从 0 起）。也可用环境变量 TFENET_DEVICE。",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="N",
        help="按「第几块 GPU」从 1 开始数，例如 4 表示第 4 张卡（等价于 --device cuda:3）。指定后覆盖 --device / TFENET_DEVICE。",
    )
    parser.add_argument(
        "--no-small",
        action="store_true",
        help="只运行完整 TfeNet（TfeNet_checkpoint），不运行 TfeNetSmall。",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="checkpoint 根目录（默认 TfeNet/checkpoint）",
    )
    parser.add_argument(
        "--checkpoint-dataset",
        default="AIIB23",
        help="权重子目录名，如 AIIB23 / ATM22 / BAS",
    )
    args = parser.parse_args()
    if args.gpu is not None:
        if args.gpu < 1:
            parser.error("--gpu 须为 >= 1 的整数（第 1 张卡为 1）")
        device = f"cuda:{args.gpu - 1}"
    else:
        device = args.device

    ckpt_kw = dict(
        checkpoint_root=args.checkpoint_root,
        checkpoint_dataset=args.checkpoint_dataset,
    )
    if not args.no_small:
        network_prediction(
            args.data_path,
            args.save_small,
            ifsmall=True,
            device=device,
            **ckpt_kw,
        )
    network_prediction(
        args.data_path,
        args.save_norm,
        ifsmall=False,
        device=device,
        **ckpt_kw,
    )
    



