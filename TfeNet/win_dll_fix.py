"""Windows: 在 import torch 之前调用，缓解 c10.dll WinError 1114（PATH/OpenMP 冲突）。"""
from __future__ import annotations

import os
import site
import sys


def prepare_torch_dll_path() -> None:
    if sys.platform != "win32":
        return
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for root in site.getsitepackages():
        torch_lib = os.path.join(root, "torch", "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
            return
