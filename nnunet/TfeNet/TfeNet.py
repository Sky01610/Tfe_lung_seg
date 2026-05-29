import os
import torch
import torch.nn as nn
import numpy as np
from model.TfeNet_model import TfeNet

# 训练/数据脚本使用；可通过环境变量 TFENET_DATASET_PATH 覆盖，勿保留原作者本机路径。
_default_dataset = os.environ.get("TFENET_DATASET_PATH", r"/path/to/your/dataset")

config = {'pad_value': 0,     
		  'augtype': {'rotate': True},
		  'startepoch': 0, 'lr_stage': np.array([20, 40, 60, 70]), 'lr': np.array([1e-2, 1e-3, 1e-4,1e-5]),	
          'dataset_path': _default_dataset}


def get_model(args=None):

	net = TfeNet(n_channels=1,number=16) 

	print('# of network parameters:', sum(param.numel() for param in net.parameters()))
	return config, net


if __name__ == '__main__':
	_, model = get_model()
