import random
import torch
import torch.nn as nn
import numpy as np
import os

from model_utils.data2 import get_data_loader
from model_utils.model2 import myBertModel
from model_utils.train import train_val

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#################################################################
seed_everything(0)
###############################################


lr = 0.0001
batchsize = 16
loss = nn.CrossEntropyLoss()
bert_path = "bert-base-chinese"
num_class = 10
data_path = "pinglun.csv"
max_acc = 0.6
device = "cuda" if torch.cuda.is_available() else "cpu"
model = myBertModel(bert_path, num_class, device).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00001)

train_loader, val_loader = get_data_loader(data_path, batchsize, val_size=0.2)


epochs = 5
save_path = "model_save/best_model.pth"

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-9)       # 余弦退火，改变学习率
val_epoch = 1
para = {
    "model": model,
    "train_loader": train_loader,
    "val_loader": val_loader,
    "scheduler": scheduler,
    "optimizer": optimizer,
    "loss": loss,
    "epoch": epochs,
    "device": device,
    "save_path": save_path,
    "max_acc": max_acc,
    "val_epoch": val_epoch,         # 训练多少轮验证一次
}

train_val(para)





















