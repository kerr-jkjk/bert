# data负责产生两个dataloader
from torch.utils.data import  DataLoader, Dataset
from sklearn.model_selection import train_test_split        # 负责给X，Y和分割比例，分割出来一个训练集和验证集的X，Y
import torch

import csv

from tqdm import tqdm


def read_file(path):
    data = []
    label = []
    label_map = {}  # 用于存储标签到数字的映射
    current_label_id = 0  # 当前标签的数字ID

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in tqdm(enumerate(reader)):
            if i % 6 == 0:
                review = row["review"].strip("\n")  # 去掉可能存在的换行符
                cat = row["cat"].strip("\n")        # 去掉可能存在的换行符

                # 如果当前标签还没有被映射，就给它分配一个新的数字ID
                if cat not in label_map:
                    label_map[cat] = current_label_id
                    current_label_id += 1

                # 将review添加到data列表，将cat对应的数字ID添加到label列表
                data.append(review)
                label.append(label_map[cat])

    print("读了%d个数据" % len(data))
    return data, label








class jdDataset(Dataset):
    def __init__(self, data, label):
        self.X = data
        self.Y = torch.LongTensor([int(i) for i in label])

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return len(self.Y)


def get_data_loader(path, batchsize, val_size=0.2):            # 读入数据，分割数据，默认把五分之一的数据作为验证集
    data, label = read_file(path)

    train_x, val_x, train_y, val_y = train_test_split(data, label, test_size=val_size, shuffle=True, stratify=label)   # stratify：各类也按val_Size的比例严格划分

    train_set = jdDataset(train_x, train_y)
    val_set = jdDataset(val_x, val_y)
    train_loader = DataLoader(train_set,  batchsize, shuffle=True)
    val_loader = DataLoader(val_set, batchsize, shuffle=True)

    return train_loader, val_loader

if __name__ == "__main__":
    get_data_loader("../pinglun.csv", 16)
