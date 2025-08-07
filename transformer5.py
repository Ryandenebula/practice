import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
import time  # 导入时间模块

os.makedirs("logs", exist_ok=True)

## 数据集加载
ori_traindata = np.loadtxt("新五种数据集/5%误差5%噪声改变丢失/丢失率40%/train.txt", delimiter=',', usecols=(0, 1, 2))
ori_traindata = torch.from_numpy(ori_traindata).float()
ori_traindata = ori_traindata.reshape(-1, 1000, 3)

ori_labels = np.loadtxt("新五种数据集/5%误差5%噪声改变丢失/丢失率40%/train.txt", delimiter=',', usecols=(3,))
train_label = torch.from_numpy(ori_labels).long()
train_label = train_label.view(-1, 1000)

ori_testdata = np.loadtxt("新五种数据集/5%误差5%噪声改变丢失/丢失率40%/train.txt", delimiter=',', usecols=(0, 1, 2))
ori_testdata = torch.from_numpy(ori_testdata).float()
ori_testdata = ori_testdata.view(-1, 1000, 3)

ori_testlabels = np.loadtxt("新五种数据集/5%误差5%噪声改变丢失/丢失率40%/train.txt", delimiter=',', usecols=(3,))
test_label = torch.from_numpy(ori_testlabels).long()
test_label = test_label.view(-1, 1000)

# 定义数据集类
class Mydataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

train_data = Mydataset(ori_traindata, train_label)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = Mydataset(ori_testdata, test_label)
test_dataloader = DataLoader(test_data, batch_size=32)

# 定义 Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.encoder(x)

# 自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(attn_out + x)

# Transformer 模型（含自注意力）
class TransformerModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super(TransformerModel, self).__init__()
        self.layers = nn.ModuleList([
            SelfAttention(d_model, nhead) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 定义 Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.decoder(x)

# 定义最终模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = Encoder()
        self.transformer = TransformerModel()
        self.decoder = Decoder()
        self.linear = nn.Linear(64, 6)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, 64)
        x = self.linear(x)
        x = x.view(-1, 1000, 6)
        return x

# 实例化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

logging.basicConfig(filename="logs/train_log.txt", level=logging.INFO, format="%(message)s")

# 训练
epochs = 100
best_acc = 0.0

# 开始时间
start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.view(-1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs).view(-1, 6)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = 100. * correct / total
    scheduler.step()

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.view(-1).to(device)
            outputs = model(inputs).view(-1, 6)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    logging.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_model.pth")

# 结束时间
end_time = time.time()

# 计算总训练时间
total_training_time = end_time - start_time
print(f"Best Test Accuracy: {best_acc:.2f}%")
print(f"Total Training Time: {total_training_time:.2f} seconds")
logging.info(f"Total Training Time: {total_training_time:.2f} seconds")