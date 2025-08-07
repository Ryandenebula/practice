import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.nn.functional as F

# 数据路径
data_path = "新五种数据集/5%误差5%噪声改变丢失/丢失率40%/train.txt"

ori_traindata = np.loadtxt(data_path, delimiter=',', usecols=(0, 1, 2))
ori_traindata = torch.from_numpy(ori_traindata).float()
ori_traindata = ori_traindata.reshape(-1, 1000, 3)
# 标签
ori_labels = np.loadtxt(data_path, delimiter=',', usecols=(3,))
train_label = torch.from_numpy(ori_labels).long()
train_label = train_label.view(-1, 1000)

ori_testdata = np.loadtxt(data_path, delimiter=',', usecols=(0, 1, 2))
ori_testdata = torch.from_numpy(ori_testdata).float()
ori_testdata = ori_testdata.view(-1, 1000, 3)
# 标签
ori_testlabels = np.loadtxt(data_path, delimiter=',', usecols=(3,))
test_label = torch.from_numpy(ori_testlabels).float()
test_label = test_label.view(-1, 1000)


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class Mydataset(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    # return len(self.labels)
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


# 加载数据集和标签
train_data = Mydataset(ori_traindata, train_label)
# DataLoader中的shuffer=False表示不打乱数据的顺序，Ture表示在每一次epoch中都打乱所有数据的顺序
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = Mydataset(ori_testdata, test_label)
test_dataloader = DataLoader(test_data, batch_size=32)


# 定义 CRRNN 模型
class CRRNN(nn.Module):
    def __init__(self, input_dim=3, cnn_channels=16, gru_hidden=64, num_classes=6):
        super(CRRNN, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        # CNN 部分
        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64))
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64))
        # GRU 部分
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            batch_first=True
        )

        # 拼接后的全连接层
        self.fc = nn.Linear(64 + gru_hidden, num_classes)

    def forward(self, x):
        # CNN 部分
        x = x.permute(0, 2, 1)
        x0 = self.conv1d(x)
        x1 = self.conv1(x0)
        out1 = F.elu(x1 + x0)

        x2 = self.conv2(out1)
        out2 = F.elu(x2 + out1)

        x3 = self.conv3(out2)
        out3 = F.elu(x3 + out2)

        x4 = self.conv4(out3)
        out4 = F.elu(x4 + out3)

        out4 = out4.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)

        out_last = self.fc(torch.cat([out4, x], dim=-1))
        out_last = out_last.reshape(-1, 6)
        return out_last


# 初始化模型、损失函数和优化器
model = CRRNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)



def evaluate_accuracy(model, dataloader, device):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度，节省内存
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_y = batch_y.reshape(-1)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        batch_y = batch_y.reshape(-1)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    test_accuracy = evaluate_accuracy(model, test_dataloader, device)
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}, Test Accuracy: {test_accuracy:.2f}%")