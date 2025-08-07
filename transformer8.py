import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import logging

# 确保 logs 目录存在
os.makedirs("logs", exist_ok=True)

## 数据集加载
ori_traindata = np.loadtxt("新8种数据集/5%丢失5%误差改变噪声/噪声10%/train.txt", delimiter=',', usecols=(0, 1, 2))
ori_traindata = torch.from_numpy(ori_traindata).float()
ori_traindata = ori_traindata.reshape(-1, 1000, 3)
# 标签
ori_labels = np.loadtxt("新8种数据集/5%丢失5%误差改变噪声/噪声10%/train.txt", delimiter=',', usecols=(3,))
train_label = torch.from_numpy(ori_labels).long()
train_label = train_label.view(-1, 1000)

ori_testdata = np.loadtxt("新8种数据集/5%丢失5%误差改变噪声/噪声10%/train.txt", delimiter=',', usecols=(0, 1, 2))
ori_testdata = torch.from_numpy(ori_testdata).float()
ori_testdata = ori_testdata.view(-1, 1000, 3)
# 标签
ori_testlabels = np.loadtxt("新8种数据集/5%丢失5%误差改变噪声/噪声10%/train.txt", delimiter=',', usecols=(3,))
test_label = torch.from_numpy(ori_testlabels).long()
test_label = test_label.view(-1, 1000)


# 定义数据集类
class Mydataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


# 加载数据集和标签
train_data = Mydataset(ori_traindata, train_label)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = Mydataset(ori_testdata, test_label)
test_dataloader = DataLoader(test_data, batch_size=32)


# 定义 Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

encoder = Encoder()

# 冻结 Encoder 参数
for param in encoder.parameters():
    param.requires_grad = False


# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return output


# 定义 Decoder，基于 Encoder 的结构
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


# 定义最终模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = encoder
        self.transformer = TransformerModel(d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1)
        self.decoder = Decoder()
        self.linear = nn.Linear(64, 9)
        self.dropout = nn.Dropout(p=0.5) # 0.5

    def forward(self, x):  # 注意这里要有冒号
        x = x.permute(0, 2, 1)  # 调整维度以适应 Encoder 的输入
        x = self.encoder(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x, x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.decoder(x)
        x = self.dropout(x)
        batch_size, channels, seq_len = x.size()
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(batch_size * seq_len, channels)
        x = self.linear(x)
        x = x.view(batch_size, seq_len, 9)

        return x


# 实例化模型
model = MyModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on the GPU" if torch.cuda.is_available() else "Running on the CPU")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 200

# 设置日志
train_log_file = "logs/train_log.txt"
train_logger = logging.getLogger("train_logger")
train_handler = logging.FileHandler(train_log_file)
train_handler.setFormatter(logging.Formatter("%(message)s"))
train_logger.addHandler(train_handler)
train_logger.setLevel(logging.INFO)

test_log_file = "logs/test_log.txt"
test_logger = logging.getLogger("test_logger")
test_handler = logging.FileHandler(test_log_file)
test_handler.setFormatter(logging.Formatter("%(message)s"))
test_logger.addHandler(test_handler)
test_logger.setLevel(logging.INFO)

# 训练和测试
for i in range(epochs):
    print("第{}轮训练".format(i + 1))
    train_step = 0
    sum_train_acc = 0
    model.train()
    for inputs, target in train_dataloader:
        inputs = inputs.to(device)
        target = target.view(-1).to(device)  # 展平目标标签

        optimizer.zero_grad()
        out = model(inputs)  # 模型输出的形状为 (batch_size, seq_len, num_classes)
        out = out.view(-1, 9)  # 展平模型输出

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        # 计算准确率
        max_probs, predicted_class = torch.max(out, dim=-1)
        accuracy = (predicted_class == target).float().mean().item()
        sum_train_acc += accuracy
        train_step += 1

    avg_train_acc = sum_train_acc / train_step
    print('平均train_acc: {:.4f}'.format(avg_train_acc * 100))
    train_logger.info(f"Epoch {i + 1}: Avg Train Acc: {avg_train_acc:.4f}")

    # 测试
    sum_test_acc = 0
    model.eval()
    with torch.no_grad():
        test_step = 0
        for inputs, target in test_dataloader:
            inputs = inputs.to(device)
            target = target.view(-1).to(device)  # 展平目标标签

            out = model(inputs)  # 模型输出的形状为 (batch_size, seq_len, num_classes)
            out = out.view(-1, 9)  # 展平模型输出

            # 计算准确率
            max_probs, predicted_class = torch.max(out, dim=-1)
            accuracy = (predicted_class == target).float().mean().item()
            sum_test_acc += accuracy
            test_step += 1

    avg_test_acc = sum_test_acc / test_step
    print('平均test_acc: {:.4f}'.format(avg_test_acc * 100))
    test_logger.info(f"Epoch {i + 1}: Avg Test Acc: {avg_test_acc:.4f}")

print('train最高准确率：{:.4f}'.format(avg_train_acc * 100))
print('test最高准确率：{:.4f}'.format(avg_test_acc * 100))