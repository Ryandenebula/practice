import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import logging


# change test the vscode git 
# 确保 logs 目录存在
os.makedirs("logs", exist_ok=True)
##数据集#数据已经归一化处理

##数据集#数据已经归一化处理
ori_traindata = np.loadtxt("train0.05_9_1000.txt", delimiter=',', usecols=(0, 1, 2))
ori_traindata = torch.from_numpy(ori_traindata).float()
ori_traindata = ori_traindata.reshape(-1, 1000, 3)
# 标签
ori_labels = np.loadtxt("train0.05_9_1000.txt", delimiter=',', usecols=(3,))
train_label = torch.from_numpy(ori_labels).long()
train_label = train_label.view(-1, 1000)

ori_testdata = np.loadtxt("train0.05_9_1000.txt", delimiter=',', usecols=(0, 1, 2))
ori_testdata = torch.from_numpy(ori_testdata).float()
ori_testdata = ori_testdata.view(-1, 1000, 3)
# 标签
ori_testlabels = np.loadtxt("train0.05_9_1000.txt", delimiter=',', usecols=(3,))
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


# 定义 Encoder 和 Decoder 模型结构（与训练时相同）
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),  # (batch, 64, 1000)
            nn.BatchNorm1d(64),  # 批量归一化
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),  # (batch, 128, 1000)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)  # (batch, 256, 1000)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


def load_model(encoder, encoder_path='encoder.pth'):
    # 加载保存的模型参数
    encoder.load_state_dict(torch.load(encoder_path))
    print("Encoder 模型已加载")

encoder = Encoder()

load_model(encoder)
# 将 Encoder 和 Decoder 的参数设置为不参与梯度计算（冻结）
for param in encoder.parameters():
    param.requires_grad = False


class TransformerModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, dim_feedforward=256):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        # 定义 Transformer 编码器和解码器
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=dim_feedforward,
                                          batch_first=True
                                          )

    def forward(self, src, tgt):
        # 使用 Transformer 进行编码和解码
        output = self.transformer(src, tgt)
        # 解码器的输出是 (batch_size, seq_len, d_model)
        return output


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = encoder
        self.transformer = TransformerModel()
        self.liner = nn.Linear(256, 9)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)  # 使用 encoder 提取特征
        x = x.permute(0, 2, 1)
        x = self.transformer(x, x)  # 通过 tansformer 进行时序建模
        x = self.liner(x)
        x = x.reshape(-1, 9)
        return x


# 实例化 MyModel
model = MyModel()
device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
print("Running on the GPU")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
epochs = 400

# 寻找最大概率
max_res_train = 0
max_res_test = 0
# 设置训练日志
train_log_file = "logs/train_log.txt"
train_logger = logging.getLogger("train_logger")
train_handler = logging.FileHandler(train_log_file)
train_handler.setFormatter(logging.Formatter("%(message)s"))
train_logger.addHandler(train_handler)
train_logger.setLevel(logging.INFO)

# 设置测试日志
test_log_file = "logs/test_log.txt"
test_logger = logging.getLogger("test_logger")
test_handler = logging.FileHandler(test_log_file)
test_handler.setFormatter(logging.Formatter("%(message)s"))
test_logger.addHandler(test_handler)
test_logger.setLevel(logging.INFO)

for i in range(epochs):
    print("第{}轮训练".format(i + 1))
    train_step = 0
    sum_train_acc = 0
    model.train()
    for inputs, target in train_dataloader:
        labeled = target.add(-1)

        labeled = labeled.to(device)
        inputs = inputs.to(device)

        labeled = labeled.reshape(-1)
        # torch.autograd.set_detect_anomaly(True)
        out = model(inputs)
        out = out.to(torch.float32)
        loss = criterion(out, labeled)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('训练次数为：{},loss为：{}'.format(train_step * 32, loss))
        max_probs, predicted_class = torch.max(out, dim=-1)
        mubiao = target.detach().cpu().numpy()
        mubiao = mubiao.reshape(-1)
        shuchu = predicted_class.detach().cpu().numpy()
        # print(type(mubiao), type(shuchu))
        shuchu = shuchu + 1
        shuchu = np.round(shuchu)
        equal_indices = mubiao == shuchu
        # 计算相等数据的数量
        equal_count = np.sum(equal_indices)
        zong = mubiao.shape[0]
        # 将对应位置正确的数目除以元素总数，得到比例（正确率）
        accuracy = equal_count / zong
        train_step = train_step + 1
        sum_train_acc = sum_train_acc + accuracy
        if accuracy > max_res_train:
            max_res_train = accuracy
        # print('训练次数为：{},loss为：{} ,正确率为：{}'.format(train_step * 32, loss, accuracy))
    print('平均train_acc: {}'.format(sum_train_acc / train_step))
    log_train_message = f"Epoch {i + 1}: Avg Train Acc: {sum_train_acc / train_step:.4f}"
    # 训练日志记录每一轮的平均准确率
    train_logger.info(log_train_message)
    sum_test_acc = 0
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        test_step = 0
        for inputs, target in test_dataloader:
            target = target.to(device)
            inputs = inputs.to(device)
            # labeled = target.add(-1)
            out = model(inputs)  # 获取预测标签
            out = out.to(torch.float32)
            max_probs, predicted_class = torch.max(out, dim=-1)
            # test_loss = criterion(out, labeled)
            mubiao = target.detach().cpu().numpy()
            mubiao = mubiao.reshape(-1)
            shuchu = predicted_class.detach().cpu().numpy()
            shuchu = shuchu + 1
            shuchu = np.round(shuchu)
            equal_indices = mubiao == shuchu
            # 计算相等数据的数量
            equal_count = np.sum(equal_indices)
            zong = mubiao.shape[0]
            # 将对应位置正确的数目除以元素总数，得到比例（正确率）
            accuracy = equal_count / zong
            sum_test_acc = sum_test_acc + accuracy
            test_step = test_step + 1
            # print('正确率为：{}'.format(accuracy))
            if accuracy > max_res_test:
                max_res_test = accuracy
    print('平均test_acc: {}'.format(sum_test_acc / test_step))
    log_test_message = f"Epoch {i + 1}: Avg test Acc: {sum_test_acc / test_step:.4f}"
    # 记录每一轮的平均准确率
    test_logger.info(log_test_message)
    if (i + 1) % 50 == 0:
        log_message_max = f"Epoch {i + 1}: Max Train Acc: {max_res_train:.4f}, Max Test Acc: {max_res_test:.4f}"
        # 训练日志记录最大训练准确率
        train_logger.info(log_message_max)
        # 测试日志记录最大测试准确率
        test_logger.info(log_message_max)

print('train最高准确率：{}'.format(max_res_train))
print('test最高准确率：{}'.format(max_res_test))
