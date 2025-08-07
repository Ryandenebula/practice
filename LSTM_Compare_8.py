import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time  # 导入时间模块

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)
# python LSTM_compare_8.py

file_path = "8种类补充实验/实验3/噪声率40%/train.txt"  # 请替换为实际的六种雷达数据文件路径

# 自定义数据集类
class RadarDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Mamba2 分类模型
class MambaClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_classes=8):
        super(MambaClassifier, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.lstm11 = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True)
        self.lstm12 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.lstm21 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.lstm22 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.lstm31 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.lstm32 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.lstm41 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.lstm42 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 转换为 [B, C, L]
        x0 = self.conv1d(x)  # 输出 [B, 64, L']
        x0 = x0.permute(0, 2, 1)  # 转换为 [B, L', 64]

        x11, _ = self.lstm11(x0)
        x12, _ = self.lstm12(x11)

        x21, _ = self.lstm21(x11)
        x22, _ = self.lstm22(x12)

        # x31, _ = self.lstm31(x21)
        # x32, _ = self.lstm32(x22)

        # x41, _ = self.lstm41(x31)
        # x42, _ = self.lstm42(x32)

        # 拼接每个时间步的输出
        out_concat = torch.cat([x21, x22], dim=-1)  # [B, L', hidden_dim * 2]

        x = self.fc(out_concat)
        x = torch.relu(x)
        x = self.out(x)  # 输出 [B, L', num_classes]

        return x


def load_and_preprocess_data(file_path):
    raw_data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
    num_samples = raw_data.shape[0]
    assert num_samples == 1000000, f"数据量必须为1000000，当前为{num_samples}"
    assert raw_data.shape[1] == 4, "每行必须有4列数据"

    features = raw_data[:, :3]
    labels = raw_data[:, 3].astype(np.int64)
    unique_labels = np.unique(labels)
    print(f"原始标签唯一值: {unique_labels}")
    if not np.all(np.isin(unique_labels, [0, 1, 2, 3, 4, 5, 6, 7])):
        raise ValueError(f"标签应为0, 1, 2, 3, 4, 5，但发现了其他值: {unique_labels}")
    print(f"标签唯一值: {np.unique(labels)}")

    num_groups = num_samples // 1000
    data = features.reshape(num_groups, 1000, 3)
    labels = labels.reshape(num_groups, 1000)

    # 检查每个组内的标签分布
    for i in range(num_groups):
        group_labels = labels[i]
        if len(np.unique(group_labels)) == 0:
            raise ValueError(f"第 {i} 组内的所有标签相同，这可能不是预期的行为")

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    return train_data, test_data, train_labels, test_labels


# 训练函数
def train_model(model, train_loader,test_loader, device, criterion, optimizer, num_epochs):
    model.to(device)  # 将模型移动到GPU上
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)  # 将数据移动到GPU上
            optimizer.zero_grad()

            outputs = model(batch_data)  # outputs 的形状是 [batch_size, seq_length, num_classes]

            # 调整输出和标签的形状
            batch_size, seq_length, num_classes = outputs.shape
            outputs = outputs.view(batch_size * seq_length, num_classes)
            batch_labels = batch_labels.view(-1)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            # 累加损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        # 计算平均损失和准确率
        avg_loss = running_loss / len(train_loader)
        avg_acc = correct / total * 100

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%")
        final_accuracy = evaluate_model(model, test_loader, device)

    return model


# 测试函数（输出每类的正确率）
def evaluate_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 6  # 每类的正确数
    class_total = [0] * 6  # 每类的总数

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)  # outputs 的形状是 [batch_size, seq_length, num_classes]
            _, predicted = torch.max(outputs, dim=2)  # predicted 的形状是 [batch_size, seq_length]
            batch_labels = batch_labels  # 标签的形状是 [batch_size, seq_length]

            # 展平预测和标签
            predicted = predicted.view(-1)
            batch_labels = batch_labels.view(-1)

            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            # 统计每类的正确率
            for i in range(6):
                class_total[i] += (batch_labels == i).sum().item()
                class_correct[i] += ((predicted == batch_labels) & (batch_labels == i)).sum().item()

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # 输出每类的正确率
    for i in range(6):
        if class_total[i] > 0:
            class_accuracy = class_correct[i] / class_total[i] * 100
            print(f"Class {i} Accuracy: {class_accuracy:.2f}% (Total: {class_total[i]})")
        else:
            print(f"Class {i} Accuracy: 0.00% (No samples)")

    return accuracy

# 主函数
def main():

    print("正在加载数据...")
    train_data, test_data, train_labels, test_labels = load_and_preprocess_data(file_path)

    train_dataset = RadarDataset(train_data, train_labels)
    test_dataset = RadarDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 确保设备定义正确
    model = MambaClassifier().to(device)  # 将模型移动到GPU上
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    start_time = time.time()  # 开始时间
    print("开始训练...")
    model = train_model(model, train_loader, test_loader,device, criterion, optimizer, num_epochs=100)
    end_time = time.time()  # 结束时间
    print(f"模型训练总耗时: {end_time - start_time:.2f}秒")
    print("最终测试结果:")
    final_accuracy = evaluate_model(model, test_loader, device)



if __name__ == "__main__":
    main()
