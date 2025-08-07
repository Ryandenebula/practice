import numpy as np
import torch
import torch.nn as nn
from mamba_ssm import Mamba2  # 导入 Mamba2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵计算工具

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# python Mamba_new.py
file_path = "误差15%/train.txt"  # 请替换为实际的六种雷达数据文件路径

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
    def __init__(self, input_dim=256, hidden_dim=256, num_classes=9, d_state=128, d_conv=4, expand=16, headdim=64):
        super(MambaClassifier, self).__init__()
        # mamba堆叠
        self.mamba1 = Mamba2(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.mamba2 = Mamba2(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        
        self.mamba3 = Mamba2(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.mamba4 = Mamba2(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)


        # 标准化
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.bn3 = nn.BatchNorm1d(input_dim)
        self.bn4 = nn.BatchNorm1d(input_dim)

        self.attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.norm = nn.LayerNorm(256)

        # CNN 部分
        self.convd = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1d = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256))
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256))
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256))
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256))

        # dropout避免过拟合 fc全连接层  out输出种类
        self.dropout = nn.Dropout(p=0.5) # 0.5
        self.fc = nn.Linear(input_dim, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.convd(x)
        x0 = self.conv1d(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        out2 = F.elu(x2 + x1)

        x3 = self.conv3(out2)
        out3 = F.elu(x3 + out2)
        x4 = self.conv4(out3)
        out4 = F.elu(x4 + out3)
        out5 = out4.permute(0, 2, 1)

        # mamba块堆叠
        x = self.mamba1(out5)
        x = self.dropout(x)
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)  # Batch Normalization

        x = self.mamba2(x)
        x = self.dropout(x)
        x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)  # Batch Normalization

        x = self.mamba3(x)
        x = self.dropout(x)
        x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)  # Batch Normalization

        x = self.mamba4(x)
        x = self.dropout(x)
        x = self.bn4(x.permute(0, 2, 1)).permute(0, 2, 1)  # Batch Normalization


        attn_output, _ = self.attn(x, x, x)  # Q, K, V 都是 x
        x = self.norm(attn_output + x)
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)  # Batch Normalization
        attn_output, _ = self.attn(x, x, x)  # Q, K, V 都是 x
        x = self.norm(attn_output + x)
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)  # Batch Normalization
        x = self.dropout(x)
        
        x = self.fc(x)  # 全连接层
        x = torch.relu(x)
        x = self.out(x)

        return x


def load_and_preprocess_data(file_path):
    raw_data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
    num_samples = raw_data.shape[0]
    assert num_samples == 1000000, f"数据量必须为1000000，当前为{num_samples}"
    assert raw_data.shape[1] == 4, "每行必须有4列数据"

    features = raw_data[:, :3]
    labels = raw_data[:, 3]
    unique_labels = np.unique(labels)
    print(f"原始标签唯一值: {unique_labels}")
    if not np.all(np.isin(unique_labels, [0, 1, 2, 3, 4, 5, 6, 7, 8])):
        raise ValueError(f"标签应为0, 1, 2, 3, 4, 5，但发现了其他值: {unique_labels}")
    print(f"标签唯一值: {np.unique(labels)}")

    num_groups = num_samples // 1000
    data = features.reshape(num_groups, 1000, 3)
    labels = labels.reshape(num_groups, 1000)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    return train_data, test_data, train_labels, test_labels


# 训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device='cuda'):
    model.train()
    best_accuracy = 0.0
    best_cm = None
    best_epoch = 0

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs.view(-1, 9), batch_labels.view(-1))  # 修改为6类
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        scheduler.step()  # 使用学习率调度器

        # 每个 epoch 结束后测试
        accuracy, cm = evaluate_model(model, test_loader, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_cm = cm
            best_epoch = epoch

    print(f"最佳测试准确率: {best_accuracy:.2f}%，在第{best_epoch + 1}个epoch")
    print("最佳混淆矩阵:")
    print(best_cm)
    return model


# 测试函数（输出每类的正确率）
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 9  # 每类的正确数
    class_total = [0] * 9  # 每类的总数

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, dim=2)
            total += batch_labels.numel()
            correct += (predicted == batch_labels).sum().item()

            # 统计每类的正确率
            for i in range(9):
                class_total[i] += (batch_labels == i).sum().item()
                class_correct[i] += ((predicted == batch_labels) & (batch_labels == i)).sum().item()

            # 收集所有标签和预测结果
            all_labels.extend(batch_labels.cpu().numpy().flatten())
            all_predictions.extend(predicted.cpu().numpy().flatten())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    # accuracy = correct / total * 100
    accuracy = (correct - class_correct[0]) / (total - class_total[0]) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # 输出每类的正确率
    for i in range(9):
        if class_total[i] > 0:
            class_accuracy = class_correct[i] / class_total[i] * 100
            print(f"Class {i} Accuracy: {class_accuracy:.2f}% (Total: {class_total[i]})")
        else:
            print(f"Class {i} Accuracy: 0.00% (No samples)")

    return accuracy, cm


# 主函数
def main():
    print("正在加载数据...")
    train_data, test_data, train_labels, test_labels = load_and_preprocess_data(file_path)

    train_dataset = RadarDataset(train_data, train_labels)
    test_dataset = RadarDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    # 生成优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

    # 计算步数
    warmup_epochs = 10
    warmup_steps  = warmup_epochs * len(train_loader)
    cosine_steps  = (60 - warmup_epochs) * len(train_loader)

    # 构造组合调度器
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-5)
        ],
        milestones=[warmup_steps]
    )

    print("开始训练...")
    model = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=60)

    print("最终测试结果:")
    final_accuracy, final_cm = evaluate_model(model, test_loader, device)
    print(f"最终测试准确率: {final_accuracy:.2f}%")
    print("最终混淆矩阵:")
    print(final_cm)


if __name__ == "__main__":
    main()