#train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_data():
    """
    加载数据集的函数
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'FER2013')
    # 数据预处理管道
    # 1. 转换为单通道灰度图
    # 2. 调整图像尺寸为48x48（模型输入要求）
    # 3. 转换为PyTorch张量
    # 4. 归一化（均值0.5，标准差0.5，将像素值从[0,1]映射到[-1,1]）
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'), # 训练集路径
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),# 验证集路径
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'), # 测试集路径
        transform=transform
    )

    return train_dataset, val_dataset, test_dataset


def setup_model_and_optimizer():
    """
        初始化模型、损失函数和优化器的函数
        :return: 模型实例、损失函数实例、优化器实例
        """
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    return model, criterion, optimizer


def evaluate_model(model, data_loader):
    """
    评估模型的函数
    :param model: 待评估的模型
    :param data_loader: 数据加载器
    :return: 模型在数据加载器上的准确率
    """
    correct = 0# 正确预测的样本数
    total = 0 # 总样本数
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            # 找到每个样本预测概率最大的类别索引
            _, predicted = outputs.max(1)
            total += labels.size(0)
            # 计算正确预测数（将预测结果与真实标签对比）
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


class SimpleCNN(nn.Module):
    """
      自定义简单卷积神经网络模型
      用于7类表情分类（Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral）
      """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积：输入通道1，输出通道16，3x3卷积核，边缘填充1（保持尺寸）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU() # ReLU激活函数
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层：2x2窗口，步长2（尺寸减半）
        # 第二层卷积：输入通道16，输出通道32，3x3卷积核，边缘填充1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 第一层全连接：输入维度32*12*12（池化后尺寸），输出128
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        # 第二层全连接：输入128，输出7（表情类别数）
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        """
               前向传播函数
               :param x: 输入张量（形状为[batch_size, 1, 48, 48]）
               :return: 输出张量（形状为[batch_size, 7]，未归一化的logits）
        """
        x = self.pool(self.relu(self.conv1(x)))# 卷积->激活->池化
        x = self.pool(self.relu(self.conv2(x)))# 第二层卷积操作
        # 将特征图展平为向量：[batch_size, 32, 12, 12] -> [batch_size, 32*12*12]
        x = x.reshape(-1, 32 * 12 * 12)
        x = self.relu(self.fc1(x))# 全连接层+激活函数
        x = self.fc2(x)# 最终分类层（输出logits）
        return x


def train_pytorch_model():
    """
        训练PyTorch模型的主函数
        :return: 训练完成后在验证集上的准确率
     """
    # 加载数据集并创建数据加载器
    train_dataset, val_dataset, _ = load_data()
    # 训练数据加载器：批量大小64，打乱数据顺序
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 验证数据加载器：批量大小64，不打乱数据
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 初始化模型、损失函数和优化器
    model, criterion, optimizer = setup_model_and_optimizer()

    num_epochs = 10# 训练轮数
    val_acc = 0  # 初始化val_acc，避免可能的未赋值前引用问题
    for epoch in range(num_epochs):
        model.train() # 设置模型为训练模式
        running_loss = 0.0# 累计训练损失
        correct = 0 # 正确预测数
        total = 0 # 总样本数
        for images, labels in train_loader:# 遍历训练批次
            optimizer.zero_grad()# 梯度清零（避免梯度累加）
            outputs = model(images)# 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播计算梯度
            optimizer.step()# 更新模型参数

            running_loss += loss.item()# 累加批次损失
            # 计算批次内准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        # 计算并打印训练集指标
        train_acc = 100. * correct / total
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_acc}%')
        # 评估验证集并打印准确率
        val_acc = evaluate_model(model, val_loader)
        print(f'Validation Accuracy: {val_acc}%')
    # 保存训练好的模型到models目录
    torch.save(model,'models/trained_model.pth')
    print("模型已成功保存！")
    return val_acc


if __name__ == "__main__":
    # 当脚本直接运行时执行训练流程
    accuracy = train_pytorch_model()
    print(f"最终验证集准确率: {accuracy}%")
