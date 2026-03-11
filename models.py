#models.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# 定义PyTorch模型
class PyTorchCNN(nn.Module):
    """
       自定义卷积神经网络模型
       用于7类情绪分类任务（Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral）
       网络结构：
       - 2层卷积+池化模块
       - 2层全连接层
       """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层卷积块：输入通道1，32个3x3卷积核（无填充），ReLU激活，2x2最大池化
            nn.Conv2d(1, 32, 3),  # 输入尺寸48x48 → 输出46x46（3x3卷积无填充）
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2),  # 池化后尺寸23x23（46/2）
            # 第二层卷积块：32→64通道，3x3卷积核（无填充），ReLU激活，2x2最大池化
            nn.Conv2d(32, 64, 3),  # 输入23x23 → 输出21x21（23-3+1=21）
            nn.ReLU(),
            nn.MaxPool2d(2),  # 池化后尺寸10x10（21//2=10）

            # 全连接层：特征展平后接入两层线性变换
            nn.Flatten(),  # 展平为一维向量：64×10×10=6400
            nn.Linear(64 * 10 * 10, 64),  # 降维至64维，ReLU激活
            nn.ReLU(),
            nn.Linear(64, 7)  # 输出7维logins（对应7种情绪类别）
        )

    def forward(self, x):
        """
               前向传播函数
               :param x: 输入张量，形状为[batch_size, 1, 48, 48]（单通道灰度图）
               :return: 输出张量，形状为[batch_size, 7]（未归一化的类别得分）
               """
        return self.net(x)


# 加载数据（从项目文件目录加载）
def load_data():
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据集根目录的相对路径
    data_dir = os.path.join(current_dir, 'FER2013')

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=transform
    )

    return train_dataset, val_dataset, test_dataset


# 数据预处理和加载
def prepare_data(train_dataset, val_dataset, test_dataset):
    """
      创建数据加载器（DataLoader）
      :param train_dataset: 训练集Dataset对象
      :param val_dataset: 验证集Dataset对象
      :param test_dataset: 测试集Dataset对象
      :return: 训练/验证/测试数据加载器
      """
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs):
    """
       训练模型主循环
       :param model: 待训练的PyTorch模型
       :param train_loader: 训练数据加载器
       :param criterion: 损失函数（如CrossEntropyLoss）
       :param optimizer: 优化器（如Adam）
       :param epochs: 训练轮数
       :return: 训练好的模型
       """
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1} Loss: {running_loss / len(train_loader):.4f}')

    return model


# 评估模型
def evaluate_model(model, val_loader, test_loader):
    # 在验证集上评估
    """
      在验证集和测试集上评估模型性能
      :param model: 训练好的PyTorch模型
      :param val_loader: 验证数据加载器
      :param test_loader: 测试数据加载器
      :return: 验证集准确率、测试集准确率
      """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f'Validation Accuracy: {val_accuracy:.4f}')

    # 在测试集上评估
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')

    return val_accuracy, test_accuracy


# 主函数
def main():
    # 加载数据
    train_dataset, val_dataset, test_dataset = load_data()
    # 数据预处理和加载
    train_loader, val_loader, test_loader = prepare_data(train_dataset, val_dataset, test_dataset)

    # 初始化模型、损失函数和优化器
    model = PyTorchCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练模型
    trained_model = train_model(model, train_loader, criterion, optimizer, epochs=5)

    # 评估模型
    val_accuracy, test_accuracy = evaluate_model(trained_model, val_loader, test_loader)


if __name__ == "__main__":
    main()