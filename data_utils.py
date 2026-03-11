#data_utils.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# 定义加载数据的函数
def load_data():
    images = []  # 用于存储加载的图片数据
    labels = []  # 用于存储图片对应的情绪标签
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    num_classes = len(emotion_labels)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FER2013', 'val')
    # 遍历每个情绪类别（0 - 6）
    for label in range(num_classes):
        # 拼接当前情绪类别的文件夹路径
        class_dir = os.path.join(data_dir, str(label))
        # 遍历当前类别文件夹中的所有图片文件
        for img_file in os.listdir(class_dir):
            # 拼接图片的完整路径
            img_path = os.path.join(class_dir, img_file)
            # 以灰度模式读取图片
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # 将图片尺寸调整为48x48像素，这是模型输入的要求尺寸
            img = cv2.resize(img, (48, 48))
            # 将图片添加到图片列表中
            images.append(img)
            # 将对应的情绪标签添加到标签列表中
            labels.append(label)
    # 将图片列表转换为NumPy数组，并调整维度以适应模型输入
    images = np.array(images, dtype='float32').reshape(-1, 48, 48, 1)
    # 对图片数据进行归一化处理，将像素值缩放到0 - 1之间
    images = images / 255.0
    print("Loaded images shape:", images.shape)
    print("Loaded labels shape:", len(labels))
    return images, labels  # 返回处理后的图片数据和标签


# 数据增强函数（PyTorch版本）
def data_augmentation_pytorch():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomShear(0.1),
        transforms.RandomZoom(0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])


# 定义自定义数据集类
class EmotionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# 划分数据集
def split_dataset(images, labels, test_size=0.2, random_state=42):
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=random_state)
    print("Train images shape:", train_images.shape)
    print("Train labels shape:", len(train_labels))
    print("Test images shape:", test_images.shape)
    print("Test labels shape:", len(test_labels))
    return train_images, test_images, train_labels, test_labels


if __name__ == "__main__":
    images, labels = load_data()
    train_images, test_images, train_labels, test_labels = split_dataset(images, labels)