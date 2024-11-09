import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OCRDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Bounding box и текстовая метка
        boxes = self.annotations.iloc[idx, 1].strip().split("\n")
        label_text = self.annotations.iloc[idx, 2]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label_text
    

train_transform = A.Compose([
    A.Resize(128, 128),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x32x32
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(32, 1), stride=(32, 1))  # добавляем MaxPool для высоты
        )
        self.rnn = nn.LSTM(256, nh, bidirectional=True)
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"Unexpected height {h} in feature map after CNN layers, expected 1."
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        return output
    

# Параметры модели
num_classes = len(set("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-/ "))  # все символы, которые могут встречаться
nh = 256  # скрытые слои для RNN

# Инициализация модели, функции потерь и оптимизатора
model = CRNN(imgH=128, nc=3, nclass=num_classes, nh=nh).to(device)
criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# DataLoader
train_data = OCRDataset(csv_file='/home/user1/AtomHackMarking-3/models/ocr_model/train/grounded true train.csv', image_dir='/home/user1/AtomHackMarking-3/models/ocr_model/train/train/imgs', transform=train_transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


# Тест
test_image_path = "models/ocr_model/train/train/imgs/19.JPG"

# Цикл обучения
for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Предсказания модели
        outputs = model(images)
        
        # Определение максимальной длины меток в текущем батче
        max_length = max(len(label) for label in labels)

        # Преобразуем метки в тензоры с учетом паддинга
        label_tensor = torch.tensor(
            [
                [ord(char) - ord(' ') for char in label] + [0] * (max_length - len(label))
                for label in labels
            ],
            dtype=torch.long
        )
        label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

        # Длины выходных предсказаний
        input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)

        # Вычисляем потерю
        loss = criterion(outputs, label_tensor, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")