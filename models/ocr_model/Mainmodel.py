import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# Определим набор символов
SYMBOL_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzAБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789. "

# Определим класс для загрузки данных
class OCRDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            return None

        try:
            image = Image.open(img_path).convert('L')
        except UnidentifiedImageError:
            print(f"Unidentified image file: {img_path}")
            return None

        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            label = f.read().strip()

        # Проверка на наличие неожиданных символов
        for c in label:
            if c not in SYMBOL_SET:
                print(f"Unexpected character '{c}' in label file: {label_path}")
                return None

        label = [SYMBOL_SET.index(c) for c in label]
        label = torch.tensor(label, dtype=torch.long)

        return image, label

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((1280, 720)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Создание DataLoader
train_dataset = OCRDataset(image_dir='/home/user1/AtomHackMarking-1/train/imgs', label_dir='/home/user1/AtomHackMarking-1/train/labels', transform=transform)

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Определение модели
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 40, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(40, 60, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

        self.classifier = nn.Linear(60, len(SYMBOL_SET))
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.permute(0, 3, 1, 2).view(x.size(0), x.size(3), -1)
        x = self.classifier(x)
        x = x.permute(1, 0, 2)
        x = self.log_softmax(x)
        return x
    
model = MyNet()


# Определение функции потерь и оптимизатора
criterion = nn.CTCLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        if batch is None:
            continue
        images, labels = batch

        optimizer.zero_grad()

        # Предсказание
        outputs = model(images)

        # Вычисление потерь
        input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
        target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        loss = criterion(outputs, labels, input_lengths, target_lengths)

        # Обратное распространение и оптимизация
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Сохранение модели
torch.save(model.state_dict(), 'ocr_model.pth')
