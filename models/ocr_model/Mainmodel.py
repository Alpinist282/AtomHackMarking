import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

print(torch.cuda.is_available())  # Должно вывести True, если CUDA доступна
print(torch.cuda.device_count())  # Покажет количество доступных GPU
print(torch.cuda.get_device_name(0))  # Покажет имя вашей видеокарты

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
        label_text = self.annotations.iloc[idx, 2]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label_text

# Инициализация временного датасета для сбора символов
full_data = OCRDataset(
    csv_file='C:/Users/ymur2/AppData/Roaming/Factorio/AtomHackMarking/models/ocr_model/train/grounded true train.csv',
    image_dir='C:/Users/ymur2/AppData/Roaming/Factorio/AtomHackMarking/models/ocr_model/train/train/imgs',
    transform=None
)

# Сбор всех символов из меток
labels = full_data.annotations.iloc[:, 2].tolist()
all_characters = set()
for label in labels:
    all_characters.update(list(label))
all_characters = sorted(list(all_characters))

# Создание отображений
char2idx = {char: idx + 1 for idx, char in enumerate(all_characters)}  # Начинаем с 1; 0 для blank label
idx2char = {idx: char for char, idx in char2idx.items()}
# Обновление num_classes

num_classes = len(all_characters) + 1  # +1 для blank label

indices = list(range(len(full_data)))

train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42) #разделение!

# Теперь можно создать трансформации и загрузчики данных
train_transform = A.Compose([
    A.Resize(512, 512),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
    A.Rotate(limit=5, p=0.3),
    A.GaussNoise(var_limit=(5.0, 10.0), p=0.1),
    A.Sharpen(p=0.1),
    A.Emboss(p=0.1),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
train_data = torch.utils.data.Subset(full_data, train_indices)
train_data.dataset.transform = train_transform

# Создаем тестовый датасет
test_data = torch.utils.data.Subset(full_data, test_indices)
test_data.dataset.transform = test_transform

# Создание DataLoader для тренировочного и тестового наборов


# Функция для визуализации аугментаций
def visualize_augmentations(dataset, idx=0, samples=5):
    import matplotlib.pyplot as plt

    images, labels = [], []
    for i in range(samples):
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)

    fig, axes = plt.subplots(1, samples, figsize=(15, 3))
    for img, lbl, ax in zip(images, labels, axes):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5) + 0.5  # Обратная нормализация
        ax.imshow(img_np)
        ax.set_title(lbl)
        ax.axis('off')
    plt.show()

# Вызов функции визуализации
visualize_augmentations(train_data, idx=0, samples=5)

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    return images, labels

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            # Первый блок
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),  # [N, 64, 512, 512]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [N, 64, 256, 256]
            # Второй блок
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [N, 128, 256, 256]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [N, 128, 128, 128]
            # Третий блок
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [N, 256, 128, 128]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [N, 256, 64, 128]
            # Четвертый блок
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # [N, 256, 64, 128]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [N, 256, 32, 128]
            # Пятый блок
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # [N, 512, 32, 128]
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [N, 512, 16, 128]
            # Шестой блок
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # [N, 512, 16, 128]
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [N, 512, 8, 128]
            # Седьмой блок
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # [N, 512, 8, 128]
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Последний MaxPool2d по высоте
            nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1)),  # [N, 512, 1, 128]
        )
        self.rnn = nn.LSTM(512, nh, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        print(f"Размеры после CNN: batch_size={b}, channels={c}, height={h}, width={w}")
        assert h == 1, f"Высота после сверточных слоев должна быть 1, но получено h={h}"
        conv = conv.squeeze(2)  # Удаляем высоту (h)
        conv = conv.permute(2, 0, 1)  # [w, batch_size, channels]
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        return output  # [w, batch_size, nclass]
    
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            # Подготовка меток (как в тренировочном цикле)
            label_indices = []
            for label in labels:
                indices = []
                for char in label:
                    if char in char2idx:
                        indices.append(char2idx[char])
                label_indices.append(indices)
            
            label_lengths = torch.tensor([len(label) for label in label_indices], dtype=torch.long).to(device)
            label_tensor = torch.cat([torch.tensor(t, dtype=torch.long) for t in label_indices]).to(device)
            
            # Подготовка длин входных последовательностей
            T, N, _ = outputs.size()
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)
            
            # Преобразуем выходы для CTC Loss
            outputs = outputs.log_softmax(2)
            
            # Вычисление потерь
            loss = criterion(outputs, label_tensor, input_lengths, label_lengths)
            total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Параметры модели
nh = 256  # размер скрытого состояния для RNN

# Инициализация модели, функции потерь и оптимизатора
model = CRNN(imgH=32, nc=3, nclass=num_classes, nh=nh).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Проверяем, существует ли сохраненная модель
model_path = 'ocr_model.pth'
start_epoch = 0
if os.path.exists(model_path):
    print("Загружаем сохраненную модель...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Возобновление обучения с эпохи {start_epoch}")

# Цикл обучения
num_epochs = 100
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(images)  # Outputs shape: [w, b, nclass]
        
        # Подготовка меток
        label_indices = []
        for label in labels:
            indices = []
            for char in label:
                if char in char2idx:
                    indices.append(char2idx[char])
                else:
                    print(f"Символ '{char}' не найден в словаре.")
            label_indices.append(indices)
        
        label_lengths = torch.tensor([len(label) for label in label_indices], dtype=torch.long).to(device)
        label_tensor = torch.cat([torch.tensor(t, dtype=torch.long) for t in label_indices]).to(device)
        
        # Подготовка длин входных последовательностей
        T, N, _ = outputs.size()
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)
        
        # Преобразуем выходы для CTC Loss
        outputs = outputs.log_softmax(2)
        
        # Вычисление потери
        loss = criterion(outputs, label_tensor, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")
    
    # Оцениваем модель на тестовом наборе
    test_loss = evaluate(model, test_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}")
    
    # Сохраняем модель после каждой эпохи
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f"Модель сохранена после эпохи {epoch+1}")

# Функция декодирования предсказаний
def decode_predictions(outputs, idx2char):
    _, preds = outputs.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    pred_text = ''
    for i in range(len(preds)):
        if preds[i] != 0 and (not (i > 0 and preds[i] == preds[i -1])):
            pred_text += idx2char[preds[i].item()]
    return pred_text


# Пример инференса на тестовом изображении
import random
idx = random.randint(0, len(test_data) - 1)
test_image, test_label = test_data[idx]

# Преобразуем тензор изображения для визуализации
test_image_np = test_image.cpu().numpy()
test_image_np = np.transpose(test_image_np, (1, 2, 0))
test_image_np = (test_image_np * 0.5) + 0.5  # Обратная нормализация

plt.imshow(test_image_np)
plt.title(f'Реальный текст: {test_label}')
plt.axis('off')
plt.show()

# Предсказание модели
model.eval()
with torch.no_grad():
    test_image_tensor = test_image.unsqueeze(0).to(device)
    outputs = model(test_image_tensor)
    pred_text = decode_predictions(outputs, idx2char)
    print("Распознанный текст:", pred_text)
import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

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
        label_text = self.annotations.iloc[idx, 2]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label_text

# Инициализация временного датасета для сбора символов
temp_data = OCRDataset(
    csv_file='/home/user1/AtomHackMarking-3/models/ocr_model/train/grounded true train.csv',
    image_dir='/home/user1/AtomHackMarking-3/models/ocr_model/train/train/imgs',
    transform=None
)

# Сбор всех символов из меток
labels = temp_data.annotations.iloc[:, 2].tolist()
all_characters = set()
for label in labels:
    all_characters.update(list(label))
all_characters = sorted(list(all_characters))

# Создание отображений
char2idx = {char: idx + 1 for idx, char in enumerate(all_characters)}  # Начинаем с 1; 0 для blank label
idx2char = {idx: char for char, idx in char2idx.items()}

# Обновление num_classes
num_classes = len(all_characters) + 1  # +1 для blank label

# Теперь можно создать трансформации и загрузчики данных
train_transform = A.Compose([
    A.Resize(512, 512),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
    A.Rotate(limit=5, p=0.3),
    A.GaussNoise(var_limit=(5.0, 10.0), p=0.1),
    A.Sharpen(p=0.1),
    A.Emboss(p=0.1),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

train_data = OCRDataset(
    csv_file='/home/user1/AtomHackMarking-3/models/ocr_model/train/grounded true train.csv',
    image_dir='/home/user1/AtomHackMarking-3/models/ocr_model/train/train/imgs',
    transform=train_transform
)
# Функция для визуализации аугментаций
def visualize_augmentations(dataset, idx=0, samples=5):
    import matplotlib.pyplot as plt

    images, labels = [], []
    for i in range(samples):
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)

    fig, axes = plt.subplots(1, samples, figsize=(15, 3))
    for img, lbl, ax in zip(images, labels, axes):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5) + 0.5  # Обратная нормализация
        ax.imshow(img_np)
        ax.set_title(lbl)
        ax.axis('off')
    plt.show()

# Вызов функции визуализации
visualize_augmentations(train_data, idx=0, samples=5)

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    return images, labels

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            # Первый блок
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),  # [N, 64, 512, 512]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [N, 64, 256, 256]
            # Второй блок
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [N, 128, 256, 256]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [N, 128, 128, 128]
            # Третий блок
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [N, 256, 128, 128]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [N, 256, 64, 128]
            # Четвертый блок
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # [N, 256, 64, 128]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [N, 256, 32, 128]
            # Пятый блок
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # [N, 512, 32, 128]
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [N, 512, 16, 128]
            # Шестой блок
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # [N, 512, 16, 128]
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [N, 512, 8, 128]
            # Седьмой блок
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # [N, 512, 8, 128]
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Последний MaxPool2d по высоте
            nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1)),  # [N, 512, 1, 128]
        )
        self.rnn = nn.LSTM(512, nh, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        print(f"Размеры после CNN: batch_size={b}, channels={c}, height={h}, width={w}")
        assert h == 1, f"Высота после сверточных слоев должна быть 1, но получено h={h}"
        conv = conv.squeeze(2)  # Удаляем высоту (h)
        conv = conv.permute(2, 0, 1)  # [w, batch_size, channels]
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        return output  # [w, batch_size, nclass]

# Параметры модели
nh = 256  # размер скрытого состояния для RNN

# Инициализация модели, функции потерь и оптимизатора
model = CRNN(imgH=32, nc=3, nclass=num_classes, nh=nh).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Проверяем, существует ли сохраненная модель
model_path = 'ocr_model.pth'
start_epoch = 0
if os.path.exists(model_path):
    print("Загружаем сохраненную модель...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Возобновление обучения с эпохи {start_epoch}")


# Цикл обучения
num_epochs = 180
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(images)  # Outputs shape: [w, b, nclass]
        
        # Подготовка меток
        label_indices = []
        for label in labels:
            indices = []
            for char in label:
                if char in char2idx:
                    indices.append(char2idx[char])
                else:
                    print(f"Символ '{char}' не найден в словаре.")
            label_indices.append(indices)
        
        label_lengths = torch.tensor([len(label) for label in label_indices], dtype=torch.long).to(device)
        label_tensor = torch.cat([torch.tensor(t, dtype=torch.long) for t in label_indices]).to(device)
        
        # Подготовка длин входных последовательностей
        T, N, _ = outputs.size()
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)
        
        # Преобразуем выходы для CTC Loss
        outputs = outputs.log_softmax(2)
        
        # Вычисление потери
        loss = criterion(outputs, label_tensor, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Сохраняем модель после каждой эпохи
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f"Модель сохранена после эпохи {epoch+1}")

# Функция декодирования предсказаний
def decode_predictions(outputs, idx2char):
    _, preds = outputs.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    pred_text = ''
    for i in range(len(preds)):
        if preds[i] != 0 and (not (i > 0 and preds[i] == preds[i -1])):
            pred_text += idx2char[preds[i].item()]
    return pred_text


# Пример инференса на тестовом изображении
test_image_path = "/home/user1/AtomHackMarking-3/models/ocr_model/train/train/imgs/19.JPG"

model.eval()
with torch.no_grad():
    test_image = cv2.imread(test_image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    augmented = train_transform(image=test_image)
    transformed_image = augmented['image']  # Получаем тензор изображения после преобразований
    
    # Отображение изображения
    test_image_np = transformed_image.cpu().numpy()
    test_image_np = np.transpose(test_image_np, (1, 2, 0))  # [C, H, W] -> [H, W, C]
    test_image_np = (test_image_np * 0.5) + 0.5  # Обратная нормализация
    plt.imshow(test_image_np)
    plt.title('Изображение после преобразований')
    plt.axis('off')
    plt.show()
    
    test_image_tensor = transformed_image.unsqueeze(0).to(device)
    outputs = model(test_image_tensor)
    pred_text = decode_predictions(outputs, idx2char)
    print("Распознанный текст:", pred_text)