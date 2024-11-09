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
import Levenshtein

from sklearn.model_selection import train_test_split

# Проверка доступности CUDA
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OCRDataset(Dataset):
    def __init__(self, annotations, image_dir, transform=None):
        self.annotations = annotations.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Изображение не найдено: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_text = self.annotations.iloc[idx, 2]  # Проверьте индекс или используйте название столбца

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label_text

# Загрузка аннотаций
annotations = pd.read_csv('C:/Users/ymur2/AppData/Roaming/Factorio/AtomHackMarking/models/ocr_model/train/grounded true train.csv')

# Вывод названий столбцов для проверки
print(annotations.columns)

# Предположим, что столбец с метками называется 'label'
labels = annotations['label_text'].tolist()  # Замените 'label' на название вашего столбца с метками

# Создание отображений
all_characters = set()
for label in labels:
    all_characters.update(list(label))
all_characters = sorted(list(all_characters))

char2idx = {char: idx + 1 for idx, char in enumerate(all_characters)}  # Начинаем с 1
idx2char = {idx: char for char, idx in char2idx.items()}

num_classes = len(all_characters) + 1  # +1 для blank label

# Разделение аннотаций на тренировочные и тестовые
train_annotations, test_annotations = train_test_split(annotations, test_size=0.2, random_state=42)

# Определение трансформаций
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

# Создание датасетов
train_data = OCRDataset(
    annotations=train_annotations,
    image_dir='C:/Users/ymur2/AppData/Roaming/Factorio/AtomHackMarking/models/ocr_model/train/train/imgs',
    transform=train_transform
)

test_data = OCRDataset(
    annotations=test_annotations,
    image_dir='C:/Users/ymur2/AppData/Roaming/Factorio/AtomHackMarking/models/ocr_model/train/train/imgs',
    transform=test_transform
)

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    return images, labels

# Создание DataLoader
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
    
def decode_batch_predictions(outputs, idx2char):
    # outputs: [T, N, C]
    _, max_indices = outputs.max(2)  # [T, N]
    max_indices = max_indices.transpose(1, 0)  # [N, T]
    decoded_texts = []
    for indices in max_indices:
        pred_text = ''
        prev_idx = None
        for idx in indices:
            idx = idx.item()
            if idx != 0 and idx != prev_idx:
                pred_text += idx2char.get(idx, '')
            prev_idx = idx
        decoded_texts.append(pred_text)
    return decoded_texts

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_cer = 0
    total_wer = 0
    num_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            # Подготовка меток
            label_indices = []
            for label in labels:
                indices = [char2idx[char] for char in label if char in char2idx]
                label_indices.append(indices)
            
            label_lengths = torch.tensor([len(label) for label in label_indices], dtype=torch.long).to(device)
            label_tensor = torch.cat([torch.tensor(t, dtype=torch.long) for t in label_indices]).to(device)
            
            # Подготовка длин входных последовательностей
            T, N, _ = outputs.size()
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)
            
            # Преобразуем выходы для CTC Loss
            outputs_log_softmax = outputs.log_softmax(2)
            
            # Вычисление потерь
            loss = criterion(outputs_log_softmax, label_tensor, input_lengths, label_lengths)
            total_loss += loss.item()
            
            # Декодирование предсказаний
            decoded_texts = decode_batch_predictions(outputs, idx2char)
            
            # Вычисление CER и WER
            for pred_text, true_text in zip(decoded_texts, labels):
                cer = Levenshtein.distance(pred_text, true_text) / max(len(true_text), 1)
                wer = Levenshtein.distance(pred_text.split(), true_text.split()) / max(len(true_text.split()), 1)
                total_cer += cer
                total_wer += wer
                num_samples += 1
            
    avg_loss = total_loss / len(dataloader)
    avg_cer = total_cer / num_samples
    avg_wer = total_wer / num_samples
    return avg_loss, avg_cer, avg_wer

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

# Инициализация списков для хранения метрик
train_losses = []
train_cers = []
train_wers = []
test_losses = []
test_cers = []
test_wers = []

# Цикл обучения
num_epochs = 3
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    total_cer = 0
    total_wer = 0
    num_samples = 0
    for images, labels in train_loader:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(images)  # Outputs shape: [w, b, nclass]
        
        # Подготовка меток
        label_indices = []
        for label in labels:
            indices = [char2idx[char] for char in label if char in char2idx]
            label_indices.append(indices)
        
        label_lengths = torch.tensor([len(label) for label in label_indices], dtype=torch.long).to(device)
        label_tensor = torch.cat([torch.tensor(t, dtype=torch.long) for t in label_indices]).to(device)
        
        # Подготовка длин входных последовательностей
        T, N, _ = outputs.size()
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)
        
        # Преобразуем выходы для CTC Loss
        outputs_log_softmax = outputs.log_softmax(2)
        
        # Вычисление потери
        loss = criterion(outputs_log_softmax, label_tensor, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Декодирование предсказаний
        decoded_texts = decode_batch_predictions(outputs, idx2char)
        
        # Вычисление CER и WER
        for pred_text, true_text in zip(decoded_texts, labels):
            cer = Levenshtein.distance(pred_text, true_text) / max(len(true_text), 1)
            wer = Levenshtein.distance(pred_text.split(), true_text.split()) / max(len(true_text.split()), 1)
            total_cer += cer
            total_wer += wer
            num_samples += 1
    
    avg_loss = epoch_loss / len(train_loader)
    avg_cer = total_cer / num_samples
    avg_wer = total_wer / num_samples

    # В конце эпохи сохраняем метрики обучения
    train_losses.append(avg_loss)
    train_cers.append(avg_cer)
    train_wers.append(avg_wer)

     # Оцениваем модель на тестовом наборе
    test_loss, test_cer, test_wer = evaluate(model, test_loader)
    test_losses.append(test_loss)
    test_cers.append(test_cer)
    test_wers.append(test_wer)

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, CER: {avg_cer:.4f}, WER: {avg_wer:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, CER: {test_cer:.4f}, WER: {test_wer:.4f}")
    
    # Сохраняем модель после каждой эпохи
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f"Модель сохранена после эпохи {epoch+1}")

# После завершения обучения строим графики
import matplotlib.pyplot as plt

epochs = range(1, num_epochs + 1)

# График функции потерь
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, test_losses, 'r-', label='Test Loss')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.title('Потери на обучении и тесте по эпохам')
plt.legend()

# График CER
plt.subplot(1, 3, 2)
plt.plot(epochs, train_cers, 'b-', label='Training CER')
plt.plot(epochs, test_cers, 'r-', label='Test CER')
plt.xlabel('Эпоха')
plt.ylabel('CER')
plt.title('CER на обучении и тесте по эпохам')
plt.legend()

# График WER
plt.subplot(1, 3, 3)
plt.plot(epochs, train_wers, 'b-', label='Training WER')
plt.plot(epochs, test_wers, 'r-', label='Test WER')
plt.xlabel('Эпоха')
plt.ylabel('WER')
plt.title('WER на обучении и тесте по эпохам')
plt.legend()

plt.tight_layout()
plt.show()

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