import torch

# Путь к вашему предобученному файлу модели
model_path = '/home/user1/AtomHackMarking-1/models/translation_model/cyrillic_g2.pth'

# Загружаем содержимое модели (веса)
model_weights = torch.load(model_path, map_location='cpu')

# Выводим содержимое
print(model_weights)
