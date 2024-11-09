Структура файлов: 
train
	imgs - папка с изображениями
	labels - папка с bbox в формате YOLO (<class_id> <x_center> <y_center> <width> <height>)
	labels_with_text - папка с bbox в формате YOLO + извлеченная маркировка

Метрика_Росатом.ipynb - файл для расчета метрики (так же будет считаться метрика и на платформе) 
sample_submission.csv - пример сабмита для тестового датасета