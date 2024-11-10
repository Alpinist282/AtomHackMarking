# AtomHackMarking
 
Репозиторий является наработкой в сфере ИИ,предназначенная для распознования маркировки на металлических изделиях. За основы взята модель CRNN для распознования текста. В репозитории хранится мобильное приложение, требующее интеграции с моделью и базой данных, размещенной локально.
## Установка
Установите все файлы для дальнейшей доработки в вашем редакторе кода
## Идея
Для распознования текста на металлической детали, пользователь фотографирует поверхность детали с содержащейся на ней маркировкой, либо же выбирает аналогичную фотографию из галлереи. После этого фотография обрабатывается нейронной сетью локально. Для этого должна быть предустановлена предобученная модель со своими весами в формате .pth. После того, как текст на изображении был распознан, передается запрос к локальной сети на обработку базы данных, для поиска нужной детали по серийному номеру.
## Доработка
Для улучшения качества модели требутеся ее доработать. На момент обучения на 100 эпохе модель имела следующие показатели:
![image](https://github.com/user-attachments/assets/966dc0b4-2c9e-44b9-aeeb-85d5500ab411)

Также для доработки следует встроить модель для поиска текста на картинке вроде yolov5, yolov8 или EAST для обнаружения текста и дальнейшего ее распознания моделью CRNN.
