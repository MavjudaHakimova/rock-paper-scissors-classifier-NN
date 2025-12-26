# Rock-Paper-Scissors Classifier

## Аннотация

### Постановка задачи

Цель проекта заключается в создании и обучении классификатора на основе CNN,
который будет классифицировать жесты рук в игре "rock-paper-scissors".
Классификатор можно будет в дальнейшем использовать для цифровой версии данной
игры.

### Зачем это нужно?

Разработка классификатора жестов "камень-ножницы-бумага" имеет практическое
применение в создании интерактивных игровых приложений, системах компьютерного
зрения и образовательных проектах по машинному обучению.

### Формат входных и выходных данных

**Входные данные**: Изображения формата .jpeg/.png, которые ресайзятся до
(224, 224) с помощью ImageDataGenerator. Генератор выдает батчи - NumPy-массивы
формата (32, 224, 224, 3).

**Выходные данные**: Вероятности принадлежности к каждому классу (Paper, Rock,
Scissors) в виде массива numpy с формой (batch_size, 3).

### Метрики

Используемые метрики:

- Accuracy (точность)
- F1-Score для каждого класса

**Целевые показатели**:

- Accuracy > 95%
- F1-Score > 0.90
- ![telegram-cloud-photo-size-2-5366557535118232856-y](https://github.com/user-attachments/assets/267aa954-2ba8-4a50-9f1c-76936b11c237)

### Датасеты

Используется датасет Rock-Paper-Scissors Dataset с Kaggle:
https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset/data

Датасет содержит 2,892 изображения (объем: 236.38 MB). Особенности:

- Разделен на train/test/validation
- Train: по 840 изображений для каждого жеста
- Test: 124 файла для каждого жеста
- Validation: суммарно 33 файла
- Все изображения на белом фоне, размером 300х300 пикселей
- Изображения созданы с использованием CGI технологий

### Моделирование

**Основная модель:** EfficientNetB0 с дополнительным классификационным слоем:

- EfficientNetB0 (предобученная на ImageNet) в качестве feature extractor
- Линейный классификационный слой для финальной классификации

### Внедрение
Проект можно собрать с помощью `Makefile`

## Структура проекта

```
rps-classifier/
├── conf/ # Hydra configs
├── data/ # Data
│ └── train
│ └── validation
│ └── test
├── models/ # Model check-points
├── plots/ # Loss and metrics visualization
├── rps/ # The main package
│ ├── init.py
│ ├── data.py # Data preprocessing
│ ├── model.py # Model architecture
│ ├── module.py # PyTorch Lightning module
│ └── train.py # Training pipeline
│ └── train_with_dvc.py # Training pipeline
├── Makefile
├── .pre-commit-config.yaml # Git hooks
├── pyproject.toml # Poetry config
├── uv.lock # Dependency locks
└── README.md # Project documentation
```

## Установка и использование

1. Клонирование репозитория

```
git clone https://github.com/MavjudaHakimova/rock-paper-scissors-classifier-NN.git
```

2. Перейдите в директорию проекта

```
cd rps-classifier
```

3. Установка и настройка (все зависимости через Makefile)

```
make setup
```

4. Чтобы запустить обучение с логгированием в одном окне терминала запустите
   MLFlow

```
make ui
```

5. Запустите обучение в другом окне терминала

```
make test
```

6. Для запустка inference создайте папку `inference_dir` с данными и запустите
   инференс:

```
make infer
```

7. Для автоматической проверки pre-commit можно сделать
   ```
   make lint
   ```
   ![telegram-cloud-photo-size-2-5366557535118233941-y](https://github.com/user-attachments/assets/6aab95e2-9bc6-4223-8d10-57c76db25229)

