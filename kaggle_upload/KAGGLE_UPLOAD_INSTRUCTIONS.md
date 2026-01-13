# Как загрузить датасет на Kaggle

## Быстрый старт

### Вариант A: Через Kaggle CLI (рекомендую)

```bash
# 1. Подготовь данные
cd kaggle_upload
chmod +x prepare_kaggle_dataset.sh
./prepare_kaggle_dataset.sh

# 2. Установи Kaggle CLI
pip install kaggle

# 3. Настрой API credentials
# - Зайди на https://www.kaggle.com/settings
# - Scroll down до "API" section
# - Нажми "Create New Token"
# - Скачается файл kaggle.json
# - Положи его в ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 4. Создай датасет на Kaggle
cd kaggle_dataset
kaggle datasets create -p .

# 5. Если нужно обновить датасет позже
kaggle datasets version -p . -m "Updated with more samples"
```

### Вариант B: Через веб-интерфейс

```bash
# 1. Подготовь данные
cd kaggle_upload
chmod +x prepare_kaggle_dataset.sh
./prepare_kaggle_dataset.sh

# 2. Создай ZIP архив
cd kaggle_dataset
zip -r ../asl-landmarks.zip .
cd ..

# 3. Загрузи через браузер
# - Зайди на https://www.kaggle.com/datasets
# - Нажми "New Dataset"
# - Upload asl-landmarks.zip
# - Заполни метаданные (или они подтянутся из dataset-metadata.json)
# - Нажми "Create"
```

---

## Структура папок для загрузки

После выполнения `prepare_kaggle_dataset.sh` будет создана папка:

```
kaggle_dataset/
├── dataset-metadata.json    # Метаданные для Kaggle
├── README.md                 # Описание датасета
├── STATISTICS.txt            # Статистика по классам
├── example-usage.py          # Пример использования
└── landmarks/                # Сами данные
    ├── A/ (404 .npy files)
    ├── B/ (400 .npy files)
    ├── C/ (403 .npy files)
    ... (26 total)
    └── Z/ (403 .npy files)
```

**Размер:** ~41 MB

---

## После загрузки на Kaggle

### 1. Создай Example Notebook

Зайди на свой датасет и нажми "New Notebook". Скопируй код из `example-usage.py`:

```python
# В Kaggle notebook пути будут:
DATA_DIR = Path('../input/asl-alphabet-hand-landmarks/landmarks')
```

### 2. Обнови README проекта

Добавь в свой `helios/README.md`:

```markdown
## Dataset

Download from **[Kaggle](https://kaggle.com/datasets/borisgraudt/asl-alphabet-hand-landmarks)** (10,508 samples)

```bash
# Via Kaggle CLI
kaggle datasets download -d borisgraudt/asl-alphabet-hand-landmarks
unzip asl-alphabet-hand-landmarks.zip -d data/raw_gestures
```
```

### 3. Поделись

- Twitter/X: "Released ASL hand landmarks dataset on Kaggle 🤟"
- LinkedIn: Пост про открытый датасет
- Reddit: r/MachineLearning, r/datasets

---

## Tips для максимальной видимости

✅ **Do:**
- Создай Kaggle Notebook с примером (визуализация + обучение)
- Добавь красивые визуализации (3D hand plots)
- Напиши подробное описание в Dataset Description
- Добавь теги: `sign-language`, `accessibility`, `computer-vision`, `mediapipe`
- Upvote свой dataset 😄

❌ **Don't:**
- Не забудь про лицензию (CC0 уже в metadata)
- Не загружай лишние файлы (.git, __pycache__, etc)

---

## Troubleshooting

**Ошибка: "Dataset already exists"**
```bash
# Обнови существующий датасет
kaggle datasets version -p . -m "Update message"
```

**Ошибка: "kaggle.json not found"**
```bash
# Проверь путь
ls -la ~/.kaggle/kaggle.json
# Должны быть права 600
chmod 600 ~/.kaggle/kaggle.json
```

**Большой размер файла**
```bash
# Kaggle поддерживает до 20GB
# Твой датасет ~41MB - норм!
```

---

## Ссылки

- Kaggle API Docs: https://github.com/Kaggle/kaggle-api
- Kaggle Dataset Guidelines: https://www.kaggle.com/datasets
- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands

---

## Пример финального URL

После загрузки датасет будет доступен по адресу:

```
https://www.kaggle.com/datasets/borisgraudt/asl-alphabet-hand-landmarks
```

Этот URL можно добавить в:
- README проекта
- MODEL_CARD.md
- Презентации
- Резюме / портфолио

