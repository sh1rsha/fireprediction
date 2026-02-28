# Wildfire AI — Система обнаружения лесных пожаров

Лёгкий конвейер детекции лесных пожаров по спутниковым снимкам.  
Работает **без ML-модели и GPU** — цветовой анализ в HSV-пространстве + NASA GIBS/FIRMS API.  
Полностью функционален **оффлайн** на синтетических данных.

---

## Содержание

- [Как это работает](#как-это-работает)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)
- [Запуск](#запуск)
- [Конфигурация](#конфигурация)
- [Описание модулей](#описание-модулей)
  - [main.py](#mainpy)
  - [data_loader.py](#data_loaderpy)
  - [inference.py](#inferencepy)
  - [utils.py](#utilspy)
- [Выходные данные](#выходные-данные)
- [Внешние API](#внешние-api)
- [Ограничения](#ограничения)
- [Бизнес-применение](#бизнес-применение)

---

## Как это работает

```
Запуск main.py
     │
     ├─► data_loader.py  ──  пробует NASA GIBS (онлайн)
     │                        если не удалось → синтетика (оффлайн)
     │
     └─► inference.py
              ├── BGR → HSV-маска красных/оранжевых зон
              ├── морфология (шум убран, очаги укрупнены)
              ├── поиск контуров → bounding box + площадь
              ├── уровень угрозы: СРЕДНИЙ / ВЫСОКИЙ / КРИТИЧЕСКИЙ
              └── визуализация (3 панели) + JSON-отчёт
```

**Почему HSV, а не нейросеть?**  
HSV позволяет запускать детекцию без GPU, без данных для обучения, прямо на ноутбуке. Архитектура спроектирована так, чтобы YOLOv8 (`ultralytics`, уже в `requirements.txt`) можно было подключить позже без переписывания пайплайна.

---

## Структура проекта

```
wildfire-ai/
├── main.py             # точка входа и оркестрация
├── inference.py        # детекция, расчёт площади, визуализация
├── data_loader.py      # NASA GIBS/FIRMS + синтетика
├── utils.py            # логирование, JSON, вспомогательные функции
├── requirements.txt
│
├── data/
│   └── sample_images/  # скачанные и синтетические PNG (создаётся автоматически)
│
└── results/            # выходные JSON и PNG (создаётся автоматически)
    ├── wildfire_report_<timestamp>.json
    └── wildfire_visualization_<timestamp>.png
```

---

## Установка

**Требования:** Python 3.9+

```bash
git clone <repo-url>
cd wildfire-ai

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python   # чтение изображений, HSV, морфология, bounding box
Pillow          # декодирование PNG из HTTP-ответа GIBS
numpy           # массивы, маски
matplotlib      # трёхпанельная визуализация
requests        # HTTP-клиент для NASA FIRMS / GIBS
```

> `ultralytics` (YOLOv8) не указан в `requirements.txt` намеренно — он подключается  
> отдельно при переходе на ML-детекцию.

---

## Запуск

### Оффлайн-демо (ключи не нужны)

```bash
python main.py
```

Что происходит:
1. Скрипт пытается загрузить снимок через NASA GIBS.
2. Если GIBS недоступен — генерирует синтетическое изображение 512×512 с 4 случайными «очагами».
3. Запускает детекцию.
4. Сохраняет результаты в `results/`.

### Онлайн-режим (с реальными данными NASA)

```bash
# Linux / macOS
export FIRMS_API_KEY="ВАШ_КЛЮЧ"

# Windows PowerShell
$env:FIRMS_API_KEY="ВАШ_КЛЮЧ"

python main.py
```

Ключ NASA FIRMS получить бесплатно: https://firms.modaps.eosdis.nasa.gov/api/area/

---

## Конфигурация

Все настройки — константы в начале файлов. Файл конфигурации не используется намеренно,  
чтобы не усложнять структуру для демо.

| Параметр | Файл | Значение по умолчанию | Описание |
|---|---|---|---|
| `FIRMS_DEFAULT_PRODUCT` | `data_loader.py` | `VIIRS_SNPP_NRT` | Продукт NASA FIRMS |
| `GIBS_LAYERS` | `data_loader.py` | список из 4 слоёв | Порядок перебора слоёв GIBS |
| `days_ago` в `run_demo()` | `main.py` | `[2, 4, 6, 8, 10]` | Даты назад при поиске снимка |
| `min_fire_area_pixels` | `inference.py` | `50` | Минимум пикселей для очага |
| `resolution_m` | `inference.py` | `30.0` | Размер пикселя в метрах (для расчёта га) |

---

## Описание модулей

### main.py

Точка входа. Две публичные функции:

#### `run_demo()`

```python
run_demo()
```

Демо-сценарий. Перебирает 5 регионов × 5 дат (2/4/6/8/10 дней назад) — итого до 25 попыток  
загрузить снимок GIBS. Берёт первый успешный. Если ни один не подошёл — уходит в синтетику.

**Регионы по умолчанию:**
- Австралия (Новый Южный Уэльс) — `-33.8, 151.0`
- Бразилия (Амазония) — `-3.0, -60.0`
- Казахстан (Акмолинская обл.) — `51.18, 71.45`
- США (Калифорния) — `36.7, -119.4`
- Индонезия (Суматра) — `-0.5, 101.5`

#### `analyze_region(image_path, lat, lon)`

```python
summary = analyze_region(image_path="data/sample_images/img.png", lat=51.18, lon=71.45)
```

Полный пайплайн для одного изображения:

1. `WildfireDetector().detect_fire_by_color(image_path)` → детекции
2. `calculate_burned_area()` на каждый очаг → добавляет `hectares`, `km2`
3. `visualize_results()` → сохраняет PNG
4. `save_json()` → сохраняет JSON-отчёт
5. Возвращает `dict` с полным резюме

---

### data_loader.py

Всё, что связано с получением входных данных.

#### `get_fire_data(country="KAZ", days=7)`

```python
records = get_fire_data(country="RUS", days=10)
# [{"latitude": "...", "longitude": "...", "brightness": "...", ...}, ...]
```

Запрашивает NASA FIRMS API, парсит CSV-ответ в список словарей.  
Требует `FIRMS_API_KEY` в переменных окружения.  
При ошибке (нет ключа / нет сети) — **тихо возвращает `[]`**, не ломает пайплайн.

#### `get_sample_satellite_image(lat, lon, width=512, height=512, output_path=None, days_ago=2)`

```python
path = get_sample_satellite_image(lat=51.18, lon=71.45, days_ago=4)
# → "data/sample_images/gibs_51.1800_71.4500_2025-02-25_VIIRS_SNPP_20250227_143201.png"
# или None, если все слои вернули пустой/тёмный кадр
```

Делает WMS `GetMap`-запрос к NASA GIBS. Перебирает 4 слоя по очереди, пока не получит  
непустой снимок (среднее значение пикселей ≥ 5). Формат BBOX для WMS 1.3.0 + EPSG:4326:  
`minLat,minLon,maxLat,maxLon` — **не** `lon,lat`.

**Порядок слоёв:**
```python
GIBS_LAYERS = [
    "VIIRS_SNPP_CorrectedReflectance_TrueColor",    # основной
    "VIIRS_NOAA20_CorrectedReflectance_TrueColor",  # второй выбор
    "MODIS_Aqua_CorrectedReflectance_TrueColor",    # резерв
    "MODIS_Terra_CorrectedReflectance_TrueColor",   # последний fallback
]
```

#### `create_synthetic_demo_image(output_path=None)`

```python
path = create_synthetic_demo_image()
# → "data/sample_images/synthetic_demo.png"
```

Создаёт PNG 512×512: зелёный OpenCV-фон + шум + 4 красно-оранжевых круга (имитация очагов).  
Всегда успешен, не требует сети.

---

### inference.py

Ядро детекции. Класс `WildfireDetector` + датакласс `Detection`.

#### `WildfireDetector(min_fire_area_pixels=50)`

```python
detector = WildfireDetector(min_fire_area_pixels=100)  # игнорировать совсем мелкие пятна
```

#### `detect_fire_by_color(image_path)`

```python
img_rgb, fire_mask, detections = detector.detect_fire_by_color("data/sample_images/img.png")
```

**Алгоритм шаг за шагом:**

```
1. cv2.imread(image_path)                        # читаем BGR
2. cv2.cvtColor(BGR → HSV)
3. cv2.inRange(HSV, lower_red1, upper_red1)      # H ∈ [0°,  15°], S/V ∈ [80, 255]
   cv2.inRange(HSV, lower_red2, upper_red2)      # H ∈ [160°,179°], S/V ∈ [80, 255]
   fire_mask = mask1 | mask2
4. MORPH_OPEN (kernel 5×5)  →  убираем шум
   dilate (kernel 5×5)      →  укрупняем очаги
5. cv2.findContours(RETR_EXTERNAL)
6. Для каждого контура:
   - area = cv2.contourArea(contour)
   - if area < min_fire_area_pixels: skip
   - bbox = cv2.boundingRect(contour)
   - area_percentage = area / (height × width) × 100
   - threat_level = _classify_threat(area_percentage)
7. Возвращаем: (img_rgb ndarray, fire_mask ndarray, [Detection.to_dict(), ...])
```

#### `_classify_threat(area_percentage)`

| Доля площади | Уровень |
|---|---|
| < 3% | `СРЕДНИЙ` |
| 3% – 10% | `ВЫСОКИЙ` |
| ≥ 10% | `КРИТИЧЕСКИЙ` |

#### `calculate_burned_area(area_pixels, resolution_m=30.0)`

```python
areas = detector.calculate_burned_area(area_pixels=5000, resolution_m=30.0)
# {"hectares": 4.5, "km2": 0.045}
```

```
area_m² = area_pixels × resolution_m²
hectares = area_m² / 10_000
km²      = area_m² / 1_000_000
```

> **Важно:** `resolution_m=30` — условное значение для расчётов в демо.  
> Реальное разрешение VIIRS ≈ 375 м, MODIS ≈ 500–1000 м.

#### `visualize_results(img_rgb, fire_mask, detections, output_path)`

```python
detector.visualize_results(img_rgb, fire_mask, detections, "results/vis.png")
```

Создаёт 3-панельный PNG (dpi=200):

| Панель | Содержимое |
|---|---|
| 1 | Оригинальное изображение |
| 2 | Бинарная маска (colormap `hot`) |
| 3 | Изображение + bounding box с подписями угрозы |

Цвета bounding box: жёлтый — СРЕДНИЙ, оранжевый — ВЫСОКИЙ, красный — КРИТИЧЕСКИЙ.

---

### utils.py

Вспомогательные функции без бизнес-логики.

```python
setup_logging()                        # basicConfig INFO+, формат с временем и модулем
ensure_directory("results/subdir")     # os.makedirs(exist_ok=True)
save_json(data_dict, "results/r.json") # json.dump с indent=2, ensure_ascii=False (UTF-8)
ts = generate_timestamp()              # "20250227_143201" (UTC)
```

---

## Выходные данные

### JSON-отчёт `results/wildfire_report_<timestamp>.json`

```json
{
  "image_path": "data/sample_images/synthetic_demo.png",
  "coordinates": { "lat": 51.18, "lon": 71.45 },
  "visualization_path": "results/wildfire_visualization_20250227_143201.png",
  "detections": [
    {
      "bbox": [102, 78, 95, 88],
      "area_pixels": 4823,
      "area_percentage": 1.84,
      "threat_level": "СРЕДНИЙ",
      "hectares": 4.34,
      "km2": 0.0434
    }
  ]
}
```

### PNG-визуализация `results/wildfire_visualization_<timestamp>.png`

Трёхпанельное изображение 200 DPI. Каждая детекция — прямоугольник с подписью уровня угрозы.

---

## Внешние API

### NASA FIRMS

- **URL:** `https://firms.modaps.eosdis.nasa.gov/api/country/csv/{key}/{product}/{country}/{days}`
- **Ответ:** CSV с заголовком, одна строка — один очаг
- **Ключ:** переменная окружения `FIRMS_API_KEY` (бесплатно на [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/api/area/))
- **Таймаут:** 20 секунд

### NASA GIBS WMS

- **URL:** `https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi`
- **Протокол:** WMS 1.3.0, CRS EPSG:4326
- **Ключ:** не требуется
- **Важно:** порядок BBOX в EPSG:4326 — `minLat,minLon,maxLat,maxLon` (не lon,lat!)
- **Параметр TIME:** дата `YYYY-MM-DD`; последние 1–2 дня могут быть ещё не загружены в архив
- **Таймаут:** 30 секунд

---

## Ограничения

| Проблема | Причина |
|---|---|
| Ложные срабатывания на закаты, пустыни, красные крыши | HSV-детекция не различает контекст |
| Площадь приблизительна | `resolution_m=30` — константа, не зависит от реального масштаба снимка |
| Нет истории и трендов | Каждый запуск независим, БД не используется |
| Однопоточный обход регионов | Нет параллельного сканирования |

**Следующие шаги:**
- YOLOv8-детекция (ultralytics уже в зависимостях)
- Реальные термальные каналы (Landsat Band 10, MODIS Band 31)
- REST API на FastAPI для интеграции с системами оповещения

---

## Бизнес-применение

- **Государственные службы** (МЧС, лесхозы) — раннее обнаружение, снижение ущерба
- **Страховые компании** — дистанционная верификация и оценка ущерба
- **Операторы инфраструктуры** — защита трубопроводов, ЛЭП
- **НКО / исследователи** — мониторинг экосистем

**Модель монетизации:** SaaS-подписка по количеству регионов, Enterprise on-premise, API-лицензия.
