## Wildfire AI – AI-powered Wildfire Detection (EN/RU)

### English

**Wildfire AI** is a lightweight, AI-inspired wildfire detection system based on satellite imagery.  
Instead of heavyweight deep learning models, it uses fast HSV color analysis to highlight potential fire zones, making it easy to demo and run completely offline.

- **Problem**: Wildfires are growing in frequency and intensity, threatening lives, infrastructure, and ecosystems. Traditional monitoring is expensive and slow.
- **Solution**: A simple yet extensible pipeline that can:
  - Ingest satellite images (NASA GIBS) or synthetic demo images.
  - Detect fire-like regions using color-based analysis.
  - Estimate burned area in hectares and km².
  - Generate visual reports and JSON summaries for further integration.

#### Tech stack

- **Python 3.9+**
- `opencv-python`, `numpy`, `Pillow`, `matplotlib`
- `requests` for NASA FIRMS / GIBS API access
- `ultralytics` is included in dependencies for future ML extensions (not required for the core HSV demo).

#### Installation

```bash
cd wildfire-ai
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### How to run

Offline demo will work out-of-the-box using a synthetic satellite-style image:

```bash
cd wildfire-ai
python main.py
```

If you configure NASA FIRMS / GIBS access, the script will try online mode first:

```bash
export FIRMS_API_KEY="YOUR_FIRMS_API_KEY"  # Windows PowerShell: $env:FIRMS_API_KEY="YOUR_FIRMS_API_KEY"
python main.py
```

The script will:

- Print Russian console output with emojis.
- Create a synthetic or real satellite image.
- Run wildfire detection.
- Save:
  - JSON report to `results/wildfire_report_<timestamp>.json`
  - Visualization figure to `results/wildfire_visualization_<timestamp>.png`

#### Project structure

```text
wildfire-ai/
├── README.md
├── requirements.txt
├── main.py
├── inference.py
├── data_loader.py
├── utils.py
├── data/
│   └── sample_images/
├── results/
└── docs/
    └── technical.md
```

#### Architecture diagram (ASCII)

```text
          +------------------------+
          |      main.py          |
          |  - run_demo()         |
          |  - analyze_region()   |
          +-----------+-----------+
                      |
                      v
      +---------------+-----------------+
      |             data_loader.py      |
      |  - get_fire_data() (FIRMS)      |
      |  - get_sample_satellite_image() |
      |  - create_synthetic_demo_image()|
      +---------------+-----------------+
                      |
                      v
          +-----------+-----------+
          |       inference.py    |
          |  - WildfireDetector   |
          |    * detect_fire_by_color() |
          |    * calculate_burned_area()|
          |    * visualize_results()    |
          +-----------+-----------+
                      |
                      v
          +-----------+-----------+
          |        utils.py       |
          | - logging, JSON, etc. |
          +-----------------------+
```

#### Business model (high-level)

- **Target customers**:
  - Forestry agencies and environmental ministries.
  - Critical infrastructure operators (pipelines, power lines).
  - Insurance and reinsurance companies.
  - NGOs and research institutions.

- **Value proposition**:
  - Near-real-time wildfire risk detection and alerts.
  - Historical risk analytics and damage estimation.
  - Lightweight deployment (on-prem, edge, or cloud).

- **Revenue model**:
  - Subscription-based SaaS tiers by monitored area / number of regions.
  - Enterprise on-prem deployments with annual maintenance.
  - Professional services: integration, custom ML models, dashboards.

#### External APIs / data sources

- **NASA FIRMS** (Fire Information for Resource Management System)  
  - Used for recent fire detections.
  - Requires an API key stored in `FIRMS_API_KEY` environment variable.
- **NASA GIBS** (Global Imagery Browse Services) WMS  
  - Used to fetch satellite imagery tiles as PNG images.
  - Accessed via standard WMS `GetMap` requests (no key in the demo).

---

### Русский

**Wildfire AI** — это лёгкая система интеллектуального обнаружения лесных пожаров по спутниковым изображениям.  
Вместо тяжёлых нейросетей используется быстрый анализ цвета в пространстве HSV, поэтому демонстрацию можно запускать полностью оффлайн.

- **Проблема**: Лесные пожары становятся всё более частыми и разрушительными. Классические методы мониторинга дороги и медленны.
- **Решение**: Минималистичный, но расширяемый конвейер, который:
  - получает спутниковые снимки (NASA GIBS) или синтетическое демо-изображение;
  - находит потенциальные зоны огня по цветовому анализу;
  - оценивает площадь в гектарах и км²;
  - формирует визуальные отчёты и JSON для последующей интеграции.

#### Технологический стек

- **Python 3.9+**
- `opencv-python`, `numpy`, `Pillow`, `matplotlib`
- `requests` для обращения к NASA FIRMS / GIBS
- `ultralytics` — в зависимостях для будущего расширения ML (в базовой HSV-демонстрации не обязателен).

#### Установка

```bash
cd wildfire-ai
python -m venv .venv
source .venv/bin/activate  # В Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### Запуск

Оффлайн-демо (используется синтетическое «спутниковое» изображение):

```bash
cd wildfire-ai
python main.py
```

Для онлайн-режима с реальными данными NASA FIRMS/GIBS:

```bash
export FIRMS_API_KEY="ВАШ_КЛЮЧ_FIRMS"  # PowerShell: $env:FIRMS_API_KEY="ВАШ_КЛЮЧ_FIRMS"
python main.py
```

Скрипт:

- выводит сообщения на русском языке с эмодзи;
- получает реальное или синтетическое изображение;
- запускает детекцию пожаров;
- сохраняет:
  - JSON-отчёт в `results/wildfire_report_<timestamp>.json`;
  - визуализацию в `results/wildfire_visualization_<timestamp>.png`.

#### Архитектура (ASCII)

```text
Пользователь
   |
   v
 main.py (run_demo, analyze_region)
   |
   +--> data_loader.py (NASA FIRMS/GIBS или синтетика)
   |
   +--> inference.py (WildfireDetector: HSV-детекция, площади, визуализация)
   |
   +--> utils.py (логирование, JSON, служебные функции)
```

#### Бизнес-модель

- **Клиенты**:
  - государственные службы лесного хозяйства и МЧС;
  - операторы критической инфраструктуры;
  - страховые компании;
  - научные и экологические организации.

- **Ценность**:
  - раннее обнаружение очагов и снижение ущерба;
  - поддержка оперативного принятия решений;
  - аналитика риска и пост-фактум оценка ущерба.

- **Монетизация**:
  - подписка SaaS по количеству контролируемых регионов / площади;
  - корпоративные on-premise‑установки;
  - консалтинг и кастомная разработка (дешборды, интеграция, ML-модели).

#### Источники API / данных

- **NASA FIRMS** — оперативная информация о возгораниях.
- **NASA GIBS (WMS)** — глобальные спутниковые снимки для визуализации и анализа.

Подробнее о реализации архитектуры и алгоритмов см. в `docs/technical.md`.

