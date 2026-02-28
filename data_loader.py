import io
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests
from PIL import Image

from utils import ensure_directory


FIRMS_API_BASE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/country/csv/{api_key}/{product}/{country}/{days}"
)
FIRMS_DEFAULT_PRODUCT = "VIIRS_SNPP_NRT"

GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"

# Русский комментарий: перебираем слои по очереди — каждый реагирует на TIME по-своему
GIBS_LAYERS = [
    "VIIRS_SNPP_CorrectedReflectance_TrueColor",    # Suomi NPP — лучше всего реагирует на TIME
    "VIIRS_NOAA20_CorrectedReflectance_TrueColor",  # NOAA-20 — второй выбор
    "MODIS_Aqua_CorrectedReflectance_TrueColor",    # Aqua — отличается от Terra
    "MODIS_Terra_CorrectedReflectance_TrueColor",   # Terra — fallback
]

LOGGER = logging.getLogger(__name__)


def get_fire_data(country: str = "KAZ", days: int = 7) -> List[Dict[str, Any]]:
    """
    Получает данные о пожарах из NASA FIRMS API для указанной страны.

    Функция безопасно обрабатывает отсутствие ключа API или сетевые ошибки и,
    в таком случае, возвращает пустой список.

    Parameters
    ----------
    country : str, optional
        Код страны (ISO Alpha-3), по умолчанию ``"KAZ"``.
    days : int, optional
        Количество последних дней для выборки, по умолчанию ``7``.

    Returns
    -------
    List[Dict[str, Any]]
        Список записей о пожарах (каждая строка CSV в виде словаря).
        При ошибках возвращается пустой список.
    """
    # Русский комментарий: пробуем обратиться к FIRMS API, но не ломаем оффлайн-демо
    api_key = os.environ.get("FIRMS_API_KEY")
    if not api_key:
        LOGGER.warning("Ключ FIRMS_API_KEY не найден в окружении, данные пожаров недоступны.")
        return []

    url = FIRMS_API_BASE_URL.format(
        api_key=api_key,
        product=FIRMS_DEFAULT_PRODUCT,
        country=country,
        days=days,
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.error("Ошибка запроса к FIRMS API: %s", exc)
        return []

    # Ответ приходит в CSV, первую строку считаем заголовком.
    text = response.text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    header = lines[0].split(",")
    records: List[Dict[str, Any]] = []
    for row in lines[1:]:
        parts = row.split(",")
        if len(parts) != len(header):
            continue
        record = dict(zip(header, parts))
        records.append(record)

    LOGGER.info("NASA FIRMS вернул %d записей о пожарах для страны %s", len(records), country)
    return records


def _build_gibs_bbox(lat: float, lon: float, half_size_deg: float = 0.5) -> str:
    """
    Строит строку BBOX для WMS-запроса NASA GIBS (EPSG:4326).

    Parameters
    ----------
    lat : float
        Широта центра.
    lon : float
        Долгота центра.
    half_size_deg : float, optional
        Половина размера окна в градусах, по умолчанию ``0.5``.

    Returns
    -------
    str
        Строка BBOX в формате ``"min_lat,min_lon,max_lat,max_lon"`` для EPSG:4326 WMS 1.3.0.
    """
    # Русский комментарий: ВАЖНО — в WMS 1.3.0 с EPSG:4326 порядок осей lat,lon (а не lon,lat!)
    min_lat = max(lat - half_size_deg, -90.0)
    max_lat = min(lat + half_size_deg, 90.0)
    min_lon = max(lon - half_size_deg, -180.0)
    max_lon = min(lon + half_size_deg, 180.0)
    # WMS 1.3.0 + EPSG:4326 = порядок: minLat, minLon, maxLat, maxLon
    return f"{min_lat},{min_lon},{max_lat},{max_lon}"


def _get_gibs_date(days_ago: int = 2) -> str:
    """
    Возвращает дату в формате YYYY-MM-DD для запроса NASA GIBS.

    NASA GIBS требует дату — снимки за сегодня/вчера могут быть ещё не готовы,
    поэтому берём данные с небольшой задержкой.

    Parameters
    ----------
    days_ago : int, optional
        Сколько дней назад брать снимок, по умолчанию ``2``.

    Returns
    -------
    str
        Дата в формате ``"YYYY-MM-DD"``.
    """
    # Русский комментарий: GIBS нужна конкретная дата, иначе вернёт ошибку или пустышку
    target_date = datetime.utcnow() - timedelta(days=days_ago)
    return target_date.strftime("%Y-%m-%d")


def _try_fetch_gibs_image(
    params: dict,
    layer: str,
) -> Optional[Image.Image]:
    """
    Делает один WMS-запрос к NASA GIBS для конкретного слоя.

    Parameters
    ----------
    params : dict
        Базовые параметры WMS-запроса (без LAYERS).
    layer : str
        Название слоя GIBS для подстановки в запрос.

    Returns
    -------
    Optional[Image.Image]
        PIL-изображение или ``None`` если запрос не удался / вернул не картинку.
    """
    # Русский комментарий: подставляем слой и делаем запрос
    params["LAYERS"] = layer
    try:
        response = requests.get(GIBS_WMS_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Ошибка запроса к GIBS (слой=%s): %s", layer, exc)
        return None

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        LOGGER.warning(
            "GIBS (слой=%s) вернул не изображение. Content-Type: %s. Ответ: %s",
            layer, content_type, response.text[:200],
        )
        return None

    try:
        pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Не удалось декодировать ответ GIBS (слой=%s): %s", layer, exc)
        return None

    # Русский комментарий: отбрасываем полностью тёмные изображения — данных нет
    if np.array(pil_image).mean() < 5:
        LOGGER.warning("GIBS (слой=%s) вернул тёмное изображение — нет данных.", layer)
        return None

    return pil_image


def get_sample_satellite_image(
    lat: float,
    lon: float,
    width: int = 512,
    height: int = 512,
    output_path: Optional[str] = None,
    days_ago: int = 2,
) -> Optional[str]:
    """
    Загружает спутниковое изображение из NASA GIBS WMS по координатам.

    Перебирает несколько слоёв GIBS пока не получит валидный снимок.
    При полной недоступности возвращает ``None`` для перехода в оффлайн-демо.

    Parameters
    ----------
    lat : float
        Широта центра.
    lon : float
        Долгота центра.
    width : int, optional
        Ширина изображения в пикселях, по умолчанию ``512``.
    height : int, optional
        Высота изображения в пикселях, по умолчанию ``512``.
    output_path : Optional[str], optional
        Путь для сохранения. Если не указан — генерируется автоматически.
    days_ago : int, optional
        Сколько дней назад брать снимок, по умолчанию ``2``.

    Returns
    -------
    Optional[str]
        Путь к сохранённому изображению или ``None`` при ошибке.
    """
    bbox = _build_gibs_bbox(lat, lon)
    gibs_date = _get_gibs_date(days_ago=days_ago)

    # Русский комментарий: базовые параметры запроса, слой подставляем в цикле
    base_params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "LAYERS": "",           # заполняется в цикле
        "STYLES": "",
        "FORMAT": "image/png",
        "TRANSPARENT": "FALSE",
        "VERSION": "1.3.0",
        "HEIGHT": str(height),
        "WIDTH": str(width),
        "CRS": "EPSG:4326",
        "BBOX": bbox,
        "TIME": gibs_date,
    }

    print(f"📡 NASA GIBS | дата={gibs_date} | координаты=({lat:.4f}, {lon:.4f})")

    pil_image: Optional[Image.Image] = None
    used_layer: str = ""

    # Русский комментарий: перебираем слои — берём первый который вернул нормальный снимок
    for layer in GIBS_LAYERS:
        print(f"   🔄 Пробуем слой: {layer}")
        result = _try_fetch_gibs_image(dict(base_params), layer)
        if result is not None:
            pil_image = result
            used_layer = layer
            print(f"   ✅ Успешно! Слой: {layer}")
            break
        print(f"   ⚠️  Слой {layer} не дал результата, пробуем следующий...")

    if pil_image is None:
        print("❌ Все слои GIBS исчерпаны — изображение не получено.")
        return None

    # Русский комментарий: уникальное имя файла — координаты + дата + слой + timestamp
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "data", "sample_images")
        ensure_directory(output_dir)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_layer = used_layer.replace("_CorrectedReflectance_TrueColor", "")
        filename = f"gibs_{lat:.4f}_{lon:.4f}_{gibs_date}_{safe_layer}_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)

    try:
        pil_image.save(output_path)
        print(f"💾 Снимок сохранён: {os.path.basename(output_path)}")
        LOGGER.info("Спутниковое изображение сохранено: %s", output_path)
    except OSError as exc:
        LOGGER.error("Ошибка сохранения изображения в %s: %s", output_path, exc)
        return None

    return output_path


def create_synthetic_demo_image(output_path: Optional[str] = None) -> str:
    """
    Создаёт синтетическое спутниковое изображение с имитацией зон возгорания.

    Для оффлайн-демо генерируется простая карта местности (зелёные и коричневые
    участки), на которой отмечены несколько условных пожаров в виде ярких пятен.

    Parameters
    ----------
    output_path : Optional[str], optional
        Путь для сохранения изображения. Если не указан, используется
        каталог ``data/sample_images`` внутри проекта.

    Returns
    -------
    str
        Путь к сохранённому синтетическому изображению.
    """
    # Русский комментарий: подготавливаем путь для сохранения
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "data", "sample_images")
        ensure_directory(output_dir)
        output_path = os.path.join(output_dir, "synthetic_demo.png")

    height, width = 512, 512

    # Русский комментарий: создаём фон, напоминающий растительность и почву
    base_image = np.zeros((height, width, 3), dtype=np.uint8)
    base_image[:] = (34, 139, 34)  # тёмно-зелёный в BGR

    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    base_image = cv2.add(base_image, noise)

    # Русский комментарий: добавляем несколько "очагов пожара" в виде ярких пятен
    num_fires = 4
    rng = np.random.default_rng()
    for _ in range(num_fires):
        center_x = int(rng.integers(50, width - 50))
        center_y = int(rng.integers(50, height - 50))
        radius = int(rng.integers(20, 60))

        color = (0, 0, 255)  # ярко-красный в BGR
        cv2.circle(base_image, (center_x, center_y), radius, color, thickness=-1)

        inner_radius = max(5, radius // 2)
        cv2.circle(base_image, (center_x, center_y), inner_radius, (0, 165, 255), thickness=-1)

    try:
        cv2.imwrite(output_path, base_image)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Ошибка сохранения синтетического изображения в %s: %s", output_path, exc)
        raise

    return output_path