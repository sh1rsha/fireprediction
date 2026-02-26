import io
import logging
import os
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
GIBS_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"

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
        Строка BBOX в формате ``"min_lon,min_lat,max_lon,max_lat"``.
    """
    # Русский комментарий: ограничиваем окно вокруг указанной точки
    min_lat = max(lat - half_size_deg, -90.0)
    max_lat = min(lat + half_size_deg, 90.0)
    min_lon = max(lon - half_size_deg, -180.0)
    max_lon = min(lon + half_size_deg, 180.0)
    return f"{min_lon},{min_lat},{max_lon},{max_lat}"


def get_sample_satellite_image(
    lat: float,
    lon: float,
    width: int = 512,
    height: int = 512,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Загружает пример спутникового изображения из NASA GIBS WMS по координатам.

    При сетевых ошибках или ошибке сервера функция возвращает ``None``,
    чтобы вызывающий код мог перейти к оффлайн-демо.

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
        Путь для сохранения изображения. Если не указан, используется
        каталог ``data/sample_images`` внутри проекта.

    Returns
    -------
    Optional[str]
        Путь к сохранённому изображению или ``None`` при ошибке.
    """
    # Русский комментарий: готовим параметры запроса WMS
    bbox = _build_gibs_bbox(lat, lon)
    params = {
        "service": "WMS",
        "request": "GetMap",
        "layers": GIBS_LAYER,
        "styles": "",
        "format": "image/png",
        "transparent": "FALSE",
        "version": "1.3.0",
        "height": str(height),
        "width": str(width),
        "crs": "EPSG:4326",
        "bbox": bbox,
    }

    try:
        response = requests.get(GIBS_WMS_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.error("Ошибка загрузки спутникового изображения из GIBS: %s", exc)
        return None

    try:
        image_bytes = io.BytesIO(response.content)
        pil_image = Image.open(image_bytes).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Не удалось декодировать ответ GIBS как изображение: %s", exc)
        return None

    # Русский комментарий: сохраняем изображение в стандартный каталог проекта
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "data", "sample_images")
        ensure_directory(output_dir)
        output_path = os.path.join(output_dir, "gibs_sample.png")

    try:
        pil_image.save(output_path)
    except OSError as exc:
        LOGGER.error("Ошибка сохранения спутникового изображения в %s: %s", output_path, exc)
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

