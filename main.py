import logging
import os
from typing import Any, Dict, List

from data_loader import create_synthetic_demo_image, get_fire_data, get_sample_satellite_image
from inference import WildfireDetector
from utils import generate_timestamp, save_json, setup_logging


LOGGER = logging.getLogger(__name__)


def _project_root() -> str:
    """
    Возвращает корневую директорию проекта.

    Returns
    -------
    str
        Абсолютный путь к корню проекта.
    """
    # Русский комментарий: считаем корнем каталог, где лежит main.py
    return os.path.dirname(os.path.abspath(__file__))


def _results_dir() -> str:
    """
    Возвращает путь к каталогу для сохранения результатов.

    Returns
    -------
    str
        Абсолютный путь к каталогу ``results``.
    """
    return os.path.join(_project_root(), "results")


def analyze_region(image_path: str, lat: float, lon: float) -> Dict[str, Any]:
    """
    Запускает полный конвейер анализа региона по спутниковому снимку.

    Выполняет:
    - детекцию зон пожара,
    - оценку площади в гектарах и км²,
    - сохранение визуализации и JSON-отчёта.

    Parameters
    ----------
    image_path : str
        Путь к изображению.
    lat : float
        Широта региона.
    lon : float
        Долгота региона.

    Returns
    -------
    Dict[str, Any]
        Структура с результатами анализа.
    """
    print("🚀 Запуск анализа региона по спутниковому изображению...")
    detector = WildfireDetector()

    try:
        img_rgb, fire_mask, detections = detector.detect_fire_by_color(image_path)
    except FileNotFoundError as exc:
        print(f"❌ Ошибка: не удалось прочитать изображение: {exc}")
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Неожиданная ошибка при детекции пожара: {exc}")
        raise

    # Русский комментарий: считаем площади для каждого очага
    enriched_detections: List[Dict[str, Any]] = []
    for det in detections:
        area_info = detector.calculate_burned_area(det["area_pixels"])
        det_with_area = {**det, **area_info}
        enriched_detections.append(det_with_area)

    # Русский комментарий: готовим визуализацию
    ts = generate_timestamp()
    vis_path = os.path.join(_results_dir(), f"wildfire_visualization_{ts}.png")
    try:
        from utils import ensure_directory

        ensure_directory(_results_dir())
        detector.visualize_results(img_rgb, fire_mask, enriched_detections, vis_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Ошибка при построении визуализации: %s", exc)
        vis_path = ""

    summary: Dict[str, Any] = {
        "image_path": image_path,
        "coordinates": {"lat": lat, "lon": lon},
        "detections": enriched_detections,
        "visualization_path": vis_path,
    }

    # Русский комментарий: выводим короткое резюме в консоль
    print(f"🛰️ Анализ изображения: {image_path}")
    print(f"📍 Координаты: широта={lat:.4f}, долгота={lon:.4f}")
    print(f"🔥 Найдено очагов пожара: {len(enriched_detections)}")
    for idx, det in enumerate(enriched_detections, start=1):
        print(
            f"   ➤ Очаг #{idx}: площадь ≈ {det['hectares']:.2f} га "
            f"({det['km2']:.4f} км²), уровень угрозы: {det['threat_level']}"
        )

    # Русский комментарий: сохраняем детальный отчёт в JSON
    report_path = os.path.join(_results_dir(), f"wildfire_report_{ts}.json")
    save_json(summary, report_path)
    print(f"📄 JSON-отчёт сохранён в: {report_path}")
    if vis_path:
        print(f"🖼️ Визуализация сохранена в: {vis_path}")

    return summary


def run_demo() -> None:
    """
    Запускает демонстрационный сценарий.

    Сначала пробует получить реальные данные из NASA FIRMS/GIBS,
    а при отсутствии сети или ключей API переходит в полностью оффлайн-режим
    с синтетическим изображением.
    """
    print("🌍 Демонстрационный режим системы обнаружения лесных пожаров.")
    print("🔎 Попытка получить реальные данные NASA (если доступно)...")

    lat: float = 51.0
    lon: float = 73.0
    image_path: str

    # Русский комментарий: сначала пробуем реальный сценарий с данными NASA
    try:
        records = get_fire_data(country="KAZ", days=3)
        if records:
            first = records[0]
            try:
                lat = float(first.get("latitude", lat))
                lon = float(first.get("longitude", lon))
            except (TypeError, ValueError):
                pass

            image_path_candidate = get_sample_satellite_image(lat, lon)
            if image_path_candidate and os.path.isfile(image_path_candidate):
                image_path = image_path_candidate
            else:
                raise RuntimeError("Не удалось получить спутниковое изображение из GIBS.")
        else:
            raise RuntimeError("Данные FIRMS отсутствуют или недоступны.")

        print("✅ Удалось получить реальные данные NASA, выполняется анализ...")
    except Exception as exc:  # noqa: BLE001
        print(
            "⚠️ Не удалось использовать NASA FIRMS/GIBS "
            f"(причина: {exc}). Переход в оффлайн-демо режим."
        )
        image_path = create_synthetic_demo_image()
        print(f"🧪 Синтетическое демо-изображение создано: {image_path}")

    # Русский комментарий: запускаем анализ для выбранного изображения
    try:
        analyze_region(image_path=image_path, lat=lat, lon=lon)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Критическая ошибка при выполнении демо: {exc}")


if __name__ == "__main__":
    setup_logging()
    run_demo()

