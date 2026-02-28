import logging
import os
from typing import Any, Dict, List

from data_loader import create_synthetic_demo_image, get_fire_data, get_sample_satellite_image
from inference import WildfireDetector
from utils import generate_timestamp, save_json, setup_logging


LOGGER = logging.getLogger(__name__)


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _results_dir() -> str:
    return os.path.join(_project_root(), "results")


def analyze_region(image_path: str, lat: float, lon: float) -> Dict[str, Any]:
    """
    Запускает полный конвейер анализа региона по спутниковому снимку.
    """
    print("🚀 Запуск анализа региона по спутниковому изображению...")
    detector = WildfireDetector()

    try:
        img_rgb, fire_mask, detections = detector.detect_fire_by_color(image_path)
    except FileNotFoundError as exc:
        print(f"❌ Ошибка: не удалось прочитать изображение: {exc}")
        raise
    except Exception as exc:
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
    except Exception as exc:
        LOGGER.error("Ошибка при построении визуализации: %s", exc)
        vis_path = ""

    summary: Dict[str, Any] = {
        "image_path": image_path,
        "coordinates": {"lat": lat, "lon": lon},
        "detections": enriched_detections,
        "visualization_path": vis_path,
    }

    # Русский комментарий: выводим короткое резюме в консоль
    print(f"\n🛰️  Анализ изображения: {image_path}")
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
        print(f"🖼️  Визуализация сохранена в: {vis_path}")

    return summary


def run_demo() -> None:
    """
    Запускает демонстрационный сценарий.

    Перебирает несколько локаций и дат, пока не найдёт рабочий снимок.
    Если NASA GIBS недоступен — переходит в оффлайн-режим с синтетикой.
    """
    print("🌍 Демонстрационный режим системы обнаружения лесных пожаров.")
    print("🔎 Поиск спутниковых снимков через NASA GIBS...\n")

    # Русский комментарий: разные регионы с высокой вероятностью найти снимок
    test_locations = [
        {"name": "Австралия (Новый Южный Уэльс)", "lat": -33.8,  "lon": 151.0},
        {"name": "Бразилия (Амазония)",            "lat": -3.0,   "lon": -60.0},
        {"name": "Казахстан (Акмолинская обл.)",   "lat": 51.18,  "lon": 71.45},
        {"name": "США (Калифорния)",                "lat": 36.7,   "lon": -119.4},
        {"name": "Индонезия (Суматра)",             "lat": -0.5,   "lon": 101.5},
    ]

    # Русский комментарий: пробуем разные даты — от 2 до 10 дней назад
    days_ago_options = [2, 4, 6, 8, 10]

    image_path = None
    chosen_lat = 51.18
    chosen_lon = 71.45

    for location in test_locations:
        lat = location["lat"]
        lon = location["lon"]

        for days_ago in days_ago_options:
            print(f"📡 {location['name']} | {days_ago} дней назад ({lat}, {lon})")
            candidate = get_sample_satellite_image(lat, lon, days_ago=days_ago)

            if candidate and os.path.isfile(candidate):
                image_path = candidate
                chosen_lat = lat
                chosen_lon = lon
                print(f"✅ Найден снимок: {os.path.basename(candidate)}\n")
                break

        if image_path:
            break

    # Русский комментарий: если ни один онлайн-снимок не получен — синтетика
    if not image_path:
        print("\n🧪 NASA GIBS не вернул данные. Запускаем оффлайн-демо с синтетическим снимком.")
        image_path = create_synthetic_demo_image()
        chosen_lat, chosen_lon = 51.18, 71.45
        print(f"🖼️  Синтетическое изображение создано: {image_path}\n")

    try:
        analyze_region(image_path=image_path, lat=chosen_lat, lon=chosen_lon)
    except Exception as exc:
        print(f"❌ Критическая ошибка при выполнении демо: {exc}")


if __name__ == "__main__":
    setup_logging()
    run_demo()