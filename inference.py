from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class Detection:
    """
    Структура для хранения информации об одном очаге пожара.

    Attributes
    ----------
    bbox : Tuple[int, int, int, int]
        Ограничивающий прямоугольник (x, y, w, h) в пикселях.
    area_pixels : int
        Площадь очага в пикселях.
    area_percentage : float
        Доля площади изображения, занимаемая очагом, в процентах.
    threat_level : str
        Уровень угрозы (КРИТИЧЕСКИЙ/ВЫСОКИЙ/СРЕДНИЙ).
    """

    bbox: Tuple[int, int, int, int]
    area_pixels: int
    area_percentage: float
    threat_level: str

    def to_dict(self) -> Dict:
        """
        Преобразует объект Detection в словарь для сериализации.

        Returns
        -------
        Dict
            Словарь с ключами ``bbox``, ``area_pixels``, ``area_percentage``,
            ``threat_level``.
        """
        # Русский комментарий: готовим структуру для JSON-отчёта
        return {
            "bbox": list(self.bbox),
            "area_pixels": int(self.area_pixels),
            "area_percentage": float(self.area_percentage),
            "threat_level": self.threat_level,
        }


class WildfireDetector:
    """
    Класс для детекции лесных пожаров по спутниковым изображениям.

    Детекция основана на простом анализе цвета в HSV-пространстве, что
    позволяет использовать систему без тяжёлых ML-моделей и работать оффлайн.
    """

    def __init__(
        self,
        min_fire_area_pixels: int = 50,
    ) -> None:
        """
        Инициализирует детектор с заданными параметрами.

        Parameters
        ----------
        min_fire_area_pixels : int, optional
            Минимальная площадь контура в пикселях, чтобы считать его очагом пожара.
        """
        # Русский комментарий: сохраняем порог площади очага
        self.min_fire_area_pixels = max(1, int(min_fire_area_pixels))

        # Диапазоны HSV для "огненных" цветов (красные/оранжевые тона)
        # Значения подобраны эмпирически и могут быть донастроены.
        self.lower_red1 = np.array([0, 80, 80])
        self.upper_red1 = np.array([15, 255, 255])
        self.lower_red2 = np.array([160, 80, 80])
        self.upper_red2 = np.array([179, 255, 255])

    def detect_fire_by_color(
        self,
        image_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Выполняет детекцию пожара на изображении на основе цветового анализа.

        Parameters
        ----------
        image_path : str
            Путь к изображению на диске.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[Dict]]
            Кортеж из:
            - изображения в RGB (np.ndarray),
            - бинарной маски пожара (np.ndarray),
            - списка детекций (каждая детекция — словарь с ключами
              ``bbox``, ``area_pixels``, ``area_percentage``, ``threat_level``).

        Raises
        ------
        FileNotFoundError
            Если изображение не найдено.
        RuntimeError
            Если изображение не удалось прочитать.
        """
        # Русский комментарий: читаем изображение и подготавливаем к анализу
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Изображение не найдено или не читается: {image_path}")

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Русский комментарий: создаём маску по HSV-диапазонам "огненных" тонов
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        fire_mask = cv2.bitwise_or(mask1, mask2)

        # Очищаем шум с помощью морфологических операций
        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fire_mask = cv2.dilate(fire_mask, kernel, iterations=1)

        # Русский комментарий: ищем контуры потенциальных очагов
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = fire_mask.shape[:2]
        total_pixels = float(height * width) if height > 0 and width > 0 else 1.0

        detections: List[Detection] = []
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < self.min_fire_area_pixels:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            area_percentage = 100.0 * area / total_pixels
            threat_level = self._classify_threat(area_percentage)

            detections.append(
                Detection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    area_pixels=area,
                    area_percentage=area_percentage,
                    threat_level=threat_level,
                )
            )

        detections_dicts: List[Dict] = [d.to_dict() for d in detections]
        return img_rgb, fire_mask, detections_dicts

    @staticmethod
    def _classify_threat(area_percentage: float) -> str:
        """
        Классифицирует уровень угрозы по размеру поражённой площади.

        Parameters
        ----------
        area_percentage : float
            Доля площади изображения в процентах.

        Returns
        -------
        str
            Один из уровней: ``"КРИТИЧЕСКИЙ"``, ``"ВЫСОКИЙ"``, ``"СРЕДНИЙ"``.
        """
        # Русский комментарий: простая эвристика для оценки угрозы
        if area_percentage >= 10.0:
            return "КРИТИЧЕСКИЙ"
        if area_percentage >= 3.0:
            return "ВЫСОКИЙ"
        return "СРЕДНИЙ"

    @staticmethod
    def calculate_burned_area(area_pixels: int, resolution_m: float = 30.0) -> Dict[str, float]:
        """
        Переводит площадь в пикселях в гектары и квадратные километры.

        Parameters
        ----------
        area_pixels : int
            Площадь очага в пикселях.
        resolution_m : float, optional
            Пространственное разрешение (размер пикселя) в метрах,
            по умолчанию ``30`` метров.

        Returns
        -------
        Dict[str, float]
            Словарь с ключами ``hectares`` и ``km2``.
        """
        # Русский комментарий: учитываем, что один пиксель покрывает resolution_m^2
        try:
            area_pixels = int(max(0, area_pixels))
            pixel_area_m2 = float(resolution_m) ** 2
            total_m2 = area_pixels * pixel_area_m2
            hectares = total_m2 / 10_000.0
            km2 = total_m2 / 1_000_000.0
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Ошибка при вычислении площади: %s", exc)
            return {"hectares": 0.0, "km2": 0.0}

        return {"hectares": float(hectares), "km2": float(km2)}

    @staticmethod
    def visualize_results(
        img_rgb: np.ndarray,
        fire_mask: np.ndarray,
        detections: List[Dict],
        output_path: str,
    ) -> str:
        """
        Визуализирует результаты детекции и сохраняет их в виде картинки.

        Создаётся фигура из трёх панелей:
        1. Оригинальное изображение.
        2. Маска пожара.
        3. Оригинальное изображение с нарисованными bounding box и подписями.

        Parameters
        ----------
        img_rgb : np.ndarray
            Исходное изображение в формате RGB.
        fire_mask : np.ndarray
            Бинарная маска пожара.
        detections : List[Dict]
            Список детекций (каждая — словарь с полями ``bbox``,
            ``area_pixels``, ``area_percentage``, ``threat_level``).
        output_path : str
            Путь для сохранения визуализации.

        Returns
        -------
        str
            Путь к сохранённой визуализации.
        """
        # Русский комментарий: копируем изображение и наносим прямоугольники
        overlay = img_rgb.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            threat = det["threat_level"]
            color = (255, 0, 0)
            if threat == "СРЕДНИЙ":
                color = (255, 255, 0)
            elif threat == "ВЫСОКИЙ":
                color = (255, 165, 0)

            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            label = f"{threat}"
            cv2.putText(
                overlay,
                label,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # Русский комментарий: строим фигуру matplotlib с тремя панелями
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_rgb)
        axes[0].set_title("Оригинальное изображение")
        axes[0].axis("off")

        axes[1].imshow(fire_mask, cmap="hot")
        axes[1].set_title("Маска пожара")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Детекции (ограничивающие прямоугольники)")
        axes[2].axis("off")

        plt.tight_layout()
        try:
            fig.savefig(output_path, dpi=200)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Ошибка при сохранении визуализации в %s: %s", output_path, exc)
            raise
        finally:
            plt.close(fig)

        return output_path

