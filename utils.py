import json
import logging
import os
from datetime import datetime
from typing import Any, Dict


def setup_logging() -> None:
    """
    Настраивает базовый логгер для приложения.

    Логирование помогает отлаживать работу системы и отслеживать ошибки в продакшене.
    """
    # Русский комментарий: настраиваем простой консольный логгер
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def ensure_directory(path: str) -> None:
    """
    Гарантирует, что каталог существует.

    Parameters
    ----------
    path : str
        Путь к каталогу, который необходимо создать при отсутствии.
    """
    # Русский комментарий: создаём каталог, если его нет
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        logging.error("Не удалось создать каталог %s: %s", path, exc)
        raise


def save_json(data: Dict[str, Any], output_path: str) -> None:
    """
    Сохраняет словарь в JSON-файл с читаемым форматированием.

    Parameters
    ----------
    data : Dict[str, Any]
        Данные для сохранения.
    output_path : str
        Путь к выходному JSON-файлу.
    """
    # Русский комментарий: сохраняем отчёт в формате JSON
    try:
        directory = os.path.dirname(output_path)
        if directory:
            ensure_directory(directory)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        logging.error("Ошибка при сохранении JSON в %s: %s", output_path, exc)
        raise


def generate_timestamp() -> str:
    """
    Генерирует строковый таймстамп в формате YYYYMMDD_HHMMSS.

    Returns
    -------
    str
        Таймстамп для использования в именах файлов.
    """
    # Русский комментарий: создаём удобный таймстамп для имени отчёта
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

