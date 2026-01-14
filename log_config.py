import logging
from typing import Optional


def _level(name: str) -> int:
    numeric = logging.getLevelName(name.upper())
    if isinstance(numeric, int):
        return numeric
    raise ValueError(f"Invalid log level: {name}")


def init_logging(log_level: str = "INFO", console_level: str = "INFO", log_file: Optional[str] = "bot.log") -> None:
    root = logging.getLogger()
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(_level(log_level))
        fh.setFormatter(fmt)
        root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(_level(console_level))
    ch.setFormatter(fmt)
    root.addHandler(ch)

    root.setLevel(min(handler.level for handler in root.handlers))

