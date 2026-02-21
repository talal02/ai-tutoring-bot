import logging
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG':    '\033[36m',   # Cyan
        'INFO':     '\033[32m',   # Green
        'WARNING':  '\033[33m',   # Yellow
        'ERROR':    '\033[31m',   # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET':    '\033[0m',
    }

    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    detailed_fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    simple_fmt = ColoredFormatter('%(levelname)s - %(message)s')

    if console_output:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper()))
        ch.setFormatter(simple_fmt)
        logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(getattr(logging, level.upper()))
        fh.setFormatter(detailed_fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
