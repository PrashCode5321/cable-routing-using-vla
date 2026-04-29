import logging
from utils import presets

logger = logging.getLogger("VLA")
logger.setLevel(presets.LOG_LEVEL)

formatter = logging.Formatter(
    fmt='[%(asctime)s] [%(levelname)s] [%(filename)s:%(funcName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)