import logging
import os
from datetime import datetime

from core import train_path


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
filename = os.path.join(train_path, 'logs.txt')
f = open(filename, 'w+')
logging.basicConfig(filename=filename, level=logger.level)


def log_msg(msg, verbose=True):
    if verbose:
        print(msg)
    logger.debug(f'{datetime.now()} : {msg}')
