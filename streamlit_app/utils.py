import os
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
import asyncio
import streamlit as st


LOG_DIR = "/home/chaichuk/Team73-Annual-Project/streamlit_app/logs"


class AsyncHandler(logging.Handler):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self.executor = ThreadPoolExecutor(max_workers=1)

    def emit(self, record):
        self.executor.submit(self.handler.emit, record)


def setup_logger(name, log_file, level=logging.DEBUG):
    """Logger setup"""

    os.makedirs(LOG_DIR, exist_ok=True)

    new_logger = logging.getLogger(name)
    new_logger.setLevel(level)

    rotating_handler = RotatingFileHandler(f'{LOG_DIR}/{log_file}', maxBytes=1024 * 1024, backupCount=5)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    rotating_handler.setFormatter(formatter)

    async_handler = AsyncHandler(rotating_handler)
    new_logger.addHandler(async_handler)

    return new_logger


def run_async(coroutine):
    """Run async function"""

    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coroutine)


def display_image_grid(images, cols=2):
    """Display multiple images as grid"""

    rows = len(images) // cols + (len(images) % cols > 0)
    for i in range(rows):
        cols_list = st.columns(cols)
        for j in range(cols):
            index = i * cols + j
            if index < len(images):
                img = images[index]

                with cols_list[j]:
                    st.image(img, use_container_width=True)
