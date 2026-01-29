from datetime import datetime
from contextlib import contextmanager


@contextmanager
def timer(logger, operation: str = "operation"):
    start = datetime.now()
    yield
    end = datetime.now()
    duration = (end - start).total_seconds()
    logger.info(f"⏱️ {operation} 耗时 {duration:.2f} 秒")
