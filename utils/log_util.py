from loguru import logger
from pathlib import Path
import sys

class LoggerPool:
    def __init__(self):
        self.logger_pool = {}
        self._initialized = False
        
    def _initialize_default(self):
        if not self._initialized:
            self.set_logger("default", "DEBUG", "", "", "")
            self._initialized = True

    def get_logger_pool(self):
        self._initialize_default()
        return self.logger_pool
    
    def get_logger(self, name):
        self._initialize_default()
        return self.logger_pool.get(name, self.logger_pool.get("default"))
    
    def set_logger(self, name: str, log_level: str, log_dir: str, retention: str, rotation: str):
        custom_logger = logger.bind(name=name)
        custom_logger.remove()
        
        # 控制台输出格式（包含完整路径，可点击）
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<magenta>{file.path:}:{line:}</magenta> | "  # 添加文件路径和行号
            "<cyan>{function}</cyan> - "
            "<level>{message}</level>"
        )
        
        # 文件输出格式（包含完整路径，可点击）
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{file.path}:{line} - "  # 添加文件路径和行号
            "{message}"
        )

        # 添加控制台输出
        custom_logger.add(
            sink=sys.stderr,
            level=log_level,
            format=console_format,
            colorize=True,
            enqueue=True,
        )

        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            # 普通日志文件
            custom_logger.add(
                sink=f"{log_dir}/{name}_{{time:YYYY-MM-DD}}.log",
                level=log_level,
                format=file_format,
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
                enqueue=True,
            )

            # 错误日志文件
            custom_logger.add(
                sink=f"{log_dir}/{name}_errors_{{time:YYYY-MM-DD}}.log",
                level="ERROR",
                format=file_format + "\n{exception}",
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
                enqueue=True,
            )

        self.logger_pool[name] = custom_logger

logger_pool = LoggerPool()