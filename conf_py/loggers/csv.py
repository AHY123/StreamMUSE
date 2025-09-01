# _target_: pytorch_lightning.loggers.CSVLogger
# save_dir: "logs"
# version: "1.0"
# name: "csv_logger"

from src.logger.base import CSVLoggerConfig
config = CSVLoggerConfig(
    save_dir="logs",
    version="1.0",
    name="csv_logger"
)