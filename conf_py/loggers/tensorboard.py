# _target_: pytorch_lightning.loggers.TensorBoardLogger
# save_dir: "logs"
# version: "1.0"
# name: "tensorboard_logger"


from src.logger.base import TensorBoardLoggerConfig
config = TensorBoardLoggerConfig(
    save_dir="logs",
    version="1.0",
    name="tensorboard_logger"
)
