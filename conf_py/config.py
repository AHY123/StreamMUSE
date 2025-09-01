# defaults:
#   - model: old_m2a_roformer
#   - loggers:
#     - union
#   - datamodule: fns
#   - trainer@trainer
#   - _self_


# project: ...
# version: "1.0"
# save_dir: "logs"

from src.project_config import ProjectConfig, TrainerConfig
from conf_py.model.old_m2a_roformer import config as model_config
from conf_py.datamodule.old_pt import config as datamodule_config
from conf_py.loggers.csv import config as csv_logger_config
from conf_py.loggers.tensorboard import config as tensorboard_logger_config
from conf_py.callback.model_checkpoint import config as model_checkpoint_config
# from conf_py
# from src.datamodule.

config = ProjectConfig(
    project_name="roformer",
    version="1.0",
    save_dir="logs",
    model=model_config,
    datamodule=datamodule_config,
    description="RoFormer model configuration",
    loggers=[
        csv_logger_config,
        tensorboard_logger_config,
    ],
    trainer=TrainerConfig(
        max_epochs=10,
        accelerator="auto",
        devices=[0],
        # val_check_interval=
        check_val_every_n_epoch=1,
        callbacks=[
            model_checkpoint_config,
        ],
    ),
    seed=42,
)
