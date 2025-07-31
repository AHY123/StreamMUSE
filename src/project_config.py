from typing import Any, Optional
from .utils.base_config import BaseConfig, Field, dataclass, ConfigDict
from pydantic import model_validator
from .logger.base import UnionLoggerConfig
from .model import UnionModelConfig
from .datamodule import UnionDataModuleConfig 
from .callback import UnionCallbackConfig
from typing import Union, Literal
import yaml
import os
import re


def get_next_version(base_dir: str, project_name: Optional[str] = None, version_prefix: Optional[str] = None) -> str:
    """
    Detects existing version directories (e.g., 'version_0', 'version_1')
    under base_dir/(project_name if exists) and returns the next available version string.
    If version_prefix is provided (e.g., "1.0"), it tries to find the max 'x' for "1.0.x"
    otherwise, it auto-increments "version_x".

    Args:
        base_dir: The base directory where logs/versions are saved.
        project_name: Optional, the name of the project, often used as a sub-directory.
        version_prefix: Optional, a version string like "1.0" to find "1.0.x".
                        If None, it looks for "version_x" directories.
    Returns:
        The next version string (e.g., "1.0.5" or "version_3").
    """
    target_dir = os.path.join(base_dir, project_name) if project_name else base_dir

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists for scanning
        if version_prefix:
            return f"{version_prefix}.0"
        return "version_0"

    max_version_num = -1
    for item in os.listdir(target_dir):
        if os.path.isdir(os.path.join(target_dir, item)):
            if version_prefix:
                # Regex for "1.0.x" or similar
                match = re.match(rf"^{re.escape(version_prefix)}\.(\d+)$", item)
                if match:
                    current_x = int(match.group(1))
                    max_version_num = max(max_version_num, current_x)
            else:
                # Regex for "version_x"
                match = re.match(r"^version_(\d+)$", item)
                if match:
                    current_x = int(match.group(1))
                    max_version_num = max(max_version_num, current_x)

    next_version_num = max_version_num + 1
    if version_prefix:
        return f"{version_prefix}.{next_version_num}"
    return f"version_{next_version_num}"

@dataclass
class TrainerConfig:
    """
    Config for the PyTorch Lightning Trainer configuration.
    """
    # _target_: str = Field("pytorch_lightning.Trainer", description="The class to instantiate for the PyTorch Lightning Trainer.")
    max_epochs: int = Field(10, description="Maximum number of epochs for training. Default is 10.")
    accelerator: Optional[str] = Field("auto", description="Accelerator to use for training (e.g., 'cpu', 'gpu'). Default is 'auto'.")
    devices: Optional[Union[int, list[int], tuple[int]]] = Field(
        None, description="Number of devices to use for training. Default is None (use all available)."
    )
    callbacks: list[UnionCallbackConfig] = Field(None, description="List of callback configurations.")
    val_check_interval: Optional[float] = Field(None, description="How often to check the validation set. Can be an int (steps) or a float (fraction of epoch).")
    check_val_every_n_epoch: Optional[int] = Field(1, description="Run validation every n epochs.")
    
    @model_validator(mode="after")
    def validate(self) -> "TrainerConfig":
        """
        Validate the configuration after initialization.
        """
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer.")
        if self.accelerator not in ["cpu", "gpu", "tpu", "ipu", "hpu", "auto"]:
            raise ValueError("Invalid accelerator specified.")
        if self.devices is not None and not isinstance(self.devices, (int, list, tuple)):
            raise ValueError("devices must be an int or a list/tuple of ints.")
        
        return self

@dataclass
class ProjectConfig:
    """
    Config for the M2A Transformer project configuration.
    """

    project: str = Field(..., description="Name of the project.")
    name: Optional[str] = Field(None, description="Sub name of the project. Default is None.")
    version: str = Field("1.0", description="Version of the project. Default is '1.0.0'.")
    save_dir: str = Field("./logs", description="Directory where project logs will be saved.")
    description: Optional[str] = Field(None, description="Description of the project.")
    loggers: list[UnionLoggerConfig] = Field(None, description="List of logger configurations.")
    model: UnionModelConfig = Field(..., description="Model configuration.")
    datamodule: UnionDataModuleConfig = Field(..., description="Data module configuration.") # Changed type
    trainer: TrainerConfig = Field(default=TrainerConfig(), description="Trainer configuration.")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility.")

    @classmethod
    def from_yaml(cls, path: str) -> "ProjectConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @model_validator(mode="after")
    def unify_logger_versions_and_names(self) -> "ProjectConfig":
        if not self.loggers:
            return self
        base_save_dir = self.save_dir
        unified_version = get_next_version(base_dir=base_save_dir, project_name=self.name, version_prefix=self.version)
        for logger_config in self.loggers:
            if logger_config.name is None:
                logger_config.name = self.name
            logger_config.version = unified_version

        return self
    
    @model_validator(mode="after")
    def validate_checkpoint_and_trainer_intervals(self) -> "ProjectConfig":
        """
        Ensures that ModelCheckpoint's step-based saving interval is compatible
        with the Trainer's validation interval.
        """
        if not self.trainer or not self.trainer.callbacks:
            return self

        # 假设你的回调配置中有一个 'type' 或类似的字段来区分不同的回调
        # 这里我们用 isinstance 来模拟，实际中你可能需要检查一个字段
        # from .callback import ModelCheckpointConfig # 你可能需要这个导入
        
        # 获取 Trainer 的验证间隔（以步数为单位）
        trainer_val_steps = self.trainer.val_check_interval
        if not isinstance(trainer_val_steps, int) or trainer_val_steps <= 0:
            # 如果 trainer 不是按 step 验证，则此检查不适用
            return self

        for callback_config in self.trainer.callbacks:
            # 假设你的 ModelCheckpoint 配置类名为 ModelCheckpointConfig
            # 你需要根据你的实现调整这个检查
            if "ModelCheckpoint" not in callback_config.__class__.__name__:
                continue

            # 假设配置中有 monitor 和 every_n_train_steps 字段
            monitor = getattr(callback_config, "monitor", None)
            every_n_steps = getattr(callback_config, "every_n_train_steps", None)

            # 检查是否监控验证指标并且是按步数保存
            if monitor and "val_" in monitor and every_n_steps is not None:
                if every_n_steps < trainer_val_steps:
                    raise ValueError(
                        f"ModelCheckpoint's 'every_n_train_steps' ({every_n_steps}) cannot be smaller than "
                        f"Trainer's 'val_check_interval' ({trainer_val_steps}) when monitoring a validation metric ('{monitor}'). "
                        "This would lead to checking stale validation metrics."
                    )
                if every_n_steps % trainer_val_steps != 0:
                    import warnings
                    warnings.warn(
                        f"For best practice, ModelCheckpoint's 'every_n_train_steps' ({every_n_steps}) should be a multiple of "
                        f"Trainer's 'val_check_interval' ({trainer_val_steps})."
                    )
        return self
