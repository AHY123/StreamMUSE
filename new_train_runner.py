from hydra import main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from src.project_config import ProjectConfig
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

# from lightning.pytorch.utilities.seed import seed_everything
from src.project_config import ProjectConfig
import os
import shutil
import logging
import sys
import re  # Added for version parsing if needed, but the current approach uses config.version directly
from typing import List, Union, Any, Optional  # Import necessary types
import hydra
from dataclasses import asdict


@main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    project_config = ProjectConfig(**OmegaConf.to_object(cfg))
    print(project_config)
    ConfigStore.instance().store()
    # ProjectConfig = ConfigStore.instance()


# if __name__ == "__main__":
#     main()

logging.basicConfig(
    level=logging.INFO,  # Changed to INFO to allow initial messages from all ranks
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # All processes print to console initially
    ],
)

# Get a logger for this module
logger = logging.getLogger(__name__)
# The first few messages might come from all ranks if this runs before DDP is fully set up.
logger.info("Application started. Initial logging setup complete.")

temp_dir = "my-temp-space"
os.environ["TMPDIR"] = temp_dir
# 确保这个目录存在
os.makedirs(temp_dir, exist_ok=True)


class ProjectRunner:
    def __init__(self, project_config: ProjectConfig):
        self.config = project_config
        # Store the version from config. This will be passed to PL loggers.
        self.logger_version = str(self.config.version)  # Ensure it's a string, e.g., "1.0.0"

        self.datamodule = None
        self.model = None
        self.loggers = None  # Renamed to avoid confusion with Trainer's .logger
        self.trainer = None
        self.callbacks = None

    def setup_datamodule(self):
        try:
            cls = hydra.utils.get_class(self.config.datamodule._target_)
            self.datamodule = cls(self.config.datamodule)
        except Exception as e:
            logger.critical(f"Fatal error setting up datamodule '{self.config.datamodule}': {e}", exc_info=True)
            raise

    def setup_model(self):
        try:
            cls = hydra.utils.get_class(self.config.model._target_)
            self.model = cls(self.config.model)
        except Exception as e:
            logger.critical(f"Fatal error setting up model of type '{self.config.model._target_}': {e}", exc_info=True)
            raise

    def setup_loggers(self):
        loggers = []
        try:
            for logger_config in self.config.loggers:
                loggers.append(hydra.utils.instantiate(logger_config))

            if not self.loggers:
                loggers.append(TensorBoardLogger("logs", name="default"))
                logger.warning("No loggers were configured or recognized. Defaulting to TensorBoardLogger.")
            self.loggers = loggers
        except Exception as e:
            logger.error(f"Error setting up PyTorch Lightning loggers: {e}", exc_info=True)
            # Fallback if logger setup fails, but try to continue with a basic logger
            self.loggers = [TensorBoardLogger("logs", name="default_error_fallback")]
            logger.info("Proceeding with a default TensorBoardLogger due to an error in logger setup.")

    def setup_callbacks(self):
        from src.callback.config import ModelCheckpointCallbackConfig

        try:
            self.callbacks = []
            if self.config.trainer.callbacks:
                for callback_config in self.config.trainer.callbacks:
                    if isinstance(callback_config, ModelCheckpointCallbackConfig):
                        # Instantiate ModelCheckpoint with the correct dirpath
                        callback_config.dirpath = os.path.join(
                            self.loggers[0].save_dir,
                            self.loggers[0].name,
                            f"version_{self.loggers[0].version}",
                            "checkpoints",
                        )
                    else:
                        pass
                    self.callbacks.append(hydra.utils.instantiate(callback_config))
            else:
                logger.warning("No callbacks configured. Proceeding without any callbacks.")
        except Exception as e:
            logger.error(f"Error setting up callbacks: {e}", exc_info=True)
            # Fallback to an empty list if callback setup fails
            self.callbacks = []
            logger.info("Proceeding without callbacks due to an error in callback setup.")

    def setup_trainer(self):
        try:
            self.trainer = Trainer(
                precision="bf16-mixed",  # data precision
                logger=self.loggers,
                # val_check_interval=5,
                # log_every_n_steps=1,
                # limit_val_batches=5,
                log_every_n_steps=5,
                **asdict(self.config.trainer),
                callbacks=self.callbacks,
                gradient_clip_algorithm="norm",
                gradient_clip_val=1.0,  # Example value, adjust as needed
                # strategy="ddp",  # Crucial for distributed training
                strategy="ddp_find_unused_parameters_true",  # Use this for DDP with unused parameters
            )
        except Exception as e:
            logger.critical(f"Fatal error setting up PyTorch Lightning Trainer: {e}", exc_info=True)
            raise

    def _configure_rank_logging(self):
        """
        Configures logging for the current DDP rank.
        Only rank 0 will have a FileHandler and its StreamHandler.
        Non-rank 0 processes will have their StreamHandler removed to suppress console output.
        """
        if self.trainer.is_global_zero:
            # Rank 0: Add a FileHandler for logging to file
            exp_log_dir = self.trainer.logger.log_dir  # Get the actual log directory
            logger.info(f"[RANK {self.trainer.global_rank}] Experiment log directory: {exp_log_dir}")
            os.makedirs(exp_log_dir, exist_ok=True)
            log_file_path = os.path.join(exp_log_dir, "experiment_log.log")

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)  # Set desired logging level for the file
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)  # Add to the specific logger used in this module
            logger.info(
                f"[RANK {self.trainer.global_rank}] All subsequent logs from rank 0 will also be saved to: {log_file_path}"
            )

            # Set the logger level for rank 0 (optional, can be INFO or DEBUG)
            logger.setLevel(logging.INFO)
        else:
            # Non-rank 0: Remove all existing StreamHandlers to suppress console output
            # Keep other handlers if any, but ensure they don't write to shared resources.
            # Also, set their logging level higher to suppress less critical messages.
            logger.setLevel(logging.ERROR)  # Suppress INFO/WARNING on non-rank 0

            # Remove StreamHandlers to prevent console output
            for handler in list(logger.handlers):  # Iterate over a copy to avoid modification issues
                if isinstance(handler, logging.StreamHandler):
                    logger.removeHandler(handler)
            logger.propagate = False  # Prevent logs from propagating to root logger after handlers are removed
            # Any specific non-rank0 file logging would need to be added here for unique files.
            # For "only rank 0 leaves log", we don't add any.

            # A final check to ensure nothing slips to console after removal
            # This line won't appear if StreamHandlers are successfully removed
            logger.info(f"[RANK {self.trainer.global_rank}] Non-rank 0 logging setup complete (console suppressed).")

    # def copy_config(self, trainer: Trainer):
    #     """Copies the configuration file to the experiment log directory.
    #     This must ONLY be called by the global_rank 0 process.
    #     """
    #     # This check is technically redundant if called within `if self.trainer.is_global_zero:` block,
    #     # but good for modularity if it were called independently.
    #     if trainer.is_global_zero:
    #         try:
    #             log_dir = trainer.logger.log_dir
    #             os.makedirs(log_dir, exist_ok=True)
    #             config_copy_path = os.path.join(log_dir, os.path.basename(self.config_path))
    #             shutil.copy(self.config_path, config_copy_path)
    #             logger.info(f"Configuration copied to {config_copy_path}")
    #         except Exception as e:
    #             logger.error(f"Failed to copy configuration file from {self.config_path} to {config_copy_path}: {e}", exc_info=True)

    def run_experiment(self):
        # Initial message from all ranks
        logger.info(f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] Starting experiment setup...")

        try:
            self.setup_datamodule()
            self.setup_model()
            self.setup_loggers()  # Loggers instantiated, but log_dir not final yet
            self.setup_trainer()  # Trainer instantiated. self.trainer.is_global_zero is now reliable.

            # --- Configure logging handlers based on rank ---
            self._configure_rank_logging()

            # --- Critical: File IO operations ONLY on rank 0 ---
            # These methods internally also check for is_global_zero for safety,
            # but calling them within this main rank 0 guard ensures clear flow.
            if self.trainer.is_global_zero:
                # `self.add_file_logger_to_exp_dir()` is now `_configure_rank_logging()`'s responsibility
                # But if you want to explicitly call it here, it will be fine as it's guarded.
                self.copy_config(self.trainer)

            # This log will only appear in rank 0's console and file log (if configured)
            # if _configure_rank_logging successfully removed handlers for non-rank 0.
            logger.info(f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] Experiment setup complete. Starting training...")
        except Exception as e:
            logger.critical(
                f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] Fatal error during experiment setup: {e}", exc_info=True
            )
            sys.exit(1)

        try:
            self.trainer.fit(self.model, datamodule=self.datamodule)
            # This log will only appear in rank 0's console and file log
            logger.info(f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] Experiment training finished successfully!")
        except Exception as e:
            # Errors from non-rank 0 will still be printed if `basicConfig` is not completely removed
            # from them. However, `logger.critical` is still level-checked.
            logger.error(
                f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] An error occurred during model training: {e}", exc_info=True
            )
            # raise # Uncomment to re-raise the exception and stop execution

        # This log will only appear in rank 0's console and file log
        logger.info(f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] Experiment run process completed.")


# if __name__ == "__main__":

#     # torch.cuda.memory._record_memory_history() # start memory snapshot

#     # Example usage
#     # runner = ProjectRunner(config_path="schema/yaml/old_m2a_transformer_aria_skyline_v0-1.2.yaml")
#     runner = ProjectRunner(config_path="conf/old_m2a_transformer_pop909-1.0.yaml")  # Use your specific config

#     runner.run_experiment()
