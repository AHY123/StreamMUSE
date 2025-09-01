# _target_: pytorch_lightning.callbacks.ModelCheckpoint
# dirpath: "checkpoints"
# filename: "{epoch:02d}-{val_loss:.2f}"
# monitor: "val_loss"
# save_top_k: 3
# mode: "min"
# save_last: True
# every_n_train_steps: 1000

from src.callback.config import ModelCheckpointCallbackConfig

config = ModelCheckpointCallbackConfig(
    dirpath="checkpoints",
    filename="{epoch:02d}-{step:05d}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
    save_top_k=5,
    save_last=True,
    every_n_epochs=None,
    every_n_steps=None
)