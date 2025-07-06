import torch
from torch.utils.data import Dataset
from .. import UnionDatasetConfig, UnionDataModuleConfig
import pytorch_lightning as pl
import hydra


class BaseDataset(Dataset):
    def __init__(self, config: UnionDatasetConfig): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int): ...


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config: UnionDataModuleConfig):
        super().__init__()
        self.config = config

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.config.train_config)
            self.val_dataset = hydra.utils.instantiate(self.config.val_config)
        elif stage == "test":
            self.test_dataset = hydra.utils.instantiate(self.config.test_config)
        elif stage == "predict":
            self.predict_dataset = hydra.utils.instantiate(self.config.predict_config)
        return super().setup(stage)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.config.train_config.batch_size, shuffle=True, collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config.val_config.batch_size, shuffle=False, collate_fn=self._collate_fn)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.config.test_config.batch_size, shuffle=False, collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch: list): ...
