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
            train_cls = hydra.utils.get_class(self.config.train_config._target_)
            self.train_dataset = train_cls(self.config.train_config)
            val_cls = hydra.utils.get_class(self.config.val_config._target_)
            self.val_dataset = val_cls(self.config.val_config)
        elif stage == "test":
            test_cls = hydra.utils.get_class(self.config.test_config._target_)
            self.test_dataset = test_cls(self.config.test_config)
        elif stage == "predict":
            predict_cls = hydra.utils.get_class(self.config.predict_config._target_)
            self.predict_dataset = predict_cls(self.config.predict_config)
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
