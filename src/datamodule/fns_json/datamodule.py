import torch
from ..base.datamodule import BaseDataset, BaseDataModule
from .config import FnsJsonDatasetConfig, FnsJsonDataModuleConfig
from .model_input import FnsInput
import pytorch_lightning as pl
from miditok.pytorch_data import DataCollator  # ,DatasetJSON,DatasetMIDI
import json
import os
import hydra
from ...tokenizer import UnionTokenizer


class FnsJsonDataset(BaseDataset):
    def __init__(self, config: FnsJsonDatasetConfig):
        super().__init__(config)
        self.config = config
        self.json_dir = config.json_dir
        self.tokenizer_config = config.tokenizer_config
        self.max_frame_length = config.max_frame_length
        self.stage = config.stage

        token_cls = hydra.utils.get_class(self.tokenizer_config._target_)
        self.tokenizer: UnionTokenizer = token_cls(self.tokenizer_config.config)
        self.FRAME = self.tokenizer.token_ids_of_type("Frame")[0]

        # --- 重构的核心逻辑 ---
        # 预处理文件路径，为 val/test 阶段创建样本块
        self.samples = []
        json_file_paths = [os.path.join(self.json_dir, f) for f in os.listdir(self.json_dir) if f.endswith(".json")]

        if self.stage in ["val", "test"]:
            # 对于验证/测试，将每个文件分割成多个样本
            for path in json_file_paths:
                with open(path, "r") as f:
                    ids = json.load(f)["ids"]
                # 计算这个文件可以被分割成多少个完整的块
                num_chunks = len(ids) // self.max_frame_length
                for i in range(num_chunks):
                    start = i * self.max_frame_length
                    # 将 (文件路径, 块的起始位置) 作为一条样本
                    self.samples.append((path, start))
        else:
            # 对于训练，每个文件就是一条样本
            self.samples = [(path, None) for path in json_file_paths]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> FnsInput:
        json_path, chunk_start = self.samples[idx]

        with open(json_path, "r") as f:
            json_content = json.load(f)

        token_ids = torch.tensor(json_content["ids"])

        if self.stage == "train":
            # 训练时，随机切片
            start_frame_idx = torch.randint(0, self.max_frame_length, (1,)).item()
            end_frame_idx = start_frame_idx + self.max_frame_length
            token_ids = self._slice_by_frames(token_ids, start_frame_idx, end_frame_idx)

        elif self.stage in ["val", "test"]:
            # 验证/测试时，使用预先计算好的块
            end_pos = chunk_start + self.max_frame_length
            token_ids = token_ids[chunk_start:end_pos]

        return FnsInput(token_ids=token_ids)
    
    
    def _slice_by_frames(self, token_ids: torch.Tensor, start_frame_idx: int, end_frame_idx: int) -> torch.Tensor:
        """
        根据 FRAME token 的位置对 token_ids 进行切片。
        它会找到指定索引的 FRAME token，并返回它们之间的序列。
        如果 FRAME token 数量不足，则返回原始序列。
        """
        # 1. 找到所有 FRAME token 的索引
        frame_indices = torch.nonzero(token_ids == self.FRAME).squeeze(-1)

        # 2. 检查是否有足够的 frame token
        if frame_indices.dim() > 0 and len(frame_indices) > end_frame_idx:
            # 3. 获取切片的开始和结束索引
            start_idx = frame_indices[start_frame_idx]
            end_idx = frame_indices[end_frame_idx]

            # 4. 对原始 token_ids 进行切片
            return token_ids[start_idx:end_idx]

        # 如果 frame 不足，则返回原始 token_ids
        return token_ids

class FnsJsonDataModule(BaseDataModule):
    def __init__(self, config: FnsJsonDataModuleConfig):
        super().__init__(config)

    def _collate_fn(self, batch:list[FnsInput]):
        return batch[0]


if __name__ == "__main__":
    # uv run -m src.datamodule.fns_json.datamodule
    from .config import FnsJsonDatasetConfig, FnsJsonDataModuleConfig

    # from miditok.pytorch_data import DatasetJSON, DataCollator
    train_config = FnsJsonDatasetConfig(json_dir="data/FNS-Seperated-POP909-Dataset/original", stage="train")
    val_config = FnsJsonDatasetConfig(json_dir="data/FNS-Seperated-POP909-Dataset/original", stage="val")
    test_config = FnsJsonDatasetConfig(json_dir="data/FNS-Seperated-POP909-Dataset/original", stage="test")
    predict_config = FnsJsonDatasetConfig(json_dir="data/FNS-Seperated-POP909-Dataset/original", stage="predict")
    config = FnsJsonDataModuleConfig(
        train_config=train_config, val_config=val_config, test_config=test_config, predict_config=predict_config
    )
    datamodule = FnsJsonDataModule(config)
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    first_batch = next(iter(train_loader))

    # 打印数据以进行检查
    print("获取到的第一批数据:")
    print(first_batch)
