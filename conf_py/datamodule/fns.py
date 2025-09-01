# _target_: src.datamodule.old_pt.datamodule.OldPtDataModule
# train_config:
#   _target_: src.datamodule.old_pt.datamodule.OldPtDataset
#   batch_size: 2
#   num_workers: 0
#   file_path: /home/ubuntu/ugrip/data/pop909/pop909_acc_cp4.pt # path to the preprocessed training data
#   target_length: 384
#   tokenization_type: XinYue's
#   split_ratio: 10
# val_config:
#   _target_: src.datamodule.old_pt.datamodule.OldPtDataset
#   batch_size: 2
#   num_workers: 0
#   file_path: /home/ubuntu/ugrip/data/pop909/pop909_acc_cp4.pt
#   target_length: 384
#   tokenization_type: XinYue's
#   split_ratio: 10

# from src.datamodule.fns.config import Fns