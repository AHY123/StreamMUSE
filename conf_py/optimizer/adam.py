from src.optimizer.base import OptimizerConfig

config = OptimizerConfig(
    _target_="pytorch.optim.Adam",
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    weight_decay=0.01,
    amsgrad=False,
    use_nesterov=False,
)
