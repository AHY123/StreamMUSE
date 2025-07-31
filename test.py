import hydra
from omegaconf import DictConfig, OmegaConf

# @hydra.main 会读取 conf/config.yaml 并构建配置对象
@hydra.main(version_base=None, config_path="conf", config_name="config")
def test_configuration(cfg: DictConfig) -> None:
    """
    加载并打印由 Hydra 组合的配置.
    """
    print("--- Hydra Composed Configuration ---")
    print(OmegaConf.to_yaml(cfg))

    print("\n--- Instantiating Network Config Object ---")
    try:
        # 使用 hydra.utils.instantiate 根据 _target_ 创建对象
        network_config_object = hydra.utils.instantiate(cfg.network)
        print(f"Successfully instantiated: {type(network_config_object)}")
        print(f"  - Num Hidden Layers: {network_config_object.num_hidden_layers}")
        print(f"  - Intermediate Size: {network_config_object.intermediate_size}")
        print(f"  - Max Position Embeddings (from base): {network_config_object.max_position_embeddings}")
    except Exception as e:
        print(f"Could not instantiate object: {e}")


# if __name__ == "__main__":
test_configuration()