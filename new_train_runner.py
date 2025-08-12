from hydra import main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
@main(config_path="conf", config_name="config",version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    # Here you would typically call your training function with the cfg
    # For example:
    # train(cfg)
    
if __name__ == "__main__":
    main()