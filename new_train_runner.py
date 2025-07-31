from hydra import main
from hydra.core.config_store import ConfigStore

@main(config_path="conf", config_name="config")
def main()