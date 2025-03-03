import hydra
from omegaconf import DictConfig
from models import get_classifier
from utils import count_model_parameters

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    model = get_classifier(cfg['model'])
    print(f"{cfg['model']['model_name']} parameter: {count_model_parameters(model):3,}")

if __name__ == "__main__":
    my_app()
