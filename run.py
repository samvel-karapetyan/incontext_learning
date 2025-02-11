import hydra
import logging
import warnings

from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

load_dotenv()

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(config: DictConfig) -> None:
    from src import (utils,
                     extract_encodings,
                     compute_encodings_avg_norm_and_generate_tokens,
                     train,
                     evaluate)

    seed_everything(config.seed)

    if config.get("print_config"):
        utils.print_configs(config, fields=tuple(config.keys()), resolve=True)

    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if config.name == "extract_encodings":
        return extract_encodings(config)

    if config.name == "compute_encodings_avg_norm_and_generate_tokens":
        return compute_encodings_avg_norm_and_generate_tokens(config)

    if config.name == "train":
        return train(config)

    if config.name == "evaluate":
        return evaluate(config)
    

if __name__ == '__main__':
    main()
