import logging

from hydra.utils import instantiate, get_class
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def evaluate(config: DictConfig):
    # Instantiate model from configuration
    log.info(f"Instantiating model <{config.model._target_}>")
    model_class = get_class(config.model._target_)
    del config.model._target_  # Remove _target_ key before instantiation

    model = model_class.load_from_checkpoint(config.checkpoint_path, **instantiate(config.model), map_location='cpu')
    # Specify map_location='cpu' to initially load the model on the CPU, overriding the default behavior of loading
    # it on the GPU it was trained on. This provides flexibility for the model to be moved to a GPU as per the
    # configuration settings of the trainer.

    # Instantiate data module from configuration
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule = instantiate(config.datamodule)

    # Instantiate trainer from configuration
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = instantiate(config.trainer)

    # Validate model using trainer and datamodule
    results = trainer.validate(model, datamodule=datamodule)

    if config.datamodule.name == "inaturalist_emb_contexts":
        # Mapping of set names to their respective dataloader indices
        dataloaders = {
            "val_inner": 0,
            "val_outer": 1,
            "val_inner_outer": 2
        }

        # Initialize an empty string to store result summaries
        result_summaries = ""

        # Iterate over each set and accumulate results in a formatted string
        for set_name in ["val_inner", "val_inner_outer", "val_outer"]:
            dataloader_idx = dataloaders[set_name]
            res = results[dataloader_idx]
            result_summaries += f"{res[f'{set_name}_accuracy_minority/dataloader_idx_{dataloader_idx}']:.2f}|"
            result_summaries += f"{res[f'{set_name}_accuracy_majority/dataloader_idx_{dataloader_idx}']:.2f}|"
            result_summaries += f"{res[f'{set_name}_accuracy/dataloader_idx_{dataloader_idx}']:.2f}\n"

        # Print the accumulated result summaries
        print(result_summaries)
