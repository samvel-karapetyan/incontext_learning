import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from hydra.utils import instantiate, get_class
from omegaconf import DictConfig
from collections.abc import Iterable

matplotlib.use('Agg')

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

    # Instantiate trainer from configuration
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = instantiate(config.trainer)

    if isinstance(config.datamodule.context_class_size, Iterable):
        val_sets = [f"val_{x}" for x in config.datamodule.val_sets] if config.datamodule.val_sets else ['val']
        combined_results = {set_name: [] for set_name in val_sets}

        context_class_sizes = list(config.datamodule.context_class_size)
        for context_class_size in context_class_sizes:
            config.datamodule.context_class_size = context_class_size
            # Instantiate data module from configuration
            log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
            datamodule = instantiate(config.datamodule)

            # Validate model using trainer and datamodule
            results = trainer.validate(model, datamodule=datamodule)
            for i, result in enumerate(results):
                set_name = val_sets[i]
                result = {key.split(f"{set_name}_")[-1].split("/")[0]: val for key, val in result.items()}
                combined_results[set_name].append(result)

        x = context_class_sizes
        for set_name, results in combined_results.items():
            fig, axs = plt.subplots(1, 3, figsize=(10, 4))
            results = {metric_name: [res[metric_name]
                                     for res in results] for metric_name in results[0]
                       if "accuracy" in metric_name}

            min_value = 1.0
            max_value = 0
            for (metric_name, result), ax in zip(results.items(), axs):
                ax.plot(x, result)
                ax.scatter(x, result)
                ax.set_title(metric_name)
                ax.set_xticks(x)
                ax.axhline(y=1.0, color='gray', linestyle='--')

                min_value = np.min(result + [min_value])
                max_value = np.max(result + [max_value])

            for ax in axs:
                ax.set_ylim(min_value - 0.005, max_value + 0.005)

            plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust subplots to fit into the figure area.
            plt.suptitle(f"{config.datamodule.name}, {set_name} | {config.spurious_setting} | {config.aim_hash}", fontsize=16)
            plt.savefig(f"{set_name}.png")