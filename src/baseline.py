import logging
import matplotlib
from itertools import product

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from collections.abc import Iterable

from src.evaluate import transform_and_filter_results, aggregate_results_by_set_and_class_size,\
    create_and_save_results_dataframe, plot_results_and_save_figure

matplotlib.use('Agg')
log = logging.getLogger(__name__)


def baseline(config: DictConfig):
    log.info(f"Calculating baselines for <{config.datamodule.name}>")
    context_class_sizes = list(config.datamodule.context_class_size) \
        if isinstance(config.datamodule.context_class_size, Iterable) \
        else [config.datamodule.context_class_size]
    
    assert config.spurious_setting in ["wb_erm", "inat_no_spurious", "inat_sum_erm"]
    
    baseline_methods = dict()

    # parameters to mention in the file name
    method_hyperparams = {
        method_name: [param_name for param_name, param in method_config.items() if isinstance(param, ListConfig)]
        for method_name, method_config in config.methods.items() 
    }

    for method_name, method_config in config.methods.items():
        # constructing a grid for the method
        method_grid = list(product(*[
            list(param) if isinstance(param, ListConfig) else [param] 
            for param in method_config.values()
        ]))

        for single_method_params in method_grid:
            single_method = dict(zip(method_config.keys(), single_method_params))

            selected_params = "_".join([
                str(single_method[param_name])
                for param_name in method_hyperparams[method_name]
            ])

            if selected_params:
                single_method_name = f"{method_name}_{selected_params}"
            else:
                single_method_name = method_name
            
            baseline_methods.update({single_method_name: instantiate(single_method)})

    for method_name, method in baseline_methods.items():    
        log.info(f"Starting to calculate metrics using {method_name.upper()}")

        val_sets = [f"val_{x}" for x in config.datamodule.val_sets] if config.datamodule.val_sets else ['val']
        list_of_results = run_evaluations_with_repetitions(method,
                                                           datamodule_config=config.datamodule,
                                                           context_class_sizes=context_class_sizes,
                                                           val_sets=val_sets,
                                                           n_repeat=config.n_repeat)

        transform_and_filter_results(list_of_results)
        combined_results = aggregate_results_by_set_and_class_size(list_of_results, val_sets, context_class_sizes)

        for set_name in val_sets:
            results_mean_sem = combined_results[set_name]

            create_and_save_results_dataframe(results_mean_sem, context_class_sizes,
                                              title=set_name,
                                              filename=f"{method_name}_{set_name}.csv")

            if len(context_class_sizes) > 1:
                title = f"{config.datamodule.name}, {set_name} | {config.spurious_setting}"
                plot_results_and_save_figure(results_mean_sem,
                                             title=title,
                                             filename=f"{method_name}_{set_name}.png")


def run_evaluations_with_repetitions(method, datamodule_config, context_class_sizes, val_sets, n_repeat):
    """
    Runs the evaluation of the model multiple times over different validation sets and context class sizes to ensure
    statistical reliability of the results.

    Args:
        method: A baseline method.
        datamodule_config: Configuration for the data module.
        context_class_sizes (list): List of context class sizes for evaluation.
        val_sets (list): List of validation sets to evaluate on.
        n_repeat (int): Number of repetitions for each evaluation setup.

    Returns:
        list: A list of dictionaries containing evaluation results for each repetition.
    """
    list_of_results = []
    for repeat_idx in range(n_repeat):
        results = {set_name: [] for set_name in val_sets}

        for context_class_size in context_class_sizes:
            log.info(f"Starting {repeat_idx=} of {context_class_size=}")
            datamodule_config.context_class_size = context_class_size
            # Instantiate data module from configuration
            log.info(f"Instantiating datamodule <{datamodule_config._target_}>")
            datamodule = instantiate(datamodule_config)
            datamodule.setup()

            # eval_results = trainer.validate(model, datamodule=datamodule)
            eval_results = method.validate(datamodule)
            for i, result in enumerate(eval_results):
                set_name = val_sets[i]
                result = {key.split(f"{set_name}_")[-1].split("/")[0]: val for key, val in result.items()}
                # key.split(f"{set_name}_")[-1].split("/")[0] is for 'val_inner_loss_epoch/dataloader_idx_0'-like cases

                results[set_name].append(result)

        list_of_results.append(results)

    return list_of_results
