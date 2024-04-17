import pytorch_lightning as pl

from omegaconf import DictConfig


def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        trainer: pl.Trainer,
) -> None:
    """ This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally, saves:
        - number of trainable model parameters
    """
    if trainer.logger is None:
        return

    hparams = dict()

    # choose which parts of hydra config will be saved to loggers
    if "seed" in config:
        hparams["seed"] = config["seed"]
    hparams["spurious_setting"] = config["spurious_setting"]
    hparams["sp_token_generation_mode"] = config["sp_token_generation_mode"]
    hparams["v1_behavior"] = config["v1_behavior"]
    hparams["datamodule"] = config["datamodule"]
    hparams["model"] = config["model"]
    hparams["optimizer"] = config["optimizer"]
    if "scheduler" in config:
        hparams["scheduler"] = config["scheduler"]
    hparams["trainer"] = config["trainer"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # xcloud stuff
    if "exp_name" in config:
        hparams["exp_name"] = config["exp_name"]
    if "run_id" in config:
        hparams["run_id"] = config["run_id"]
    if "tb_exp_name" in config:
        hparams["tb_exp_name"] = config["tb_exp_name"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def empty(*args, **kwargs):
    pass
