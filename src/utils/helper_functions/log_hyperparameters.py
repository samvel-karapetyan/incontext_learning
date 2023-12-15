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
    hparams["trainer"] = config["trainer"]
    hparams["datamodule"] = config["datamodule"]
    hparams["model"] = config["model"]
    hparams["optimizer"] = config["optimizer"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]
    if "scheduler" in config:
        hparams["scheduler"] = config["scheduler"]

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
