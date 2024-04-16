import logging

from hydra.utils import instantiate
from pytorch_lightning.utilities import model_summary
from omegaconf import DictConfig

from src.utils import log_hyperparameters, setup_aim_logger

log = logging.getLogger(__name__)


def train(config: DictConfig):
    log.info(f"Instantiating model <{config.model._target_}>")
    model = instantiate(config.model,
                        optimizer_conf=config.optimizer,
                        scheduler_conf=config.scheduler
                        )
    log.info(repr(model_summary.summarize(model)))
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule = instantiate(config.datamodule)

    # Init lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(instantiate(cb_conf))

    # Init lightning loggers
    loggers = []
    if "loggers" in config:
        for name, lg_conf in config.loggers.items():
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger = instantiate(lg_conf)
            loggers.append(logger)

            if name == 'aim':
                setup_aim_logger(logger)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = instantiate(config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial")

    log_hyperparameters(config=config, model=model, trainer=trainer)

    # Train the model
    log.info("Starting training!")
    trainer.fit(model, datamodule=datamodule)
