import torch.nn as nn
import torchmetrics

from pytorch_lightning import LightningModule
from hydra.utils import instantiate

from src.utils.custom_metrics import MinorityMajorityAccuracy, GroupAccuracy


class InContextLearner(LightningModule):
    """
    A PyTorch Lightning module for in-context learning.
    """

    def __init__(self, network, loss_fn, val_sets, spurious_setting, dataset_name, optimizer_conf=None, scheduler_conf=None):
        """
        Initializes the InContextLearner module with a network, loss function, validation sets, optimizer, and scheduler configurations.

        Args:
            network: The neural network to be used.
            loss_fn: The loss function for training.
            val_sets: A list of validation dataset names.
            spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                    Options include 'separate_token'(x,c) , 'no_spurious'(x), 'sum'(x+c)
            dataset_name (str): Name of the dataset.
            optimizer_conf: Configuration dictionary for the optimizer.
            scheduler_conf: Configuration dictionary for the scheduler.
        """
        super(InContextLearner, self).__init__()

        self._network = network
        self._with_spurious_token = (spurious_setting == 'separate_token')
        self._dataset_name = dataset_name

        self._optimizer_conf = optimizer_conf
        self._scheduler_conf = scheduler_conf
        self._fc = nn.Linear(network.embed_dim, 1)
        self._loss_fn = loss_fn
        self._val_sets = [f"val_{x}" for x in val_sets] if val_sets else ['val']

        self.accuracy = dict()
        self.accuracy_minority = dict()
        self.accuracy_majority = dict()

        if dataset_name == "waterbirds_emb_contexts":
            self.group_accuracies = [dict() for _ in range(4)]

        self._initialize_metrics()

    def forward(self, input_embeds, *args, **kwargs):
        """
        Defines the forward pass for the model.

        Args:
            input_embeds: The input embeddings for the model.

        Returns:
            The output predictions of the model.
        """
        out = self._network(inputs_embeds=input_embeds).last_hidden_state

        pred_embeddings = out[:, 1::3] if self._with_spurious_token else out[:, ::2]

        pred_y = self._fc(pred_embeddings)

        return pred_y

    def _step(self, batch, set_name):
        """
        A step for training or validation.

        Args:
            batch: The batch of data for the step.
            set_name: The name of the dataset (e.g., 'train', 'val_inner', ...).

        Returns:
            The loss for the batch.
        """
        input_seq, spurious_labels, class_labels, image_indices = batch

        pred_y_logit = self.forward(input_seq).squeeze()

        loss = self._loss_fn(pred_y_logit, class_labels.float())

        pred_y = nn.functional.sigmoid(pred_y_logit)
        last_pred_y = pred_y[:, -1]
        last_class_labels = class_labels[:, -1]
        last_spurious_class = spurious_labels[:, -1]

        self.accuracy[set_name].update(last_pred_y, last_class_labels)
        for min_maj_metric in [self.accuracy_minority[set_name],
                               self.accuracy_majority[set_name]]:
            min_maj_metric.update(
                query_prediction_batch=last_pred_y,
                query_target_batch=last_class_labels,
                query_spurious_batch=last_spurious_class,
                context_targets_batch=class_labels[:, :-1],
                context_spurious_vals_batch=spurious_labels[:, :-1]
            )

        self.log(f"{set_name}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{set_name}_accuracy", self.accuracy[set_name], on_step=False, on_epoch=True)
        self.log(f"{set_name}_accuracy_minority", self.accuracy_minority[set_name], on_step=False, on_epoch=True)
        self.log(f"{set_name}_accuracy_majority", self.accuracy_majority[set_name], on_step=False, on_epoch=True)

        if self._dataset_name == "waterbirds_emb_contexts":
            for i in range(4):
                self.group_accuracies[i][set_name].update(
                    query_prediction_batch=last_pred_y,
                    query_target_batch=last_class_labels,
                    query_spurious_batch=last_spurious_class)
                self.log(f"{set_name}_group_{i}_accuracy", self.group_accuracies[i][set_name], on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch):
        """
        The training step. Processes a single batch during training.

        Args:
            batch: The batch to process.

        Returns:
            The loss for the training batch.
        """
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        The validation step. Processes a single batch during validation.

        Args:
            batch: The batch to process.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            The loss for the validation batch.
        """
        set_name = self._val_sets[dataloader_idx]

        return self._step(batch, set_name)

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers.

        Returns:
            The optimizer and (optionally) the learning rate scheduler.
        """
        target = self._optimizer_conf.pop('target')
        optimizer_conf = dict(**self._optimizer_conf, params=self.parameters())
        optimizer = instantiate(optimizer_conf, _target_=target)

        if self._scheduler_conf.target is None:
            return optimizer
        else:
            monitor = self._scheduler_conf.pop('monitor', None)
            interval = self._scheduler_conf.pop('interval', None)
            scheduler_target = self._scheduler_conf.pop('target')
            scheduler = instantiate(self._scheduler_conf, optimizer=optimizer, _target_=scheduler_target)

            ret_opt = dict(optimizer=optimizer,
                           lr_scheduler={"scheduler": scheduler, "monitor": monitor, "interval": interval})

            return ret_opt

    def _initialize_metrics(self):
        """
        Initializes metrics for training and validation.
        """

        for set_name in ["train"] + self._val_sets:
            if self._dataset_name == "waterbirds_emb_contexts":
                for i in range(4):
                    self.group_accuracies[i][set_name] = GroupAccuracy(group=i)

            self.accuracy[set_name] = torchmetrics.Accuracy(task="binary")
            self.accuracy_minority[set_name] = MinorityMajorityAccuracy(group_type="minority")
            self.accuracy_majority[set_name] = MinorityMajorityAccuracy(group_type="majority")

            self._set_metric_attributes(set_name)

    def _set_metric_attributes(self, set_name):
        """
        Sets metric attributes for a given set name.

        Args:
            set_name: The name of the dataset (e.g., 'train', 'val_inner', ...).
        """
        setattr(self, f"{set_name}_accuracy", self.accuracy[set_name])
        setattr(self, f"{set_name}_accuracy_minority", self.accuracy_minority[set_name])
        setattr(self, f"{set_name}_accuracy_majority", self.accuracy_majority[set_name])

        if self._dataset_name == "waterbirds_emb_contexts":
            for i in range(4):
                setattr(self, f"{set_name}_group_{i}_accuracy", self.group_accuracies[i][set_name])
