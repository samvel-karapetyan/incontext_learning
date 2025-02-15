"""In-context learning transformer implementation.

We use the following naming conventions:
  B: batch size
  L: sequence length
  C: number of context examples per class (class_context_size)
  D: model dimensionality
  Q: number of queries
"""
from hydra.utils import instantiate
from transformers import GPTJModel
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torchmetrics

from src.utils.custom_metrics import MinorityMajorityAccuracy, GroupAccuracy, WorstGroupAccuracy

torch.set_float32_matmul_precision('high')


class InContextLearner(LightningModule):
    """In-context learner with different query prediction at each position.

    This transformer expects a list of tokens some of which are query tokens, and produces predictions on queries.
    Importantly, query tokens are not attended to by tokens on their right. For example, this ICL transformer can
    handle a sequences like this:
    (a) [x1, y1, q1, x2, y2, q2, ..., xn, yn, qn]
    (b) [x1, c1, y1, q1, x2, c2, y2, q2, ..., xn, cn, yn, qn]
    where xi (context example) and qi (query example) are expected to be representations, likely containing information
    about spurious features; ci are binary spurious features encoded in R^D; and yi are binary labels encoded in R^D.
    """

    def __init__(self,
                 embedding_size: int,
                 network: GPTJModel,
                 loss_fn,
                 val_sets,
                 dataset_name: str,
                 optimizer_conf=None,
                 scheduler_conf=None,
                 input_layer_norm: bool = False,
                 ):
        """
        Args:
            embedding_size: The size of image representation.
            network: The neural network to be used.
            loss_fn: The loss function for training.
            val_sets: A list of validation dataset names.
            dataset_name (str): Name of the dataset.
            optimizer_conf: Configuration dictionary for the optimizer.
            scheduler_conf: Configuration dictionary for the scheduler.
        """
        super(InContextLearner, self).__init__()

        if input_layer_norm:
            self._input_ln = nn.LayerNorm(embedding_size, eps=1e-5)
        else:
            self._input_ln = None

        if embedding_size != network.embed_dim:
          self._proj = nn.Linear(embedding_size, network.embed_dim)
        else:
          self._proj = None

        self._network = torch.compile(network, fullgraph=True)
        self._fc = nn.Linear(network.embed_dim, 1)

        self._loss_fn = loss_fn
        self._val_sets = [f"val_{x}" for x in val_sets] if val_sets else ['val']
        self._dataset_name = dataset_name
        self._optimizer_conf = optimizer_conf
        self._scheduler_conf = scheduler_conf

        self.accuracy = dict()
        self.accuracy_minority = dict()
        self.accuracy_majority = dict()

        if dataset_name in ["waterbirds_emb_contexts", "multinli_emb_contexts"]:
            self.group_accuracies = [dict() for _ in range(4)]
            self.worst_group_accuracy = dict()

        self._initialize_metrics()

    def forward(
            self,
            input_embeds: torch.Tensor,
            query_indices: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_embeds: Torch tensor of shape (B, L, D).
            query_indices: Torch tensor of shape (Q,) describing query token positions.

        Returns: a torch tensor of shape (B, Q, 1) consisting of query prediction logits.
        """
        if self._input_ln is not None:
            input_embeds = self._input_ln(input_embeds)
        if self._proj is not None:
            input_embeds = self._proj(input_embeds)
        out = self._network(
            inputs_embeds=input_embeds,
            # output_attentions=True,
        )
        out = out.last_hidden_state
        pred_embeddings = out[:, query_indices]
        pred_y = self._fc(pred_embeddings)
        return pred_y

    def _step(self, batch, set_name):
        """A step for training or validation.

        Args:
            batch: The batch of data for the step. Should be (input_seq, context, queries, query_indices).
                input_seq should be a tensor of shape (B, L, D). context and queries should be tensors of shape
                (B, 2*C, 3) describing context/query examples with (id, spurious_label, class_label) triplets.
                query_indices should be a tensor of shape (B, Q) with equal rows.
            set_name: The name of the dataset (e.g., 'train', 'val_inner', ...).

        Returns:
            The loss for the batch.
        """
        input_seq, context, queries, query_indices = batch

        pred_y_logit = self.forward(input_seq, query_indices[0]).squeeze()
        query_class_labels = queries[:, :, 2]
        loss = self._loss_fn(pred_y_logit, query_class_labels.float())

        with torch.no_grad():
            last_pred_y = nn.functional.sigmoid(pred_y_logit[:, -1])
            last_spurious_class = queries[:, -1, 1]
            last_class_labels = queries[:, -1, 2]

            self.accuracy[set_name].update(last_pred_y, last_class_labels)

            for min_maj_metric in [self.accuracy_minority[set_name],
                                   self.accuracy_majority[set_name]]:
                min_maj_metric.update(
                    query_prediction_batch=last_pred_y,
                    query_target_batch=last_class_labels,
                    query_spurious_batch=last_spurious_class,
                    context_targets_batch=context[:, :, 2],
                    context_spurious_vals_batch=context[:, :, 1],
                )

            self.log(f"{set_name}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"{set_name}_accuracy", self.accuracy[set_name], on_step=False, on_epoch=True)
            self.log(f"{set_name}_accuracy_minority", self.accuracy_minority[set_name], on_step=False, on_epoch=True)
            self.log(f"{set_name}_accuracy_majority", self.accuracy_majority[set_name], on_step=False, on_epoch=True)

            if self._dataset_name in ["waterbirds_emb_contexts", "multinli_emb_contexts",]:
                self.worst_group_accuracy[set_name].update(
                    preds=last_pred_y,
                    targets=last_class_labels,
                    spurious_labels=last_spurious_class,
                )
                self.log(f"{set_name}_worst_group_accuracy", self.worst_group_accuracy[set_name], on_step=False,
                         on_epoch=True)

                for i in range(4):
                    self.group_accuracies[i][set_name].update(
                        query_prediction_batch=last_pred_y,
                        query_target_batch=last_class_labels,
                        query_spurious_batch=last_spurious_class)
                    self.log(f"{set_name}_group_{i}_accuracy", self.group_accuracies[i][set_name], on_step=False,
                             on_epoch=True)

        return loss

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        set_name = self._val_sets[dataloader_idx]
        return self._step(batch, set_name)

    def configure_optimizers(self):
        """Configures the optimizers and learning rate schedulers.

        Returns:
            The optimizer and (optionally) the learning rate scheduler.
        """
        target = self._optimizer_conf.pop('target')
        optimizer_conf = dict(**self._optimizer_conf, params=self.parameters())
        optimizer = instantiate(optimizer_conf, _target_=target)

        if self._scheduler_conf.get('target', None) is None:
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
        """Initializes metrics for training and validation."""

        for set_name in ["train"] + self._val_sets:
            if self._dataset_name in ["waterbirds_emb_contexts", "multinli_emb_contexts"]:
                self.worst_group_accuracy[set_name] = WorstGroupAccuracy()
                for i in range(4):
                    self.group_accuracies[i][set_name] = GroupAccuracy(group=i)

            self.accuracy[set_name] = torchmetrics.Accuracy(task="binary")
            self.accuracy_minority[set_name] = MinorityMajorityAccuracy(group_type="minority")
            self.accuracy_majority[set_name] = MinorityMajorityAccuracy(group_type="majority")

            self._set_metric_attributes(set_name)

    def _set_metric_attributes(self, set_name):
        """Sets metric attributes for a given set name."""
        setattr(self, f"{set_name}_accuracy", self.accuracy[set_name])
        setattr(self, f"{set_name}_accuracy_minority", self.accuracy_minority[set_name])
        setattr(self, f"{set_name}_accuracy_majority", self.accuracy_majority[set_name])

        if self._dataset_name in ["waterbirds_emb_contexts", "multinli_emb_contexts"]:
            setattr(self, f"{set_name}_worst_group_accuracy", self.worst_group_accuracy[set_name])
            for i in range(4):
                setattr(self, f"{set_name}_group_{i}_accuracy", self.group_accuracies[i][set_name])
