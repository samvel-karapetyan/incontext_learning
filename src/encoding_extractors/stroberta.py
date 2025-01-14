import os
import torch
import torch.nn as nn

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import CrossEncoder

class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        return x

class STRoBERTa(nn.Module):
    def __init__(self,
                 model_name,
                 *args, **kwargs):
        super().__init__()
        self.model = CrossEncoder(model_name)

        self.model.model.classifier.out_proj = Identity()

    def forward(self, x):
        emb = self.model.predict(x, convert_to_tensor=True, show_progress_bar=False)

        return emb