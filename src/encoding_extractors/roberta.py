import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel


class RoBERTa(nn.Module):
    def __init__(self, 
            model_name, 
            *args, 
            **kwargs
        ):
        super().__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

    def forward(self, x: list[tuple[str, str]]):

        premises, hypotheses = zip(*x)
        inputs = self.tokenizer(premises, hypotheses, return_tensors="pt", padding=True, truncation=True)

        inputs = {key: value.to(self.model.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}

        outputs = self.model(**inputs)

        sentence_embeddings = outputs.last_hidden_state[:, 0, :] # Shape: (batch_size, hidden_size)

        return sentence_embeddings
