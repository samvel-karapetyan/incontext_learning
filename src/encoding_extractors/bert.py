import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BERT(nn.Module):
    def __init__(self, 
            model_name, 
            *args, 
            **kwargs
        ):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, x):
        if not isinstance(x[0], str): 
            x = [self.tokenizer.pad_token.join(pair) for pair in x]

        inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        
        inputs = {key: value.to(self.model.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}

        outputs = self.model(**inputs)

        sentence_embeddings = outputs.last_hidden_state[:, 0, :] # Shape: (batch_size, hidden_size)

        return sentence_embeddings