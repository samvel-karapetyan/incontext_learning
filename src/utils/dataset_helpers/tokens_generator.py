import numpy as np


class TokenGenerator:
    def __init__(self,
                 tokens_data):
        """
        Initialize the TokenGenerator with token data and configuration flags.

        Parameters:
        tokens_data (dict): A dictionary containing token-related data such as 'token_len', 'avg_norm', etc.
        """
        self.tokens_data = tokens_data

    def __call__(self):
        """Creates and returns token generators based on the configuration.

        Returns: A tuple containing 2 generators: query, class.
        """

        def fixed_class_tokens_fn():
            return self.tokens_data["opposite_class_tokens"]

        return fixed_class_tokens_fn
