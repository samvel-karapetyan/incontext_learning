import numpy as np


class TokenGenerator:
    def __init__(self,
                 tokens_data,
                 sp_token_generation_mode: str):
        """
        Initialize the TokenGenerator with token data and configuration flags.

        Parameters:
        tokens_data (dict): A dictionary containing token-related data such as 'token_len', 'avg_norm', etc.
        sp_token_generation_mode (str): Mode of spurious (representation) token generation. Accepts 'random' or
            'opposite'. 'random' generates tokens with normal distribution, and 'opposite' generates a pair of
            tokens where the second is the negative of the first.
        """
        self.tokens_data = tokens_data
        self.sp_token_generation_mode = sp_token_generation_mode

    def __call__(self):
        """Creates and returns token generators based on the configuration.

        Returns: A tuple containing 3 generators: spurious (representation), spurious (label), class.
        """

        def random_sp_tokens_generator():
            """
            A generator function that yields random tokens continuously.

            Generates tokens with a normal distribution and normalizes them based on 'avg_norm'.
            """
            while True:
                if self.sp_token_generation_mode == "random":
                    tokens = np.random.randn(2, self.tokens_data["token_len"]).astype(np.float32)
                else:
                    first_row = np.random.randn(1, self.tokens_data["token_len"]).astype(np.float32)
                    tokens = np.vstack((first_row, -first_row))

                tokens = (tokens / np.linalg.norm(tokens, axis=1, keepdims=True)) * self.tokens_data["avg_norm"]
                yield tokens

        def fixed_c_spurious_tokens_generator():
            while True:
                yield self.tokens_data[f"opposite_spurious_tokens_2"]

        def fixed_class_tokens_generator():
            while True:
                yield self.tokens_data[f"opposite_class_tokens"]

        x_spurious_tokens_generator = random_sp_tokens_generator

        c_spurious_tokens_generator = fixed_c_spurious_tokens_generator

        class_tokens_generator = fixed_class_tokens_generator

        return x_spurious_tokens_generator, c_spurious_tokens_generator, class_tokens_generator
