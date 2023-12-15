import numpy as np


class TokenGenerator:
    def __init__(self, tokens_data, are_spurious_tokens_fixed, are_class_tokens_fixed):
        """
        Initialize the TokenGenerator with token data and configuration flags.

        Parameters:
        tokens_data (dict): A dictionary containing token-related data such as 'token_len',
                            'avg_norm', 'fixed_spurious_tokens', and 'fixed_class_tokens'.
        are_spurious_tokens_fixed (bool): Flag indicating whether to use fixed spurious tokens.
        are_class_tokens_fixed (bool): Flag indicating whether to use fixed class tokens.
        """
        self.tokens_data = tokens_data
        self.are_spurious_tokens_fixed = are_spurious_tokens_fixed
        self.are_class_tokens_fixed = are_class_tokens_fixed

    def __call__(self):
        """
        Creates and returns token generators based on the configuration.

        When an instance of TokenGenerator is called, it will return generators
        that yield either fixed or random tokens for spurious and class tokens, depending on the configuration.

        Returns:
        tuple: A tuple containing two generators. The first generates spurious tokens,
               and the second generates class tokens.
        """

        def random_tokens_generator():
            """
            A generator function that yields random tokens continuously.

            Generates tokens with a normal distribution and normalizes them based on 'avg_norm'.
            """
            while True:
                tokens = np.random.randn(2, self.tokens_data["token_len"]).astype(np.float32)
                tokens = (tokens / np.linalg.norm(tokens, axis=1, keepdims=True)) * self.tokens_data["avg_norm"]
                yield tokens

        def fixed_spurious_tokens_generator():
            """
            A generator function that yields the same fixed spurious tokens continuously.
            """
            while True:
                yield self.tokens_data["fixed_spurious_tokens"]

        def fixed_class_tokens_generator():
            """
            A generator function that yields the same fixed class tokens continuously.
            """
            while True:
                yield self.tokens_data["fixed_class_tokens"]

        # Choose the appropriate generators based on the flags
        spurious_tokens_generator = (fixed_spurious_tokens_generator
                                     if self.are_spurious_tokens_fixed
                                     else random_tokens_generator)()

        class_tokens_generator = (fixed_class_tokens_generator
                                  if self.are_class_tokens_fixed
                                  else random_tokens_generator)()

        return spurious_tokens_generator, class_tokens_generator
