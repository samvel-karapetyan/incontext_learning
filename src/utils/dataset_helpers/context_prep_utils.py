import numpy as np


def generate_spurious_labels(
        primary_label: int,
        secondary_label: int,
        class_size: int,
        minority_proportion: float) -> list[int]:
    """Generates spurious labels based on given labels and proportions.

    Args:
        primary_label (int): The primary label for the majority of tokens.
        secondary_label (int): The secondary label for the minority of tokens.
        class_size (int): The total number of examples in a class.
        minority_proportion (float): The proportion of the minority group in the class.
        extra_token (bool): Whether to add an extra token. Default is False.

    Returns:
        list: A list of spurious labels.
    """
    majority_count = int(class_size * (1 - minority_proportion))
    minority_count = class_size - majority_count

    spurious_tokens = [primary_label] * majority_count + \
                      [secondary_label] * minority_count

    return spurious_tokens


def get_context_example_tokens(
        img_encoding: np.ndarray,
        spurious_token: np.ndarray,
        class_token: np.ndarray,
        spurious_setting: str,
) -> list[np.ndarray]:
    if spurious_setting == 'no_spurious':
        return [img_encoding, class_token]
    if spurious_setting == 'sum':
        return [img_encoding + spurious_token, class_token]
    if spurious_setting == 'separate_token':
        return [img_encoding, spurious_token, class_token]
    if spurious_setting == 'sum_with_spurious':
        return [img_encoding + spurious_token, spurious_token, class_token]
    raise ValueError(
        f"Invalid spurious setting: '{spurious_setting}'. "
        f"Expected 'no_spurious', 'sum', 'separate_token' or 'sum_with_spurious'.")


def get_query_example_tokens(
        img_encoding: np.ndarray,
        spurious_token: np.ndarray,
        spurious_setting: str,
) -> list[np.ndarray]:
    if spurious_setting == 'no_spurious':
        return [img_encoding]
    if spurious_setting == 'sum':
        return [img_encoding + spurious_token]
    if spurious_setting == 'separate_token':
        return [img_encoding]
    if spurious_setting == 'sum_with_spurious':
        return [img_encoding + spurious_token]
    raise ValueError(
        f"Invalid spurious setting: '{spurious_setting}'. "
        f"Expected 'no_spurious', 'sum', 'separate_token' or 'sum_with_spurious'.")


def get_group_counts_based_on_proportions(
        num_examples: int,
        group_proportions: list[float]) -> list[int]:
    """Computes group sizes based on proprtions."""
    assert np.allclose(np.sum(group_proportions), 1.0)
    group_counts = [int(num_examples * p) for p in group_proportions]
    cur_sum = sum(group_counts)
    cur_idx = 0
    while cur_sum < num_examples:
        group_counts[cur_idx] += 1
        cur_sum += 1
        cur_idx = (cur_idx + 1) % len(group_proportions)
    return group_counts
