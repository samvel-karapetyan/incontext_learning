import numpy as np


class PartlySwapper:
    """
    A class that performs partial swapping of points in image representations to create augmented examples for minority groups.

    Args:
        swapping_minority_proportion_context (float): The proportion of the minority group's to create via swapping in context.
        swapping_minority_proportion_query (float): The proportion of the minority group's to create via swapping in queries.
        points_to_swap_range (list): A list containing the range of the number of points to swap in the selected vectors.

    Returns:
        tuple: A tuple containing the modified encoding vectors of the minority and majority groups,
            along with the corresponding spurious labels.
    """

    def __init__(
            self, 
            swapping_minority_proportion_context, 
            swapping_minority_proportion_query, 
            points_to_swap_range):
        self._context_prop = swapping_minority_proportion_context
        self._query_prop = swapping_minority_proportion_query
        self._low, self._high = list(points_to_swap_range) # convert ListConfig to List

    def _swap_points(
            self,
            image_enc: np.ndarray, 
            labels: np.ndarray, 
            spurs: np.ndarray,
            minority_count: int,
            selected_dims: np.ndarray,
            ):
        seq_len, n_dim = image_enc.shape

        selected_first_class = np.random.choice(np.where(labels == 0)[0], size=minority_count, replace=False)
        selected_sec_class = np.random.choice(np.where(labels == 1)[0], size=minority_count, replace=False)

        # Creating masks over image_enc to easily apply swapping
        first_class_points_mask = np.in1d(np.arange(seq_len), selected_first_class).reshape(-1, 1) \
            @ np.in1d(np.arange(n_dim), selected_dims).reshape(1, -1)

        sec_class_points_mask = np.in1d(np.arange(seq_len), selected_sec_class).reshape(-1, 1) \
            @ np.in1d(np.arange(n_dim), selected_dims).reshape(1, -1)

        # Swapping points in the selected positions
        temp = image_enc[first_class_points_mask].copy()
        image_enc[first_class_points_mask] = image_enc[sec_class_points_mask]
        image_enc[sec_class_points_mask] = temp
        
        temp = spurs[selected_first_class].copy() 
        spurs[selected_first_class] = spurs[selected_sec_class]
        spurs[selected_sec_class] = temp

        return image_enc, spurs

    def __call__(
            self, 
            image_enc: np.ndarray, 
            context_labels: np.ndarray, 
            context_spurs: np.ndarray,
            query_img_encodings: np.ndarray,
            query_labels: np.ndarray,
            query_spurs: np.ndarray,
        ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Perform a partial swap of points between classes.

        Returns:
            tuple: A tuple containing the modified encoding vectors and the corresponding 
                spurious labels of context and queries.
        """

        assert (context_labels == context_spurs).all()
        assert (context_labels == 0).sum() == (context_labels == 1).sum()

        assert (query_labels == query_spurs).all()
        points_to_swap = np.random.randint(low=self._low, high=self._high)

        seq_len, n_dim = image_enc.shape
        # NOTE: First 3 points each representation denote type (image/label/query)
        selected_dims = np.random.choice(np.arange(3, n_dim), size=points_to_swap, replace=False)
 
        context_class_size = (context_labels == 0).sum()
        context_minority_count = int(self._context_prop * context_class_size)

        image_enc, context_spurs = self._swap_points(
                                            image_enc=image_enc,
                                            labels=context_labels,
                                            spurs=context_spurs,
                                            minority_count=context_minority_count,
                                            selected_dims=selected_dims)
        
        query_minority_count = int(0.5 * self._query_prop * seq_len)

        query_img_encodings, query_spurs = self._swap_points(
                                            image_enc=query_img_encodings,
                                            labels=query_labels,
                                            spurs=query_spurs,
                                            minority_count=query_minority_count,
                                            selected_dims=selected_dims)

        return image_enc, context_spurs, query_img_encodings, query_spurs
