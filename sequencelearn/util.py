import numpy as np
import torch
from typing import List, Tuple, Union, Optional


def pad_and_mark(
    embeddings: List[List[Union[float, List[float]]]],
    constant_outside: str,
    labels: Optional[List[Union[str, List[str]]]] = None,
) -> Tuple[np.array, Optional[np.array], np.array]:
    """_summary_

    Args:
        embeddings (List[List[Union[float, List[float]]]]): _description_
        constant_outside (str): _description_
        labels (Optional[List[Union[str, List[str]]]], optional): _description_. Defaults to None.

    Returns:
        Tuple[np.array, Optional[np.array], np.array]: _description_
    """

    dim_0 = len(embeddings)
    dim_1 = max([len(vector) for vector in embeddings])
    dim_2 = len(embeddings[0][0])

    not_padded = np.zeros([dim_0, dim_1])

    embeddings_padded = np.zeros([dim_0, dim_1, dim_2])
    for idx_matrix, embedding_matrix in enumerate(embeddings):
        for idx_vector, embedding_vector in enumerate(embedding_matrix):
            not_padded[idx_matrix, idx_vector] = 1
            embeddings_padded[idx_matrix][idx_vector] = embedding_vector

    not_padded = not_padded.ravel().astype(bool)

    if labels is not None:
        labels_padded = np.full([dim_0, dim_1], constant_outside, dtype="object")
        for idx_vector, label_vector in enumerate(labels):
            for idx_scalar, label_scalar in enumerate(label_vector):
                labels_padded[idx_vector][idx_scalar] = label_scalar
    else:
        labels_padded = None

    return embeddings_padded, labels_padded, not_padded


def convert_to_entropy(labels: np.array) -> torch.tensor:
    """_summary_

    Args:
        labels (np.array): _description_

    Returns:
        torch.tensor: _description_
    """

    label_options = np.unique([item for sublist in labels for item in sublist]).tolist()

    longest_sequence = max([len(sequence) for sequence in labels])
    label_tensor = np.zeros([len(labels), longest_sequence, len(label_options)])
    for row_idx, label_list in enumerate(labels):
        for column_idx, label in enumerate(label_list):
            label_tensor[row_idx][column_idx][label_options.index(label)] = 1
    return label_tensor.argmax(axis=2)
