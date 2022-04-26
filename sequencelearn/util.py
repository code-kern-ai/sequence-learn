import numpy as np


def pad_and_mark(embeddings, constant_outside, labels=None):
    """
    pad ragged array to the max length of the item length (i.e. [num records x item length x embedding dim] of the embeddings)
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
