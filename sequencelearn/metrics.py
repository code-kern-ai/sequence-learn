from typing import List, Optional, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix

from sequencelearn import CONSTANT_OUTSIDE, util


def get_confusion_matrix(
    predictions: List[List[str]],
    labels: List[List[str]],
    constant_outside: Optional[str] = CONSTANT_OUTSIDE,
) -> Tuple[np.array, List[str]]:
    """Calculates the confusion matrix on BIO-level

    Args:
        predictions (List[List[str]]): Plain list of the model predictions
        labels (List[List[str]]): Plain list of labels
        constant_outside (Optional[str], optional): Placeholder for out-of-scope entities. Defaults to "OUTSIDE".

    Returns:
        Tuple[np.array, List[str]]: Confusion matrix, and the names of the labels (in order)
    """

    predictions_bio = util.convert_to_bio(predictions, constant_outside, flat=True)
    labels_bio = util.convert_to_bio(labels, constant_outside, flat=True)

    label_options = np.unique(labels_bio).tolist()
    labels_sorted = set()
    for label in label_options:
        if label != constant_outside:
            labels_sorted.add(label[2:])
    labels_sorted = sorted(list(labels_sorted))
    labels_sorted_bio = []
    for label in labels_sorted:
        labels_sorted_bio.append(f"B-{label}")
        labels_sorted_bio.append(f"I-{label}")
    labels_sorted_bio.append(constant_outside)

    cm = confusion_matrix(labels_bio, predictions_bio, labels=labels_sorted_bio)
    return cm, labels_sorted_bio
