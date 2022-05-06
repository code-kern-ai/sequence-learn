import numpy as np
from abc import ABC, abstractmethod
from sequencelearn.util import pad_and_mark
from typing import List, Tuple

CONSTANT_OUTSIDE = "OUTSIDE"
# you can choose any kind of marker for your out-of-scope labels


class BaseTagger(ABC):
    @abstractmethod
    def __init__(self, constant_outside: str):
        super().__init__()
        self.CONSTANT_OUTSIDE = constant_outside

    def convert_labels_and_create_mappings(self, labels: np.array) -> np.array:
        """Puts numpy array of strings into numpy array of indices, i.e., integers, and stores a mapping in the instance.

        Args:
            labels (np.array): Array containing the labels as strings on token-level

        Returns:
            np.array: Array containing indices
        """
        self.idx2label = None
        self.label2idx = None
        if labels.dtype not in [float, int]:
            self.idx2label = {idx: label for idx, label in enumerate(np.unique(labels))}
            self.label2idx = {label: idx for idx, label in self.idx2label.items()}
            self.classes_ = [label for label in self.label2idx.keys()]
            labels = np.vectorize(self.label2idx.get)(labels)
        return labels

    @abstractmethod
    def fit(
        self,
        embeddings: List[List[List[float]]],
        labels: List[List[str]],
    ):
        """Starts the training procedure for the given tagger.

        Args:
            embeddings (List[List[List[float]]]): Plain list of the embeddings (e.g. created via the code-kern-ai/embedders library)
            labels (List[List[str]]): Plain list of the labels
        """
        pass

    @abstractmethod
    def predict_proba(
        self, embeddings: List[List[List[float]]]
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """Forwards the input data through the tagger and creates label sequence predictions and probabilities

        Args:
            embeddings (List[List[List[float]]]): Plain list of the embeddings (e.g. created via the code-kern-ai/embedders library)

        Returns:
            Tuple[List[List[str]], List[List[float]]]: Plain list of the predicted labels sequence, plain list of the prediction probabilities
        """
        pass

    def predict(
        self, embeddings: List[List[List[float]]]
    ) -> List[List[str]]:
        """Forwards the input data through the tagger and creates label sequence predictions

        Args:
            embeddings (List[List[List[float]]]): Plain list of the embeddings (e.g. created via the code-kern-ai/embedders library)

        Returns:
            List[List[str]]: plain list of the predicted labels sequence
        """
        predictions_unsqueezed, _ = self.predict_proba(embeddings)
        return predictions_unsqueezed


class PointTagger(BaseTagger):
    """Tagging algorithm based on point-based predictions. Probability distributions are independent from one another (from the algorithm point of view); context can still be given via the embeddings.

    Args:
        constant_outside (str): Placeholder value for predictions that are out-of-scope.
    """
    def __init__(self, constant_outside):
        super().__init__(constant_outside)

    def predict_proba(self, embeddings):
        embeddings, _, not_padded = pad_and_mark(embeddings, self.CONSTANT_OUTSIDE)
        embeddings_padded = np.concatenate(embeddings)[not_padded]

        predictions_expanded = self.model.predict_proba(embeddings_padded)

        predictions = np.argmax(predictions_expanded, axis=-1)
        if self.idx2label:
            predictions = np.vectorize(self.idx2label.get)(predictions)

        confs = []
        for idx, argmax in enumerate(predictions_expanded.argmax(axis=1)):
            confs.append(predictions_expanded[idx][argmax])
        confs = np.array(confs)

        start_idx = 0
        predictions_unsqueezed = []
        confs_unsqueezed = []
        for length in [len(vector) for vector in embeddings]:
            end_idx = start_idx + length
            predictions_unsqueezed.append(predictions[start_idx:end_idx].tolist())
            confs_unsqueezed.append(confs[start_idx:end_idx].tolist())
            start_idx = end_idx

        return predictions_unsqueezed, confs_unsqueezed

    def fit(self, embeddings, labels):
        embeddings, labels, not_padded = pad_and_mark(
            embeddings, self.CONSTANT_OUTSIDE, labels=labels
        )

        labels = self.convert_labels_and_create_mappings(labels)

        embeddings = np.concatenate(embeddings)[not_padded]
        labels = np.concatenate(labels)[not_padded]

        self.model.fit(embeddings, labels)

class SequenceTagger(BaseTagger):
    """Tagging algorithm based on sequence-based predictions. Probability distributions are dependent to one another.

    Args:
        constant_outside (str): Placeholder value for predictions that are out-of-scope.
    """
    def __init__(self, constant_outside):
        super().__init__(constant_outside)

    @abstractmethod
    def predict_proba(self, embeddings):
        pass

    @abstractmethod
    def fit(self, embeddings, labels):
        pass
    