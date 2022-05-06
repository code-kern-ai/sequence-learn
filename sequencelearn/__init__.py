import numpy as np
from abc import ABC, abstractmethod
from sequencelearn.util import pad_and_mark
from typing import List, Union, Tuple

CONSTANT_OUTSIDE = "OUTSIDE"
# you can choose any kind of marker for your out-of-scope labels


class BaseTagger(ABC):
    @abstractmethod
    def __init__(self, constant_outside: str):
        super().__init__()
        self.CONSTANT_OUTSIDE = constant_outside

    def convert_labels_and_create_mappings(self, labels: np.array) -> np.array:
        """_summary_

        Args:
            labels (np.array): _description_

        Returns:
            np.array: _description_
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
        embeddings: List[List[Union[float, List[float]]]],
        labels: List[Union[str, List[str]]],
    ):
        """_summary_

        Args:
            embeddings (List[List[Union[float, List[float]]]]): _description_
            labels (List[Union[str, List[str]]]): _description_
        """
        pass

    @abstractmethod
    def predict_proba(
        self, embeddings: List[List[Union[float, List[float]]]]
    ) -> Tuple[List[Union[str, List[str]]], List[Union[float, List[float]]]]:
        """_summary_

        Args:
            embeddings (List[List[Union[float, List[float]]]]): _description_

        Returns:
            Tuple[List[Union[str, List[str]]], List[Union[float, List[float]]]]: _description_
        """
        pass

    def predict(
        self, embeddings: List[List[Union[float, List[float]]]]
    ) -> List[Union[str, List[str]]]:
        """_summary_

        Args:
            embeddings (List[List[Union[float, List[float]]]]): _description_

        Returns:
            List[Union[str, List[str]]]: _description_
        """
        predictions_unsqueezed, _ = self.predict_proba(embeddings)
        return predictions_unsqueezed


class PointTagger(BaseTagger):
    """_summary_

    Args:
        constant_outside (str): _description_
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
    """_summary_

    Args:
        constant_outside (str): _description_
    """
    def __init__(self, constant_outside):
        super().__init__(constant_outside)

    @abstractmethod
    def predict_proba(self, embeddings):
        pass

    @abstractmethod
    def fit(self, embeddings, labels):
        pass
    