import numpy as np
from abc import ABC, abstractmethod
from sequencelearn.util import pad_and_mark

CONSTANT_OUTSIDE = "OUTSIDE"


class BaseTagger(ABC):
    @abstractmethod
    def __init__(self, constant_outside) -> None:
        super().__init__()
        self.CONSTANT_OUTSIDE = constant_outside

    @abstractmethod
    def fit(self, embeddings, labels) -> None:
        pass

    @abstractmethod
    def _predict(self, embeddings: np.array) -> np.array:
        pass

    def convert_labels_and_create_mappings(self, labels):
        self.idx2label = None
        self.label2idx = None
        if labels.dtype not in [float, int]:
            self.idx2label = {idx: label for idx, label in enumerate(np.unique(labels))}
            self.label2idx = {label: idx for idx, label in self.idx2label.items()}
            labels = np.vectorize(self.label2idx.get)(labels)
        return labels

    def predict_proba(self, embeddings: np.array) -> np.array:
        predictions_expanded = self._predict(embeddings)
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

    def predict(self, embeddings: np.array) -> np.array:
        predictions_unsqueezed, _ = self.predict_proba(embeddings)
        return predictions_unsqueezed


class SequenceTagger(BaseTagger):
    def __init__(self, constant_outside) -> None:
        super().__init__(constant_outside)

    def _predict(self, embeddings: np.array) -> np.array:
        embeddings, _, not_padded = pad_and_mark(embeddings, self.CONSTANT_OUTSIDE)
        return np.concatenate(self.model.predict(embeddings))[not_padded]

    @abstractmethod
    def fit(self, embeddings, labels):
        pass


class PointTagger(BaseTagger):
    def __init__(self, constant_outside) -> None:
        super().__init__(constant_outside)

    def _predict(self, embeddings: np.array) -> np.array:
        embeddings, _, not_padded = pad_and_mark(embeddings, self.CONSTANT_OUTSIDE)
        embeddings = np.concatenate(embeddings)[not_padded]
        return self.model.predict_proba(embeddings)

    def fit(self, embeddings, labels):
        embeddings, labels, not_padded = pad_and_mark(
            embeddings, self.CONSTANT_OUTSIDE, labels=labels
        )

        labels = self.convert_labels_and_create_mappings(labels)

        embeddings = np.concatenate(embeddings)[not_padded]
        labels = np.concatenate(labels)[not_padded]

        self.model.fit(embeddings, labels)
