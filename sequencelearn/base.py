import numpy as np

from abc import ABC, abstractmethod


class BaseTagger(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, embeddings, labels) -> None:
        pass

    def _predict(self, embeddings: np.array) -> np.array:
        if self.model is None:
            raise Exception("Model has not been trained yet. Call .fit()")
        return self.model.predict(embeddings)

    def predict(self, embeddings: np.array) -> np.array:
        """
        Forwards tensor through network to create hard predictions.
        Args:
            embeddings (np.array): Input tensor with dimensions [number records x padding length x embedding dimension].
        Returns:
            np.array: Hard predictions for the given input tensor.
        """
        predictions = self._predict(embeddings)
        predictions = np.argmax(predictions, axis=-1)
        if self.idx2label:
            predictions = np.vectorize(self.idx2label.get)(predictions)
        return predictions

    def predict_confidence(self, embeddings: np.array) -> np.array:
        """
        Calculates confidence scores for a prediction given some input tensor.
        Args:
            embeddings (np.array): Input tensor with dimensions [number records x padding length x embedding dimension].
        Returns:
            np.array: Confidence scores (without actual prediction) of the respective prediction for the given input tensor.
        """
        pred_confs = []
        for pred in self._predict(embeddings):
            confs = []
            argmax_indices = pred.argmax(axis=1)
            for argmax_idx, pred_i in zip(argmax_indices, pred):
                confs.append(pred_i[argmax_idx])
            pred_confs.append(confs)
        return np.array(pred_confs)

    def predict_proba(self, embeddings: np.array) -> np.array:
        """
        Combines hard prediction with respective confidence scores for a given input tensor.
        Args:
            embeddings (np.array): Input tensor with dimensions [number records x padding length x embedding dimension].
        Returns:
            np.array: Zipped prediction and confidence scores for a given input tensor.
        """
        preds_proba = []
        for pred, conf in zip(
            # this can be improved performance-wise, as it calls self.model.predict twice with the same input
            self.predict(embeddings),
            self.predict_confidence(embeddings),
        ):
            preds_proba.append(list(zip(pred, conf)))
        prediction_probabilities = np.array(preds_proba, dtype="object, float")
        return prediction_probabilities
