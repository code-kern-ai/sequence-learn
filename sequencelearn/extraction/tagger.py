from typing import Dict, List
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense
from keras.callbacks import EarlyStopping
import numpy as np


class NamedEntityTagger:
    def __init__(self):
        self.model: Sequential = None

    def fit(
        self,
        embeddings: np.array,
        labels: np.array,
        hidden_dim=128,
        batch_size=32,
        num_epochs=5,
    ) -> Dict[str, List[float]]:
        """
        Instantiates and trains a neural named entity tagger.

        Args:
            embeddings (np.array): Input tensor with dimensions [number records x padding length x embedding dimension].
            labels (np.array): Target tensor with dimensions [number records x padding length].
            hidden_dim (int, optional): Dimensionality of the internal RNN. Defaults to 128.
            batch_size (int, optional): Number of samples in a single batch. Defaults to 32.
            num_epochs (int, optional): Number of epochs to be trained. Defaults to 5.

        Returns:
            History: Callback containing information about the training loop.
        """

        self.idx2label = None
        self.label2idx = None
        if labels.dtype not in [float, int]:
            self.idx2label = {idx: label for idx, label in enumerate(np.unique(labels))}
            self.label2idx = {label: idx for idx, label in self.idx2label.items()}
            labels = np.vectorize(self.label2idx.get)(labels)

        padding_length = embeddings.shape[1]
        embedding_dim = embeddings.shape[2]
        num_classes = labels.max() + 1  # 0 is also included
        self.model = self._build(
            padding_length,
            embedding_dim,
            num_classes,
            hidden_dim,
        )

        self.model.compile(
            optimizer="Adam",
            loss="sparse_categorical_crossentropy",
            metrics="accuracy",
        )
        callback = EarlyStopping(monitor="loss", patience=3)
        history = self.model.fit(
            embeddings,
            labels,
            validation_split=0.2,
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[callback],
        )
        return history.history

    def _build(
        self, padding_length: int, embedding_dim: int, num_classes: int, hidden_dim: int
    ) -> Sequential:
        lstm = LSTM(hidden_dim, return_sequences=True)
        bi_lstm = Bidirectional(lstm, input_shape=(padding_length, embedding_dim))
        tag_classifier = Dense(num_classes, activation="softmax")
        sequence_labeller = TimeDistributed(tag_classifier)
        return Sequential([bi_lstm, sequence_labeller])

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
        return np.array(preds_proba)
