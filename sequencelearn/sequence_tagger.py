from typing import Dict, List
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense
from keras.callbacks import EarlyStopping
import numpy as np
from sequencelearn.base import SequenceTagger, CONSTANT_OUTSIDE
from sequencelearn.util import pad_and_mark


class NeuralSequenceTagger(SequenceTagger):
    def __init__(
        self,
        constant_outside=CONSTANT_OUTSIDE,
    ):
        super().__init__(constant_outside)
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

        embeddings, labels, not_padded = pad_and_mark(
            embeddings, self.CONSTANT_OUTSIDE, labels=labels
        )
        labels = self.convert_labels_and_create_mappings(labels)

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
