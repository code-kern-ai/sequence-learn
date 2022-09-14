import numpy as np
import torch
from typing import Optional
from sequencelearn import CONSTANT_OUTSIDE, SequenceTagger
from sequencelearn.modules.crf_head import CRFHead
from sequencelearn.util import convert_to_entropy, pad_and_mark, batch


class CRFTagger(SequenceTagger):
    def __init__(
        self,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        random_seed: Optional[int] = None,
        batch_size: int = 128,
        print_every: int = 10,
        verbose: bool = False,
        constant_outside=CONSTANT_OUTSIDE,
        **kwargs
    ):
        super().__init__(constant_outside)
        self.model = CRFHead(**kwargs)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.print_every = print_every
        self.verbose = verbose

    def predict_proba(self, embeddings: np.array) -> np.array:
        preds_ = []
        confs_ = []
        for embeddings_batch in batch(embeddings, self.batch_size):
            embeddings_padded, _, _ = pad_and_mark(
                embeddings_batch,
                self.CONSTANT_OUTSIDE,
            )
            embeddings_padded = torch.tensor(embeddings_padded.astype(np.float32))
            preds, confs = self.model.forward(embeddings_padded, inference=True)
            for embedding, pred, conf in zip(embeddings_batch, preds, confs):
                pred_ids = pred[: len(embedding)].tolist()
                preds_.append([self.idx2label[pred_id] for pred_id in pred_ids])
                confs_.append([conf for _ in pred_ids])
        return preds_, confs_

    def fit(self, embeddings, labels):
        embeddings_padded, labels_padded, _ = pad_and_mark(
            embeddings, self.CONSTANT_OUTSIDE, labels
        )
        labels_padded = self.convert_labels_and_create_mappings(labels_padded)
        labels_padded = convert_to_entropy(labels_padded)
        embeddings_padded = torch.tensor(embeddings_padded.astype(np.float32))
        labels_padded = torch.tensor(labels_padded)
        self.model.fit(
            embeddings_padded,
            labels_padded,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            random_seed=self.random_seed,
            print_every=self.print_every,
            verbose=self.verbose,
        )
