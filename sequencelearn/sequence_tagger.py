import numpy as np
import torch
from sequencelearn import CONSTANT_OUTSIDE, SequenceTagger
from sequencelearn.modules.crf_head import CRFHead
from sequencelearn.util import convert_to_entropy, pad_and_mark


class CRFTagger(SequenceTagger):
    def __init__(self, constant_outside=CONSTANT_OUTSIDE, **kwargs):
        super().__init__(constant_outside)
        self.model = CRFHead(**kwargs)

    def predict_proba(self, embeddings: np.array) -> np.array:
        embeddings_padded, _, _ = pad_and_mark(embeddings, self.CONSTANT_OUTSIDE)
        embeddings_padded = torch.tensor(embeddings_padded.astype(np.float32))
        preds, confs = self.model.forward(embeddings_padded, inference=True)
        preds_ = []
        confs_ = []
        for embedding, pred, conf in zip(embeddings, preds, confs):
            pred_ids = pred[: len(embedding)].tolist()
            preds_.append([self.idx2label[pred_id] for pred_id in pred_ids])
            confs_.append([conf for _ in pred_ids])
        return preds_, confs_

    def fit(self, embeddings, labels, **kwargs):
        embeddings_padded, labels_padded, _ = pad_and_mark(
            embeddings, self.CONSTANT_OUTSIDE, labels
        )
        labels_padded = self.convert_labels_and_create_mappings(labels_padded)
        labels_padded = convert_to_entropy(labels_padded)
        embeddings_padded = torch.tensor(embeddings_padded.astype(np.float32))
        labels_padded = torch.tensor(labels_padded)
        self.model.fit(embeddings_padded, labels_padded, **kwargs)
