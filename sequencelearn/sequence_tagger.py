import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import torch.optim as optim


def pad_embeddings(embeddings):
    longest_sequence = max([len(sequence) for sequence in embeddings])
    embedding_dim = len(embeddings[0][0])
    embeddings_padded = torch.zeros([len(embeddings), longest_sequence, embedding_dim])
    padding_mask = torch.ones([len(embeddings), longest_sequence])
    for idx, sequence in enumerate(embeddings):
        sequence_tensor = torch.tensor(sequence)
        embeddings_padded[idx][: len(sequence_tensor)] = sequence_tensor
        padding_mask[idx][len(sequence_tensor) :] = 0
    return embeddings_padded, padding_mask


def convert_labels(labels, constant_outside="OUTSIDE"):
    label_options = [constant_outside] + np.unique(
        [item for sublist in labels for item in sublist if item != constant_outside]
    ).tolist()

    longest_sequence = max([len(sequence) for sequence in labels])
    label_tensor = np.zeros([len(labels), longest_sequence, len(label_options)])
    for row_idx, label_list in enumerate(labels):
        for column_idx, label in enumerate(label_list):
            label_tensor[row_idx][column_idx][label_options.index(label)] = 1
    label_tensor = torch.tensor(label_tensor.argmax(axis=2))
    return label_tensor


class CRFTagger(nn.Module):
    def __init__(self, hidden_dim: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim

    def _build(self, embedding_dim, num_classes):
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.linear = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.class_linear = nn.Linear(self.hidden_dim, self.num_classes)
        self.crf = CRF(self.num_classes, batch_first=True)

    def loss_fn(self, x, y):
        return -self.crf(x, y)

    def forward(self, x, binarize=False):
        x = F.relu(self.linear(x))
        x = self.class_linear(x)

        if binarize:
            x = torch.LongTensor(self.crf.decode(x))
            x = F.one_hot(x, self.num_classes).float()

        return x

    def fit(self, x, y, num_epochs=100):

        embedding_dim = x.shape[-1]
        num_classes = int(y.max()) + 1

        self._build(embedding_dim, num_classes)

        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(num_epochs):

            optimizer.zero_grad()
            predictions = self.forward(x, binarize=False)
            loss = self.loss_fn(predictions, y)
            loss.backward()
            optimizer.step()
