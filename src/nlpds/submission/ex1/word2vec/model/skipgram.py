from typing import Self

import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nlpds.abc.ex1.word2vec.model.skipgram import (
    SkipGramNegativeSamplingABC,
    SkipGramSoftMaxABC,
)


class SkipGramSoftMax(SkipGramSoftMaxABC):

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embed_dim)

    @classmethod
    def new(
            cls,
            vocab_size: int,
            embedding_dim: int,
            input_weights: Tensor = None,
            output_weights: Tensor = None,
    ) -> 'SkipGramSoftMaxABC':
        """
        Create a new instance of the model, optionally with existing embedding weights.
        If given, you may assume that the weights are compatible with the given vocabulary size and embedding dimension.
        """
        instance = cls(vocab_size, embedding_dim)

        if input_weights is not None:
            instance.input_embeddings.weight.data.copy_(input_weights)

        if output_weights is not None:
            instance.output_embeddings.weight.data.copy_(output_weights)

        return instance

    def embed_input(
            self,
            target_ids: Tensor,
    ) -> Tensor:
        """Get the input embedding of the given target tokens."""
        return self.input_embedding(target_ids)

    def embed_output(
            self,
            context_ids: Tensor,
    ) -> Tensor:
        """Get the output embeddings of the given context tokens."""
        return self.output_embedding(context_ids)

    def forward(
            self,
            input_id: Tensor,
            context_ids: Tensor,
    ) -> Tensor:
        """
        Perform a skip-gram forward pass.
        Should return one score for each input-context-id pair.
        """
        input_embedding = self.embed_input(input_id)
        output_embeddings_weights = self.output_embedding.weight

        input_context_product = torch.matmul(input_embedding, torch.t(output_embeddings_weights))

        log_probs = F.log_softmax(input_context_product, dim=1)

        return log_probs.gather(1, context_ids.view(-1, 1)).squeeze(1)


class SkipGramNegativeSampling(SkipGramNegativeSamplingABC):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__(vocab_size, embed_dim)
        self.input_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embed_dim)

    @classmethod
    def new(
            cls,
            vocab_size: int,
            embedding_dim: int,
            input_weights: Tensor = None,
            output_weights: Tensor = None,
    ) -> 'SkipGramSoftMaxABC':
        """
        Create a new instance of the model, optionally with existing embedding weights.
        If given, you may assume that the weights are compatible with the given vocabulary size and embedding dimension.
        """
        instance = cls(vocab_size, embedding_dim)

        if input_weights is not None:
            instance.input_embeddings.weight.data.copy_(input_weights)

        if output_weights is not None:
            instance.output_embeddings.weight.data.copy_(output_weights)

        return instance

    def embed_input(self, target_ids: Tensor) -> Tensor:
        """Get the input embedding of the given target tokens."""
        return self.input_embeddings(target_ids)

    def embed_output(self, context_ids: Tensor) -> Tensor:
        """Get the output embeddings of the given context tokens."""
        return self.output_embeddings(context_ids)

    def forward(
        self,
        input_id: Tensor,
        context_ids: Tensor,
        negative_ids: Tensor,
    ) -> Tensor:
        input_embedding = self.input_embedding(input_id)  # (num_samples, embedding_dim)

        context_embeddings = self.output_embedding(context_ids)  # (num_samples, embedding_dim)
        negative_embeddings = self.output_embedding(negative_ids)  # (num_samples, num_negative_words, embedding_dim)

        pos_val = self.log_sigmoid(torch.sum(context_embeddings * input_embedding, dim=1)).squeeze()
        neg_vals = torch.bmm(negative_embeddings, input_embedding.unsqueeze(2)).squeeze(2)
        neg_vals = self.log_sigmoid(-torch.sum(neg_vals, dim=1)).squeeze()

        total_probs = pos_val + neg_vals

        return -total_probs.mean()
