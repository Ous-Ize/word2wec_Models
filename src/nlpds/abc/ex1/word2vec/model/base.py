from abc import ABC, abstractmethod
from typing import Self

import torch
from torch import Tensor


class Word2VecBaseABC(ABC, torch.nn.Module):

    @classmethod
    @abstractmethod
    def new(
        cls,
        vocab_size: int,
        embedding_dim: int,
        input_weights: Tensor | None = None,
        output_weights: Tensor | None = None,
    ) -> Self:
        """
        Create a new instance of the model, optionally with existing embedding weights.
        If given, you may assume that the weights are compatible with the given vocabulary size and embedding dimension.
        """
        raise NotImplementedError(f"{cls.__name__}.new() is not implemented")
