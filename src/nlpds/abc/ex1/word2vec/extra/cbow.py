from abc import abstractmethod
from typing import NamedTuple

import torch
from torch import Tensor

from nlpds.abc.ex1.word2vec.data.dataset import DatasetABC
from nlpds.abc.ex1.word2vec.data.negative import DatasetNegativeSamplingABC
from nlpds.abc.ex1.word2vec.model.base import Word2VecBaseABC
from nlpds.abc.ex1.word2vec.model.negative import NegativeSamplingMixin


class CbowSample(NamedTuple):
    input: torch.Tensor  # shape: (window,)
    context: torch.Tensor  # shape: (1,)
    input_offsets: torch.Tensor | None  # shape: (window,)


class CbowDatasetABC(DatasetABC[CbowSample]):
    pass


class CbowSoftMaxABC(Word2VecBaseABC):
    # Hint: Use torch.nn.EmbeddingBag

    @abstractmethod
    def embed_input(
        self,
        input_ids: Tensor,
        input_offsets: Tensor | None = None,
    ) -> Tensor:
        """Get the input embedding of the given tokens."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.embed_output() is not implemented"
        )

    @abstractmethod
    def embed_output(
        self,
        context_ids: Tensor,
    ) -> Tensor:
        """Get the output embeddings of the given tokens."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.embed_output() is not implemented"
        )

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        output_id: Tensor,
        input_offsets: Tensor | None = None,
    ) -> Tensor:
        """Perform a CBOW model forward pass."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() is not implemented"
        )


class CbowNegativeSamplingABC(NegativeSamplingMixin, CbowSoftMaxABC):
    pass


class CbowNegativeSample(NamedTuple):
    input: torch.Tensor  # shape: (window,)
    context: torch.Tensor  # shape: (1,)
    negative: torch.Tensor  # shape: (k,)


class CbowDatasetNegativeSamplingABC(DatasetNegativeSamplingABC[CbowNegativeSample]):
    pass
