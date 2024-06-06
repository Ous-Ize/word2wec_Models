from abc import abstractmethod

from torch import Tensor

from src.nlpds.abc.ex1.word2vec.model.base import Word2VecBaseABC
from src.nlpds.abc.ex1.word2vec.model.negative import NegativeSamplingMixin


class SkipGramSoftMaxABC(Word2VecBaseABC):
    """
    Skip-Gram-SoftMax model Abstract Base Class.
    Implement the objective function in the .forward() method.
    """

    # Hint: use torch.nn.Embedding

    @abstractmethod
    def embed_input(
        self,
        target_ids: Tensor,
    ) -> Tensor:
        """Get the input embedding of the given target tokens."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.embed_target() is not implemented"
        )

    @abstractmethod
    def embed_output(
        self,
        context_ids: Tensor,
    ) -> Tensor:
        """Get the output embeddings of the given context tokens."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.embed_output() is not implemented"
        )

    @abstractmethod
    def forward(
        self,
        input_id: Tensor,
        context_ids: Tensor,
    ) -> Tensor:
        """
        Perform a skip-gram forward pass.
        Should return one score for each input-context-id pair.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() is not implemented"
        )


class SkipGramNegativeSamplingABC(NegativeSamplingMixin, SkipGramSoftMaxABC):
    pass
