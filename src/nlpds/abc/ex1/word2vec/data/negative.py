from abc import ABC, abstractmethod
from typing import NamedTuple, Self

import torch
from numpy import ndarray as NDArray

from src.nlpds.abc.ex1.word2vec.data.dataset import DatasetABC
from src.nlpds.abc.ex1.word2vec.data.tokenizer import TokenizedSentence, TokenizerABC


class NegativeSamplerABC(ABC):
    """
    Negative Sampler Abstract Base Class.
    Calculate the 'unigram distribution' from the token frequencies in your constructor.
    """

    @classmethod
    @abstractmethod
    def new(
        cls,
        token_frequencies: NDArray,
        samples_to_draw: int,
        power: float = 0.75,
    ) -> Self:
        """
        Create a negative sampler instance parameterized by the 'unigram distribution' from token frequencies.

        Args:
            token_frequencies (NDArray): The token frequencies.
            samples_to_draw (int): The number of samples to draw.
            power (float, optional): The power for the negative sampling distribution. Defaults to 0.75.

        Returns:
            Self: The sampler.
        """
        raise NotImplementedError(f"{cls.__name__}.new() is not implemented")

    @property
    @abstractmethod
    def distribution(self) -> NDArray:
        """Get the unigram distribution."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.distribution getter is not implemented"
        )

    @property
    @abstractmethod
    def samples_to_draw(self) -> int:
        """Get the number of samples to draw."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.samples_to_draw getter is not implemented"
        )

    @staticmethod
    @abstractmethod
    def _draw(
        distribution: NDArray, samples_to_draw: int, exclude: set[int]
    ) -> list[int]:
        """Actual implementation of the .draw() method."""
        raise NotImplementedError("_draw() is not implemented")

    def draw(self, exclude: set[int]) -> list[int]:
        """Draw k negative samples from the distribution while excluding the given set of words."""
        return self._draw(self.distribution, self.samples_to_draw, exclude)


class NegativeSample(NamedTuple):
    """
    A single training sample of one target and context word, and $k$ negative samples.
    """

    target: torch.Tensor  # shape: (1,)
    context: torch.Tensor  # shape: (1,)
    negative: torch.Tensor  # shape: (k,)


class DatasetNegativeSamplingABC[Item: NegativeSample](DatasetABC[Item]):
    """
    Negative Sampling variant of the Dataset Abstract Base Class.
    """

    @classmethod
    @abstractmethod
    def new(
        cls,
        sentences: list[str],
        tokenizer: TokenizerABC,
        window_size: int,
        threshold: float,
        sampler: NegativeSamplerABC,
    ) -> Self:
        """Create a new dataset object by loading the given file, tokenizing it with the given tokenizer, and creating context windows for the given window size.
        Use the given threshold to calculate the sub-sampling probabilities from the token frequencies in the input corpus.
        You may expect that the input corpus is compatible with the given tokenizer.

        Args:
            sentences (list[str]): The list of sentences to create the dataset from.
            tokenizer (TokenizerABC): The tokenizer to use.
            window_size (int): The window size to the left and right of the target word.
            threshold (float): The sub-sampling threshold.
            sampler (NegativeSamplerABC): The sampler for the negative samples.

        Returns:
            Self: The dataset.
        """
        raise NotImplementedError(f"{cls.__name__}.new() is not implemented")

    @property
    @abstractmethod
    def sampler(self) -> NegativeSamplerABC:
        """Get the negative sampler."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.sampler getter is not implemented"
        )

    @staticmethod
    @abstractmethod
    def _sentence_to_samples(
        sentence: TokenizedSentence,
        window_size: int,
        sub_sampling_probs: NDArray,
        sampler: NegativeSamplerABC,
    ) -> list[Item]:
        """
        Convert a tokenized sentence into a list of training samples.
        """
        raise NotImplementedError("_sentence_to_samples() is not implemented")
