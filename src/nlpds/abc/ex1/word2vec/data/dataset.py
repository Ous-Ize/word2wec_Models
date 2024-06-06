from abc import abstractmethod
from typing import NamedTuple, Self, Sequence

from numpy import ndarray as NDArray
from torch import Tensor

from src.nlpds.abc.ex1.word2vec.data.tokenizer import TokenizedSentence, TokenizerABC


class Word2VecSample(NamedTuple):
    """
    A single training sample of one target and context word.
    """

    target: Tensor  # shape: (1,)
    context: Tensor  # shape: (1,)


class DatasetABC[Item: Word2VecSample](Sequence[Item]):
    """
    The dataset object should store training samples.
    """

    @classmethod
    @abstractmethod
    def new(
        cls,
        sentences: list[str],
        tokenizer: TokenizerABC,
        window_size: int,
        threshold: float,
    ) -> Self:
        """Create a new dataset object by loading the given file, tokenizing it with the given tokenizer, and creating training samples for the given window size.
        Use the given threshold to calculate the sub-sampling probabilities from the token frequencies in the input corpus.
        You may expect that the input corpus is compatible with the given tokenizer.

        Args:
            sentences (list[str]): The list of sentences to create the dataset from.
            tokenizer (TokenizerABC): The tokenizer to use.
            window_size (int): The window size to the left and right of the target word.
            threshold (float): The sub-sampling threshold.

        Returns:
            Self: The dataset.
        """
        raise NotImplementedError(f"{cls.__name__}.new() is not implemented")

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the size of this dataset, i.e. the number of training samples.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.__len__() is not implemented"
        )

    @abstractmethod
    def __getitem__(self, index):
        """
        Get a single training sample.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.__iter__() is not implemented"
        )

    @property
    @abstractmethod
    def token_counts(self) -> NDArray:
        """
        Get the raw counts of each token from the given input corpus.
        Should return an array of counts, where each entry (row) corresponds to the token id.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.token_counts property getter is not implemented"
        )

    @property
    @abstractmethod
    def token_frequencies(self) -> NDArray:
        """
        Get the frequency of each token in the given input corpus.
        Should return an array of floats, where each entry (row) corresponds to the token id.
        All values should be between 0.0 and 1.0, and sum up to 1.0.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.token_frequencies property getter is not implemented"
        )

    @property
    @abstractmethod
    def window_size(self) -> int:
        """
        Get the window size.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.window_size property getter is not implemented"
        )

    @property
    @abstractmethod
    def sub_sampling_probs(self) -> NDArray:
        """
        Get the probabilities for *keeping* the tokens.
        Should return an array of floats, where each entry (row) corresponds to the token id.
        All values should be between 0.0 and 1.0.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.sub_sampling_probs property getter is not implemented"
        )

    @staticmethod
    @abstractmethod
    def _sentence_to_samples(
        sentence: TokenizedSentence,
        window_size: int,
        sub_sampling_probs: NDArray,
    ) -> list[Item]:
        """
        Convert a tokenized sentence into a list of training samples.
        """
        raise NotImplementedError("_sentence_to_samples() is not implemented")

    def sentence_to_samples(
        self,
        sentence: TokenizedSentence,
    ) -> list[Item]:
        """
        Convert a tokenized sentence into a list of training samples.
        """
        return self._sentence_to_samples(
            sentence,
            self.window_size,
            self.sub_sampling_probs,
        )
