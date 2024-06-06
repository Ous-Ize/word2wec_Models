from abc import ABC, abstractmethod
from typing import NamedTuple, Self, Sequence

import numpy as np
from numpy import ndarray as NDArray


class TokenizedSentence(NamedTuple):
    """
    Simple container for tokenized sentences.
    Note: Gets a list of token ids but returns a numpy array!
    """

    input_ids: NDArray

    @classmethod
    @abstractmethod
    def new(cls, input_ids: Sequence[int]) -> Self:
        """Construct a new tokenized sentence object from the given input ids.

        Args:
            input_ids (Sequence[int]): The input token ids.
        """
        return cls(np.array(input_ids))


class PreTokenizerABC(ABC):
    """
    PreTokenizer Abstract Base Class.
    Implement the pre_tokenize() method to split a sentence into 'words',
    separating punctuation marks from words and splitting on whitespace.
    """

    @classmethod
    @abstractmethod
    def with_case(cls, lowercase: bool) -> Self:
        raise NotImplementedError(f"{cls.__name__}.with_case() is not implemented")

    @property
    @abstractmethod
    def lowercase(self) -> bool:
        raise NotImplementedError(
            f"{self.__class__.__name__}.lowercase property getter is not implemented"
        )

    @abstractmethod
    def pre_tokenize(self, sentence: str) -> list[str]:
        """Pre-tokenize the given sentence."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.pre_tokenize() is not implemented"
        )

    def __call__(self, sentence: str) -> list[str]:
        return self.pre_tokenize(sentence)


class TokenizerABC(ABC):
    """
    Tokenizer Abstract Base Class.
    The Fitter inner class is a builder-style object for fitting a tokenizer to a corpus.
    """

    class Fitter[Parent: TokenizerABC](ABC):
        """A builder-style object for fitting a tokenizer to a corpus."""

        @abstractmethod
        def with_lowercase(self, value: bool) -> Self:
            """
            Set the lowercase flag for this fitter.
            Default: False.
            """
            raise NotImplementedError(
                f"{self.__class__.__name__}.with_lowercase() is not implemented"
            )

        @abstractmethod
        def with_limit(self, max_size: int) -> Self:
            """
            Set the maximum vocabulary size for this fitter.
            Default: unlimited.
            """
            raise NotImplementedError(
                f"{self.__class__.__name__}.with_limit() is not implemented"
            )

        @abstractmethod
        def fit(self, corpus: list[str]) -> Parent:
            """Fit a tokenizer to the given corpus.

            Args:
                corpus (list[str]): The corpus to fit the tokenizer to.

            Returns:
                Parent (Generic[TokenizerABC]): The fitted tokenizer.
            """
            raise NotImplementedError(
                f"{self.__class__.__name__}.fit() is not implemented"
            )

    @classmethod
    @abstractmethod
    def fitter(cls) -> Fitter[Self]:
        """Create a new Fitter object for this tokenizer.

        Returns:
            Fitter[Self]: The Fitter object for this tokenizer.
        """
        raise NotImplementedError(f"{cls.__name__}.fitter() is not implemented")

    @classmethod
    def builder(cls) -> Fitter[Self]:
        """Alias for the cls.fitter() method."""
        return cls.fitter()

    @classmethod
    @abstractmethod
    def new(
        cls,
        vocabulary: dict[str, int],
        pre_tokenizer: PreTokenizerABC,
    ) -> Self:
        """Construct a new tokenizer object from the given vocabulary and pre-tokenizer."""
        raise NotImplementedError(f"{cls.__name__}.new() is not implemented")

    @property
    @abstractmethod
    def vocabulary(self) -> dict[str, int]:
        """Get the vocabulary of this tokenizer."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.vocabulary property getter is not implemented"
        )

    @property
    @abstractmethod
    def pre_tokenizer(self) -> PreTokenizerABC:
        """Get the pre-tokenizer of this tokenizer."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.pre_tokenizer property getter is not implemented"
        )

    @abstractmethod
    def tokenize(self, sentence: str) -> TokenizedSentence:
        """
        Tokenize the given sentence.
        See _tokens_to_input_ids() method!
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.tokenize() is not implemented"
        )

    def _tokens_to_input_ids(self, tokens: list[str]) -> list[int]:
        """
        Convert a list of token strings to their corresponding input ids.
        All tokens not present in the vocabulary will be removed.

        Args:
            tokens (list[str]): The tokens to convert to their respective ids.

        Returns:
            list[int]: A list of token ids.
        """
        return [self.vocabulary[token] for token in tokens if token in self.vocabulary]

    def __call__(self, sentence: str) -> TokenizedSentence:
        return self.tokenize(sentence)
