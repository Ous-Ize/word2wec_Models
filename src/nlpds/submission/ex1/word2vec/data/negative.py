from src.nlpds.abc.ex1.word2vec.data.negative import (
    DatasetNegativeSamplingABC,
    NegativeSample,
    NegativeSamplerABC,
)
from src.nlpds.abc.ex1.word2vec.data.dataset import TokenizedSentence, TokenizerABC, Word2VecSample
from src.nlpds.submission.ex1.word2vec.data.dataset import Tokenizer
from typing import Self, List
from numpy import ndarray as NDArray
from random import random
import torch
import numpy as np


class NegativeSampler(NegativeSamplerABC):
    """
    Negative Sampler Abstract Base Class.
    Calculate the 'unigram distribution' from the token frequencies in your constructor.
    """
    def __init__(
        self,
        distribution: NDArray,
        samples_to_draw: int,
    ) -> None:
        self.distribution = distribution
        self.samples_to_draw = samples_to_draw

    @classmethod
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
        distribution = (token_frequencies ** power) / np.sum(token_frequencies ** power)

        return cls(distribution, samples_to_draw)

    @property
    def distribution(self) -> NDArray:
        """Get the unigram distribution."""
        return self.distribution

    @property
    def samples_to_draw(self) -> int:
        """Get the number of samples to draw."""
        return self.samples_to_draw

    @staticmethod
    def _draw(
        distribution: NDArray, samples_to_draw: int, exclude: set[int]
    ) -> list[int]:
        """Actual implementation of the .draw() method."""
        updated_distribution = distribution.copy()
        for id in exclude:
            updated_distribution[id] = 0
        negative_samples = np.random.choice(len(updated_distribution), samples_to_draw, p=updated_distribution, replace=False)
        return list(negative_samples)

    @distribution.setter
    def distribution(self, value):
        self._distribution = value

    @samples_to_draw.setter
    def samples_to_draw(self, value):
        self._samples_to_draw = value


class DatasetNegativeSampling(DatasetNegativeSamplingABC[NegativeSample]):
    def __init__(
            self, samples: List[NegativeSample],
            token_counts: NDArray,
            token_frequencies: NDArray,
            sub_sampling_probs: NDArray,
            window_size: int,
            sampler: NegativeSamplerABC) -> None:
        """
        Negative Sampling variant of the Dataset Abstract Base Class.
        """
        super().__init__()
        self.samples = samples
        self.token_counts = token_counts
        self.token_frequencies = token_frequencies
        self.sub_sampling_probs = sub_sampling_probs
        self.window_size = window_size
        self.sampler = sampler

    @classmethod
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
        fitter = Tokenizer.fitter().with_lowercase(True).with_limit(10000)
        tokenizer_fitted = fitter.fit(sentences)
        tokenized_sentences = []
        token_counts = np.zeros(max(tokenizer_fitted.vocabulary.values()) + 1, dtype=int)
        for sentence in sentences:
            tokenized_sentence = tokenizer_fitted.tokenize(sentence)
            tokenized_sentences.append(tokenized_sentence)
            token_counts[tokenized_sentence] += 1

        total_token_count = token_counts.sum()

        token_frequencies = token_counts / total_token_count

        sub_sampling_probs = (np.sqrt(token_frequencies / threshold) + 1) * threshold / token_frequencies

        samples = [sample for tokenized_sentence in tokenized_sentences
                   for sample in
                   cls._sentence_to_samples(tokenized_sentence, window_size,
                                            sub_sampling_probs, sampler)]

        return cls(samples, token_counts, token_frequencies, sub_sampling_probs, window_size, sampler)

    def __len__(self) -> int:
        """
        Get the size of this dataset, i.e. the number of training samples.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Get a single training sample.
        """
        input_id, context_ids, negativeids = self.samples[index]
        return input_id, context_ids, negativeids

    @property
    def token_counts(self) -> NDArray:
        """
        Get the raw counts of each token from the given input corpus.
        Should return an array of counts, where each entry (row) corresponds to the token id.
        """
        return self.token_counts

    @property
    def token_frequencies(self) -> NDArray:
        """
        Get the frequency of each token in the given input corpus.
        Should return an array of floats, where each entry (row) corresponds to the token id.
        All values should be between 0.0 and 1.0, and sum up to 1.0.
        """
        return self.token_frequencies

    @property
    def window_size(self) -> int:
        """
        Get the window size.
        """
        return self.window_size

    @property
    def sub_sampling_probs(self) -> NDArray:
        """
        Get the probabilities for *keeping* the tokens.
        Should return an array of floats, where each entry (row) corresponds to the token id.
        All values should be between 0.0 and 1.0.
        """
        return self.sub_sampling_probs

    @property
    def sampler(self) -> NegativeSamplerABC:
        """Get the negative sampler."""
        return self.sampler

    @staticmethod
    def _sentence_to_samples(
        sentence: TokenizedSentence,
        window_size: int,
        sub_sampling_probs: NDArray,
        sampler: NegativeSamplerABC,
    ) -> list["NegativeSample"]:
        """
        Convert a tokenized sentence into a list of training samples.
        """
        samples = list()
        sampled_sentence = list()
        for tokenID in sentence.input_ids:
            if random() < sub_sampling_probs[tokenID]:
                token_vector = torch.tensor(tokenID)
                sampled_sentence.append(token_vector)

        list_context_ids = list()
        list_negative_ids = list()

        for index in range(len(sampled_sentence)):
            start = max(0, index - window_size)
            end = min(index + window_size, len(sampled_sentence)-1)
            for context in range(start, end + 1):
                if context != index:
                    list_context_ids.append(sampled_sentence[context])

            list_negative_ids = sampler.draw(set(sentence.input_ids))

            input_id = sampled_sentence[index]
            context_ids = torch.tensor(list_context_ids)
            negative_ids = torch.tensor(list_negative_ids).t()

            sample = NegativeSample(input_id, context_ids, negative_ids)
            samples.append(sample)

            list_context_ids.clear()
            list_negative_ids.clear()

        return samples

    @sampler.setter
    def sampler(self, value):
        self._sampler = value

    @token_counts.setter
    def token_counts(self, value):
        self._token_counts = value

    @token_frequencies.setter
    def token_frequencies(self, value):
        self._token_frequencies = value

    @sub_sampling_probs.setter
    def sub_sampling_probs(self, value):
        self._sub_sampling_probs = value

    @window_size.setter
    def window_size(self, value):
        self._window_size = value

