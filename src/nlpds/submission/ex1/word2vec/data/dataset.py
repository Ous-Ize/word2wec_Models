from random import random

import numpy as np
import torch

from src.nlpds.abc.ex1.word2vec.data.dataset import DatasetABC, Word2VecSample, TokenizedSentence, TokenizerABC
from src.nlpds.submission.ex1.word2vec.data.tokenizer import Tokenizer
from typing import Self, List
from numpy import ndarray as NDArray


class Dataset(DatasetABC[Word2VecSample]):

    def __init__(self, samples: List[Word2VecSample],
                 token_counts: NDArray,
                 token_frequencies: NDArray,
                 sub_sampling_probs: NDArray,
                 window_size: int
                 ):
        """
           The dataset object should store training samples.
        """

        self.samples = samples
        self.token_counts = token_counts
        self.token_frequencies = token_frequencies
        self.sub_sampling_probs = sub_sampling_probs
        self.window_size = window_size

    @classmethod
    def new(
            cls,
            sentences: list[str],
            tokenizer: TokenizerABC,
            window_size: int,
            threshold: float = 0.001,
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
        fitter = Tokenizer.fitter().with_lowercase(True).with_limit(10000)
        tokenizer_fitted = fitter.fit(sentences)
        tokenized_sentences = []
        token_counts = np.zeros(max(tokenizer_fitted.vocabulary.values()) + 1, dtype=int)
        for sentence in sentences:
            tokenized_sentence = tokenizer_fitted.tokenize(sentence).input_ids
            tokenized_sentences.append(tokenized_sentence)
            token_counts[tokenized_sentence] += 1

        total_token_count = token_counts.sum()

        token_frequencies = token_counts / total_token_count

        sub_sampling_probs = (np.sqrt(token_frequencies / threshold) + 1) * threshold / token_frequencies

        samples = [sample for tokenized_sentence in tokenized_sentences
                   for sample in cls._sentence_to_samples(tokenized_sentence, window_size, sub_sampling_probs)]

        return cls(samples, token_counts, token_frequencies, sub_sampling_probs, window_size)

    def __len__(self) -> int:
        """
        Get the size of this dataset, i.e. the number of training samples.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Get a single training sample.
        """
        return self.samples[index]

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

    @staticmethod
    def _sentence_to_samples(
            sentence: TokenizedSentence,
            window_size: int,
            sub_sampling_probs: NDArray,
    ) -> list['Word2VecSample']:
        """
        Convert a tokenized sentence into a list of training samples.
        """
        samples = list()
        sampled_sentences = list()
        for tokenID in sentence.input_ids:
            if random() < sub_sampling_probs[tokenID]:
                token_vector = torch.tensor(tokenID)
                sampled_sentences.append(token_vector)

        for index in range(len(sampled_sentences)):
            start = max(0, index - window_size)
            end = min(index + window_size, len(sampled_sentences)-1)
            for context in range(start, end + 1):
                if context != index:
                    samples.append(Word2VecSample(sampled_sentences[index], sampled_sentences[context]))

        return samples

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
