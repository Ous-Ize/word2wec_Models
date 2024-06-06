import string
from typing import Self, NamedTuple, Sequence
from collections import Counter
import re
import numpy as np

from src.nlpds.abc.ex1.word2vec.data.tokenizer import PreTokenizerABC, TokenizerABC, TokenizedSentence


class PreTokenizer(PreTokenizerABC):
    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase

    @classmethod
    def with_case(cls, lowercase: bool) -> Self:
        return cls(lowercase)

    @property
    def lowercase(self) -> bool:
        return self._lowercase

    def pre_tokenize(self, sentence: str) -> list[str]:
        """Pre-tokenize the given sentence."""
        converted_sentence = sentence

        if self.lowercase:
            converted_sentence = converted_sentence.lower()

        return re.sub(r'([{}])'.format(re.escape(string.punctuation)), r' \1 ', converted_sentence).strip().split()



    @lowercase.setter
    def lowercase(self, value):
        self._lowercase = value


class Tokenizer(TokenizerABC):
    def __init__(self, vocabulary: dict[str, int], pre_tokenizer: PreTokenizerABC):
        self.vocabulary = vocabulary
        self.pre_tokenizer = pre_tokenizer

    class Fitter(TokenizerABC.Fitter['Tokenizer']):
        def __init__(self):
            self.lowercase = False
            self.max_size = -1

        def with_lowercase(self, value: bool) -> Self:
            """
            Set the lowercase flag for this fitter.
            Default: False.
            """
            self.lowercase = value
            return self

        def with_limit(self, max_size: int) -> Self:
            """
            Set the maximum vocabulary size for this fitter.
            Default: unlimited.
            """
            self.max_size = max_size
            return self

        def fit(self, corpus: list[str]) -> 'Tokenizer':
            """Fit a tokenizer to the given corpus.

            Args:
                corpus (list[str]): The corpus to fit the tokenizer to.

            Returns:
                Parent (Generic[TokenizerABC]): The fitted tokenizer.
            """
            pre_tokenizer = PreTokenizer().with_case(self.lowercase)
            token_counts = Counter()
            vocabulary = dict()
            id = 0
            for sentence in corpus:
                tokens = pre_tokenizer.pre_tokenize(sentence)
                '''for token in tokens:
                    if vocabulary.get(token) is None:
                        vocabulary.update({token: id})
                        id += 1
                    if len(vocabulary) >= self.max_size:
                        break'''
                token_counts.update(tokens)
                if self.max_size == -1:
                    most_common_tokens = token_counts.most_common()
                else:
                    most_common_tokens = token_counts.most_common(self.max_size)

                vocabulary = {token: idx for idx, (token, _) in enumerate(most_common_tokens)}

            return Tokenizer(vocabulary, pre_tokenizer)

    @classmethod
    def fitter(cls) -> Fitter:
        """Create a new Fitter object for this tokenizer.

        Returns:
            Fitter[Self]: The Fitter object for this tokenizer.
        """
        return cls.Fitter()

    @classmethod
    def new(
        cls,
        vocabulary: dict[str, int],
        pre_tokenizer: PreTokenizerABC,
    ) -> Self:
        """Construct a new tokenizer object from the given vocabulary and pre-tokenizer."""
        return cls(vocabulary, pre_tokenizer)

    @property
    def vocabulary(self) -> dict[str, int]:
        """Get the vocabulary of this tokenizer."""
        return self._vocabulary

    @property
    def pre_tokenizer(self) -> PreTokenizerABC:
        """Get the pre-tokenizer of this tokenizer."""
        return self._pre_tokenizer

    def tokenize(self, sentence: str) -> TokenizedSentence:
        """
        Tokenize the given sentence.
        See _tokens_to_input_ids() method!
        """
        tokens = self.pre_tokenizer.pre_tokenize(sentence)
        ids = self._tokens_to_input_ids(tokens)
        return TokenizedSentence.new(ids)

    @vocabulary.setter
    def vocabulary(self, value):
        self._vocabulary = value

    @pre_tokenizer.setter
    def pre_tokenizer(self, value):
        self._pre_tokenizer = value


