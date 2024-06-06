import unittest
from typing import Final

import numpy as np

tiny_corpus: Final[list[str]] = [
    "The quick brown fox jumps over the lazy-dog.",
    "A quick brown dog outpaces a quick fox.",
    "Most dogs are brown.",
    "Most foxes are quick.",
    "Some lazy dogs have brown fur.",
    "Some quick foxes have lazy fur.",
]


class DummyPreTokenizer:
    def pre_tokenize(self, sentence: str) -> list[str]:
        return sentence.split()

    def __call__(self, sentence: str) -> list[str]:
        return sentence.split()


class TestTokenizer(unittest.TestCase):
    def test_tokenize(self):
        from src.nlpds.submission.ex1.word2vec.data.tokenizer import Tokenizer

        tokenizer = Tokenizer.new({"a": 0, "b": 1, "c": 2}, DummyPreTokenizer())  # type: ignore
        self.assertEqual(list(tokenizer.tokenize("a b c").input_ids), [0, 1, 2])

    def test_fit(self):
        from src.nlpds.submission.ex1.word2vec.data.tokenizer import Tokenizer

        abcd = ["a a a b", "c c c d d"]

        tokenizer = Tokenizer.fitter().fit(abcd)
        self.assertEqual(len(tokenizer.vocabulary), 4)

        tokenizer = Tokenizer.fitter().with_limit(-1).fit(abcd)
        self.assertEqual(len(tokenizer.vocabulary), 4)

        tokenizer = Tokenizer.fitter().with_limit(3).fit(abcd)
        self.assertEqual(len(tokenizer.vocabulary), 3)
        self.assertEqual(set(tokenizer.vocabulary.keys()), {"a", "c", "d"})

    def test_fit_tiny_corpus(self):
        from src.nlpds.submission.ex1.word2vec.data.tokenizer import Tokenizer

        tokenizer = Tokenizer.fitter().fit(tiny_corpus)
        self.assertEqual(len(tokenizer.vocabulary), 21)

    def test_fit_tiny_corpus_lower(self):
        from src.nlpds.submission.ex1.word2vec.data.tokenizer import Tokenizer

        tokenizer = Tokenizer.fitter().with_lowercase(True).fit(tiny_corpus)
        self.assertEqual(len(tokenizer.vocabulary), 19)


class DummyTokenizedSentence:
    def __init__(self, ll):
        self.input_ids = np.array(ll)


class DummyTokenizer:
    sub_sampling_probs = np.array([1.0, 1.0, 1.0])

    def tokenize(self, sentence: str):
        return DummyTokenizedSentence(list(map(int, sentence.split())))

    def __call__(self, sentence: str):
        return self.tokenize(sentence)


class TestDataset(unittest.TestCase):

    @staticmethod
    def get_dataset():
        from src.nlpds.submission.ex1.word2vec.data.dataset import Dataset

        dataset = Dataset.new(
            [
                "0 1 2",
                "0 0 0",
                "0 1 0",
            ],
            DummyTokenizer(),
            1,
            1.0,
        )

        return dataset

    def test_sentence_to_samples(self):
        from src.nlpds.submission.ex1.word2vec.data.dataset import Dataset

        window_size = 1
        sub_sampling_probs = np.ones(3)
        samples = Dataset._sentence_to_samples(
            DummyTokenizedSentence([0, 1, 2]),
            window_size,
            sub_sampling_probs,
        )
        self.assertEqual(
            {tuple(map(int, sample)) for sample in samples},
            {
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 1),
            },
        )

    def test_dataset_samples(self):
        dataset = self.get_dataset()
        sample_set = {tuple(map(int, sample)) for sample in dataset}
        self.assertEqual(
            sample_set,
            {
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 1),
                (0, 0),
            },
        )

    def test_dataloader(self):
        from torch.utils.data.dataloader import DataLoader

        dataset = self.get_dataset()
        dl = DataLoader(dataset, batch_size=8, shuffle=False)
        for batch in dl:
            x, y = batch
            self.assertEqual(tuple(x.squeeze().shape), (8,))
            self.assertEqual(tuple(y.squeeze().shape), (8,))
            break
