import unittest

import torch


class TestSkipGram(unittest.TestCase):
    def setUp(self):
        from src.nlpds.submission.ex1.word2vec.model.skipgram import SkipGramSoftMax

        self.vocab_size = 25
        self.emb_dim = 10
        one = torch.ones((self.vocab_size, self.emb_dim), requires_grad=True)
        two = torch.full_like(one, 2.0, requires_grad=True)
        self.model = SkipGramSoftMax.new(self.vocab_size, self.emb_dim, one, two)

    def test_embed_input_1d_1(self):
        emb_input = self.model.embed_input(torch.tensor([0]))
        self.assertEqual((1, self.emb_dim), emb_input.shape)

    def test_embed_input_2d_2_1(self):
        emb_input = self.model.embed_input(torch.tensor([[0], [1]]))
        self.assertEqual((2, 1, self.emb_dim), emb_input.shape)

    def test_embed_output_1d_4(self):
        emb_output = self.model.embed_output(torch.tensor([0, 1, 2, 3]))
        self.assertEqual((4, self.emb_dim), emb_output.shape)

    def test_embed_output_2d_1_4(self):
        emb_output = self.model.embed_output(torch.tensor([[0, 1, 2, 3]]))
        self.assertEqual((1, 4, self.emb_dim), emb_output.shape)

    def test_embed_output_2d_2_2(self):
        emb_output = self.model.embed_output(torch.tensor([[0, 1], [2, 3]]))
        self.assertEqual((2, 2, self.emb_dim), emb_output.shape)

    def test_calculate_objective(self):
        expected = -3.2189
        actual = self.model.forward(
            torch.tensor([0]),
            torch.tensor([1]),
        ).item()
        self.assertAlmostEqual(expected, actual, places=4)
