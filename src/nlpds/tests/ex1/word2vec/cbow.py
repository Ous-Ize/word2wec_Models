import unittest

import torch


class TestCBOW(unittest.TestCase):

    def setUp(self):
        from nlpds.submission.ex1.word2vec.extra.cbow import CbowSoftMax

        self.vocab_size = 25
        self.emb_dim = 10
        self.model = CbowSoftMax.new(
            torch.ones((self.vocab_size, self.emb_dim), requires_grad=True),
            2 * torch.ones((self.vocab_size, self.emb_dim), requires_grad=True),
        )

    def test_embed_input_1d(self):
        emb_input = self.model.embed_input(torch.tensor([0, 1, 2, 3]))
        self.assertEqual((1, self.emb_dim), emb_input.shape)

    def test_embed_input_2d_1_4(self):
        emb_input = self.model.embed_input(torch.tensor([[0, 1, 2, 3]]))
        self.assertEqual((1, self.emb_dim), emb_input.shape)

    def test_embed_input_2d_2_2(self):
        emb_input = self.model.embed_input(torch.tensor([[0, 1], [2, 3]]))
        self.assertEqual((2, self.emb_dim), emb_input.shape)

    def test_embed_input_1d_offsets(self):
        emb_input = self.model.embed_input(
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 2]),
        )
        self.assertEqual((2, self.emb_dim), emb_input.shape)

    def test_embed_output_1d_1(self):
        emb_output = self.model.embed_output(torch.tensor([0]))
        self.assertEqual((1, self.emb_dim), emb_output.shape)

    def test_embed_output_2d_2_1(self):
        emb_output = self.model.embed_output(torch.tensor([[0], [1]]))
        self.assertEqual((2, 1, self.emb_dim), emb_output.shape)

    def test_calculate_objective(self):
        expected = -3.2189
        actual = self.model.forward(
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([0]),
        ).item()
        self.assertAlmostEqual(expected, actual, places=4)
