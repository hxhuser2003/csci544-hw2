"""Run: python -m unittest -v test_public (no real dataset is needed)."""
import copy
import importlib
import math
import os
import unittest
import torch
from torch import nn

hw = importlib.import_module(os.environ.get("HW2_MODULE", "hw2"))


class PublicTests(unittest.TestCase):
    def test_vocab(self):
        tokens = [["pen", "good", "pen"], ["bad", "good", "bad"], ["rare"]]
        self.assertEqual(hw.build_vocab(tokens, max_words=2, min_freq=2),
                         {"<pad>": 0, "<unk>": 1, "<mask>": 2, "bad": 3, "good": 4})

    def test_encoding(self):
        v = {"<pad>": 0, "<unk>": 1, "<mask>": 2, "good": 3, "pen": 4}
        self.assertEqual(hw.encode_reviews([["good", "new", "pen"], []], v, 2),
                         [[3, 1], [1]])

    def test_collation(self):
        ids, mask, lengths, labels = hw.collate_batch([([3, 4], 1), ([1], 0)])
        self.assertEqual(ids.tolist(), [[3, 4], [1, 0]])
        self.assertEqual(mask.tolist(), [[True, True], [True, False]])
        self.assertEqual(lengths.tolist(), [2, 1])
        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(mask.dtype, torch.bool)

    def test_masked_mean(self):
        x = torch.tensor([[[2., 4.], [4., 8.], [100., 100.]]])
        mask = torch.tensor([[True, True, False]])
        torch.testing.assert_close(hw.masked_mean(x, mask), torch.tensor([[3., 6.]]))

    def test_attention_padding(self):
        q = k = torch.zeros(1, 3, 2)
        v = torch.tensor([[[2., 0.], [0., 4.], [90., 90.]]])
        out, weights = hw.scaled_attention(q, k, v, torch.tensor([[True, True, False]]))
        torch.testing.assert_close(out, torch.tensor([[[1., 2.], [1., 2.], [0., 0.]]]))
        torch.testing.assert_close(weights, torch.tensor([[[.5, .5, 0.], [.5, .5, 0.], [0., 0., 0.]]]))

    def test_attention_causal(self):
        q = k = torch.zeros(1, 3, 2)
        v = torch.tensor([[[2., 0.], [0., 4.], [90., 90.]]])
        out, _ = hw.scaled_attention(q, k, v, torch.tensor([[True, True, False]]), causal=True)
        torch.testing.assert_close(out, torch.tensor([[[2., 0.], [1., 2.], [0., 0.]]]))

    def test_classifier_shapes(self):
        ids, mask, lengths, _ = hw.collate_batch([([3, 4], 1), ([1], 0)])
        models = [hw.MeanFFNN(6), hw.LSTMClassifier(6),
                  hw.TransformerClassifier(hw.MiniEncoder(6))]
        for model in models:
            logits = model(ids, mask, lengths)
            self.assertEqual(logits.shape, (2, 2))
            self.assertTrue(torch.isfinite(logits).all())

    def test_corruption_branches(self):
        ids = torch.tensor([[3, 4, 5, 6, 1, 0]])
        select = torch.tensor([[.10, .10, .10, .15, .00, .00]])
        replace = torch.tensor([[.10, .80, .90, .10, .10, .10]])
        random_ids = torch.full_like(ids, 7)
        original = ids.clone()
        corrupted, targets = hw.corrupt_for_mlm(ids, select, replace, random_ids)
        self.assertEqual(corrupted.tolist(), [[2, 7, 5, 6, 1, 0]])
        self.assertEqual(targets.tolist(), [[3, 4, 5, -100, -100, -100]])
        self.assertTrue(torch.equal(ids, original))

    def test_uniform_mlm_loss(self):
        logits = torch.zeros(1, 3, 8, requires_grad=True)
        targets = torch.tensor([[3, -100, 5]])
        loss = hw.masked_token_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), math.log(8), places=6)
        loss.backward()
        self.assertEqual(logits.grad[0, 1].abs().sum().item(), 0)

    def test_all_ignored_mlm(self):
        logits = torch.zeros(2, 3, 8, requires_grad=True)
        loss = hw.masked_token_loss(logits, torch.full((2, 3), -100))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0.0)

    def test_transfer_does_not_alias(self):
        source = hw.MiniEncoder(8)
        template = hw.TransformerClassifier(hw.MiniEncoder(8))
        head = copy.deepcopy(template.output.state_dict())
        result = hw.transfer_encoder(source, template)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(result.encoder.state_dict()[key], value)
        for key, value in head.items():
            torch.testing.assert_close(result.output.state_dict()[key], value)
        self.assertNotEqual(result.encoder.embedding.weight.data_ptr(), source.embedding.weight.data_ptr())
        self.assertTrue(all(p.requires_grad for p in result.parameters()))

    def test_training_and_evaluation_finite(self):
        from hw2_support import make_loader, make_optimizer
        loader = make_loader([([3, 4], 1), ([4], 0), ([3], 1)], hw.collate_batch, batch_size=2)
        model = hw.MeanFFNN(5)
        loss = hw.train_one_epoch(model, loader, make_optimizer(model.parameters()))
        scores = hw.evaluate_model(model, loader)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(set(scores), {"loss", "accuracy", "precision", "recall", "f1"})
        self.assertTrue(all(math.isfinite(x) for x in scores.values()))


if __name__ == "__main__":
    unittest.main()
