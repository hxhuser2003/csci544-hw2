"""Additional checks for assignment boundary cases, separate from submission."""
import copy
import math
import unittest
import torch
from torch import nn
import hw2 as hw


class Contracts(unittest.TestCase):
    def test_mlm_special_tokens_boundaries_and_no_input_alias(self):
        ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        select = torch.tensor([[0., 0., 0., .14, .14, .14, .15]])
        replace = torch.tensor([[0., 0., 0., .79, .8, .9, 0.]])
        random_ids = torch.full_like(ids, 7)
        snapshots = [tensor.clone() for tensor in (ids, select, replace, random_ids)]
        corrupted, targets = hw.corrupt_for_mlm(ids, select, replace, random_ids)
        self.assertEqual(corrupted.tolist(), [[0, 1, 2, 2, 7, 5, 6]])
        self.assertEqual(targets.tolist(), [[-100, -100, -100, 3, 4, 5, -100]])
        for tensor, snapshot in zip((ids, select, replace, random_ids), snapshots):
            torch.testing.assert_close(tensor, snapshot)
            self.assertNotEqual(tensor.data_ptr(), corrupted.data_ptr())
            self.assertNotEqual(tensor.data_ptr(), targets.data_ptr())
        self.assertNotEqual(corrupted.data_ptr(), targets.data_ptr())

    def test_transfer_preserves_template_and_enables_training(self):
        source = hw.MiniEncoder(7, d_model=3, ff_dim=5)
        template = hw.TransformerClassifier(hw.MiniEncoder(7, d_model=3, ff_dim=5))
        for parameter in template.parameters():
            parameter.requires_grad_(False)
        before = copy.deepcopy(template.state_dict())
        result = hw.transfer_encoder(source, template)
        self.assertTrue(all(parameter.requires_grad for parameter in result.parameters()))
        for name, parameter in result.named_parameters():
            self.assertNotEqual(parameter.data_ptr(), dict(template.named_parameters())[name].data_ptr())
        with torch.no_grad():
            result.encoder.embedding.weight.add_(5)
            result.output.bias.add_(5)
        for key, value in before.items():
            torch.testing.assert_close(template.state_dict()[key], value)
        self.assertFalse(torch.equal(result.encoder.embedding.weight, source.embedding.weight))

    def test_vocab_empty_and_input_preservation(self):
        tokens = [[], ["z", "a", "z", "a", "rare"]]
        before = copy.deepcopy(tokens)
        self.assertEqual(hw.build_vocab(tokens, 1),
                         {"<pad>": 0, "<unk>": 1, "<mask>": 2, "a": 3})
        self.assertEqual(hw.build_vocab([[]]),
                         {"<pad>": 0, "<unk>": 1, "<mask>": 2})
        self.assertEqual(tokens, before)

    def test_ffnn_hand_computed_logits_and_mean_gradient(self):
        model = hw.MeanFFNN(5, d_model=2, hidden_dim=2).double()
        with torch.no_grad():
            model.embedding.weight.copy_(torch.tensor([[0., 0.], [0., 0.],
                                                       [0., 0.], [2., -4.], [4., 2.]]))
            model.hidden.weight.copy_(torch.eye(2))
            model.hidden.bias.zero_()
            model.output.weight.copy_(torch.eye(2))
            model.output.bias.zero_()
        logits = model(torch.tensor([[3, 4, 0]]),
                       torch.tensor([[True, True, False]]), torch.tensor([2]))
        torch.testing.assert_close(logits, torch.tensor([[3., 0.]], dtype=torch.float64))
        hidden = torch.tensor([[[2., 4.], [4., 8.], [99., 99.]]],
                              dtype=torch.float64, requires_grad=True)
        hw.masked_mean(hidden, torch.tensor([[True, True, False]])).sum().backward()
        torch.testing.assert_close(hidden.grad,
            torch.tensor([[[.5, .5], [.5, .5], [0., 0.]]], dtype=torch.float64))

    def test_lstm_uses_last_real_token(self):
        torch.manual_seed(8)
        model = hw.LSTMClassifier(9, d_model=3, hidden_dim=5).double()
        ids = torch.tensor([[3, 4, 5], [6, 0, 0]])
        mask, lengths = ids.ne(0), torch.tensor([3, 1])
        states, _ = model.lstm(model.embedding(ids))
        expected = model.output(torch.stack([states[0, 2], states[1, 0]]))
        torch.testing.assert_close(model(ids, mask, lengths), expected)
        torch.testing.assert_close(model(ids, mask, lengths), expected)

    def test_encoder_padding_and_classifier_pooling(self):
        torch.manual_seed(9)
        encoder = hw.MiniEncoder(9, d_model=4, ff_dim=7, max_length=5).double()
        ids = torch.tensor([[3, 4, 0, 0]])
        hidden = encoder(ids, ids.ne(0))
        short = encoder(ids[:, :2], ids[:, :2].ne(0))
        torch.testing.assert_close(hidden[:, :2], short)
        self.assertEqual(hidden[:, 2:].abs().sum().item(), 0.)
        model = hw.TransformerClassifier(encoder).double()
        expected = model.output(short.mean(dim=1))
        torch.testing.assert_close(model(ids, ids.ne(0), torch.tensor([2])), expected)
        hidden.sum().backward()
        self.assertIsNotNone(encoder.embedding.weight.grad)

    def test_attention_scaling_and_gradients(self):
        q = torch.tensor([[[1., 1.], [0., 0.]]], dtype=torch.float64, requires_grad=True)
        k = torch.tensor([[[1., 1.], [0., 0.]]], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([[[2., 0., 1.], [0., 4., 3.]]],
                         dtype=torch.float64, requires_grad=True)
        context, weights = hw.scaled_attention(q, k, v, torch.ones(1, 2, dtype=torch.bool))
        p = math.exp(math.sqrt(2)) / (math.exp(math.sqrt(2)) + 1)
        torch.testing.assert_close(weights,
            torch.tensor([[[p, 1-p], [.5, .5]]], dtype=torch.float64))
        self.assertEqual(context.shape, (1, 2, 3))
        context.sum().backward()
        for tensor in (q, k, v):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_evaluation_global_metrics_and_state_preservation(self):
        # Fixed logits from real embedding weights allow hand-checked metrics.
        class Lookup(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Embedding(6, 2)
                with torch.no_grad():
                    self.logits.weight.zero_()
                    self.logits.weight[3] = torch.tensor([0., math.log(3)])
                    self.logits.weight[4] = torch.tensor([math.log(3), 0.])
            def forward(self, ids, mask, lengths):
                return self.logits(ids[:, 0])
        model = Lookup().double()
        batches = [hw.collate_batch([([3], 1), ([4], 1)]),
                   hw.collate_batch([([5], 0)])]
        model.logits.weight.grad = torch.ones_like(model.logits.weight)
        before = copy.deepcopy(model.state_dict())
        gradients = model.logits.weight.grad.clone()
        for training in (True, False):
            model.train(training)
            metrics = hw.evaluate_model(model, batches)
            self.assertEqual(model.training, training)
            self.assertAlmostEqual(metrics['loss'],
                (math.log(4/3) + math.log(4) + math.log(2))/3)
            for key, value in {'accuracy': 2/3, 'precision': 1., 'recall': .5, 'f1': 2/3}.items():
                self.assertAlmostEqual(metrics[key], value)
            torch.testing.assert_close(model.logits.weight.grad, gradients)
            for key, value in before.items():
                torch.testing.assert_close(model.state_dict()[key], value)

    def test_training_weighted_loss_and_parameter_update(self):
        torch.manual_seed(10)
        model = hw.MeanFFNN(6, d_model=3, hidden_dim=4).double()
        batches = [hw.collate_batch([([3], 1), ([4], 0)]),
                   hw.collate_batch([([5], 1)])]
        model.eval()
        with torch.no_grad():
            expected = sum(nn.functional.cross_entropy(model(*b[:3]), b[3]).item()*len(b[3])
                           for b in batches) / 3
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        self.assertAlmostEqual(hw.train_one_epoch(model, batches, optimizer), expected)
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        hw.train_one_epoch(model, batches, optimizer)
        self.assertTrue(model.training)
        self.assertTrue(any(not torch.equal(value, model.state_dict()[key])
                            for key, value in before.items()))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
