"""CSCI 544 HW2, Fall 2026: student starter.
Implement TODOs only. Constructors and experiment driver are provided.
"""
from __future__ import annotations
import copy
import math
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from hw2_support import load_reviews, set_seed, make_loader, fit_classifier, pretrain_mlm

PAD, UNK, MASK = 0, 1, 2
IGNORE = -100


def build_vocab(train_tokens, max_words=5000, min_freq=2):
    """Count provided training tokens. The driver truncates first, to 64 tokens."""
    counts = Counter(word for tokens in train_tokens for word in tokens)
    words = sorted((word for word, count in counts.items() if count >= min_freq),
                   key=lambda word: (-counts[word], word))[:max_words]
    vocab = {"<pad>": PAD, "<unk>": UNK, "<mask>": MASK}
    vocab.update({word: index for index, word in enumerate(words, start=3)})
    return vocab


def encode_reviews(token_lists, vocab, max_length=64):
    return [[vocab.get(word, UNK) for word in tokens[:max_length]] or [UNK]
            for tokens in token_lists]


def collate_batch(examples):
    lengths = torch.tensor([len(sequence) for sequence, _ in examples], dtype=torch.long)
    ids = torch.full((len(examples), int(lengths.max().item())), PAD, dtype=torch.long)
    for row, (sequence, _) in enumerate(examples):
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    mask = ids.ne(PAD)
    labels = torch.tensor([label for _, label in examples], dtype=torch.long)
    return ids, mask, lengths, labels


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1)


class MeanFFNN(nn.Module):
    """The constructor is provided. Implement forward only."""
    def __init__(self, vocab_size, d_model=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.hidden = nn.Linear(d_model, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, ids, mask, lengths):
        embedded = self.embedding(ids)
        pooled = masked_mean(embedded, mask)
        return self.output(F.relu(self.hidden(pooled)))


class LSTMClassifier(nn.Module):
    """One unidirectional LSTM, reset to zero states on every call."""
    def __init__(self, vocab_size, d_model=32, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.lstm = nn.LSTM(d_model, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=False, dropout=0.0)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, ids, mask, lengths):
        states, _ = self.lstm(self.embedding(ids))
        # Each review ends before the right-padding suffix.
        rows = torch.arange(ids.size(0), device=ids.device)
        last_real = states[rows, lengths - 1]
        return self.output(last_real)


def scaled_attention(q, k, v, mask, causal=False):
    """Return (context, weights). q,k: B,T,d; v: B,T,dv; mask: B,T.
    True means a real position. Padding is a right suffix; every row is nonempty.
    """
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    # Keys must be masked before normalizing over the key dimension.
    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    if causal:
        future = torch.ones(q.size(1), k.size(1), dtype=torch.bool,
                            device=q.device).triu(diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = weights.masked_fill(~mask.unsqueeze(-1), 0.0)
    return weights @ v, weights


class MiniEncoder(nn.Module):
    """Provided one-head, one-block post-norm encoder; not a BERT replica."""
    def __init__(self, vocab_size, d_model=32, ff_dim=64, max_length=64):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position = nn.Embedding(max_length, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, ids, mask):
        positions = torch.arange(ids.size(1), device=ids.device)
        real_positions = mask.unsqueeze(-1)
        x = (self.embedding(ids) + self.position(positions)) * real_positions
        q, k, z = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        context, _ = scaled_attention(q, k, z, mask, causal=False)
        h1 = self.norm1(x + self.out_proj(context))
        h2 = self.norm2(h1 + self.ff2(F.relu(self.ff1(h1))))
        return h2 * real_positions


class TransformerClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.output = nn.Linear(encoder.d_model, 2)

    def forward(self, ids, mask, lengths):
        hidden = self.encoder(ids, mask)
        return self.output(masked_mean(hidden, mask))


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, total_examples = 0.0, 0
    for ids, mask, lengths, labels in loader:
        optimizer.zero_grad()
        logits = model(ids, mask, lengths)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        count = labels.size(0)
        total_loss += loss.detach().item() * count
        total_examples += count
    return total_loss / total_examples


def evaluate_model(model, loader):
    was_training = model.training
    total_loss, total_examples = 0.0, 0
    actual, predicted = [], []
    model.eval()
    try:
        with torch.no_grad():
            for ids, mask, lengths, labels in loader:
                logits = model(ids, mask, lengths)
                count = labels.size(0)
                total_loss += F.cross_entropy(logits, labels).item() * count
                total_examples += count
                actual.extend(labels.tolist())
                predicted.extend(logits.argmax(dim=-1).tolist())
        return {
            "loss": total_loss / total_examples,
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision": float(precision_score(actual, predicted, average="binary",
                                                pos_label=1, zero_division=0)),
            "recall": float(recall_score(actual, predicted, average="binary",
                                          pos_label=1, zero_division=0)),
            "f1": float(f1_score(actual, predicted, average="binary",
                                  pos_label=1, zero_division=0)),
        }
    finally:
        model.train(was_training)


def corrupt_for_mlm(ids, select_u, replace_u, random_ids, mask_prob=0.15):
    """Deterministic given random arrays. Never mutate caller-owned tensors."""
    selected = ids.ge(3) & select_u.lt(mask_prob)
    corrupted = ids.clone()
    targets = torch.full_like(ids, IGNORE)
    targets[selected] = ids[selected]
    masked = selected & replace_u.lt(0.8)
    randomized = selected & replace_u.ge(0.8) & replace_u.lt(0.9)
    corrupted[masked] = MASK
    corrupted[randomized] = random_ids[randomized]
    return corrupted, targets


def masked_token_loss(logits, targets):
    selected = targets.ne(IGNORE)
    if not selected.any():
        # Keep a path to logits so backward() remains valid in the empty case.
        return logits.sum() * 0.0
    return F.cross_entropy(logits[selected], targets[selected])


def transfer_encoder(pretrained_encoder, template_classifier):
    classifier = copy.deepcopy(template_classifier)
    classifier.encoder.load_state_dict(pretrained_encoder.state_dict())
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)
    return classifier


# The experiment driver below is provided in the student starter; do not modify it.
def run_experiment(path="data.tsv", n_per_class=5000, epochs=5, mlm_epochs=2,
                   batch_size=64, max_length=64, max_words=5000):
    """All full-assignment defaults are fixed; smaller inputs support public tests."""
    torch.set_num_threads(2)
    train, val, test = load_reviews(path, n_per_class)
    # Only model-visible training tokens may determine the vocabulary.
    train_tokens = [tokens[:max_length] for tokens, _ in train]
    vocab = build_vocab(train_tokens, max_words=max_words, min_freq=2)
    def encoded(data):
        sequences = encode_reviews([tokens for tokens, _ in data], vocab, max_length)
        return list(zip(sequences, [label for _, label in data]))
    train_data, val_data, test_data = encoded(train), encoded(val), encoded(test)
    def train_loader(seed=42):
        return make_loader(train_data, collate_batch, batch_size, True, seed)
    val_loader = make_loader(val_data, collate_batch, batch_size)
    test_loader = make_loader(test_data, collate_batch, batch_size)
    records = []
    def fit_and_record(name, model):
        fit = fit_classifier(model, train_loader(), val_loader,
                             train_one_epoch, evaluate_model, epochs=epochs)
        metrics = evaluate_model(model, test_loader)
        records.append((name, fit, metrics))

    set_seed(42)
    fit_and_record("FFNN", MeanFFNN(len(vocab)))
    set_seed(42)
    fit_and_record("LSTM", LSTMClassifier(len(vocab)))
    set_seed(42)
    template = TransformerClassifier(MiniEncoder(len(vocab), max_length=max_length))
    fit_and_record("Transformer", copy.deepcopy(template))

    # These parameters are the SAME initial encoder as the scratch comparison.
    pretrained_encoder = copy.deepcopy(template.encoder)
    set_seed(43)
    lm_head = nn.Linear(pretrained_encoder.d_model, len(vocab))
    history = pretrain_mlm(pretrained_encoder, lm_head, train_loader(seed=43),
                          corrupt_for_mlm, masked_token_loss,
                          epochs=mlm_epochs, seed=43)
    transferred = transfer_encoder(pretrained_encoder, template)
    fit_and_record("MLM-Transformer", transferred)

    print(f"Training reviews: {len(train_data)}")
    print(f"Validation reviews: {len(val_data)}")
    print(f"Testing reviews: {len(test_data)}")
    print(f"Vocabulary size: {len(vocab)}")
    for name, fit, metrics in records:
        print(f"{name} Best epoch: {fit['best_epoch']}")
        print(f"{name} Validation Loss: {fit['validation']['loss']:.4f}")
        print(f"{name} Testing Loss: {metrics['loss']:.4f}")
        for key, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                           ("recall", "Recall"), ("f1", "F1-score")]:
            print(f"{name} Testing {label}: {metrics[key]:.4f}")
    print(f"MLM final training loss: {history[-1]['loss']:.4f}")


if __name__ == "__main__":
    run_experiment()
