# CSCI 544 HW2: Neural Sentiment Analysis

Compare a mean-pooled feedforward network, an LSTM, a small Transformer,
and a Transformer initialized through masked language model (MLM) pretraining
on Amazon US Office Products reviews.

The 14 starter implementations in `hw2.py` are complete. The provided model
constructors, function signatures, experiment driver, and `hw2_support.py`
remain unchanged. All model parameters are trained locally; no pretrained
checkpoint or external model download is used.

## Setup

Use Python 3.12 and install the pinned dependencies before running:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the original Amazon US Office Products TSV used in HW1 in the project
directory as `data.tsv`. Do not substitute another category, a processed HW1
feature file, or the synthetic test data. The original dataset is not tracked
in this repository.

## Run

```bash
python -m unittest -v test_public test_contracts
python run_smoke.py
python hw2.py
```

The first command runs 12 public tests and 9 additional contract tests. The
smoke command exercises all four configurations on synthetic data with smaller
settings. Only the final command runs the required full-data experiment.

The full experiment runs on CPU with two PyTorch threads. It samples 5,000
reviews per class and uses a new 8,000/1,000/1,000 train/validation/test split.
Each review is limited to its first 64 tokens. The vocabulary contains up to
5,000 ordinary words plus three special tokens and is fitted only on training
tokens. Each classifier trains for five epochs; MLM adds two pretraining
epochs using only training reviews. Checkpoints are selected by validation
loss, then evaluated on the test set.

## Recorded results

The verified full run used 8,000 training reviews, 1,000 validation reviews,
1,000 test reviews, and a vocabulary of 5,003 tokens.

| Configuration | Best epoch | Test loss | Test accuracy | Test F1 |
|---|---:|---:|---:|---:|
| FFNN | 5 | 0.4130 | 0.8320 | 0.8310 |
| LSTM | 4 | 0.4253 | 0.8180 | 0.8150 |
| Transformer | 5 | 0.4443 | 0.8100 | 0.8061 |
| MLM-Transformer | 5 | 0.4345 | 0.8020 | 0.8004 |

The final MLM training loss was 6.2845. These are results from one fixed run;
pretraining is not required to outperform training from scratch. MLM adds
training work, so the two Transformer configurations do not use equal total
training time.

The full run exited successfully, printed exactly the required 33 lines,
and produced no standard-error output. A fresh import also succeeded without
`data.tsv` and produced no output. An independent code review found no
correctness issues. See [full results](results/full_experiment.txt) and the
[implementation guide](docs/implementation.md).

## Project files

| File | Purpose |
|---|---|
| `hw2.py` | Completed implementation and unchanged experiment driver |
| `hw2_support.py` | Unmodified supplied data and training helpers |
| `test_public.py` | Unmodified supplied public tests |
| `test_contracts.py` | Additional boundary, gradient, and state checks |
| `run_smoke.py` | Unmodified supplied synthetic end-to-end check |
| `toy_data.tsv` | Synthetic data for the smoke check only |
| `requirements.txt` | Assignment-pinned direct dependencies |
| `CSCI544_HW2_Fall2026.pdf` | Authoritative assignment handout |
| `docs/implementation.md` | Step-by-step explanation |
| `results/full_experiment.txt` | The 33 computed full-experiment output lines |

## Submission

Submit only `hw2.py` to the course assignment. The dataset, environment,
tests, documentation, support module, and experiment results are not part of
the submission. The program does not install dependencies or download data
or models at runtime.
