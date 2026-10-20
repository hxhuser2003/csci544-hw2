"""Synthetic smoke run, NOT the required Amazon-data experiment."""
from hw2 import run_experiment

if __name__ == "__main__":
    run_experiment("toy_data.tsv", n_per_class=20, epochs=2, mlm_epochs=1,
                   batch_size=8, max_length=12, max_words=80)
