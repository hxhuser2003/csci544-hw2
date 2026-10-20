# Implementation guide

## 1. Vocabulary, encoding, and batches

`build_vocab` counts exactly the supplied training tokens. It removes words
below the frequency threshold, sorts the remaining words by decreasing count
and alphabetically for ties, and retains at most `max_words` ordinary words.
The special IDs are `<pad>=0`, `<unk>=1`, and `<mask>=2`; ordinary IDs start
at 3. The experiment driver truncates training reviews before this function
is called. Validation and test reviews never influence the vocabulary.

`encode_reviews` keeps the first `max_length` tokens of each review and maps
unknown words to ID 1. An empty review becomes `[1]`, preserving the review
and ensuring every sequence contains a real token. The function returns
unpadded lists and does not modify its inputs.

`collate_batch` right-pads each sequence to the longest length in that batch.
It returns integer IDs, a Boolean mask, original lengths, and integer labels.
A true mask entry marks a real token. Padding must not affect pooling,
attention normalization, or the selected LSTM state.

## 2. Mean-pooled feedforward classifier

Each word ID is mapped to a trainable embedding. `masked_mean` sums vectors
only at real positions and divides by the number of real tokens. The pooled
vector passes through the provided hidden linear layer, ReLU, and output
linear layer. The result contains two raw logits, not probabilities.

This representation is insensitive to word order. It provides a simple
baseline for comparison with models that represent sequence information.

## 3. LSTM classifier

Embedded tokens pass through the supplied single-layer, unidirectional LSTM.
Omitting explicit initial states gives zero hidden and cell states on each
call, so information is not carried between batches.

For each review, the representation is the sequence hidden state at
`lengths[b] - 1`. Selecting the final padded position would be incorrect.
The output linear layer maps the selected state to two logits. Neither
state averaging nor the LSTM cell state is used as the review representation.

## 4. Training and evaluation

Each training batch clears old gradients, computes logits and mean
cross-entropy, backpropagates, and updates the supplied optimizer. Epoch loss
is weighted by batch size, including the final smaller batch.

Evaluation saves the prior mode, enables evaluation mode and `no_grad`, and
computes loss without updating parameters or existing gradients. Predictions
use logit argmax; a tie selects class 0. Metrics are calculated from all
predictions across the split. Precision, recall, and F1 use positive label 1
with `zero_division=0`. The previous mode is restored before returning.

The supplied helper selects the checkpoint with the lowest validation loss.
Test data does not participate in vocabulary fitting, pretraining, training,
or checkpoint selection.

## 5. Attention and Transformer encoder

Scaled dot-product attention computes `Q @ K.transpose(-2, -1) / sqrt(d)`.
Padded keys are assigned negative-infinity scores before softmax. Optional
causal masking also blocks keys after the query position. Softmax operates
over keys, then padded query rows are set to zero. The context is the matrix
product of these weights with V. Sentiment classification and MLM both use
noncausal attention; causal mode is supported for the function contract.

The encoder adds token and learned position embeddings, clears padding
positions, and projects Q, K, and V. It then applies attention, the output
projection, a residual addition, and layer normalization. The feedforward
network is followed by another residual addition and layer normalization.
Padding outputs are cleared again. The order matches the handout exactly.

The Transformer classifier takes a masked mean of the encoder outputs and
applies the binary sentiment head. No additional blocks, dropout, or
pretrained vectors are introduced.

## 6. MLM corruption and loss

Only ordinary token IDs of at least 3 are eligible for selection. Selection
uses the supplied random array and a strict comparison with `mask_prob`.
Selected tokens retain their original IDs as targets; other targets are -100.

For selected positions, replacement values below 0.8 produce `<mask>`,
values from 0.8 up to 0.9 use the supplied random word IDs, and values from
0.9 onward leave the input unchanged. Unchanged selected tokens still
contribute to the loss. The function generates no random numbers and changes
none of its inputs.

`masked_token_loss` averages cross-entropy only over selected positions and
uses all vocabulary columns. When every target is ignored, it returns a
differentiable scalar zero instead of NaN. The provided pretraining loop
skips optimizer updates for batches with no selected targets.

## 7. Encoder transfer and comparison

`transfer_encoder` deep-copies the template classifier, loads the pretrained
encoder weights, preserves the template sentiment-head weights, and makes
all returned parameters trainable. The result shares no parameter storage
with either input.

The scratch Transformer and the encoder before MLM start from identical
encoder weights. Fine-tuning uses the same initial sentiment head as the
scratch configuration and a fresh supervised optimizer. Both the encoder
and sentiment head are updated during fine-tuning.

## 8. Verification

The completed project passed 12 supplied public tests and 9 additional
contract tests. The additional tests cover vocabulary ordering and empty
inputs, hand-computed feedforward logits and masked-mean gradients, real-token
LSTM indexing, padding invariance, attention scaling and gradients,
evaluation state and gradient preservation, weighted training loss,
MLM replacement boundaries, and independent parameter transfer.

The synthetic smoke run and the full original-data run both completed.
The full run produced exactly 33 correctly formatted lines and no
standard-error output. Importing the module in a fresh process without the
dataset succeeded silently. Structural checks confirmed that the supplied
constructors, required function signatures, experiment driver, and support
module were preserved. An independent review found no correctness issues.

## 9. Reproduction

Follow the environment setup in the README, supply the original `data.tsv`,
and run the listed test and experiment commands. Direct dependencies are
pinned to torch 2.10.0, numpy 2.3.5, pandas 2.2.3, and scikit-learn 1.8.0.
The program runs on CPU with float32 parameters and two PyTorch CPU threads.
Keep the supplied seeds and all experiment defaults unchanged.
