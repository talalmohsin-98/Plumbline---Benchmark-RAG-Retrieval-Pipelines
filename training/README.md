# Training

Exact reproduction steps for the fine-tuned reranker: environment, data
preparation, hyperparameters, and the push to Hugging Face Hub.

The protocol — success criterion, mining rules, hyperparameters — is
pre-registered in [`docs/02_EVALUATION_SPEC.md` §3](../docs/02_EVALUATION_SPEC.md),
written and committed **before** any of this ran. Read it first. The short version:
if lane 6 loses to lane 4, that is the published result, and nothing here gets
retuned in pursuit of a win.

---

## Step 1 — mine the negatives (local)

```bash
python -m backend.goldset.split        # 70/30 by group, seed 42
python -m training.mine_negatives      # train split only
```

Writes `data/train_pairs.jsonl` (400 pairs) and `data/mining_report.json`.

Needs a live `DATABASE_URL`, because mining runs lane 3 over the 80 train
questions for real. It takes about a minute, most of it the network round-trip to
hosted Supabase.

The miner refuses to write a pair file that contains anything from the test split.
That is enforced three ways — a proxy that raises the moment a non-train question
reaches the retriever or the embedder, a check that every train row was actually
issued, and a re-read of the finished file for test qids and test question text.
If any of them fires, nothing is written.

**What the last run measured** (`data/mining_report.json`):

| | |
|---|---|
| Pairs | 400 — 80 positive, 320 negative |
| Negatives per positive | 4.00, no row short |
| Gold inside the mined top-20 | 71/80 rows, median gold rank 2 |
| Distinct chunks used as negatives | 260 |
| **Negatives that are gold for a test question** | **24/320 (7.5%), covering 15 of 35 test questions** |

That last row is the pre-registered decision-5 diagnostic and it is not small. It
is reported rather than filtered: removing those chunks would mean reading the
test answer key to shape training. See EVALUATION_SPEC §3, decision 5.

---

## Step 2 — train on Colab (free T4)

Local torch is CPU-only, so training runs on Colab. `training/train_reranker.ipynb`
is the notebook; open it at [colab.research.google.com](https://colab.research.google.com)
via **File → Upload notebook**, or paste its cells into a blank one.

**Runtime → Change runtime type → T4 GPU** before running anything. The notebook's
first cell prints the device; if it says `cpu`, stop.

### The five files to upload

Cell 3 opens a file picker. Select all five at once; they must land in `/content`.

| File | From |
|---|---|
| `train_pairs.jsonl` | `data/` |
| `mining_report.json` | `data/` |
| `train_reranker.py` | `training/` |
| `model_card.md.tmpl` | `training/` |
| `model_card_data.md.tmpl` | `training/` |

`train_reranker.py` finds the two templates next to itself, so they have to travel
together. It imports nothing from this repository, which is why it runs standalone
on Colab as `python train_reranker.py` rather than needing the package installed.

Without `mining_report.json` the run still trains, but the model card loses its
training-data section — including the contamination diagnostic, which is the part
most worth publishing.

### Then

1. **Cell 4** — `notebook_login()`. Needs a token with **write** scope from
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. **Cell 5** — trains and pushes. About two minutes on a T4. If it takes an hour,
   the runtime is on CPU.
3. **Cell 6** — downloads `training_metrics.json`. Put it in `data/` and commit it;
   it is the provenance for the model card's training table.
4. **Cell 7** — verifies the published checkpoint loads with the same `num_labels`
   and the same `CrossEncoder` activation as the stock one. **This must match.**
   If it does not, lane 6 differs from lane 4 in its scoring convention as well as
   its weights, and the comparison is no longer one-variable.

### Hyperparameters

Frozen. From EVALUATION_SPEC §3, and the defaults in `train_reranker.py`:

```
base            cross-encoder/ms-marco-MiniLM-L-6-v2   (num_labels=1, head inherited)
epochs          3
batch_size      16
learning_rate   2e-5
warmup          10% of total steps
loss            BCEWithLogitsLoss
max_length      512
seed            42
holdout         8 of 80 questions, monitor only — never selects the checkpoint
```

Overriding `--epochs` or `--learning-rate` prints a warning naming the
pre-registration. A checkpoint trained with different settings must not be reported
against the test split without a new pre-registration in the spec.

### Why a plain `transformers` loop rather than `CrossEncoderTrainer`

The lane 4 / lane 6 comparison is one-variable only if the two checkpoints are the
same kind of artefact. The stock model is a plain HF sequence-classification repo
with no sentence-transformers configuration, so `CrossEncoder` gives it the default
activation for a one-logit head. A checkpoint written by the ST trainer carries
`config_sentence_transformers.json`, which can set a different activation at
inference — and then part of the measured delta is a scoring convention rather than
the weights. `save_pretrained` keeps the artefact the same shape as the stock one.
The loop it costs is forty lines.

---

## Step 3 — score it (local)

```bash
python -m backend.evaluate --split test   # lanes 1-6
python -m backend.significance            # McNemar + paired bootstrap vs lane 4
```

**Check the tripwire before reading the result.** Lane 6 above 0.95 recall@10
(≥ 34/35) means test questions reached training. `backend.significance` prints it
and refuses to declare a verdict; audit `data/train_pairs.jsonl` against
`data/test.jsonl` and report the audit before the metric.

---

## Reproducibility, honestly

Seeds fix the split (42), the negative sampling (`42:{qid}` per row), the training
data order and the dropout masks. They do **not** make a CUDA run bit-identical —
cuBLAS kernel selection and non-deterministic reductions are not controlled here,
so two runs of the training script produce slightly different weights and slightly
different test-split numbers in the third decimal place.

The library versions the training actually ran on are recorded in
`data/training_metrics.json` and on the model card, because this is the one step
that does not run on the pinned environment in `requirements.txt`.
