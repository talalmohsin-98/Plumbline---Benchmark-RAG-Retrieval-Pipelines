"""Cross-encoder fine-tuning driver: run on Google Colab free T4, not locally.

Hyperparameters are pre-registered in `docs/02_EVALUATION_SPEC.md` §3 and are
frozen: 3 epochs, batch 16, lr 2e-5, 10% warmup, BinaryCrossEntropy, max_length
512, seed 42. If lane 6 loses to lane 4, that is the published result and none
of these numbers moves in pursuit of a win.

    python -m training.train_reranker --push-to talalmohsin-98/plumbline-reranker-v1

**Why this is a plain `transformers` loop and not `SentenceTransformer`'s
`CrossEncoderTrainer`.** The lane 4 / lane 6 comparison is only one-variable if
the two checkpoints are the same *kind* of artefact. `cross-encoder/ms-marco-
MiniLM-L-6-v2` is a plain HF sequence-classification repo with `num_labels=1`
and no sentence-transformers configuration, so `CrossEncoder` gives it the
default activation for a one-logit head. A checkpoint written by the ST trainer
carries `config_sentence_transformers.json`, which can set a *different*
activation at inference -- and then the lane 4 to lane 6 delta is partly a
scoring convention rather than the model. Saving through `save_pretrained` keeps
the artefact byte-for-byte the same shape as the stock one. The loop it costs is
forty lines.

The head is inherited, not re-initialised: the base checkpoint already has
`num_labels=1`, so `from_pretrained` keeps the MS MARCO relevance head and this
is a fine-tune rather than a fresh head on a frozen trunk.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_PAIRS = Path("data/train_pairs.jsonl")
DEFAULT_MINING_REPORT = Path("data/mining_report.json")
DEFAULT_OUT = Path("data/reranker-v1")
DEFAULT_METRICS = Path("data/training_metrics.json")

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REPO_ID = "talalmohsin-98/plumbline-reranker-v1"

# --- Pre-registered hyperparameters. Frozen. --------------------------------
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WARMUP_FRACTION = 0.1
MAX_LENGTH = 512  # bge-small and MiniLM both stop here; matches rerank.py
SEED = 42

# 10% of the *questions*, not of the pairs. Splitting pairs would put a
# question's positive in train and its negatives in eval, so the eval slice
# would be scoring a question the model had already been told the answer to.
# Costs 8 of 80 questions, which is real at this size and is why the slice
# monitors rather than selects -- see `evaluate_slice`.
HOLDOUT_FRACTION = 0.1


@dataclass(frozen=True)
class Example:
    """One training pair as the loop sees it."""

    qid: str
    question: str
    text: str
    label: float


def load_pairs(path: Path) -> list[Example]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m training.mine_negatives` first, "
            "or upload the pair file if you are on Colab."
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
    return [
        Example(qid=r["qid"], question=r["question"], text=r["text"], label=float(r["label"]))
        for r in rows
    ]


def split_by_question(
    examples: list[Example], holdout_fraction: float = HOLDOUT_FRACTION, seed: int = SEED
) -> tuple[list[Example], list[Example]]:
    """Hold out whole questions, keeping every pair of a question on one side."""
    qids = sorted({e.qid for e in examples})
    shuffled = list(qids)
    random.Random(seed).shuffle(shuffled)
    holdout = set(shuffled[: max(1, round(len(qids) * holdout_fraction))])
    train = [e for e in examples if e.qid not in holdout]
    held = [e for e in examples if e.qid in holdout]
    return train, held


def set_seeds(seed: int) -> None:
    """Seed python, numpy and torch.

    Honest about what this buys: it makes the data order, the shuffle and the
    dropout masks reproducible. It does not make a CUDA run bit-identical --
    cuBLAS kernel selection and non-deterministic reductions are outside this.
    Two runs of this script on a T4 will produce very slightly different
    weights, which is stated in the model card rather than papered over.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Evaluation on the held-out slice
# --------------------------------------------------------------------------


def evaluate_slice(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Example],
    device: str,
    batch_size: int = 32,
) -> dict[str, float]:
    """Loss and a ranking check on held-out train questions.

    A *monitor*, not a selector. The slice is 8 questions; choosing the best
    epoch on 8 questions would be selecting on noise, so the published
    checkpoint is always the last epoch and this only tells us whether the run
    went sideways. EVALUATION_SPEC §3 asks for per-epoch eval on a held-out
    slice of train, which is what this is.

    `positive_ranked_first` is the useful one: for each held-out question, does
    the positive outscore all four of its own mined negatives? That is exactly
    the job the reranker does at inference, measured over five candidates
    instead of twenty.
    """
    import torch

    model.eval()
    scores: list[float] = []
    labels = [e.label for e in examples]
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            encoded = tokenizer(
                [e.question for e in batch],
                [e.text for e in batch],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits.squeeze(-1)
            scores.extend(logits.float().cpu().tolist())

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(scores), torch.tensor(labels)
    ).item()

    by_question: dict[str, list[tuple[float, float]]] = {}
    for example, score in zip(examples, scores, strict=True):
        by_question.setdefault(example.qid, []).append((score, example.label))
    wins = 0
    scored = 0
    for candidates in by_question.values():
        positives = [s for s, label in candidates if label == 1.0]
        negatives = [s for s, label in candidates if label == 0.0]
        if not positives or not negatives:
            continue
        scored += 1
        wins += max(positives) > max(negatives)

    model.train()
    return {
        "bce_loss": round(loss, 4),
        "positive_ranked_first": round(wins / scored, 4) if scored else 0.0,
        "questions": scored,
        "pairs": len(examples),
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train(
    examples: list[Example],
    *,
    base_model: str = BASE_MODEL,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    warmup_fraction: float = WARMUP_FRACTION,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, dict[str, Any]]:
    """Fine-tune the base cross-encoder. Returns (model, tokenizer, metrics)."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and verbose:
        print(
            "WARNING: no CUDA device. This is written for a Colab T4; on CPU it "
            "will work but slowly, and EVALUATION_SPEC §3 says not to.",
            file=sys.stderr,
        )

    train_examples, held_out = split_by_question(examples, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # num_labels=1 matches the base checkpoint's existing head, so the MS MARCO
    # relevance weights are kept rather than replaced by a random projection.
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1).to(device)
    model.train()

    def collate(batch: list[Example]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        encoded = tokenizer(
            [e.question for e in batch],
            [e.text for e in batch],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return encoded, torch.tensor([e.label for e in batch], dtype=torch.float)

    loader = DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    total_steps = len(loader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * warmup_fraction),
        num_training_steps=total_steps,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Epoch 0: the stock model on the held-out slice. Without it the per-epoch
    # table has no baseline and a run that made the model worse looks the same
    # as one that made it better.
    per_epoch = [{"epoch": 0, **evaluate_slice(model, tokenizer, held_out, device)}]
    if verbose:
        print(f"epoch 0 (stock)  {per_epoch[0]}", file=sys.stderr)

    for epoch in range(1, epochs + 1):
        running = 0.0
        for encoded, labels in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.squeeze(-1)
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
        stats = {
            "epoch": epoch,
            "train_loss": round(running / len(loader), 4),
            **evaluate_slice(model, tokenizer, held_out, device),
        }
        per_epoch.append(stats)
        if verbose:
            print(f"epoch {epoch}          {stats}", file=sys.stderr)

    metrics = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model": base_model,
        "device": device,
        # Recorded because this is the one step of the pipeline that does not
        # run on the pinned environment in `requirements.txt`: it runs on
        # whatever Colab has that day. The checkpoint format is stable across
        # these versions -- it is a plain HF sequence-classification model --
        # but a reader comparing two runs deserves to see what produced each.
        "environment": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "python": sys.version.split()[0],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_fraction": warmup_fraction,
            "warmup_steps": round(total_steps * warmup_fraction),
            "max_length": MAX_LENGTH,
            "loss": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "scheduler": "linear with warmup",
            "seed": seed,
        },
        "data": {
            "pairs_total": len(examples),
            "pairs_train": len(train_examples),
            "pairs_holdout": len(held_out),
            "questions_train": len({e.qid for e in train_examples}),
            "questions_holdout": len({e.qid for e in held_out}),
            "holdout_fraction": HOLDOUT_FRACTION,
            "holdout_qids": sorted({e.qid for e in held_out}),
        },
        "steps": {"per_epoch": len(loader), "total": total_steps},
        "per_epoch": per_epoch,
        "checkpoint_selected": "final epoch",
        "checkpoint_selection_note": (
            "The last epoch, always. The held-out slice is 8 questions; picking "
            "the best epoch on 8 questions would be selecting on noise, and "
            "EVALUATION_SPEC §3 pre-registers 3 epochs. The slice monitors, it "
            "does not select."
        ),
    }
    return model, tokenizer, metrics


# --------------------------------------------------------------------------
# The model card
# --------------------------------------------------------------------------


CARD_TEMPLATE = Path(__file__).with_name("model_card.md.tmpl")
DATA_TEMPLATE = Path(__file__).with_name("model_card_data.md.tmpl")


def build_model_card(metrics: dict[str, Any], mining: dict[str, Any] | None) -> str:
    """Fill the card templates from measured numbers, never from typed-in ones.

    Same rule as everything else here: if a figure appears, a command in this
    repository produced it. The prose lives in `model_card.md.tmpl` rather than
    in an f-string because it is a document -- a reader opens it on the Hub
    without cloning anything -- and documents are easier to review as markdown
    than as escaped Python.

    `string.Template` rather than `str.format`, so the markdown can contain
    braces without escaping them.
    """
    hyper = metrics["hyperparameters"]
    data = metrics["data"]
    epoch_rows = "\n".join(
        "| {epoch} | {train_loss} | {bce} | {first} |".format(
            epoch=e["epoch"],
            train_loss=e.get("train_loss", "— (stock)"),
            bce=e["bce_loss"],
            first=e["positive_ranked_first"],
        )
        for e in metrics["per_epoch"]
    )

    if mining:
        collisions = mining["test_gold_chunks_mined_as_negatives"]
        positives = mining["positive_in_mined_depth"]
        affected = collisions["distinct_test_questions_affected"]
        of_test = collisions["of_test_questions"]
        mining_section = Template(DATA_TEMPLATE.read_text(encoding="utf-8")).substitute(
            mine_depth=mining["parameters"]["mine_depth"],
            mining_seed=mining["parameters"]["seed"],
            questions_mined=mining["questions_mined"],
            pairs_total=mining["pairs"]["total"],
            pairs_positive=mining["pairs"]["positives"],
            pairs_negative=mining["pairs"]["negatives"],
            negatives_mean=mining["negatives_per_positive"]["mean"],
            rows_below_target=mining["negatives_per_positive"]["rows_below_target"],
            gold_in_depth=positives["rows"],
            gold_of=positives["of"],
            median_gold_rank=positives["median_gold_rank"],
            collision_count=collisions["count"],
            collision_of=collisions["of_negatives"],
            collision_pct=f"{collisions['fraction']:.1%}",
            affected_questions=affected,
            of_test_questions=of_test,
            affected_pct=f"{affected / of_test:.0%}" if of_test else "an unknown share of",
        ).strip()
        contamination = (
            f"**{affected} of {of_test} test questions have one of their gold chunks in "
            "the training negatives** (see above). Not a leak that flatters this model -- "
            "a handicap, reported rather than removed."
        )
    else:
        mining_section = (
            "_The mining report was not supplied at training time, so this section is "
            "missing its numbers. Re-run with `--mining-report data/mining_report.json`._"
        )
        contamination = (
            "**The training-data contamination diagnostic is missing** because the mining "
            "report was not supplied when this card was generated."
        )

    return Template(CARD_TEMPLATE.read_text(encoding="utf-8")).substitute(
        base_model=metrics["base_model"],
        mining_section=mining_section,
        epochs=hyper["epochs"],
        batch_size=hyper["batch_size"],
        learning_rate=hyper["learning_rate"],
        warmup_pct=f"{hyper['warmup_fraction']:.0%}",
        warmup_steps=hyper["warmup_steps"],
        total_steps=metrics["steps"]["total"],
        loss=hyper["loss"],
        optimizer=hyper["optimizer"],
        scheduler=hyper["scheduler"],
        max_length=hyper["max_length"],
        seed=hyper["seed"],
        device=metrics["device"],
        questions_train=data["questions_train"],
        pairs_train=data["pairs_train"],
        questions_holdout=data["questions_holdout"],
        pairs_holdout=data["pairs_holdout"],
        questions_total=data["questions_train"] + data["questions_holdout"],
        epoch_rows=epoch_rows,
        contamination_bullet=contamination,
        repo_id=REPO_ID,
        generated_at=metrics["generated_at"],
    )


# --------------------------------------------------------------------------
# Saving and publishing
# --------------------------------------------------------------------------


def save(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, out: Path) -> None:
    """Write the checkpoint in the same shape as the stock one.

    `save_pretrained` for both, and nothing else -- no sentence-transformers
    configuration files. See this module's docstring: an extra config here
    would make lane 6 differ from lane 4 in its scoring convention as well as
    its weights, and the comparison would stop being one-variable.
    """
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)


def push(out: Path, repo_id: str, token: str | None, card: str) -> str:
    """Upload the checkpoint folder and the generated card to the Hub."""
    from huggingface_hub import HfApi

    (out / "README.md").write_text(card, encoding="utf-8")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.train_reranker",
        description="Fine-tune the lane 6 cross-encoder on mined hard negatives.",
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--mining-report", type=Path, default=DEFAULT_MINING_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--push-to",
        default=None,
        help=f"HF repo id to publish to, e.g. {REPO_ID}. Omit to train without publishing.",
    )
    parser.add_argument("--hf-token", default=None, help="defaults to HF_TOKEN in the environment")
    args = parser.parse_args(argv)

    if args.epochs != EPOCHS or args.learning_rate != LEARNING_RATE:
        print(
            "NOTE: you are overriding a pre-registered hyperparameter. "
            "EVALUATION_SPEC §3 freezes epochs=3 and lr=2e-5, and a checkpoint "
            "trained otherwise must not be reported against the test split "
            "without a new pre-registration.",
            file=sys.stderr,
        )

    examples = load_pairs(args.pairs)
    print(f"pairs      {len(examples)} from {args.pairs}", file=sys.stderr)

    model, tokenizer, metrics = train(
        examples,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    save(model, tokenizer, args.out)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    mining = None
    if args.mining_report.exists():
        mining = json.loads(args.mining_report.read_text(encoding="utf-8"))
    else:
        print(
            f"WARNING: {args.mining_report} not found. The model card will be "
            "missing its training-data section -- upload it alongside the pairs.",
            file=sys.stderr,
        )
    card = build_model_card(metrics, mining)
    (args.out / "README.md").write_text(card, encoding="utf-8")

    print(f"\ncheckpoint {args.out}")
    print(f"metrics    {args.metrics}")

    if args.push_to:
        import os

        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            print(
                "HF_TOKEN is not set and --hf-token was not given. The checkpoint is "
                "saved locally; publish it later.",
                file=sys.stderr,
            )
            return 1
        url = push(args.out, args.push_to, token, card)
        print(f"published  {url}")
    else:
        print("not published (pass --push-to to upload)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
