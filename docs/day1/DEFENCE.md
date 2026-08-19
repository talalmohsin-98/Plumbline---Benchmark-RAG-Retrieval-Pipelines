# Day 1 — interview brief

Eight questions I expect, and what I would say. Every answer is checkable
against a committed artifact; the ones that are not are quarantined in the last
section rather than smoothed over.

Counts throughout: 115 gold rows, 139 screened, 137 decisions, 78% screener
agreement. `PIPELINE.md` shows where the rest went.

---

### 1. Why LLM screening instead of verifying everything by hand?

Attention, not money. Reading a few hundred drafts properly is a full day, and I
had a day for the whole gold set including ingest. Screening replaces the *labour* of
reading, not the judgement: I decided all 137 verdicts myself, and the
screener's agreement with me — 78% — is published beside every metric the gold
set produces. What I get back is uniformity. The four rules get applied the same
way to row 130 as to row 3, which is not true of a human at hour six, and every
verdict carries a written reason I can argue with. The fully-manual path,
`verify.py`, is still in the repo — it is the ground truth the screener is
measured against.

### 2. How do you know your gold set is any good?

I don't know it's good. I know how it was built and where it's weak. Every one
of the 115 rows carries my verdict — 137 decisions over 139 screened rows, 73 of
them taken fully blind, with no screener verdict shown at any point. Two
independent screeners scored every draft before I saw it, and I overrode the
first one on 30 of 137. The label itself is checked by a separate
`temperature=0` call against the chunk's two document neighbours; 12 gold rows
were relabelled by that check. The rejection rate is a direct census, 13.7%, not
an estimate. What none of that establishes is that these questions actually
discriminate between retrieval lanes. Day 2 measures that.

### 3. Your screener rejected 9 questions and you kept 7 of them. So what?

It means the screener's drops are worthless while its keeps are mostly fine, and
only a stratified draw could have shown me that. A false drop is invisible: a
wrongly discarded question never appears in the finished gold set, so reading
that file cannot find one. Agreement on drops was 0/7. On keeps it was 106/128.
Had I audited keeps only — the obvious thing to do — I would have published 83%
agreement and quietly lost every good question the screener disliked. All 7 are
in the gold set now. The two I could not judge, q115 and q117, had only a
stand-in model's verdict, so I excluded them rather than import one the real
screener never gave.

### 4. Why did you adjudicate blind for some rows and not others?

Sequence, not design. The first pass was the disputed queue — every row the two
screeners split on, plus 12 controls — and it revealed each verdict *after* the
keypress, because on a disputed row the thing worth recording is which screener
I sided with, and the reveal is what records it. The blind pass came later, once
two models agreeing stopped being warrant enough to accept a row: all 73 agreed
keeps, nothing shown before or after, and the running agreement rate withheld
from the progress line. Two rejection-stratum rows were decided last with the
screener's reasoning visible and are stamped `blind: false` for that reason. The
mode is on every decision, so the two are never averaged.

### 5. Why can't you compare the 96% blind agreement to the 60% revealed one?

The revealed rate is 58%, not 60%; the spec's figure is stale, see the last
section. Either way, mode was never randomised, so the gap is population, not
method. The revealed stratum is the disputed rows plus controls, selected
*because* two screeners contradicted each other, so the screener is wrong on a
large fraction by construction. The blind stratum is the rows both agreed to
keep — the easy majority. Any protocol opens a gap there. The comparison
that holds is same-population, different-mode: agreed keeps judged revealed (12
controls, 11/12) against agreed keeps judged blind (70/73). Fisher exact,
two-tailed, p = 0.46 — no detectable anchoring effect, though n=12 rules out
only a large one.

### 6. Why is the train/test split grouped instead of random?

Because multi-labelling makes a row-wise shuffle leak. 26 of the 115 rows carry
more than one gold chunk, and two questions can share one — q023 and q056 both
ask which package forms need, sharing all five of theirs. A random shuffle put 8
chunks on both sides. Worse, Day 3 mines hard negatives from top-20 retrieval
with gold removed, and "gold" is per-question: a chunk that is gold for a *test*
question can be sampled as a negative for a *train* question, training the
reranker to push down something the test set will expect it to return. That is a
leak in the direction that lowers the score — harder to spot than one that
flatters.

### 7. Why did you exclude nDCG?

It needs graded relevance and this gold set has binary labels. With a single
relevance level, nDCG@10 collapses toward a noisier restatement of MRR: same
ranking signal, more machinery, no more information. So I report recall@5,
recall@10 and MRR. Recall answers "did we find it", MRR answers "did we rank it
first", and the gap between recall@5 and recall@10 says whether a lane ranks
well or merely retrieves broadly — which is the whole demonstration, a reranker
moving MRR while barely moving recall@10. If the gold set ever gets graded
labels, nDCG earns its place. It is a stated exclusion in the evaluation spec,
not an oversight I am rationalising afterwards.

### 8. What's wrong with your gold set?

Four things that matter. The screening layer is entirely LLM — "two screeners
agreed" means two models agreed — and the adjudication on top is one reader, me,
so a bias all three share is invisible to every check I ran. The test split is 35
questions: one question moves recall 2.9 points, and small lane differences will
not be separable. It is not reproducible from the corpus — I tightened the
eligibility filter mid-day, the pool went 405 → 392, so `--n 350` now draws a
different sample. And one row, q017, sits in the gold set although its chunk
fails that same filter. I found that writing this brief.

---

## The full limitations list

The answer above is the 30-second version. This is the list.

| # | limitation | where it is visible |
|---|---|---|
| 1 | Two screeners are two models, not two people. The whole screening layer is LLM work. | `data/claude_screen.jsonl`, `screen_model` on every screened row |
| 2 | One human reader. A question that reads as reasonable to both models *and* to me still gets through. | by construction |
| 3 | `q017` is in the gold set but was drafted from `langchain_tools_ch_000`, which is 80% frontmatter and fails the scaffolding filter — the exact criterion used to hand-exclude q061, q064 and q080. It was restored by the rejection-stratum pass on 2026-08-19, three days after `exclusions.json` was written, and the reproducibility check was not re-run. | reproduce with the snippet in `PIPELINE.md` |
| 4 | Not reproducible from the corpus. The eligible pool was 405 chunks at drafting time and is 392 now. Everything from `goldset_screened.jsonl` forward reproduces exactly. | `PIPELINE.md` note 3 |
| 5 | Where the drafting run stopped is not recorded anywhere. 350 sampled, 148 drafted; the 202-row gap is arithmetic, not a measurement. | no artifact — see note 1 |
| 6 | Rephrased rows are not re-scored against rules 1, 3 and 4 after the rewrite. 2 rows were rephrased; 5 carry a `fix` verdict. | `rephrase_attempts` in `goldset_screened.jsonl` |
| 7 | The audit queue file is not redacted — it carries the verdicts the pass was blind to. Blindness held structurally at the terminal (4 tests) and by procedure at the file. | `02_EVALUATION_SPEC.md`, "What the blind pass guaranteed" |
| 8 | The corpus is three Python framework doc sites. Nothing here says how these lanes rank on prose, legal text, or anything without code fences and frontmatter. | `data/demo_corpus/MANIFEST.json` |
| 9 | 2 of 139 screened rows carry no human verdict at all (q115, q117), so the rejection stratum is covered but not exhaustively. | `data/audit_decisions.jsonl` has 137 rows |

---

## Where this brief and the repo disagree

Flagged rather than written into an answer above, because these are claims the
artifacts do not currently support. Each needs a decision before Day 2 closes.

1. **"The screener agreed with me 60% of the time on the disputed rows"**
   (`02_EVALUATION_SPEC.md`, "What the adjudication measured"). The artifacts say
   **52%** — 26 of 50, from `audit_results.json` `by_queue_kind.disputed`. The
   60% is not reproducible from anything committed; it looks like a figure from
   an earlier, smaller disputed pass that survived an edit. Q5 above quotes the
   revealed-stratum rate (58%, 37 of 64), which is a different quantity again.
   The spec line needs correcting to 52%.
2. **The multi-label paragraph is stale.** The spec says the rule "flagged 77 of
   98 kept rows", "21 rows survived", "27 extra chunks", and "two candidates
   failed that check". The artifacts say 242 candidates across **92 of 115**
   rows, **26** rows accepted, **35** extra chunks. `multilabel_review.jsonl`
   records `answer_token_present` false on 62 of 74, so "two failed" is not
   reproducible either. `04_README_TEMPLATE.md` repeats the stale "21".
3. **`02_EVALUATION_SPEC.md` §3 still says train 84 / test 36.** The split is
   80/35. §4's results schema example likewise carries superseded numbers
   (total 120, screened 116, adjudicated 51, rejection rate 0.155).
4. **`split.py` says q023 and q056 share four chunks.** They share five.
5. **"The queue file was never opened before or during the blind run"** is my
   statement of fact, not an artifact. It is the right thing to say and it
   cannot be evidenced from the repository. If asked, say so in those words.
