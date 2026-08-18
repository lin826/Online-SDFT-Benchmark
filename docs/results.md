# Results

[← Main README](../README.md)

## Current sharpened semantic three-seed ablation

The checked-in six-method comparison uses `semantic-title-body-sharp-t001`,
NumPy `2.4.6`,
and dataset fingerprint
`986cdf1a7d5fcc04c2b33f1bf90a1fc4f24a97ee85e663370382d8a67e4c932d`.
It was run on Apple MPS for three paired evaluation seeds 0–2 and 240
chronological decisions per seed using `LiquidAI/LFM2.5-230M`. The six methods are
Base, ICL, RAG, REINFORCE, RFT, and Online-SDFT. Student-visible context
contains realistic synthetic title/body text, category, local time, regime,
and local importance. ICL and RAG never call the hindsight teacher and
prompt only reliable singleton callbacks using the selected `causal_demos`
format. ICL uses the three latest eligible interactions; RAG retrieves three
interactions with equal weights on metadata similarity and visible title/body
similarity (`rag_text_weight=0.5`).

The evaluator shifts negative utility vectors when needed, normalizes them to
base probabilities $p$, and samples the hidden action from
$q_i\propto p_i^{100}$, corresponding to temperature `0.01`. Accuracy is exact
equality with that sampled hidden action; only regret compares numeric utility.

Online-SDFT and REINFORCE were tuned directly on the reported seeds 0–2 and
have no disjoint confirmation result, so the table ranks methods on the
streams that selected them. REINFORCE candidates had to exceed Base
in both pooled exact matches and total regret, then were ranked by accuracy
first and regret second. RFT's temperature and learning rate were screened on
seed 1200. RAG uses the fixed `K=3` configuration. This provenance makes the
report a descriptive six-method simulator ablation rather than an independent validation
study.

REINFORCE, RFT, and Online-SDFT use the same frozen LFM base and reset PEFT
LoRA architecture: rank `4`, alpha `8`, dropout `0`, and adapters on the Q, K,
V, and attention-output projections in the six attention layers. Each trains the same
172,032 adapter parameters. One physical model is reused
sequentially, with the identical initial adapter state restored before every
method and seed.

Online-SDFT uses learning rate `1e-3`, a 64-record selection-balanced replay
window with recency half-life `32`, batch size eight, two update steps, a
four-example warmup, gradient-norm clipping at `1.0`, and ambiguous replay-group
weight `0.05`. Its behavior starts with epsilon `0.02`. When the student's
maximum action probability is at most `0.60`, it mixes in an `INTERRUPT` probe
with initial weight `0.15` and half-life `80`. That sampling distribution is
passed through without an additional argmax taper through step 160, then
tapered toward argmax with half-life `5`. Its hindsight forward uses the same
physical model with the adapter disabled and
runs at temperature `1.0`. The target is conditioned on callback reliability:
reliable singleton callbacks use 0.05 teacher, 0.05 frozen decision
distribution, and 0.90 maximum-entropy behavior support. For an
ambiguous digest open, the frozen decision distribution is projected onto
`{INTERRUPT, LATER}` with 0/1/0 teacher/decision/behavior weights; `UNKNOWN`
creates no target. Training replay is not serving memory:
`replay_prompt_examples=0`.

REINFORCE has no teacher. It samples from the current action-token LoRA policy
and applies AdamW at learning rate `1e-4` to batches of eight matured known
factual outcomes. The learner-only outcome map is immediate push open `+5`,
delayed push open `-1`, push deletion `-5`, digest open `0`, and digest deletion
`-5`; `UNKNOWN` is censored before mapping. It uses a fixed zero baseline,
entropy coefficient `1.0`, and gradient-norm clipping at `1.0`. Each callback
is consumed once, with no replay or incomplete end-of-horizon batch flush. The
reported observable-reward column still uses the shared environment map of
`+5`, `-1`, `-2`, `+0.25`, `-1`, and `0` for `UNKNOWN` rather than the learner's
shaped scalar.

Rejection Fine-Tuning (RFT) is the capacity-matched hard-target distillation
baseline. It draws one
categorical route at proposal temperature `8.0` from the same physical model
with its adapter disabled for each matured
non-`UNKNOWN` callback, accepts it only when reliable singleton causal evidence
verifies the sampled route, and trains ordinary one-hot cross-entropy on
accepted replay rows. Ambiguous digest opens, teacher mismatches, and `UNKNOWN`
produce no target or update. RFT uses the same reset 172,032-parameter LoRA
architecture, initialization, and teacher-forward budget as Online-SDFT, but
its student schedule is independently fixed at replay size `32`, epsilon `0.06`,
and learning rate `7e-4`, without Online-SDFT's recency, probe, or taper. The
proposal temperature and hard sample/filter rule are fixed parts of the RFT
baseline; the report therefore includes its accepted-example budget and
acceptance rate alongside its prequential metrics.

Mean ± 95% confidence interval across seeds:

| Method | Sampled-preference accuracy | Cumulative utility gap | Observable reward / decision |
| --- | ---: | ---: | ---: |
| Base | 28.19% ± 2.33 | 164.67 ± 2.66 | 0.256 ± 0.140 |
| ICL | 42.22% ± 3.57 | 131.48 ± 0.60 | 0.008 ± 0.055 |
| RAG | 50.00% ± 7.37 | 106.62 ± 30.45 | 0.316 ± 0.389 |
| REINFORCE | 32.08% ± 2.06 | 157.99 ± 2.56 | -0.090 ± 0.176 |
| RFT | 52.78% ± 2.60 | 105.26 ± 15.32 | 0.364 ± 0.250 |
| **Online-SDFT** | **70.28% ± 2.60** | **44.76 ± 4.57** | **0.955 ± 0.082** |

Tuned REINFORCE records 231 exact matches versus Base's 203, a
`+3.89`-point accuracy difference, and reduces mean cumulative utility gap by
`6.67`. Its shared observable-reward mean remains below Base's, so the gain shows up
in accuracy and regret rather than in every reported metric.

Online-SDFT has the highest mean accuracy and lowest mean cumulative utility
gap on these streams. Relative to RFT, the strongest baseline by both means,
the mean differences are `+17.50` accuracy percentage points and `-60.50`
cumulative utility-gap units. Relative to Base, they are `+42.08` points and
`-119.90` gap units. Its 506 exact action matches among 720 decisions give the
displayed 70.28% mean.
Across the three streams RFT attempted 311 candidates, accepted 75 (24.12%),
rejected 102 ambiguous callbacks and 134 teacher mismatches, and performed 132
optimizer steps after its per-seed warmups. Another 380 released callbacks
were censored `UNKNOWN`, while 29 callbacks were still pending at the horizon.
Observable reward reflects executed-surface outcomes rather than agreement with
either evaluator-only answer, so it is not interchangeable with accuracy or
regret.
Accuracy is exact equality between the frozen action and the hidden sampled
preference; only regret directly compares numeric evaluator utilities.
These are differences between reported means, not claims of statistical
significance with only three streams.

The exact configuration and per-seed records are in
[`outputs/bandit/`](../outputs/bandit), and the aggregate figure is
[`figures/bandit_accuracy.png`](../figures/bandit_accuracy.png). These are synthetic semantic-dataset
results; phone latency, energy, and production behavior come from the device
implementation.

## Required production follow-up

A larger simulator report should include:

- paired per-seed sampled-preference accuracy, utility regret, and executed-outcome
  reward;
- learning curves over the chronological stream;
- the fraction of feedback still pending at the horizon;
- the full executed-route × hidden-preference observation matrix;
- results by executed route, observed selection, and feedback delay;
- teacher-distribution entropy for ambiguous callbacks; and
- ablations for missing notification access and longer update deferral.

Base, ICL, RAG, REINFORCE, and RFT should be reported primarily as ablations of
the Online-SDFT loop: no adaptation, prompt-only recency memory, prompt-only
retrieval, scalar-reward LoRA adaptation, and rejection-filtered hard-target
distillation. All must use the same stream,
decision-time features, deterministic callback matrix, delayed release queue,
reward mapping, sampled-preference evaluator, and utility-optimal regret
comparator. A result is invalid if any baseline reads either evaluator-only
answer or if the same event/action pair produces different feedback across
methods.

A future Android report should additionally include:

- inference and teacher-forward latency;
- peak memory during teacher inference and LoRA adapter training;
- energy and thermal behavior per maintenance job;
- number of lessons combined per device wake-up;
- background-job completion and cancellation rates; and
- student-state validation, activation, and rollback time.

No phone deployment claim should be based on simulator accuracy alone.
