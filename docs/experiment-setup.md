# Experiment Setup

[← Main README](../README.md)

## Protocol

```text
notification posted
  → save decision-time OS snapshot
  → student chooses and commits one route
  → wait for that route's callback or timeout
  → same model produces a soft hindsight distribution
  → queue a small local LoRA-adapter update for future requests
```

Every method sees the same decision-time inputs: title and body text, channel
or category, a local importance estimate, local time and usage regime, and
matured feedback from only the actions it executed itself. Exact deadline,
deadline-derived urgency, and affinity stay private to the simulator. Title and
body carry coarse timing and salience cues, as real content would, but never
the sampled preference or the utility-optimal route, and no raw latent value
reaches a prompt, memory record, or retrieval key.

The hindsight teacher additionally reads the executed route and the later
observable selection — `INTERRUPT`, `LATER`, `ARCHIVE`, or `UNKNOWN`. It never
reads the simulator reward, hidden user state, sampled preference,
utility-optimal route, busy state, shadow-policy label, or any unexecuted
outcome. Its prompt states the delivery-surface semantics explicitly, including
that a digest open after `LATER` is not definitive evidence for `LATER`.

## Delayed feedback

The causal callback clock advances 15 minutes per decision and releases feedback
only when the observation window closes. Displayed local times are separate,
ordered contextual samples within each regime.

| Executed route | Observable event | Delay |
| --- | --- | ---: |
| `INTERRUPT` | immediate open | 1 min |
| `INTERRUPT` | deletion | 15 min |
| `INTERRUPT` | delayed read | 120 min |
| `LATER` | digest read or deletion | 120 min |
| `ARCHIVE` | no delivered surface: `UNKNOWN` | 240 min |

The sampled preference scores every method and drives the simulated user, but is
never copied into a callback: execute `LATER` on an item that wanted an
immediate alert and the digest read still reports `LATER`. REINFORCE's scalar
comes only from the executed surface (`+5` immediate open, `-1` delayed open,
`-2` push deletion, `+0.25` digest open, `-1` digest deletion, `0` for
`UNKNOWN`) and never compares against hidden gold. ICL and RAG may retain an
`UNKNOWN` interaction as unlabeled audit history, but only reliable singleton
callbacks enter a serving prompt. Feedback still pending at the horizon is not
flushed into the score.

## Learner configuration

These are the checked-in defaults, which is what `run.py` and the notebook use;
the released benchmark tunes serving and replay on top of them, and the
article's ablation quantifies both.

Online-SDFT wraps [`LiquidAI/LFM2.5-230M`](https://huggingface.co/LiquidAI/LFM2.5-230M)
with a PEFT rank-4 LoRA adapter (`alpha=8`) on `q_proj`, `k_proj`, `v_proj`, and
`self_attn.out_proj` in attention layers 2/4/6/8/10/12 — 172,032 trainable
parameters across 48 tensors, with the hybrid model's convolutional `out_proj`
modules left frozen. The base never changes and the adapter is never merged.
Student passes enable the adapter; teacher passes disable it, so the teacher
stays the fixed initial policy without a second checkpoint.

Callback reliability picks the target. A reliable singleton fuses 5% teacher, 5%
frozen decision-time distribution, and 90% maximum-entropy delivery-surface
support. An ambiguous digest open uses the frozen decision distribution
projected onto `{INTERRUPT, LATER}` (`0/1/0`). `UNKNOWN` creates no target.
Training uses AdamW at `1e-3`, a 32-record selection-balanced replay window,
batches of at most eight, two update steps, and a four-example warmup, with
ambiguous rows at group weight `0.05`. Replay is training storage only
(`replay_prompt_examples=0`): no past callback enters a future prompt.

## Benchmark configuration

Three paired streams, seeds 0–2, NumPy `2.4.6`, Apple MPS. ICL and RAG use the
`causal_demos` prompt, ICL keeping the three latest reliable callbacks and RAG
retrieving three on a 50/50 metadata and title/body blend. RAG `K=3` is a
disclosed post-selection override; the earlier `K=1` validation does not confirm
it. REINFORCE, RFT, and Online-SDFT all start from the same reset adapter.

See [Baselines](baselines.md), [Methods](methods.md), and
[Evaluation](evaluation.md).
