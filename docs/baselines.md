# Baselines

[← Main README](../README.md)

Distilling hindsight into an adapter is not the only way to keep a model
improving on someone's phone. You could leave the weights alone and put recent
history in the prompt, retrieve similar past episodes instead, skip the teacher
and learn from a scalar, or keep the teacher but demand a verified hard label.
Each is one of the methods below, and each shares the stream,
decision-time inputs, callback matrix, delayed release queue, and evaluator from
[Experiment setup](experiment-setup.md). Only the learning rule differs.

| Method | Adaptation |
| --- | --- |
| Base | Frozen `LiquidAI/LFM2.5-230M` |
| ICL | Prompts the three latest reliable singleton callbacks |
| RAG | Retrieves three, on a 50/50 metadata and title/body similarity blend |
| REINFORCE | No teacher; factual-reward policy gradient on the same LoRA adapter |
| RFT | Samples one teacher route, keeps it only if verified, trains a matched LoRA adapter on a hard target |

**ICL and RAG.** Neither calls the teacher. Both use the `causal_demos`
prompt and keep every matured interaction for audit, but only a callback whose
public support pins one route becomes a demonstration; ambiguous digest opens
and `UNKNOWN` never do. RAG's metadata component blends category, importance,
circular time-of-day distance, and regime; deadline, affinity, sampled
preference, and utility-optimal action are never retrieval keys.

**REINFORCE.** Same frozen base and same reset LoRA adapter, but no teacher: it
samples an action token from its own policy and applies an action-token policy
gradient scaled by the factual reward, with an EMA reward baseline and a small
entropy bonus. Each matured known callback is consumed once and never replayed,
so it isolates what a scalar buys when nothing reranks or verifies it.

**RFT.** For each matured non-`UNKNOWN` callback,
[Rejection Fine-Tuning](https://arxiv.org/abs/2605.10674) softens the same
fixed-initial teacher with sampling temperature `8.0`, draws one route, and
keeps it only when the callback has reliable singleton support *and* the sample
matches it. Mismatches, ambiguous digest opens, and `UNKNOWN` produce nothing.
Accepted rows train with one-hot cross-entropy. Everything else is shared with
Online-SDFT — frozen base, the same 172,032-parameter adapter architecture and
initialization, replay, optimizer shell, warmup, clipping, adapter-disabled
teacher, and teacher-forward budget, so the comparison isolates hard verified
supervision from a reliability-conditioned soft target. Candidate sampling runs
on its own per-seed stream and cannot shift serving, feedback, or replay draws.
This is a causal online adaptation of offline rejection sampling, not a
reproduction of it; see [Yuan et al.](https://arxiv.org/abs/2308.01825) and the
[original implementation](https://github.com/OFA-Sys/gsm8k-ScRel).

Lined up this way they answer one question: given the same delayed,
action-dependent evidence, how much does it matter whether you remember it,
reward it, verify it, or distill it? See [Results](results.md).
