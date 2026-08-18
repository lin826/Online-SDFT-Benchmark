# Methods

[← Main README](../README.md)

The repository centers Online-SDFT under one mobile evidence and timing
contract. The comparison methods are ablations of the same loop, not claims
that each is a state-of-the-art notification system:

| Method | Adaptation |
| --- | --- |
| Base | Frozen Liquid LFM2.5-230M |
| ICL | Adds the latest reliable singleton factual interactions to the prompt |
| RAG | Retrieves similar reliable singleton factual interactions into the prompt |
| REINFORCE | Removes the teacher; one factual-reward action-token LoRA update per known callback |
| RFT | Samples one adapter-disabled teacher route, keeps it only after reliable singleton verification, and trains LoRA on a hard target |
| **Online-SDFT** | Delayed action-token LoRA update using a reliability-conditioned soft target from the same model with its adapter disabled |

## Code boundaries

| Module | Responsibility |
| --- | --- |
| `environment.py` | chronological stream, executed outcome, observation delay, evaluator-only utility |
| `privilege.py` | phone-observable decision snapshot and callback encoding |
| `methods.py` | frozen Liquid base policy, hindsight teacher prompt, memory baselines, and LoRA updates |
| `experiment.py` | release matured feedback → predict → score → execute → queue |
| `reporting.py` | post-run aggregation and figures |

Methods receive `StudentObservation`, never the full simulator `Event`. Given
the same event and executed route, every method receives the identical callback
and reward. They differ only in how they retain or learn from that information.
The observation now includes the original synthetic notification title/body
alongside category, time, regime, and local importance, allowing the LFM to use
semantic content rather than routing from metadata alone.

## Student and teacher

The deployed policy starts from
[`LiquidAI/LFM2.5-230M`](https://huggingface.co/LiquidAI/LFM2.5-230M). Its
next-token logits for `A/B/C` define the three route probabilities. Every
adaptive method—REINFORCE, RFT, and Online-SDFT—uses the same PEFT LoRA
architecture. The frozen base parameters are $\theta_0$, the current adapter
parameters are $\Delta_t$, and the student policy is

$$P_t(a) = \operatorname{softmax}_a\left(\ell_{\theta_0, \Delta_t}(x_t)_{c(a)}\right)$$

where $c(a)$ is the single action token for route $a$. The adapter has rank
`4`, alpha `8`, dropout `0`, and targets `q_proj`, `k_proj`, `v_proj`, and
`self_attn.out_proj` in attention layers 2, 4, 6, 8, 10, and 12. It has 172,032
trainable parameters; the LFM base weights stay frozen, and the adapter is
never merged. One physical model is reused sequentially. Its adapter is
restored to the identical PEFT initialization before every method and seed, so
each adaptive method starts from the same frozen-base policy and capacity.

After the chosen route's callback matures, Online-SDFT uses that same physical
LFM with its adapter disabled. The resulting fixed-base teacher receives the
sealed decision snapshot plus the factual trajectory:

$$Q_t(a) = p_{\theta_0}(a \mid x_t, a_t, s_t, f_t)$$

Here $s_t\in\{\text{INTERRUPT},\text{LATER},\text{ARCHIVE},\text{UNKNOWN}\}$
is the user selection observable from the executed surface. The teacher does
not receive teacher demonstrations, a shadow router, or the dataset answer. Its
prompt explicitly states that a digest open does not resolve `INTERRUPT` versus
`LATER`, and that `UNKNOWN` is censored feedback rather than evidence for
`ARCHIVE`.

The teacher runs at temperature `1.0` with no separate reasoning-token pass;
disabling the adapter prevents student updates from changing it. Let $D_t$ be
the student distribution frozen when the action was committed, $B_t$ the
maximum-entropy distribution over routes compatible with the callback, and
$\Pi_{S_t}(D_t)$ the decision distribution renormalized on public causal
support $S_t$. The selected target is:

When the callback pins one route, so that $\lvert S_t \rvert = 1$:

$$T_t = 0.05\,Q_t + 0.05\,D_t + 0.90\,B_t$$

When a digest open leaves it ambiguous, so that
$S_t = \lbrace \text{INTERRUPT}, \text{LATER} \rbrace$:

$$T_t = \Pi_{S_t}(D_t)$$

An immediate or delayed interrupt open and a deletion from either delivered
surface provide reliable singleton evidence and use the `.05/.05/.90`
teacher/decision/behavior fusion. A digest open remains ambiguous between
`INTERRUPT` and `LATER`; it uses the projected decision distribution with
`0/1/0` fusion weights, so neither the teacher nor an invented one-hot label
resolves that ambiguity. `UNKNOWN` produces no target or update.

Online-SDFT minimizes soft cross-entropy against that target:

$$\mathcal{L}_t = -\sum_a T_t(a) \log P_t(a)$$

The selected optimizer uses learning rate `1e-3`, a 64-record
selection-balanced replay window with recency half-life `32`, batches of at
most eight, two update steps, a four-example warmup, and gradient-norm clipping
at `1.0`. Ambiguous callbacks remain eligible training rows but receive
replay-group weight `0.05`.
The replay window is local training storage, not student context:
`replay_prompt_examples=0`, so serving predictions query no callback history.
The teacher's argmax remains diagnostic; its complete distribution enters the
reliable-singleton target.

Online-SDFT's selected behavior policy begins with epsilon-greedy sampling at
epsilon `0.02`. If the student's largest action probability is at most `0.60`,
it mixes an `INTERRUPT` probe into that baseline with weight
$0.15\,2^{-(t-1)/80}$. No additional argmax taper is applied through decision
160. After that point, the behavior distribution is tapered toward the student
argmax with weight $2^{-(t-160)/5}$ on the sampling distribution. The policy
uses only the current student probabilities and the public decision index; it
cannot read a hidden label or utility.

## Rejection Fine-Tuning baseline

[Rejection Fine-Tuning (RFT)](https://arxiv.org/abs/2605.10674) provides the
hard-target practical-distillation comparison. When a non-`UNKNOWN` callback
matures, RFT obtains the same fixed-base teacher distribution $Q_t$ used by
Online-SDFT by disabling the adapter on the same physical model. The teacher
forward remains at temperature `1.0`; RFT flattens the proposal to temperature
`8.0` and draws exactly one candidate:

$$Q_t^{(8)}(a) = \frac{Q_t(a)^{1/8}}{\sum_b Q_t(b)^{1/8}}, \qquad y_t \sim \operatorname{Categorical}(Q_t^{(8)}), \qquad K = 1$$

The candidate uses an event-keyed deterministic uniform draw with inverse-CDF
sampling, isolated from action, feedback, and replay randomness. It is accepted
only if the public causal
support is a reliable singleton and $y_t$ equals that singleton. Teacher
mismatches are rejected. Every ambiguous digest open is rejected because the
executed surface cannot verify `INTERRUPT` versus `LATER`; `UNKNOWN` is censored
and is not a candidate attempt. Rejected and censored observations contribute
no target, negative example, reward, KL term, replay row, replacement update,
or backfill step.

For an accepted candidate, RFT uses the hard target
$T_t(a)=\mathbf{1}[a=y_t]$ and ordinary one-hot cross-entropy. It never reads
the sampled preference, utility-optimal action, evaluator utilities, regret,
scalar reward, correctness,
or an unchosen outcome. The trace records the sampled candidate, acceptance
decision, rejection reason, and optimizer count; aggregate results report the
attempted and accepted example budget and acceptance rate.

RFT uses the same frozen LFM base, reset rank-4/alpha-8 LoRA architecture,
172,032 trainable parameters, adapter initialization, batch size, update steps,
warmup, and norm cap as Online-SDFT. Its student schedule is independently
fixed at replay size `32`, epsilon-greedy serving with epsilon `0.06`, and
learning rate `7e-4`; it has no replay-recency weighting, uncertainty probe, or
late exploration taper. RFT also spends one adapter-disabled
hindsight-teacher forward on every matured non-`UNKNOWN` callback. It can
perform fewer optimizer steps because rejection is intrinsic to the recipe.
The shared architecture isolates temperature-`8.0` sample-and-filter hard
supervision from Online-SDFT's reliability-conditioned soft target while
retaining the explicitly configured optimizer and behavior-policy differences.

This benchmark is a causal online/hindsight adaptation of offline
rejection-sampling fine-tuning, not a literal reproduction of an offline
teacher-trajectory collection phase with a full environment verifier. See the
original [RFT paper](https://arxiv.org/abs/2308.01825) and
[official implementation](https://github.com/OFA-Sys/gsm8k-ScRel) for that
setting.

## Delayed memory baselines

ICL and RAG do not call the hindsight teacher. They retain the interaction
produced by their own executed route, but only callbacks whose public causal
support identifies one route become prompt demonstrations. Ambiguous digest
opens and censored `UNKNOWN` outcomes remain available for audit but are not
prompted. The selected `causal_demos` format renders each eligible past
notification as a user turn followed by its observed route code as the
assistant turn.

ICL includes the three most recent eligible interactions. RAG retrieves three
eligible interactions using a 50/50 blend of metadata similarity and visible
title/body token similarity. The metadata component combines category, local
importance, circular time-of-day distance, and regime. Exact deadline,
affinity, sampled preference, utility-optimal action, and post-action
selections are never retrieval keys. The canonical benchmark fixes `K=3` for
both memory baselines and `rag_text_weight=0.5`.

REINFORCE has no hindsight teacher. It samples directly from the current
action-token policy $P_t$ with the same reset rank-4/alpha-8 LoRA architecture
and 172,032 trainable parameters used by RFT and Online-SDFT. For executed
action $a_i$ and learner-only factual-outcome reward $\tilde r_i$, its
eight-row objective is

$$\mathcal{L}^{\mathrm{RL}}_B = -\frac{1}{|B|} \sum_{i \in B} \left[ \tilde r_i \log P_i(a_i) + 1.0\, H(P_i) \right], \qquad |B| = 8$$

It applies AdamW at learning rate `1e-4`, uses a fixed zero baseline, and clips
the gradient norm at `1.0`. Every matured known callback is consumed once, and
there is no replay or end-of-horizon batch flush.

Its training scalar comes only from that executed-surface callback: immediate
push open `+5`, delayed push open `-1`, push deletion `-5`, digest open `0`, and
digest deletion `-5`. A censored `UNKNOWN` event is logged but is excluded
before reward mapping, so it produces no gradient row. This learner-only map
uses neither evaluator-only answer nor any unchosen outcome. It is distinct
from the shared map used in the reported observable-reward metric: `+5`, `-1`,
`-2`, `+0.25`, `-1`, and `0` for `UNKNOWN`. Online-SDFT creates no target from
`UNKNOWN`; ICL and RAG may retain the completed interaction as explicitly
unlabeled audit history, but do not place it in a serving prompt. No method
invents a route label or gradient target from `UNKNOWN`.
This holds the causal feedback source and timing constant while ablating the
learning rule.

## Configuration provenance

Online-SDFT and REINFORCE were tuned directly on the reported seeds 0–2 without
a disjoint confirmation set. The REINFORCE screen required pooled exact-action
matches above Base and pooled total regret below Base, then ranked passing
candidates by accuracy first and regret second. RFT's proposal temperature and
learning rate were screened on seed 1200. The checked-in comparison therefore ranks
methods on the streams that selected them.

## Relationship to the SDFT paper

[Shenfeld et al.](https://arxiv.org/abs/2601.19897) condition a same-model
teacher on expert demonstrations. This repository explores a different,
SDFT-inspired mobile application: a teacher conditioned on a completed local
trajectory. Claims from the paper do not automatically transfer to this
setting; the current protocol requires its own benchmark and phone validation.
