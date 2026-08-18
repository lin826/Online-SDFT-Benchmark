# Evaluation and Regret

[← Main README](../README.md)

## Prequential scoring

Every action is scored before its outcome or any resulting update is available.
The run reports accuracy and regret over the complete chronological stream,
including cold start and delayed adaptation.
The checked-in report evaluates all six methods on
`semantic-title-body-sharp-t001` seeds 0–2. Online-SDFT and REINFORCE were tuned
on these same streams, with no disjoint confirmation set, so the report ranks
methods on the streams that selected them. REINFORCE
candidates had to beat Base in pooled exact matches and total regret, then were
ranked by accuracy first and regret second. RFT's temperature and learning rate
were selected on seed 1200.

The current context includes templated mobile-style title/body text. Content
scenarios coarsely influence the latent event properties, enabling semantic
within-category decisions. This tests language sensitivity under
fixed conditions, not generalization to naturally occurring prose; that requires a
separate real or de-identified corpus.

For simulator event $t$, the evaluator has a utility vector $u_t(a)$. Let
$m_t=\min(0,\min_a u_t(a))$. The evaluator converts utilities into nonnegative
weights and normalizes them to base probabilities. It then applies sampling
temperature $\tau=0.01$ and makes one seeded draw:

$$w_t(a) = u_t(a) - m_t, \qquad p_t(a) = \frac{w_t(a)}{\sum_b w_t(b)}$$

$$q_t(a) = \frac{p_t(a)^{1/\tau}}{\sum_b p_t(b)^{1/\tau}}, \quad \tau = 0.01, \qquad y_t \sim \operatorname{Categorical}(q_t)$$

If every weight is zero, the distribution is uniform. Thus an already
nonnegative vector such as $(20,30,50)$ first maps to
$(0.20,0.30,0.50)$ and is then sharpened proportionally to
$(0.20^{100},0.30^{100},0.50^{100})$. The implementation operates in stable
log space and preserves exact zero support.

The sampled preference $y_t$ drives the synthetic user's behavior and online
accuracy, but is never exposed to a method. Utility regret remains a separate,
nonnegative comparison with the utility-optimal route. If the assistant chose
$a_t$:

$$a_t^{\mathrm{opt}} = \arg\max_a u_t(a), \qquad r_t = u_t(a_t^{\mathrm{opt}}) - u_t(a_t)$$

Cumulative regret is $\sum_t r_t$. Online accuracy is the fraction of frozen
actions exactly equal to the hidden sampled preference $y_t$. A different
action receives no accuracy credit because its utility is close or tied; only
regret directly compares the numeric evaluator utilities.

The second reported signal is observation-grounded reward. It is determined by
the factual outcome on the executed delivery surface: immediate push open `+5`,
delayed push open `-1`, push deletion `-2`, digest open `+0.25`, digest deletion
`-1`, and censored `UNKNOWN` `0`. This shared reported map applies to every
method's trace.

For training only, REINFORCE remaps the same matured factual outcomes to `+5`,
`-1`, `-5`, `0`, and `-5`, respectively. `UNKNOWN` is censored before this map
and supplies no gradient row. The learner-only scalar is never substituted
into the reported observable-reward column. Neither mapping compares against
evaluator-only answers, and neither replaces sampled-preference accuracy.

Online-SDFT uses the same public support to condition target reliability.
Reliable singleton callbacks use 0.05 teacher, 0.05 frozen decision, and 0.90
behavior support. For the ambiguous digest open, the frozen decision
distribution is projected and renormalized on `{INTERRUPT, LATER}`; its
teacher/decision/behavior weights are 0/1/0. `UNKNOWN` creates no target. These
weights depend on callback type, not reward or hidden evaluator state.

RFT uses the same delayed teacher view and public support, but converts it into
a rejection-filtered hard lesson. It samples one categorical route at proposal
temperature `8.0` from the same physical model with its adapter disabled for
each matured non-`UNKNOWN` callback, accepts it only
when reliable singleton evidence verifies that route, and otherwise performs no
update. Ambiguous digest opens, teacher mismatches, and `UNKNOWN` never become
training examples. The comparison fixes RFT to Online-SDFT's reset rank-4,
alpha-8 LoRA architecture, 172,032 trainable parameters, adapter
initialization, and adapter-disabled teacher-forward budget. RFT has its own
replay-32, epsilon-`0.06` student schedule and learning rate `7e-4`; it does not
inherit Online-SDFT's replay-recency, probing, or exploration-taper schedule.
Its proposal temperature is also part of the fixed RFT configuration. Its
smaller accepted-example and optimizer-step budget is an intrinsic result of
rejection and is reported separately.

## Fixed memory-baseline configuration

ICL and RAG retain every matured interaction for audit but prompt only reliable
singleton callbacks. Both use the `causal_demos` prompt and
`rag_text_weight=0.5`; ICL keeps the three most recent eligible interactions,
while RAG retrieves three with a fixed `K=3` configuration.

## Fixed adaptive-LoRA configuration

REINFORCE, RFT, and Online-SDFT use the same frozen LFM base and the same PEFT
LoRA architecture: rank `4`, alpha `8`, dropout `0`, and Q, K, V, and
attention-output adapters in the six attention layers. Each trains 172,032
adapter parameters. The same physical model is reused sequentially, and its
adapter is restored to the identical initialization before every method and
seed.

Online-SDFT uses learning rate `1e-3`, replay size 64, batch size eight, two
update steps, a four-example warmup, selection-balanced replay with recency
half-life `32` and ambiguous-group weight `0.05`, and no replay-history examples
in the serving prompt. Its behavior starts with epsilon `0.02`; when the
student's maximum action probability is at most `0.60`, an `INTERRUPT` probe is
mixed in with initial weight `0.15` and half-life `80`. The resulting sampling
distribution receives no additional argmax taper through step 160, then is
tapered toward argmax with half-life `5`. Its hindsight forward uses the same physical model with the
adapter disabled. RFT uses the same adapter architecture but an independent
replay-32, epsilon-`0.06` schedule, learning rate `7e-4`, and temperature-`8.0`
verified hard targets. REINFORCE has no teacher; it uses learning rate `1e-4`
and one action-token LoRA update per batch of eight matured known factual
outcomes, with a fixed zero baseline, entropy coefficient `1.0`, and no replay
or incomplete horizon-batch flush.

## Timing gates

The benchmark is valid only if:

1. the decision-time snapshot is sealed before the student acts;
2. the action and score are frozen before execution;
3. only the executed route produces a callback;
4. the callback contains an observed selection or `UNKNOWN`, never a copied
   dataset answer;
5. the callback is hidden until its event-specific delay expires;
6. the REINFORCE, RFT, and SDFT LoRA updates, and the RFT/SDFT teacher forwards,
   run only after release;
7. causal guidance may expose only the delivery-surface support set, never the
   event's sampled preference or utility-optimal route; and
8. feedback still pending at the horizon is not flushed into reported scores.

The output trace records decision time, feedback availability time, actual
release time, teacher evidence, and lesson status for audit.

Serving draws and simulator draws use common per-seed random streams across
methods. RFT candidate sampling and training replay use separate method-specific
streams, so a sampled candidate or extra update cannot shift a later
exploration draw. REINFORCE still samples from its policy while the other methods
use their declared behavior rule; that behavior-policy difference is intrinsic
to the learning ablation and is recorded in the configuration.

## Measured elsewhere

The simulator covers the learning protocol. Android wall-clock latency, memory,
battery drain, thermal behavior, background-job reliability, encrypted-store
overhead, and student-state swap safety come from the device implementation.

See [Mobile evidence and permission audit](privileged-information.md).
