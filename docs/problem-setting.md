# Problem Setting

[← Main README](../README.md)

## A chronological contextual bandit

At decision $t$, the assistant receives a compact context $x_t$, commits one
route $a_t$, and eventually observes only the callback $f_t$ caused by that
route. Outcomes for the two unchosen routes do not exist in the log.

```text
release matured older feedback
  → observe x_t
  → commit a_t
  → freeze the score
  → execute a_t
  → queue f_t until its observation window closes
```

The evaluator can inspect simulator state to grade the frozen action, but the
student and teacher never receive its utility vector, sampled preference, or
utility-optimal route.

## Semantic notification stream

Each event now contains a mobile-style title and body, for example a calendar
meeting starting soon, a routine shipment update, or a production alert. Every
category cycles through low-salience, routine, and time-sensitive scenarios.
Each scenario owns explicit importance, deadline, affinity, semantic-tier, and
useful-horizon metadata, so wording is decision-relevant rather than
decorative. Relative times such as “within 10 minutes” monotonically affect the
latent deadline, and `LATER` loses time-sensitive value when the next digest
would arrive after that useful horizon. Bounded logit-normal sampling avoids
the old endpoint masses caused by clipping Gaussian values—or by floating-point
beta draws—to zero or one.

Names, work projects, merchants, products, and interests come from separate
seeded vocabularies. Stable per-seed entity preferences make repeated people or
topics informative, while busy state changes gradually along the stream.
Context bonuses are scenario-aware: a normal health report is not treated as
an on-call incident, and suggested social posts do not receive the same
off-hours boost as a live call. Routine work and commerce items also pay an
off-hours quiet-period cost that shrinks as real urgency rises.

The text never contains `INTERRUPT`, `LATER`, `ARCHIVE`, a utility value, or a
sampled preference or utility-optimal route. Exact latent floats remain private; the learner receives
only the kind of coarse timing and salience evidence a person could infer from
the original notification itself. The content is templated synthetic text, not
a claim of coverage over real-world writing or private phone data.

## What becomes available when

| Moment | Learner may use | Still unavailable |
| --- | --- | --- |
| Notification posted | title/body, channel/category, local importance estimate, time, regime | busy/interruption state, exact deadline, exact affinity, future interaction, sampled preference, utility-optimal route |
| Decision | compact context and past matured lessons | current callback, sampled preference, utility-optimal route |
| Observation window closes | executed route and its factual callback | outcomes of unchosen routes |
| Future update | same-model rejection-filtered hard target or reliability-conditioned soft target | retroactive change to the completed decision |

## Action-dependent feedback

The simulator samples one hidden preferred route per event from a sharpened
evaluator distribution. Signed utility vectors are shifted by their negative
minimum when needed and normalized into base probabilities $p$. The published
temperature is `0.01`, so the draw uses $q_i\propto p_i^{100}$. It uses that
seeded draw to materialize the synthetic user's behavior, then censors the
result according to the route that actually ran:

| Executed route | Hidden preference: `INTERRUPT` | Hidden preference: `LATER` | Hidden preference: `ARCHIVE` |
| --- | --- | --- | --- |
| `INTERRUPT` | immediate open → observed `INTERRUPT` | delayed read → observed `LATER` | deletion → observed `ARCHIVE` |
| `LATER` | digest read → observed `LATER`; hidden immediate preference stays unavailable | digest read → observed `LATER` | digest deletion → observed `ARCHIVE` |
| `ARCHIVE` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` |

The hidden column headings remain simulator-only. The observable implications
of each delivery surface are public learning guidance: the Online-SDFT teacher
is explicitly told when a surface leaves multiple routes possible, but never
receives the hidden column heading for an event.
Thus a digest read can provide positive evidence even when the hidden evaluator
would have preferred `INTERRUPT`; observable learning reward and
sampled-preference accuracy intentionally measure different things. The
executed-surface outcomes
map to immediate push open `+5`, delayed push open `-1`, push deletion `-2`,
digest open `+0.25`, digest deletion `-1`, and censored `UNKNOWN` `0`. This
mapping never reads the hidden preferred route. ICL and RAG may retain an
unknown interaction as explicitly unlabeled history, but it creates no route
label or gradient update. RFT and Online-SDFT likewise treat it as censored and
perform no update. RFT also rejects an ambiguous digest open or a sampled
teacher route that does not match reliable singleton evidence.

## Prequential evaluation

There is no shuffled train/test split. Every action counts while the model is
still learning, so cold start, delayed feedback, exploration, and preference
drift all enter the result. Feedback that would arrive after the last scored
request remains pending and cannot improve the reported run.

See [Methods](methods.md), the [mobile evidence audit](privileged-information.md),
and [Evaluation](evaluation.md).
