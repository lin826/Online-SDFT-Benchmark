# Mobile Evidence and Permission Audit

[← Main README](../README.md)

## Deployment scope

The live protocol models a first-party Android app, a user-authorized
notification listener, or an OS-integrated/managed-device assistant. It does
not claim that an arbitrary third-party app can silently inspect or reroute all
notifications.

## Fields supplied to the hindsight teacher

| Field | Captured | Practical source | Constraint |
| --- | --- | --- | --- |
| notification title/body | decision time | [`Notification.extras`](https://developer.android.com/reference/android/app/Notification#extras) or first-party payload | notification access or first-party ownership; sensitive content must remain local |
| notification channel/category | decision time | [`StatusBarNotification`](https://developer.android.com/reference/android/service/notification/StatusBarNotification) and notification metadata | notification access or first-party ownership |
| importance estimate | decision time | [`NotificationListenerService.Ranking.getImportance`](https://developer.android.com/reference/android/service/notification/NotificationListenerService.Ranking#getImportance()) plus an optional local content score | ranking access; content score must remain on-device |
| executed route | commit time | assistant's own local decision log | always known to the assistant |
| immediate open, delayed read, or deletion | later callback | [`onNotificationRemoved(..., reason)`](https://developer.android.com/reference/android/service/notification/NotificationListenerService#onNotificationRemoved(android.service.notification.StatusBarNotification,%20android.service.notification.NotificationListenerService.RankingMap,%20int)) plus first-party open timing | notification listener; precise open timing is strongest for first-party notifications |
| first-party open/dismiss | later callback | [`Notification.contentIntent` and `deleteIntent`](https://developer.android.com/reference/android/app/Notification) | only for notifications the app owns |
| digest open, deletion, or timeout | later callback | assistant-owned digest UI and local timer | only for the assistant's digest |
| observed user selection | after callback | deterministic mapping from the factual event: immediate open → `INTERRUPT`, delayed/digest read → `LATER`, deletion → `ARCHIVE` | `UNKNOWN` whenever the executed route cannot expose the preference |

Calendar, contacts, app-usage history, location, and enterprise roster data are
not part of the live protocol. A product may add them only through a separate,
explicit permission or managed integration.

## Fields never supplied

- simulator reward;
- interruption filter or current busy/interruption state;
- exact deadline, deadline-derived urgency, or affinity values (title/body may contain coarse human-readable timing or relevance cues);
- evaluator utility, sampled preference, utility-optimal route, or hidden
  interruptibility float;
- shadow-policy recommendation;
- a hand-written demonstration label;
- outcomes for unexecuted routes; or
- future callbacks before their observation window closes.

## Evidence format

The teacher prompt combines the original notification with a short natural
language callback such as:

```text
Notification:
The notification title is Checkout review starts in 10 minutes. The message
says Maya asked you to join on time. Tap to open the video call. This is a calendar
notification that arrived at 14:50 local time during the weekday period. Its
on-device importance score is 0.88 out of 1.

Observed callback:
The router placed the notification in a later digest. The user opened it from
the digest 120 minutes later. This behavior revealed LATER as the observed
user selection on the executed surface.
```

If the sampled hidden preference were `INTERRUPT` in this example, the same
digest read would still produce `observed_user_selection=LATER`; the code does
not retrieve or serialize that missed preference. This records what the user
did on the digest surface, not what they might have done under an immediate
push. Executing `ARCHIVE` always produces `UNKNOWN`, regardless of the sampled
preference.

The shared observable-reward metric is derived only from the factual outcome on
this executed surface: immediate push open `+5`, delayed push open `-1`, push
deletion `-2`, digest open `+0.25`, digest deletion `-1`, and censored
`UNKNOWN` `0`. For learning only, REINFORCE remaps the same known outcomes to
`+5`, `-1`, `-5`, `0`, and `-5`; `UNKNOWN` is censored before mapping and
cannot become a gradient row. This shaping never changes the shared reported
metric, never compares against the sampled preference or utility-optimal
route, and never enters the SDFT teacher prompt. `UNKNOWN` is still logged and
may be inspected
in the completed trace, but cannot become a route label or gradient update. ICL
and RAG may retain the factual interaction as explicitly unlabeled audit
history, but only reliable singleton callbacks enter a serving prompt.

RFT and Online-SDFT do not use the scalar reward. Their teacher prompt receives
the public delivery-surface semantics in prose: an immediate open, delayed
interrupt read, or deletion has singleton causal support; a digest open remains
compatible with `{INTERRUPT, LATER}`; and `UNKNOWN` provides no target or
update. Both teacher forwards use the same physical LFM as the student with its
LoRA adapter disabled. RFT draws one categorical teacher candidate for each
matured non-`UNKNOWN` callback and accepts a one-hot target only when reliable
singleton support verifies that route. It rejects teacher mismatches and every
ambiguous digest open. Online-SDFT instead retains a reliability-conditioned
soft target.
This guidance prevents either method from mistaking a digest open for definitive
evidence of `LATER` without exposing either evaluator-only answer.

## Production storage and scheduling

A deployment should encrypt the local event queue, bound its retention, expose
a way for the user to erase stored data, and minimize raw-content
retention. This semantic
benchmark requires title/body during inference and hindsight, but a production
system should delete or transform that text as soon as its learning design
allows. Local model-state training should be deferred with Android
[WorkManager constraints](https://developer.android.com/develop/background-work/background-tasks/persistent/getting-started/define-work#work-constraints)
for charging, idle, battery-not-low, and storage-not-low conditions. Thermal
checks and atomic student-state rollback are still application responsibilities.

## Timing enforced in code

```text
t0  snapshot phone-observable metadata
t1  student commits route
t2  evaluator freezes score
t3  route executes
t4  route-specific callback or timeout matures
t5  same physical model, with its LoRA adapter disabled, produces fixed-base Q_t
t6  RFT filters one categorical candidate or SDFT forms its soft target
t7  a causal learner update can affect only later requests
```

[`tests/test_environment.py`](../tests/test_environment.py) checks the field
boundary. [`tests/test_experiment.py`](../tests/test_experiment.py) checks that
teacher calls and updates occur only after the declared delay.
