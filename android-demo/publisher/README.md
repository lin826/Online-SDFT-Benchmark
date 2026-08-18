# Online SDFT demo publishers

This module produces three explicitly labeled synthetic publisher apps. They are
separate packages so Android's notification listener reports distinct sources;
none of them impersonates a real chat, calendar, or mail product.

| Flavor | Application ID | Launcher label | Default category |
| --- | --- | --- | --- |
| `chat` | `ai.onlinesdft.publisher.chat` | Chat | `chat` |
| `calendar` | `ai.onlinesdft.publisher.calendar` | Calendar | `calendar` |
| `mail` | `ai.onlinesdft.publisher.mail` | Mail | `mail` |

Build all debug APKs from `android-demo/`:

```sh
./gradlew :publisher:assembleChatDebug \
  :publisher:assembleCalendarDebug \
  :publisher:assembleMailDebug
```

Open each installed app once and grant notification permission. A controller can
then publish a case with a package-targeted broadcast (shown here for Chat):

```sh
adb shell am broadcast \
  -a ai.onlinesdft.publisher.POST_CASE \
  -p ai.onlinesdft.publisher.chat \
  --es case_id case-001 \
  --es title 'Maya · Dinner moved to 7:30' \
  --es body 'Can you confirm the reservation still works?' \
  --es category chat \
  --es importance high \
  --es regime demo-personal \
  --es event_id event-001
```

The accepted extras are `case_id`, `title`, `body`, `category`, `importance`,
`regime`, and `event_id`. They are copied onto the resulting Android notification
so the demo notification listener can correlate its routing decision with the
on-device audit event. Importance values `urgent`, `critical`, `max`, `high`, `4`, and `5`
select the high-alert channel; `min`, `low`, `silent`, `1`, and `2` select the
low-alert channel; all other values use the default channel.
