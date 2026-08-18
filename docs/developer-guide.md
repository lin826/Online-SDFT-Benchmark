# Developer Guide

[← Main README](../README.md)

How to run your own learning rule on this benchmark, and how to put a different
model behind it.

## Add a learning rule

Every method is a subclass of `OnlineAgent` in
[`online_sdft/methods.py`](../online_sdft/methods.py). Two hooks matter:
`action_probs` decides, and `observe` learns.

```python
class MyAgent(OnlineAgent):
    name = "MyMethod"
    learning_rate = 1e-3
    uses_teacher = False        # True gets you a hindsight teacher forward

    def observe(self, observation, action, teacher_distribution,
                teacher_action, feedback, rng, teacher_observation=None,
                decision_distribution=None, candidate_action=None,
                behavior_distribution=None):
        if feedback.get("observed_user_selection") == "UNKNOWN":
            return                       # censored: no target, no update
        target = my_target(feedback, decision_distribution)
        self.policy.update([(observation.text, target)])
```

Register it in two places: `AGENT_CLASSES` at the bottom of `methods.py`, and
`METHODS` in [`online_sdft/config.py`](../online_sdft/config.py). `create_agent`
dispatches on the class, so add a branch there if your constructor takes a
settings object.

What `observe` receives is the whole causal boundary. `feedback` is the matured
callback for the action you executed, `decision_distribution` is your own
distribution frozen at commit time, and `teacher_distribution` is the
adapter-disabled forward if `uses_teacher` is set. There is no route into the
simulator's `Event`, and the harness calls `observe` only after the callback's
delay has elapsed, so timing is enforced for you rather than by convention.

The policy behind `self.policy` offers `probs` (serve), `base_probs` (frozen
reference), `teacher_probs` (hindsight forward), `update` and `update_support`
(soft-target gradient steps), and `reinforce_update` (action-token policy
gradient). Any of them can back a new rule.

Then run it:

```bash
.venv/bin/python run.py --device cpu --seeds 3
```

Three test files gate a new method: `tests/test_experiment.py` for update
timing, `tests/test_environment.py` for the observation boundary, and
`tests/test_results.py` for the configuration the run records.

## Swap the model

```bash
.venv/bin/python run.py --model-id your-org/your-model --device cpu
```

Three things have to line up:

- **Route decoding.** Scores come from the next-token logits for the codes
  `A`, `B`, `C`, so the tokenizer must encode each as exactly one token.
  `LiquidLLMPolicy` raises at load time when it does not.
- **Adapter placement.** `lora_target_modules` and `lora_layers_to_transform`
  in `OnlineSDFTSettings` name modules in the LFM2 architecture
  (`q_proj`, `k_proj`, `v_proj`, `self_attn.out_proj` on layers 2–12, even).
  A different family needs its own module names, and the parameter count in
  the reported configuration moves with them.
- **The loader.** `LiquidLLMPolicy.__init__` calls `Lfm2ForCausalLM`
  directly. Another family needs `AutoModelForCausalLM` there, or a sibling
  policy class implementing the same `StudentPolicy` protocol.

A one-seed run is enough to confirm the wiring before spending a full sweep:

```bash
.venv/bin/python run.py --model-id your-org/your-model --seeds 1 --device cpu
```

For the on-device counterpart, see the
[Android integration guide](../android-demo/README.md#integrate-online-sdft-into-your-own-app).
