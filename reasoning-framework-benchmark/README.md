# Reasoning Framework Benchmark — GitHub Pilot

This branch contains an executable pilot for the Reasoning Framework experiment.

## Conditions

The same target model is tested under four conditions:

- `BASELINE` — no reasoning controller.
- `COMPACT` — Problem → First Principle → Mechanism → Evidence → Solution.
- `FULL` — Observe → Diagnose → Derive → Hypothesize → Predict → Test → Revise → Engineer.
- `ADAPTIVE` — a routing call selects DIRECT, COMPACT, or FULL before answering.

## Pilot cases

Six representative cases are included:

- RF-A02: correlation / employee training
- RF-B01: competing mechanisms / mobile battery drain
- RF-C01: sequential contradictory evidence / memory-leak diagnosis
- RF-D01: false first principle / Kafka requirement
- RF-E04: risk, reversibility, and second-order effects / fraud threshold
- RF-F01: trivial arithmetic / mode-selection control

## One-time setup

Add a GitHub Actions repository secret named `OPENAI_API_KEY`.

Do not commit an API key to the repository.

Then open **Actions → Reasoning Benchmark Pilot → Run workflow** and choose:

- target model;
- evaluator model.

For a first plumbing run the evaluator can equal the target model. For a stronger experiment, use an independent evaluator model when practical.

## Outputs

The workflow uploads an artifact containing:

- `pilot_report.json`
- `pilot_runs.jsonl`
- `pilot_pairwise.jsonl`

The report includes mean blinded outcome scores, mean protocol-behavior scores, target-model token/latency cost, violation labels, adaptive routing choices, and pairwise wins/ties/losses against baseline.

## Experimental hygiene

- Calls are stateless (`store=False`) and do not inherit this ChatGPT conversation.
- Evaluators are not told which condition generated a response.
- Sequential RF-C01 turns are judged only against evidence available up to that turn.
- The current OpenAI Responses API does not expose a seed parameter, so repeated runs should be treated as stochastic replicates rather than deterministic seeded generations.
- This six-case pilot is a plumbing and directional-signal experiment, not sufficient evidence to validate the framework. The validated 30-case v0.2 suite should follow after the pilot succeeds.

## Local execution

```bash
cd reasoning-framework-benchmark
python -m pip install -r requirements.txt
export OPENAI_API_KEY=...
export OPENAI_TARGET_MODEL=gpt-5.6-sol
export OPENAI_JUDGE_MODEL=gpt-5.6-sol
python pilot.py
```

Results are written to `reasoning-framework-benchmark/results/`.
