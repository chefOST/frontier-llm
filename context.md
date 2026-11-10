# Evaluation context

We are evaluating Claude full-video transition runs in W&B project `chefOST/claude - full evals` against MOSCATO annotations from `chefOST-data`.

## Current example

- Source: `CMU__S08_Eggs_7150991-727` (`f6c8545d36644243b9108c865615ec64`)
- Latest eval: `CMU__S08_Eggs_7150991-727__eval_v3` (`ir1umiyg`)
- Code branch: `raf/evaluate-claude-runs`

## Pipeline

1. Normalize Claude object names using MOSCATO aliases (`egg`, `beaten egg`, and `scrambled egg` map to `eggs` for S08).
2. Ignore objects without a GT match.
3. Build alphabetically sorted Claude and per-object/video GT state vocabularies.
4. Use an LLM once to create and freeze the state mapping; attach it to the eval run.
5. Preserve MOSCATO state sets as multi-label sets and remove duplicates.
6. Log precision, recall/detail, Qwen3-Embedding-8B cosine similarity, and timing F1.

## V2 versus V3

- V2 compares Claude’s carried-forward state against every concurrent GT state at each frame.
- V3 matches transitions in time, then compares Claude’s `state_after` only with states newly added by that GT transition. Out-of-vocabulary detail is reported but not scored as false.
- V3 results: precision `0.2857`, recall `0.2833`, cosine `0.3301`, timing F1 `0.5714`.

MOSCATO officially maps `scrambled` and `whisked` to canonical state `beaten`. V3 is a transition-focused diagnostic; it is not directly comparable to V2’s framewise state-set scores.
