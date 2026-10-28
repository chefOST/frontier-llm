# Frontier / ChefOST

ChefOST evaluates object-state tracking in procedural cooking videos. Use this repository for frontier vision-language model baselines, including Claude and OpenAI models.

`main` contains the shared project scaffold. Code and saved runs are on [`raf/full-runs`](https://github.com/chefOST/frontier/tree/raf/full-runs/ego4d-pseudo-labels), copied with its original history from `chefOST-data`. Start with that branch's README for API credentials, video inputs and run commands.

## Metrics

For each object, `P_t` is the predicted state set and `T_t` the ground-truth state set at frame `t`.

- Precision: fraction of predicted states that are correct, `|P_t ∩ T_t| / |P_t|`, averaged over frames with predictions.
- Recall (detail): fraction of ground-truth states recovered, `|P_t ∩ T_t| / |T_t|`, averaged over frames where the object is present.

Average per-frame ratios equally; missing predictions give zero recall when GT states exist. Record the dataset, split and sampling; document empty-GT handling and aggregation across objects/videos. Keep published paper results distinct from ChefOST experiments.

## Context

Read the [project goals](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md#project-goals) and the paper notes relevant to your task. Use the [organization guide](https://github.com/chefOST/.github/blob/main/profile/README.md) for shared project context and the [data repository](https://github.com/chefOST/chefOST-data) for shared annotations and vocabularies.

## Experiment tracking

Follow [W&B setup](weights_and_biases/README.md) to configure the appropriate team and project. Verify the destination used by your actual run command before reporting results.

Details: [documentation](docs/README.md). Agents: [instructions](AGENTS.md). Follow the [branch and documentation style](https://github.com/chefOST/.github/blob/main/CONTRIBUTING.md).
