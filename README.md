# Frontier-LLM / ChefOST

Frontier vision-language model baselines for object-state tracking in procedural cooking videos.

## Shared guidance

- [Organization guide](https://github.com/chefOST/.github/blob/main/profile/README.md).
- [Project goals](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md#project-goals) and [paper notes](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md).
- [Evaluation definitions](https://github.com/chefOST/.github/blob/main/docs/Evaluation.md).
- [Contribution rules](https://github.com/chefOST/.github/blob/main/CONTRIBUTING.md) and [agent instructions](AGENTS.md).

## Data

Get annotations and vocabularies from [chefOST-data](https://github.com/chefOST/chefOST-data). Get videos from the shared [R2 bucket](https://github.com/chefOST/chefOST-data#videos).

## Outputs

Log runs and outputs to the `chefOST` W&B projects:

- Experiments: [`claude-experiments`](https://wandb.ai/chefOST/claude-experiments).
- Full evaluations: [`claude - full evals`](https://wandb.ai/chefOST/claude%20-%20full%20evals).

Use the exact destinations in [settings.json](weights_and_biases/settings.json).
