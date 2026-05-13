# Publishing to Hugging Face Datasets

This folder is a ready-to-upload Hugging Face dataset repository. Once
you create the empty dataset on the Hub, follow these steps.

## One-time setup

```bash
pip install huggingface_hub
huggingface-cli login           # paste a write token from https://huggingface.co/settings/tokens
```

## Create the dataset repo

On the Hub UI (https://huggingface.co/new-dataset) or via CLI:

```bash
huggingface-cli repo create gbfs-audit-catalogue --type dataset
```

Repo will live at:
<https://huggingface.co/datasets/rohanfosse/gbfs-audit-catalogue>

## Upload

```bash
cd papers/01_gold_standard/huggingface

# Copy the production parquet next to the dataset card.
cp ../../../data/stations_gold_standard_final.parquet .

# Push everything in one command.
huggingface-cli upload rohanfosse/gbfs-audit-catalogue . \
  --repo-type dataset \
  --commit-message "v1.0 release"
```

## Verify

```python
from datasets import load_dataset
gs = load_dataset("rohanfosse/gbfs-audit-catalogue", split="train")
gs.to_pandas().head()
```

## What this gives you

- One-line `load_dataset()` access for any researcher.
- Automatic indexing in **Google Dataset Search** (via the Croissant
  metadata that the Hub generates from the dataset card).
- Versioned releases : tag the parquet at v1.0, release new tags as
  the audit corpus grows.
- Discussions tab for issues and reuse questions.
- Cross-link with the Zenodo concept DOI in the dataset card.

## Tag your release

```bash
huggingface-cli upload rohanfosse/gbfs-audit-catalogue . \
  --repo-type dataset \
  --revision v1.0.0
```

## Updating

To publish v1.1:

```bash
huggingface-cli upload rohanfosse/gbfs-audit-catalogue . \
  --repo-type dataset \
  --commit-message "v1.1 release"
```
