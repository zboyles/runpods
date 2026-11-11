# Runpod workers

Serverless worker configurations for [Runpod](https://www.runpod.io), one folder each.

| Folder | What it does |
| --- | --- |
| [`rvc-training/`](rvc-training/) | Trains an RVC voice model with Applio: a dataset on a network volume in, a `.pth` + `.index` out. |

Each folder has its own `Dockerfile` (build context: this repo's root, as
Runpod's GitHub builder uses), a `handler.py`, and a README with the endpoint
settings. Credentials never go in the repo: clients read a git-ignored `.env`,
and endpoints use Runpod secrets.
