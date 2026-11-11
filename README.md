# VoLoAgent on RoboCasa

Server deployment for the official RoboCasa365 `pi05_pretrain_human300` checkpoint with VoLoAgent and DeepSeek visual monitoring.

Code is versioned here. Models, kitchen assets, environments, caches, API credentials, and run outputs stay outside this repository on the server.

Runtime root: `/root/autodl-tmp/robocasa-repro`.

Upstream revisions are pinned in `configs/upstreams.json`. The model is downloaded from the official `robocasa/robocasa365_checkpoints` Hugging Face repository. Optimizer state and demonstration datasets are not required for inference.

The server proxy is an SSH reverse forward to the operator computer. Before setup:

```bash
ssh -NT -p SSH_PORT -R 127.0.0.1:17897:127.0.0.1:7897 USER@SERVER
```

On the server, clone this repository into the runtime root's `code` directory, source `scripts/env.sh`, then use the deployment scripts. No LIBERO installation is needed.

Deployment and first-episode validation are in progress.
