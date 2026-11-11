# VoLoAgent on RoboCasa

Server deployment for the official RoboCasa365 `pi05_pretrain_human300` checkpoint with VoLoAgent and DeepSeek visual monitoring.

Code is versioned here. Models, kitchen assets, environments, caches, API credentials, and run outputs stay outside this repository on the server.

Runtime root: `/root/autodl-tmp/robocasa-repro`.

Upstream revisions are pinned in `configs/upstreams.json`. The model is downloaded from the official `robocasa/robocasa365_checkpoints` Hugging Face repository. Optimizer state and demonstration datasets are not required for inference.

The server proxy is an SSH reverse forward to the operator computer. Before setup:

```bash
ssh -NT -p SSH_PORT -R 127.0.0.1:17897:127.0.0.1:7897 USER@SERVER
```

On the server, clone this repository into the runtime root's `code` directory. No LIBERO installation is needed.

## Setup

The server needs Python/uv, Git, curl, and NVIDIA EGL libraries. The validated server has Python and uv under `/root/miniconda3/bin`.

```bash
cd /root/autodl-tmp/robocasa-repro
git clone https://github.com/Ranpeng0615/voloagent-robocasa.git code
source code/scripts/env.sh
python code/scripts/checkout.py
bash code/scripts/install_runtime.sh
envs/sim/bin/python code/scripts/download_model.py
envs/sim/bin/python code/scripts/download_assets.py
```

The DeepSeek API key belongs in `$ROBOCASA_ROOT/secrets/deepseek.key`, outside the code checkout. `DEEPSEEK_KEY_FILE` can select another file.

For large public files, accessible CDN hosts can use direct connections while origins that need the VPN use the SSH proxy:

```bash
export NO_PROXY="$NO_PROXY,hf-mirror.com,cas-bridge.xethub.hf.co,us.aws.cdn.hf.co,public.boxcloud.com,dl.boxcloud.com"
export no_proxy="$NO_PROXY"
REPRO_HF_DOWNLOAD_BASE_URL=https://hf-mirror.com envs/sim/bin/python code/scripts/download_model.py
```

Model revision metadata comes from the official Hugging Face repository. Model weights are verified against its SHA-256 values; kitchen archives are checked during ZIP extraction.

## Run one task

```bash
source /root/autodl-tmp/robocasa-repro/code/scripts/env.sh
python "$REPRO_CODE_DIR/scripts/configure_macros.py"
"$ROBOCASA_ROOT/envs/sim/bin/python" "$REPRO_CODE_DIR/scripts/run_episode.py" --task PickPlaceCounterToStove --seed 7 --video
```

The launcher loads the model, verifies an inference call, starts DeepSeek and VoLoAgent, then runs one native episode. All services it starts are stopped afterward. It writes the result, policy chunks, VLM requests and responses, final images, and video under `$ROBOCASA_ROOT/runs/`.

The wall-clock limit is the native task horizon divided by the native control frequency, plus 20 seconds. It includes task-time VLA and VLM requests and simulation. Model loading/compilation and scene initialization happen before this timer. Video encoding happens afterward. Video shows simulation frames at the native frequency and omits reasoning waits.

## Validated configuration

- GPU: RTX PRO 6000 Blackwell Server Edition, about 96 GB.
- Model: official RoboCasa365 `pi05_pretrain_human300`, revision in `configs/validated-versions.json`.
- Three policy cameras: left external, right external, wrist.
- π inference runs on the GPU through JAX. CPU PyTorch is used for helper/dataset imports.
- Monitor: DeepSeek Flash, reasoning effort `max`, every 40 simulation steps.
- Recovery: instruction replanning. Additional geometric grasp/place tools have not been validated in this deployment.
- Online simulator ground-truth state is not sent to the monitor; native success is authoritative.
- The VoLoAgent `libero` CLI profile selects the MuJoCo wire format; no LIBERO environment or assets are installed.
- Calls remain serial: monitoring, policy inference, then simulation execution.

The inference patch loads normalization statistics directly from the checkpoint before building the data config. Serving therefore does not require the human demonstration datasets.

First native episode passed: `PickPlaceCounterToStove`, seed 7, "Pick the broccoli from the plate and place it in the pan." Native success requires the object inside the pan and the gripper away from it. Result: **331 steps, 45.41 seconds, within the 50-second limit**. This is one episode, not an aggregate benchmark score.

Dependency versions are captured in `configs/*-constraints.txt`. Third-party code and models retain their upstream licenses.
