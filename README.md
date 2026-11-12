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

## First serial validation

- GPU: RTX PRO 6000 Blackwell Server Edition, about 96 GB.
- Model: official RoboCasa365 `pi05_pretrain_human300`, revision in `configs/validated-versions.json`.
- Three policy cameras: left external, right external, wrist.
- π inference runs on the GPU through JAX. CPU PyTorch is used for helper/dataset imports.
- Monitor: DeepSeek Flash, reasoning effort `max`, every 40 simulation steps.
- Recovery: instruction replanning. Additional geometric grasp/place tools have not been validated in this deployment.
- Online simulator ground-truth state is not sent to the monitor; native success is authoritative.
- The VoLoAgent `libero` CLI profile selects the MuJoCo wire format; no LIBERO environment or assets are installed.
- The original `run_episode.py` mode keeps serial calls. The optional duplex mode below runs independent workers.

The inference patch loads normalization statistics directly from the checkpoint before building the data config. Serving therefore does not require the human demonstration datasets.

First native episode passed: `PickPlaceCounterToStove`, seed 7, "Pick the broccoli from the plate and place it in the pan." Native success requires the object inside the pan and the gripper away from it. Result: **331 steps, 45.41 seconds, within the 50-second limit**. This is one episode, not an aggregate benchmark score.

Dependency versions are captured in `configs/*-constraints.txt`. Third-party code and models retain their upstream licenses.

## Duplex execution with SAM3 and GraspGen

```bash
source /root/autodl-tmp/robocasa-repro/code/scripts/env.sh
python "$REPRO_CODE_DIR/scripts/checkout.py"
python "$REPRO_CODE_DIR/scripts/apply_patches.py"
bash "$REPRO_CODE_DIR/scripts/install_tools.sh"
"$ROBOCASA_ROOT/envs/sim/bin/python" "$REPRO_CODE_DIR/scripts/download_tool_models.py"
"$ROBOCASA_ROOT/envs/sim/bin/python" "$REPRO_CODE_DIR/scripts/run_episode.py" --duplex --task BreadAndCheese --seed 7 --video
```

SAM3 needs licensed access to `facebook/sam3` for a fresh download. Existing local weights are accepted and SHA-256 checked. Checkpoint revisions and paths are in `configs/tool-models.json`. Tool inference uses a separate GPU Torch environment; π keeps its JAX environment.

VLA starts immediately with the native task instruction. The high-level VLM reads the latest three-camera snapshot, short task memory, and executor feedback in parallel. It can continue, refine the VLA instruction, switch executors, hold, or stop. Only one VLM request is in flight, and old observations do not queue. VLA does not wait for the initial VLM decision.

The first valid VLM decision uses DeepSeek Flash with thinking enabled at `max` and establishes an ordered task plan. Later requests keep that plan and compact progress memory, disable thinking (`reasoning_effort=none`), and cap decision output at 1024 tokens. The local bridge preserves this selection instead of forcing every request to max. This reduces repeated reasoning work; it does not make a request accept new frames while it is generating, or guarantee a response deadline.

VLA is the default executor; tools are available when the VLM observes an anomaly or insufficient progress. The VLM chooses recovery, retries, placement after recovery, and return to VLA from images, recent executor events, and each tool's limits. The executor does not require a failure label to accept a tool call, or automatically route a reported failure to grasp/hold. The previous external view is included for visual progress comparison.

The supervisor reports `stage=acquire|transport|place|other`, `target`, and `grasp_status=unverified|confirmed|failed|lost` for context and logging. These assessments do not override its chosen command. The VLM chooses instruction boundaries and verifies holding from images rather than following a mandatory verification-lift sequence. The executor validates command structure, required tool arguments and revision freshness; it does not independently classify grasp success.

The simulation/control thread owns MuJoCo. VLA inference and tool preparation run in separate workers. Commands are applied at control boundaries without waiting for a model response. A changed VLA instruction or tool/target increments a revision, clears pending actions, and invalidates late results from the previous revision. Position holds preserve the last commanded gripper state. RGB, depth, camera calibration and robot proprioception are available; simulator object state and segmentation are not.

`scripts/duplex_contract.py` defines the shared command and feedback semantics:

| Guidance | Execution behavior |
| --- | --- |
| `continue`, same VLA instruction | Keep current buffer and inference. |
| `continue`, changed VLA instruction | Apply as `refine`; clear old actions and reject old inference results. |
| `continue`, changed tool or geometric args | Apply as `switch_tool`, with required arguments validated. |
| `continue`, same geometric tool/args, changed prose | Update its description without restarting the same motion. |
| VLM reports failed/stalled acquisition or unverified holding | Preserve the selected command; the VLM decides whether to hold, refine or switch tools. |
| Unchanged `continue` after tool failure/completion | Reject with explicit feedback; the VLM must verify and select/retry an executor. |

Feedback names the commanded gripper state `open`/`close`, raw finger separation in meters. It supplies no open/closed classification from finger spacing and no synthetic object-possession label; the VLM assesses holding from images. Native `action[6] < 0.5` means open (including `-0.999`); it is not a closure strength. The VLA action queue is a continuously refilled control buffer, not a task-progress queue. Feedback includes actual queued gripper commands and the last guidance application/rejection result. These fields do not prove that an object was grasped.

Grasping reuses VoLoAgent's SAM3 server, `GraspClient`, depth projection, GraspGen sampling and collision filtering. If an external view is occluded, it tries another calibrated external/wrist RGB-D pair. The RoboCasa adapter converts the predicted Panda hand pose to the controlled grip site using the robot's rigid transform. Placement localizes its support with SAM3 and depth. Grasp motion completion is explicitly unverified until the VLM checks the visual result. The current grasp tool opens the gripper during clearance/approach and uses a fixed pose from its preparation snapshot; the VLM is told about the opening behavior so it can avoid restarting grasp on an already held object. Geometric placement currently needs the held-object offset established by a preceding geometric grasp; VLA remains available for other placements.

`whole_body.py` allocates each Cartesian tracking step across 11 robot DOFs: base translation/yaw, torso height and seven arm joints. It uses copied robot Jacobians, bounded damped least squares, joint limits, and native controller-frame/actuator scaling. Arm increments become OSC deltas; base increments become velocity commands with model-derived Coulomb-friction compensation. The base can reposition locally within 35 cm per horizontal axis and 0.5 rad from the tool's starting configuration; it stops during finger closure/release. VLA's native action interface and robot physics stay unchanged. This is local differential IK, not global navigation, cuRobo, or a collision-free whole-body trajectory planner. No simulator object pose/contact state is supplied to the planner.

The sparse-cloud patch preserves measured points when GraspGen's fixed outlier rule would leave fewer than 64. This handles small objects in 256px RGB-D observations without fabricating dense geometry or changing model weights. SAM3 also uses CUDA autocast and a local checkpoint path.

This is asynchronous system execution. Cameras update with simulator steps, independently of the VLM worker. DeepSeek still receives one fixed observation set per request; new frames cannot enter an in-flight request. After command application and feedback publication, the next request takes the latest observation. Logs include the observation's simulator-return timestamp, its age when submitted and when a decision returns, and the latest available frame age. π is the original checkpoint, without RTC training; its asynchronous action chunks use an executed-action prefix offset. Buffer exhaustion produces a visible position hold.

`rollout_realtime.mp4` starts at the first non-hold control action and preserves wall time, including model/tool waits. A separate worker encodes timestamped frames, with no speed-up or removal of idle intervals. `result.json` includes first-action delay and both measured wall/video durations. Execution stops on native success, native step horizon, wall deadline, or a VLM stop, whichever comes first. The wall deadline remains native horizon/control rate plus 20 seconds, including initial task reasoning. `BreadAndCheese` has a 900-step horizon at 20 Hz (65-second wall limit) and requires both foods on the board and the gripper away.

For diagnostics:

```bash
"$ROBOCASA_ROOT/envs/sim/bin/python" -m unittest discover -s "$REPRO_CODE_DIR/tests" -v
"$ROBOCASA_ROOT/envs/sim/bin/python" "$REPRO_CODE_DIR/scripts/analyze_duplex.py" "$RUN_DIR"
"$ROBOCASA_ROOT/envs/sim/bin/python" "$REPRO_CODE_DIR/scripts/check_whole_body.py"
```

`duplex_events.jsonl` records observations, decisions, applied commands, invalidated results, tool evidence and every control step. `analysis.json` measures how many control/motion steps happened during each VLM call. Task results must be read from the native success flag; a VLM stop message is not a successful benchmark result.

## Duplex validation result

Control/regression checks cover action invalidation, command acknowledgement, gripper polarity, `continue` guidance changes, VLM-controlled tool selection, raw gripper feedback, whole-body allocation, failed-tool feedback, concurrent observation publication during a blocked VLM request, camera/controller frames and wall-time recording. The benchmark below predates these changes; no new benchmark result is claimed for them. This historical `SteamInMicrowave` run used code `74e81cc`, native/Python/NumPy seed 7, and tool model seed 7. It attempted broccoli from sink to bowl, bowl to microwave, close door, and start.

Native result: **failed**, 2100 steps, 119.56 seconds within a 125-second wall limit. The video lasts 116.60 seconds versus 116.58 measured seconds from the first action. There were 17 completed VLM calls (mean 6.63 seconds) and 400 VLA calls (mean 74 milliseconds). During one 15.66-second VLM call, 270 non-hold control steps continued. The maximum measured delivery delay after a VLM command arrived was 43.7 milliseconds; this excludes the VLM's own reasoning time.

SAM3 and GraspGen inference, tool preparation and native motion execution ran on the deployed stack. The geometric grasp failed to converge above the recessed sink, with about 5 cm horizontal error, and the VLM returned control to VLA. VLA also did not complete the grasp within the native horizon. The VLM repeatedly allowed continued approach despite little object progress. Thus the asynchronous routing is demonstrated; reliable grasp recovery and completion of this long task are not yet demonstrated. Earlier trials also show VLA-to-tool switching and rejection of an obsolete in-flight VLA result. Earlier trials did not set the global Python/NumPy seeds and are not controlled A/B comparisons.

Run videos, model/tool inputs, decisions and manifests remain under the server runtime root. They are not committed to the public repository.
