# VoLoAgent on RoboCasa

Server deployment for the official RoboCasa365 `pi05_pretrain_human300` checkpoint with VoLoAgent and Codex Sol visual monitoring.

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

The default VLM is `gpt-6.1-sol`, reasoning `high`, service tier `fast`, for both planning and monitoring. Run `bash scripts/start_codex_relay.sh` on the authenticated local computer before server episodes. The relay uses the existing Codex CLI login and forwards localhost port 18111 over SSH; credentials remain local. `CODEX_RELAY_PYTHON` selects a local Python with FastAPI, httpx and uvicorn. `CODEX_MODEL_CATALOG` can select a local model catalog. The server's `CODEX_BRIDGE_UPSTREAM` defaults to `http://127.0.0.1:18111`.

Server startup checks relay health before loading the policy, and saves `vlm_config.json`. Visual requests and responses are recorded in each run's `codex_calls/`. Fast is requested explicitly; no task success or latency improvement is claimed by this configuration change. The old DeepSeek bridge remains only for historical reproducibility.

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

The launcher loads the model, verifies an inference call, starts the Codex bridge and VoLoAgent, then runs one native episode. All services it starts are stopped afterward. It writes the result, policy chunks, VLM requests and responses, final images, and video under `$ROBOCASA_ROOT/runs/`.

The wall-clock limit is the native task horizon divided by the native control frequency, plus 20 seconds. It includes task-time VLA and VLM requests and simulation. Model loading/compilation and scene initialization happen before this timer. Video encoding happens afterward. Video shows simulation frames at the native frequency and omits reasoning waits.

## First serial validation (historical DeepSeek configuration)

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

VLA starts immediately with the native task instruction. The VLM receives the latest three-camera snapshot, previous external view, memory and executor feedback in parallel. The first valid VLM call uses max thinking; later calls disable thinking and cap output at 1024 tokens. Each call sees fixed images; new frames cannot enter a running request.

The VLM chooses subgoals and tool order. There is no built-in grasp/place sequence, verification-lift requirement, holding gate, forced recovery switch or finger-spacing classification. The native success check determines task success. This is a custom RoboCasa adapter using the author's perception tools, not the author's unmodified motion executor.

VLA instructions are short imperative action goals, passed verbatim to the policy. Scene descriptions, explanations, grasp assessments and future plans belong in `reason/memory`. For `continue`, the VLM omits `instruction` or returns an empty string to retain the active goal without flushing queued actions. It selects `refine` for an intentional goal/correction change, rather than paraphrasing the same goal each observation. Explicitly changed instructions are still applied; the executor does not silently discard them.

The monitor also receives a timestamped pair of historical external views, selected near ten seconds before the current observation when available. This visual baseline remains across instruction refinements. It helps the VLM distinguish holding an object from approaching a destination. Frame age is context only: no timer marks failure or forces a tool switch. The prompt asks the VLM to compare available tools when progress stalls, including destination localization and pose movement for an already held object. Omitted `tool` fields on `continue/refine/hold/stop` are resolved only where the command already determines the executor; a new tool call/switch still needs an explicit tool.

| Tool | Input and behavior |
| --- | --- |
| `vla` | A language instruction; continues asynchronously until changed or cancelled. |
| `sam3` | `call_tool`, `args.target`; returns SAM3 segmentation and calibrated RGB-D position, without motion. |
| `graspgen` | `call_tool`, `args.target`; returns a converted grasp pose and `pose_ref`, without motion. Optional `topdown_only=true`; no directional filter is forced by the adapter. |
| `move_ee` | `switch_tool`; tracks exactly one requested pose using mobile base, torso and arm. No intermediate waypoints, automatic opening, closing or lifting. |
| `gripper` | `switch_tool`, `args.command=open|close`; operates the gripper while holding position. Optional `duration_s`, default 0.5 seconds. |
| `hold` | Retains position and the last commanded gripper state. |

`move_ee` accepts `position_world=[x,y,z]` in meters, or `pose_ref="current"`, or the exact `grasp:<id>` reference returned by GraspGen. Optional `quaternion_xyzw` sets orientation; otherwise position-only moves retain the current orientation and grasp references use the predicted orientation. `offset_world` and `offset_local` are translations in world and selected-pose axes. Gripper state is preserved unless `args.gripper=open|close` is explicitly supplied. There is no per-motion timeout unless the VLM sets `timeout_s`; the overall task budget still applies. Numerical completion uses 9 mm position and 0.12 rad rotation tolerance and is reported as `motion_done_unverified`, without a grasp-success claim.

`call_tool` submits a read-only perception query without changing the active executor, queued VLA actions or gripper command. One tool preparation/query is pending at once. Results include source observation time, request id and errors in `latest_tool_result`; the latest successful GraspGen pose is retained separately in `grasp_pose`. The VLM decides whether to hold for a query, keep moving, refresh an old pose, execute a pose, close, lift, place or return to VLA. Perception results never start motion by themselves.

`switch_tool` with `sam3/graspgen` is accepted as an alias for the same read-only query. This accommodates a model selecting the correct tool with the alternate command name; it does not select a tool or target for the model. Response validation errors are returned to the next monitor input in `last_response_error` so the model can correct missing arguments instead of repeating a silently rejected request.

Commands validate structure, required arguments and revision freshness. `continue` with changed VLA guidance applies as `refine`; changed motion tool/args apply as `switch_tool`. Changing only prose for an unchanged motion does not restart it. A motion switch clears old actions and invalidates late motion results. A finished motion requires a new explicit motion command. Visual assessments are recorded without overriding the selected command.

Discrete perception submissions/results and executor completion/failure advance `feedback.feedback_version`. VLM responses based on an older version are discarded and immediately reconsidered using the latest observation, without committing their memory/plan updates. The control thread checks the version again at delivery and consumes ready tool results before applying guidance. This prevents an old "query pending" hold from cancelling motion after that query has already completed. Ordinary camera frames and VLA action refills do not advance this version. Current explicit holds remain available, and perception results do not automatically resume VLA or execute grasp poses.

Raw feedback supplies the commanded gripper state and measured finger separation in meters; no open/closed or object-possession classification is derived from separation. Native `action[6] < 0.5` means open, including `-0.999`. Action-buffer length is not task progress. The VLM assesses possession and task progress from images and observed object motion.

SAM3 and GraspGen run in the author's model server. The adapter handles calibrated RGB-D projection and converts the predicted Panda hand pose to RoboCasa's controlled grip site using the robot's rigid transform. No simulator object identities, poses, contacts or segmentation are supplied to perception or the VLM.

`whole_body.py` converts one requested Cartesian tracking step into base, torso and seven arm-joint actions using robot Jacobians, bounded damped least squares, native joint limits and actuator scaling. Numerical speed limits remain; the adapter's extra 35 cm/0.5 rad base-displacement budget is removed. This is a low-level differential IK controller, not global navigation or a collision-free trajectory planner. VLA's native interface and robot physics are unchanged.

The sparse-cloud patch preserves measured points when GraspGen's fixed outlier rule would leave fewer than 64. This handles small objects in 256px RGB-D observations without fabricating dense geometry or changing model weights. SAM3 also uses CUDA autocast and a local checkpoint path.

This is asynchronous system execution. Cameras update with simulator steps, independently of the VLM worker. The VLM still receives one fixed observation set per request; new frames cannot enter an in-flight request. After command application and feedback publication, the next request takes the latest observation. Logs include the observation's simulator-return timestamp, its age when submitted and when a decision returns, and the latest available frame age. π is the original checkpoint, without RTC training; its asynchronous action chunks use an executed-action prefix offset. Buffer exhaustion produces a visible position hold.

`rollout_realtime.mp4` starts at the first non-hold control action and preserves wall time, including model/tool waits. A separate worker encodes timestamped frames, with no speed-up or removal of idle intervals. `result.json` includes first-action delay and both measured wall/video durations. Execution stops on native success, native step horizon, wall deadline, or a VLM stop, whichever comes first. The wall deadline remains native horizon/control rate plus 20 seconds, including initial task reasoning. `BreadAndCheese` has a 900-step horizon at 20 Hz (65-second wall limit) and requires both foods on the board and the gripper away.

For diagnostics:

```bash
"$ROBOCASA_ROOT/envs/sim/bin/python" -m unittest discover -s "$REPRO_CODE_DIR/tests" -v
"$ROBOCASA_ROOT/envs/sim/bin/python" "$REPRO_CODE_DIR/scripts/analyze_duplex.py" "$RUN_DIR"
"$ROBOCASA_ROOT/envs/sim/bin/python" "$REPRO_CODE_DIR/scripts/check_whole_body.py"
```

`duplex_events.jsonl` records observations, decisions, applied commands, invalidated results, tool evidence and every control step. `analysis.json` measures how many control/motion steps happened during each VLM call. Task results must be read from the native success flag; a VLM stop message is not a successful benchmark result.

## Duplex validation result

Control/regression checks cover action invalidation, command acknowledgement, gripper polarity, `continue` guidance changes, VLM-controlled primitive selection, nonblocking perception, raw gripper feedback, whole-body allocation, failed-tool feedback, concurrent observation publication during a blocked VLM request, camera/controller frames and wall-time recording. The benchmark below predates these changes; no new benchmark result is claimed for them. This historical `SteamInMicrowave` run used code `74e81cc`, native/Python/NumPy seed 7, and tool model seed 7. It attempted broccoli from sink to bowl, bowl to microwave, close door, and start.

Native result: **failed**, 2100 steps, 119.56 seconds within a 125-second wall limit. The video lasts 116.60 seconds versus 116.58 measured seconds from the first action. There were 17 completed VLM calls (mean 6.63 seconds) and 400 VLA calls (mean 74 milliseconds). During one 15.66-second VLM call, 270 non-hold control steps continued. The maximum measured delivery delay after a VLM command arrived was 43.7 milliseconds; this excludes the VLM's own reasoning time.

SAM3 and GraspGen inference, tool preparation and native motion execution ran on the deployed stack. The geometric grasp failed to converge above the recessed sink, with about 5 cm horizontal error, and the VLM returned control to VLA. VLA also did not complete the grasp within the native horizon. The VLM repeatedly allowed continued approach despite little object progress. Thus the asynchronous routing is demonstrated; reliable grasp recovery and completion of this long task are not yet demonstrated. Earlier trials also show VLA-to-tool switching and rejection of an obsolete in-flight VLA result. Earlier trials did not set the global Python/NumPy seeds and are not controlled A/B comparisons.

Run videos, model/tool inputs, decisions and manifests remain under the server runtime root. They are not committed to the public repository.
