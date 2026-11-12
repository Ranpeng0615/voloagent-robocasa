"""Thin command validation and raw executor feedback for VLM orchestration."""
from __future__ import annotations
import math

PERCEPTION_TOOLS = {'sam3', 'graspgen'}
MOTION_TOOLS = {'move_ee', 'gripper'}
TOOLS = {'vla', 'hold'} | PERCEPTION_TOOLS | MOTION_TOOLS


def clean_instruction(value):
    return ' '.join(value.split())


def vector(value, size, field):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'{field} must contain {size} finite numbers')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError(f'{field} must contain {size} finite numbers')


def validate_tool_args(tool, args):
    if tool in PERCEPTION_TOOLS:
        if not isinstance(args.get('target'), str) or not args['target'].strip():
            raise ValueError('Missing tool argument: target')
        if 'topdown_only' in args and not isinstance(args['topdown_only'], bool):
            raise ValueError('topdown_only must be boolean')
    elif tool == 'gripper':
        if args.get('command') not in {'open', 'close'}:
            raise ValueError('gripper requires command=open|close')
        duration = args.get('duration_s', .5)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError('duration_s must be finite and nonnegative')
    elif tool == 'move_ee':
        ref = args.get('pose_ref')
        if ref is not None and (not isinstance(ref, str) or not (ref == 'current' or ref.startswith('grasp:'))):
            raise ValueError('pose_ref must be current or a returned grasp reference')
        if ref is None and 'position_world' not in args:
            raise ValueError('move_ee requires position_world or pose_ref')
        for field, size in [('position_world', 3), ('quaternion_xyzw', 4), ('offset_world', 3), ('offset_local', 3)]:
            if field in args:
                vector(args[field], size, field)
        if 'quaternion_xyzw' in args and sum(v*v for v in args['quaternion_xyzw']) == 0:
            raise ValueError('quaternion_xyzw must be nonzero')
        if args.get('gripper', 'keep') not in {'keep', 'open', 'close'}:
            raise ValueError('gripper must be keep|open|close')
        if 'timeout_s' in args:
            timeout = args['timeout_s']
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError('timeout_s must be finite and positive')


def validate_decision(value):
    if not isinstance(value, dict) or value.get('command') not in {'continue', 'refine', 'switch_tool', 'call_tool', 'hold', 'stop'}:
        raise ValueError('Invalid command')
    tool = value.get('tool')
    if not isinstance(tool, str) or tool not in TOOLS:
        raise ValueError('Invalid tool')
    if not isinstance(value.get('instruction', ''), str) or not isinstance(value.get('args', {}), dict):
        raise ValueError('Invalid instruction/args')
    if not isinstance(value.get('reason', ''), str):
        raise ValueError('Invalid reason')
    if value['command'] == 'refine' and tool != 'vla':
        raise ValueError('refine is for VLA; use switch_tool for motion tools')
    if value['command'] == 'call_tool':
        if tool not in PERCEPTION_TOOLS:
            raise ValueError('call_tool is for sam3/graspgen queries')
        validate_tool_args(tool, value.get('args', {}))
    elif value['command'] == 'switch_tool':
        validate_tool_args(tool, value.get('args', {}))
    return value


def resolve_decision(value, active, phase):
    """Apply changed guidance; visual assessments never rewrite tool choices."""
    validate_decision(value)
    result = dict(value)
    kind = value['command']
    tool = 'hold' if kind == 'hold' else value['tool']
    instruction = clean_instruction(value.get('instruction', '')) or active['instruction']
    args = value.get('args', {})
    if kind == 'continue':
        if tool == active['tool'] and not args:
            args = active['args']
        if tool != active['tool'] or args != active['args']:
            kind = 'switch_tool'
        elif tool == 'vla' and instruction != clean_instruction(active['instruction']):
            kind = 'refine'
        elif phase.startswith('failed') or phase in {'motion_done_unverified', 'awaiting_initial_guidance'}:
            raise ValueError('No active motion to continue; select or restart an executor')
    if kind == 'switch_tool':
        if tool in PERCEPTION_TOOLS:
            kind = 'call_tool'
        validate_tool_args(tool, args)
    result.update(command=kind, tool=tool, instruction=instruction, args=dict(args))
    return result


def gripper_command(raw_value):
    # RoboCasa GymWrapper: action.gripper_close < .5 -> native -1 (open).
    return 'close' if float(raw_value) >= .5 else 'open'


def executor_feedback(guide, execution, qpos, ee_world, control_hz):
    return {
        'schema_version': 2,
        'executor': guide['tool'], 'phase': execution.phase,
        'gripper': {'commanded': gripper_command(execution.last_close),
            'finger_separation_m': abs(float(qpos[0])-float(qpos[1]))},
        'action_buffer': {'kind': 'rolling_control_buffer', 'queued_steps': len(execution.actions),
            'buffered_sim_s': len(execution.actions)/control_hz,
            'auto_refilled': guide['tool'] == 'vla', 'indicates_task_progress': False,
            'gripper_preview': [gripper_command(action[6]) for action in execution.actions]},
        'vla_pending': execution.vla_pending, 'tool_pending': execution.tool_pending,
        'robot_ee_world': [float(v) for v in ee_world],
        'tool_motion': None if execution.motion is None else dict(execution.motion.tracking_error),
        'latest_tool_result': execution.latest_tool_result,
        'grasp_pose': execution.grasp_pose,
        'mobile_manipulation_available': True,
        'motion_success_is_unverified': True,
        'last_guidance_result': execution.last_guidance_result,
        'last_visual_assessment': execution.last_visual_assessment,
        'recent_executor_events': list(execution.recent_events)}
