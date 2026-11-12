"""Small shared command/feedback contract for the supervisor and executor."""
from __future__ import annotations


def clean_instruction(value):
    return ' '.join(value.split())


def validate_decision(value):
    if not isinstance(value, dict) or value.get('command') not in {'continue','refine','switch_tool','hold','stop'}:
        raise ValueError('Invalid command')
    if value.get('tool') not in {'vla','grasp','place','hold'}:
        raise ValueError('Invalid tool')
    if not isinstance(value.get('instruction',''), str) or not isinstance(value.get('args',{}), dict):
        raise ValueError('Invalid instruction/args')
    if not isinstance(value.get('reason',''),str):
        raise ValueError('Invalid visual evidence')
    if not isinstance(value.get('target',''),str):
        raise ValueError('Invalid recovery target')
    if value.get('progress','uncertain') not in {'normal','uncertain','stalled','failed'}:
        raise ValueError('Invalid progress assessment')
    if value.get('grasp_status','unverified') not in {'unverified','confirmed','failed','lost'}:
        raise ValueError('Invalid grasp assessment')
    if value.get('stage','other') not in {'acquire','transport','place','other'}:
        raise ValueError('Invalid stage')
    if value['command']=='refine' and value['tool']!='vla':
        raise ValueError('refine is for VLA; use switch_tool to restart geometric tools')
    if value['command']=='switch_tool':
        validate_tool_args(value['tool'], value.get('args',{}))
    return value


def validate_tool_args(tool, args):
    field={'grasp':'target','place':'destination'}.get(tool)
    if field and (not isinstance(args.get(field),str) or not args[field].strip()):
        raise ValueError('Missing tool argument: '+field)


def resolve_decision(value, active, phase):
    """Resolve continue with changed guidance; never silently discard changes.

    Geometric tools consume args, VLA consumes instruction. Updating only the
    description of an unchanged geometric motion does not restart that motion.
    Visual assessments are supervisor reports, not executor routing rules.
    """
    validate_decision(value)
    result=dict(value)
    kind=value['command']
    tool='hold' if kind=='hold' else value['tool']
    instruction=clean_instruction(value.get('instruction','')) or active['instruction']
    args=value.get('args',{})
    if kind=='continue':
        if tool==active['tool'] and not args:
            args=active['args']
        if tool!=active['tool'] or args!=active['args']:
            kind='switch_tool'
        elif tool=='vla' and instruction!=clean_instruction(active['instruction']):
            kind='refine'
        elif phase.startswith('failed') or phase in {'motion_done_unverified','awaiting_initial_guidance'}:
            raise ValueError('No active motion to continue; verify outcome and select/refine an executor')
    if kind=='switch_tool':
        validate_tool_args(tool,args)
    result.update(command=kind,tool=tool,instruction=instruction,args=dict(args))
    return result


def gripper_command(raw_value):
    # RoboCasa GymWrapper: action.gripper_close < .5 -> native -1 (open).
    return 'close' if float(raw_value)>=.5 else 'open'


def executor_feedback(guide, execution, qpos, ee_world, control_hz):
    separation=abs(float(qpos[0])-float(qpos[1]))
    return {
        'schema_version':1,
        'executor':guide['tool'], 'phase':execution.phase,
        'gripper':{'commanded':gripper_command(execution.last_close),
            'finger_separation_m':separation},
        'action_buffer':{'kind':'rolling_control_buffer', 'queued_steps':len(execution.actions),
            'buffered_sim_s':len(execution.actions)/control_hz,
            'auto_refilled':guide['tool']=='vla', 'indicates_task_progress':False,
            'gripper_preview':[gripper_command(action[6]) for action in execution.actions]},
        'vla_pending':execution.vla_pending, 'tool_pending':execution.tool_pending,
        'robot_ee_world':[float(v) for v in ee_world],
        'tool_motion':None if execution.motion is None else dict(execution.motion.tracking_error),
        'mobile_manipulation_available':True,
        'motion_success_is_unverified':True,
        'last_guidance_result':execution.last_guidance_result,
        'last_visual_assessment':execution.last_visual_assessment,
        'recent_executor_events':list(execution.recent_events)}
