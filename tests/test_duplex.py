"""Control invariants that matter when inference and motion overlap."""
import sys
from itertools import product
import json
import threading
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from duplex_core import Exchange, EventLog, TimedVideo
from duplex_vlm import parse_decision, supervise
from duplex_contract import executor_feedback
from run_duplex import ExecutionState, policy_wire
from robocasa_tools import MotionExecutor, ToolPlan, Waypoint, make_plan


class DuplexControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bus=Exchange('pick bread')
        self.log=EventLog(Path(self.temp.name)/'events.jsonl')
        self.addCleanup(self.log.file.close)
        self.owner=ExecutionState(self.bus,self.log)

    def command(self, command, tool='vla', revision=0, **extra):
        return {'command':command,'tool':tool,'based_revision':revision,'received_at':time.monotonic(),
                'instruction':'retry grasping bread','args':{},
                'progress':'stalled' if tool in {'grasp','place'} else 'normal',
                'stage':'place' if tool=='place' else 'acquire','grasp_status':'unverified','target':'bread',
                'reason':'Object remained on its support over repeated observations',**extra}

    def test_refine_flushes_immediately_without_waiting_for_inflight_vla(self):
        self.owner.actions.extend([np.ones(12) for _ in range(5)])
        self.owner.vla_pending=True
        guide=self.owner.apply(self.command('refine'),20)
        self.assertEqual(len(self.owner.actions),0)
        self.assertEqual(guide['revision'],1)
        self.assertTrue(self.owner.vla_pending)
        # A response remains outstanding; the current control action is hold.
        np.testing.assert_array_equal(self.owner.hold()[:6],np.zeros(6))

    def test_stale_high_level_command_cannot_restart_cancelled_tool(self):
        self.owner.apply(self.command('hold',tool='hold'),21)
        stale=self.command('switch_tool',tool='grasp',args={'target':'bread'})
        self.assertIsNone(self.owner.apply(stale,22))
        self.assertEqual(self.bus.guide['tool'],'hold')
        self.assertEqual(self.bus.guide['revision'],1)

    def test_hold_preserves_closed_gripper(self):
        self.owner.last_close=1.
        self.owner.apply(self.command('hold',tool='hold'),23)
        self.assertEqual(self.owner.hold()[6],1.)

    def test_inflight_old_vla_and_tool_results_are_rejected_after_refine(self):
        self.owner.apply(self.command('refine'),24)
        self.assertFalse(self.owner.accepts(0,'vla',25))
        self.assertFalse(self.owner.accepts(0,'tool',25))
        self.assertTrue(self.owner.accepts(1,'vla',25))
        self.assertEqual(self.owner.stale_results,2)

    def test_latest_observation_replaces_old_one(self):
        for i in range(100): self.bus.publish({'step':i})
        self.assertEqual(self.bus.read()[0]['step'],99)

    def test_continue_with_new_vla_instruction_applies_and_invalidates_old_work(self):
        self.owner.actions.extend([np.ones(12) for _ in range(5)])
        self.owner.vla_pending=True
        guide=self.owner.apply(self.command('continue'),42)
        self.assertEqual(guide['instruction'],'retry grasping bread')
        self.assertEqual(guide['revision'],1)
        self.assertFalse(self.owner.actions)
        self.assertFalse(self.owner.accepts(0,'vla',43))
        self.assertEqual(self.owner.last_guidance_result['effective_command'],'refine')
        snap={key:np.zeros((16,16,3),np.uint8) for key in ['left','right','wrist']}
        snap['state']=np.zeros(16)
        self.assertEqual(policy_wire(snap,guide['instruction'])['prompt'],'retry grasping bread')

    def test_unchanged_continue_keeps_buffer_and_inflight_work(self):
        self.owner.actions.append(np.ones(12));self.owner.vla_pending=True
        self.owner.apply(self.command('continue',instruction=' pick   bread '),42)
        self.assertEqual(self.bus.guide['revision'],0)
        self.assertEqual(len(self.owner.actions),1)
        self.assertTrue(self.owner.vla_pending)

    def test_continue_tool_change_is_applied_as_switch(self):
        guide=self.owner.apply(self.command('continue',tool='grasp',args={'target':'bread'}),42)
        self.assertEqual(guide['tool'],'grasp')
        self.assertEqual(self.owner.phase,'preparing')
        self.assertEqual(self.owner.last_guidance_result['effective_command'],'switch_tool')

    def test_continue_new_geometric_target_restarts_but_new_description_does_not(self):
        self.owner.apply(self.command('switch_tool',tool='grasp',args={'target':'bread'}),40)
        motion=object();self.owner.motion=motion;self.owner.phase='approach'
        self.owner.apply(self.command('continue',tool='grasp',revision=1,
            instruction='Approach the bread carefully',args={'target':'bread'}),41)
        self.assertIs(self.owner.motion,motion)
        self.assertEqual(self.bus.guide['revision'],1)
        self.assertEqual(self.bus.guide['instruction'],'Approach the bread carefully')
        guide=self.owner.apply(self.command('continue',tool='grasp',revision=1,args={'target':'cheese'}),42)
        self.assertEqual(guide['args'],{'target':'cheese'})
        self.assertEqual(guide['revision'],2)
        self.assertIsNone(self.owner.motion)

    def test_unchanged_continue_cannot_resume_failed_or_finished_tool(self):
        self.owner.apply(self.command('switch_tool',tool='grasp',args={'target':'bread'}),40)
        for phase in ['failed_above_target','motion_done_unverified']:
            self.owner.phase=phase
            self.owner.apply(self.command('continue',tool='grasp',revision=1,args={'target':'bread'}),41)
            self.assertEqual(self.owner.last_guidance_result['status'],'rejected')
            self.assertEqual(self.bus.guide['revision'],1)
            self.assertEqual(self.owner.phase,phase)

    def test_continue_cannot_bypass_required_tool_arguments(self):
        self.owner.apply(self.command('continue',tool='grasp'),42)
        self.assertEqual(self.owner.last_guidance_result['status'],'rejected')
        self.assertEqual(self.bus.guide['tool'],'vla')

    def test_visual_assessments_do_not_override_vlm_continue_refine_or_hold(self):
        assessments=product(['normal','uncertain','stalled','failed'],
            ['unverified','confirmed','failed','lost'],['acquire','transport','place'],
            [('continue','vla'),('refine','vla'),('hold','hold')])
        for progress,grasp_status,stage,(command,tool) in assessments:
            with self.subTest(progress=progress,grasp_status=grasp_status,stage=stage,command=command):
                previous_tool=self.bus.guide['tool']
                revision=self.bus.guide['revision']
                instruction=self.bus.guide['instruction'] if command=='continue' else 'carry bread to bowl'
                self.owner.apply(self.command(command,tool=tool,revision=revision,
                    instruction=instruction,progress=progress,grasp_status=grasp_status,
                    stage=stage,target=''),42)
                self.assertEqual(self.bus.guide['tool'],tool)
                self.assertEqual(self.owner.last_guidance_result['status'],'applied')
                effective=self.owner.last_guidance_result['effective_command']
                self.assertEqual(effective,'switch_tool' if command=='continue' and tool!=previous_tool else command)

    def test_vlm_selects_grasp_recovery_and_clears_previous_actions(self):
        self.owner.actions.extend([np.ones(12) for _ in range(5)])
        guide=self.owner.apply(self.command('switch_tool',tool='grasp',args={'target':'bread'},
            progress='failed',grasp_status='failed'),42)
        self.assertEqual(guide['tool'],'grasp')
        self.assertEqual(guide['args'],{'target':'bread'})
        self.assertFalse(self.owner.actions)
        self.assertEqual(self.owner.phase,'preparing')

    def test_normal_place_after_grasp_is_not_rejected_as_non_recovery(self):
        self.owner.apply(self.command('switch_tool',tool='grasp',args={'target':'bread'}),40)
        self.owner.phase='motion_done_unverified'
        guide=self.owner.apply(self.command('switch_tool',tool='place',revision=1,
            args={'destination':'bowl'},progress='normal',grasp_status='confirmed'),42)
        self.assertEqual(guide['tool'],'place')
        self.assertEqual(guide['args'],{'destination':'bowl'})
        self.assertEqual(self.owner.last_guidance_result['status'],'applied')

    def test_failed_target_history_does_not_force_executor_choice(self):
        self.owner.recent_events.append({'executor':'grasp','args':{'target':'bread'},'status':'failed_above_target'})
        guide=self.owner.apply(self.command('switch_tool',tool='grasp',args={'target':'bread'}),42)
        self.assertEqual(guide['tool'],'grasp')
        guide=self.owner.apply(self.command('refine',revision=1,progress='failed',grasp_status='failed'),43)
        self.assertEqual(guide['tool'],'vla')

    def test_native_gripper_sign_and_rolling_buffer_are_explicit(self):
        self.owner.last_close=-.999212
        action=np.zeros(12);action[6]=.999
        self.owner.actions.append(action)
        feedback=executor_feedback(self.bus.guide,self.owner,[.039,-.039],[0,0,0],20)
        self.assertEqual(feedback['gripper']['commanded'],'open')
        self.assertAlmostEqual(feedback['gripper']['finger_separation_m'],.078)
        self.assertNotIn('measured',feedback['gripper'])
        self.assertNotIn('holding_object',feedback['gripper'])
        self.assertEqual(feedback['action_buffer']['gripper_preview'],['close'])
        self.assertTrue(feedback['action_buffer']['auto_refilled'])
        self.assertFalse(feedback['action_buffer']['indicates_task_progress'])
        self.assertNotIn('last_gripper_close_command',feedback)
        self.owner.last_close=.999
        feedback=executor_feedback(self.bus.guide,self.owner,[.0014,-.0013],[0,0,0],20)
        self.assertEqual(feedback['gripper']['commanded'],'close')
        self.assertAlmostEqual(feedback['gripper']['finger_separation_m'],.0027)
        self.assertNotIn('measured',feedback['gripper'])

    def test_slow_vlm_does_not_block_capture_and_next_request_uses_latest_frame(self):
        entered=threading.Event();release=threading.Event();second=threading.Event()
        requests=[];errors=[]
        def frame(step):
            return {'step':step,'wall_s':step/20,'captured_at_monotonic':time.monotonic(),
                'feedback':{'phase':'executing'},
                **{k:np.full((16,16,3),step,np.uint8) for k in ['left','right','wrist']}}
        def post(url,**kwargs):
            requests.append(kwargs['json'])
            if len(requests)==1:
                entered.set()
                if not release.wait(3): raise TimeoutError('Test did not release model response')
            else:
                second.set();self.bus.stop.set()
            decision={'command':'continue','tool':'vla','instruction':'pick bread','args':{},
                'plan':['grasp bread','verify holding','place bread']}
            response=mock.Mock()
            response.json.return_value={'choices':[{'message':{'content':json.dumps(decision)}}]}
            return response
        def supervise_checked():
            try: supervise(self.bus,'pick bread',self.log)
            except Exception as exc: errors.append(exc)
        self.bus.publish(frame(0))
        with mock.patch('duplex_vlm.httpx.Client') as client:
            client.return_value.__enter__.return_value.post.side_effect=post
            thread=threading.Thread(target=supervise_checked,daemon=True);thread.start()
            try:
                self.assertTrue(entered.wait(3))
                for step in range(1,21): self.bus.publish(frame(step))
                self.assertEqual(self.bus.read()[0]['step'],20)
                self.assertEqual(json.loads(requests[0]['messages'][1]['content'][0]['text'])['observed_step'],0)
                release.set()
                with self.bus.condition:
                    self.assertTrue(self.bus.condition.wait_for(lambda: bool(self.bus.commands),timeout=3))
                command=self.bus.take_commands()[0]
                self.owner.apply(command,20)
                latest=frame(21);latest['feedback']['last_guidance_result']=self.owner.last_guidance_result
                self.bus.publish(latest);self.bus.acknowledge(command['command_id'])
                self.assertTrue(second.wait(3))
            finally:
                release.set();self.bus.stop.set()
                with self.bus.condition:self.bus.condition.notify_all()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        next_input=json.loads(requests[1]['messages'][1]['content'][0]['text'])
        self.assertEqual(next_input['observed_step'],21)
        self.assertEqual(next_input['feedback']['last_guidance_result']['status'],'applied')
        self.assertEqual(requests[0]['thinking'],{'type':'enabled'})
        self.assertEqual(requests[0]['reasoning_effort'],'max')
        self.assertEqual(requests[1]['thinking'],{'type':'disabled'})
        self.assertEqual(requests[1]['reasoning_effort'],'none')
        self.assertEqual(requests[1]['max_tokens'],1024)
        self.assertEqual(next_input['request_kind'],'monitor')
        self.assertEqual(next_input['initial_plan'],['grasp bread','verify holding','place bread'])
        self.log.file.flush()
        events=[json.loads(line) for line in (Path(self.temp.name)/'events.jsonl').read_text().splitlines()]
        first_end=next(event for event in events if event['event']=='vlm_end')
        self.assertEqual(first_end['observed_step'],0)
        self.assertEqual(first_end['current_step'],20)
        self.assertGreaterEqual(first_end['input_frame_age_s'],first_end['latest_frame_age_s'])

    def test_dequeue_does_not_acknowledge_before_new_state_is_published(self):
        command_id=self.bus.command({'command':'hold','tool':'hold'},0,5)
        self.bus.take_commands()
        self.assertLess(self.bus.acknowledged,command_id)
        self.bus.publish({'step':6,'revision':1})
        self.bus.acknowledge(command_id)
        self.assertEqual(self.bus.acknowledged,command_id)

    def test_geometric_action_uses_native_rotated_base_frame(self):
        goal=Waypoint('approach',np.array([.02,0,0]),np.eye(3),1.)
        motion=MotionExecutor(ToolPlan('grasp',[goal],{}))
        origin=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        action=motion.tick({'ee_pos':np.zeros(3),'ee_rot':np.eye(3),
            'controller_origin_rot':origin,'controller_scale':np.array([.05]*3+[.5]*3)})
        self.assertAlmostEqual(action[0],0)
        self.assertLess(action[1],0)
        self.assertEqual(action[6],1)

    def test_completed_motion_is_not_reported_as_verified_grasp(self):
        motion=MotionExecutor(ToolPlan('grasp',[Waypoint('lift',np.zeros(3),np.eye(3),1.)],{}))
        action=motion.tick({'ee_pos':np.zeros(3),'ee_rot':np.eye(3)})
        self.assertIsNone(action)
        self.assertEqual(motion.status,'motion_done_unverified')

    def test_high_initial_arm_does_not_force_unreachable_horizontal_approach(self):
        pose=np.eye(4);pose[:3,:3]=np.diag([1,-1,-1]);pose[2,3]=.9
        client=mock.Mock()
        client.detect_and_segment.return_value=(np.ones((16,16),bool),.9,(0,0,16,16))
        client.compute_grasp.return_value=(pose,.8)
        mount=np.eye(4);mount[2,3]=.097
        snap={'step':0,'left':np.zeros((16,16,3),np.uint8),'depth':np.full((16,16),.8),
            'K':np.array([[16,0,8],[0,16,8],[0,0,1]]),'T_camera':np.eye(4),
            'ee_pos':np.array([-.3,0,1.3]),'ee_rot':np.diag([1,-1,-1]),'T_hand_to_site':mount}
        with mock.patch.dict('os.environ',{'RUN_DIR':self.temp.name}):
            plan=make_plan({'tool':'grasp','revision':1,'args':{'target':'vegetable'}},snap,self.log,client=client)
        above=next(w for w in plan.waypoints if w.phase=='above_target')
        self.assertLess(above.pos[2],1.1)
        self.assertGreater(above.pos[2],.8)
        grasp=next(w for w in plan.waypoints if w.phase=='approach')
        self.assertAlmostEqual(grasp.pos[2],.803)

    def test_video_keeps_wall_clock_pause(self):
        import cv2
        path=Path(self.temp.name)/'video.mp4'
        video=TimedVideo(path,self.log,fps=20)
        video.update(np.zeros((64,64,3),dtype=np.uint8),{'phase':'waiting_for_vlm'})
        video.start(time.monotonic())
        time.sleep(.55)
        report=video.close(time.monotonic())
        cap=cv2.VideoCapture(str(path))
        duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS);cap.release()
        self.assertLess(abs(duration-report['wall_from_first_action_s']),.08)
        self.assertIsNone(report['error'])

    def test_invalid_tool_decision_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_decision('{"command":"switch_tool","tool":"grasp","args":{}}')


if __name__=='__main__': unittest.main()
