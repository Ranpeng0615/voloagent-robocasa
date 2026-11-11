"""Control invariants that matter when inference and motion overlap."""
import sys
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from duplex_core import Exchange, EventLog, TimedVideo
from duplex_vlm import parse_decision
from run_duplex import ExecutionState
from robocasa_tools import MotionExecutor, ToolPlan, Waypoint


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
                'instruction':'retry grasping bread','args':{},**extra}

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
