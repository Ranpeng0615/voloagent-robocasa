"""Thread-safe exchange between observation, reasoning and execution loops."""
from __future__ import annotations
import collections
import json
import threading
import time
from pathlib import Path


class EventLog:
    def __init__(self, path: Path):
        self.file = path.open('a', buffering=1)
        self.lock = threading.Lock()
        self.start = time.monotonic()

    def emit(self, event: str, **fields):
        with self.lock:
            self.file.write(json.dumps({'event': event, 'monotonic': time.monotonic(),
                'elapsed_s': time.monotonic()-self.start, **fields}, ensure_ascii=False)+'\n')


class Exchange:
    """Latest observation only. Revisions invalidate queued and in-flight work."""
    def __init__(self, instruction: str):
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.snapshot = None
        self.guide = {'revision': 0, 'tool': 'vla', 'instruction': instruction, 'args': {}}
        self.commands = collections.deque()
        self.command_seq = 0
        self.acknowledged = 0

    def publish(self, snapshot):
        with self.condition:
            self.snapshot = snapshot
            self.condition.notify_all()

    def read(self):
        with self.condition:
            return self.snapshot, dict(self.guide)

    def command(self, decision, based_revision, observed_step):
        with self.condition:
            self.command_seq += 1
            self.commands.append({**decision, 'based_revision': based_revision,
                                  'observed_step': observed_step, 'received_at': time.monotonic(),
                                  'command_id': self.command_seq})
            return self.command_seq

    def acknowledge(self, command_id):
        with self.condition:
            self.acknowledged = max(self.acknowledged, command_id)
            self.condition.notify_all()

    def take_commands(self):
        with self.condition:
            result = list(self.commands)
            self.commands.clear()
            return result

    def revise(self, tool, instruction, args):
        with self.condition:
            self.guide = {'revision': self.guide['revision']+1, 'tool': tool,
                          'instruction': instruction, 'args': args}
            self.condition.notify_all()
            return dict(self.guide)


class TimedVideo:
    """Sample the latest simulator frame on the actual wall clock, including waits.

    The renderer is owned by the simulation thread. This worker only copies and
    encodes its last completed frame; it never calls MuJoCo or changes its clock.
    """
    def __init__(self, path, log, fps=20):
        self.path, self.log, self.fps = path, log, fps
        self.lock = threading.Lock()
        self.finish = threading.Event()
        self.frame = None
        self.status = {}
        self.history = collections.deque()
        self.start_time = None
        self.end_time = None
        self.frames = 0
        self.thread = None
        self.error = None

    def update(self, frame, status):
        with self.lock:
            self.frame = frame.copy()
            self.status = dict(status)
            self.history.append((time.monotonic(), self.frame, self.status))

    def start(self, when):
        self.start_time = when
        self.thread = threading.Thread(target=self._record, daemon=True, name='wall-video')
        self.thread.start()

    def _record(self):
        import cv2
        import imageio.v2 as imageio
        writer = None
        try:
            writer = imageio.get_writer(self.path, fps=self.fps, codec='libx264',
                                       quality=7, macro_block_size=2)
            while True:
                target = self.start_time+self.frames/self.fps
                if self.finish.is_set() and target >= self.end_time:
                    break
                delay = target-time.monotonic()
                if delay > 0:
                    self.finish.wait(delay)
                    if self.finish.is_set() and target >= self.end_time:
                        break
                with self.lock:
                    # Keep original capture timestamps even if the encoder
                    # falls behind. Never fill earlier time with future frames.
                    while len(self.history)>1 and self.history[1][0]<=target:
                        self.history.popleft()
                    captured, saved_frame, saved_status = self.history[0]
                    frame = saved_frame.copy()
                    status = dict(saved_status)
                canvas = cv2.copyMakeBorder(frame, 0, 52, 0, 0, cv2.BORDER_CONSTANT)
                labels = [f'Wall {self.frames/self.fps:.2f}s | step {status.get("step",0)} | rev {status.get("revision",0)}',
                          f'{status.get("executor", "hold")} | {status.get("phase", "")}']
                for i, label in enumerate(labels):
                    cv2.putText(canvas, label[:62], (5, frame.shape[0]+18+i*22),
                                cv2.FONT_HERSHEY_SIMPLEX, .37, (255,255,255), 1, cv2.LINE_AA)
                writer.append_data(canvas)
                self.frames += 1
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.log.emit('video_error', error=self.error)
        finally:
            if writer is not None:
                writer.close()

    def close(self, when):
        self.end_time = when
        self.finish.set()
        if self.thread is not None:
            self.thread.join(timeout=20)
            if self.thread.is_alive():
                self.error = 'Video encoder did not finish within 20 seconds'
        return {'fps': self.fps, 'frames': self.frames,
                'duration_s': self.frames/self.fps,
                'wall_from_first_action_s': None if self.start_time is None else when-self.start_time,
                'error': self.error}
