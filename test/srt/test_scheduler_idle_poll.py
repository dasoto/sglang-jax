"""Rank-0 idle throttle of the multi-host scheduler loop (Scheduler._wait_for_requests_if_idle)."""

import time
import unittest

import zmq

from sgl_jax.srt.managers.scheduler import Scheduler


class _FakeScheduler:
  _wait_for_requests_if_idle = Scheduler._wait_for_requests_if_idle

  def __init__(self, idle, poll_ms, recv_from_tokenizer, recv_from_rpc=None):
    self._idle = idle
    self._idle_poll_ms = poll_ms
    self._idle_poller = None
    self.recv_from_tokenizer = recv_from_tokenizer
    self.recv_from_rpc = recv_from_rpc

  def is_fully_idle(self):
    return self._idle


class IdlePollTest(unittest.TestCase):

  def setUp(self):
    self.ctx = zmq.Context()
    self.pull = self.ctx.socket(zmq.PULL)
    self.pull.bind("inproc://idle-poll-test")
    self.push = self.ctx.socket(zmq.PUSH)
    self.push.connect("inproc://idle-poll-test")

  def tearDown(self):
    self.push.close(0)
    self.pull.close(0)
    self.ctx.term()

  def _elapsed(self, sched):
    t0 = time.perf_counter()
    sched._wait_for_requests_if_idle()
    return time.perf_counter() - t0

  def test_idle_waits_for_timeout(self):
    self.assertGreaterEqual(self._elapsed(_FakeScheduler(True, 50, self.pull)), 0.04)

  def test_idle_wakes_up_on_request(self):
    self.push.send(b"req")
    self.assertLess(self._elapsed(_FakeScheduler(True, 5000, self.pull)), 1.0)
    self.assertEqual(self.pull.recv(zmq.NOBLOCK), b"req")  # message left for recv_requests

  def test_busy_scheduler_does_not_wait(self):
    self.assertLess(self._elapsed(_FakeScheduler(False, 5000, self.pull)), 0.1)

  def test_disabled_by_env_value(self):
    self.assertLess(self._elapsed(_FakeScheduler(True, 0, self.pull)), 0.1)


if __name__ == "__main__":
  unittest.main()
