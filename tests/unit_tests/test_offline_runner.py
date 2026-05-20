import importlib.util

import pytest
import torch

RAY_AVAILABLE = importlib.util.find_spec("ray") is not None


class _Handle:
    def __init__(self, result):
        self._result = result

    def wait(self):
        return self._result


class _Env:
    def __init__(self, result):
        self.result = result
        self.input_channel = None
        self.output_channel = None

    def evaluate(self, input_channel, output_channel):
        self.input_channel = input_channel
        self.output_channel = output_channel
        return _Handle(self.result)


class _Rollout:
    def __init__(self):
        self.input_channel = None
        self.output_channel = None

    def evaluate(self, input_channel, output_channel):
        self.input_channel = input_channel
        self.output_channel = output_channel
        return _Handle(None)


@pytest.mark.skipif(not RAY_AVAILABLE, reason="rlinf runner imports require ray")
def test_offline_runner_evaluate_uses_online_eval_channel_wiring():
    from rlinf.runners.offline_runner import OfflineRunner

    runner = object.__new__(OfflineRunner)
    runner.env_channel = object()
    runner.rollout_channel = object()
    runner.env = _Env([{"success": torch.tensor([1.0, 0.0])}])
    runner.rollout = _Rollout()

    metrics = runner.evaluate()

    assert runner.env.input_channel is runner.rollout_channel
    assert runner.env.output_channel is runner.env_channel
    assert runner.rollout.input_channel is runner.env_channel
    assert runner.rollout.output_channel is runner.rollout_channel
    assert metrics["success"] == 0.5
    assert metrics["num_trajectories"] == 2
