# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.dataset_env.dataset_env import DatasetEnv


def _build_dataset(tmp_path):
    frames = []
    for step in range(4):
        frames.append(
            {
                "image": np.full((8, 8, 3), fill_value=step, dtype=np.uint8),
                "state": np.linspace(0, 1, 7, dtype=np.float32) + step,
                "actions": np.full((7,), fill_value=0.5, dtype=np.float32),
                "task": "pick up the block",
            }
        )
    np.save(tmp_path / "episode_000.npy", np.array(frames, dtype=object), allow_pickle=True)


def _build_env_cfg(tmp_path):
    return OmegaConf.create(
        {
            "data_path": str(tmp_path),
            "data_format": "npy",
            "reward_fn": "mae",
            "reward_type": "improvement",
            "error_fn": "mae",
            "residual_penalty": "l1",
            "residual_coef": 0.1,
            "reward_coef": 1.0,
            "action_key": "actions",
            "state_key": "state",
            "camera_name": "image",
            "chunk_size": 2,
            "group_size": 2,
            "auto_reset": False,
            "use_rel_reward": False,
            "ignore_terminations": False,
            "seed": 0,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "/tmp",
            },
        }
    )


class TestOfflineAdapterReward(unittest.TestCase):
    def test_dataset_env_improvement_reward_with_adapter_payload(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _build_dataset(tmp_path)
            env = DatasetEnv(
                cfg=_build_env_cfg(tmp_path),
                num_envs=2,
                seed_offset=0,
                total_num_processes=1,
            )
            env.reset()

            gt_actions = env._gt_actions.copy()
            base_actions = np.zeros_like(gt_actions, dtype=np.float32)
            final_actions = gt_actions.copy()
            delta_actions = final_actions - base_actions

            _, chunk_rewards, _, _, infos_list = env.chunk_step(
                {
                    "actions": final_actions,
                    "base_actions": base_actions,
                    "delta_actions": delta_actions,
                }
            )

            expected_reward = 0.5 - 0.1 * 0.5
            self.assertEqual(chunk_rewards.shape, (2, 2))
            self.assertTrue(
                np.allclose(chunk_rewards[:, -1].numpy(), expected_reward, atol=1e-6)
            )

            episode_info = infos_list[0]["episode"]
            self.assertTrue(
                np.allclose(episode_info["base_mae"].numpy(), 0.5, atol=1e-6)
            )
            self.assertTrue(
                np.allclose(episode_info["final_mae"].numpy(), 0.0, atol=1e-6)
            )
            self.assertTrue(
                np.allclose(episode_info["improvement"].numpy(), 0.5, atol=1e-6)
            )
            self.assertTrue(
                np.allclose(episode_info["delta_norm"].numpy(), 0.5, atol=1e-6)
            )

    def test_prepare_actions_preserves_delta_payload_for_dataset_env(self):
        raw_actions = np.array([[[0.1] * 6 + [0.75]]], dtype=np.float32)
        raw_base_actions = np.array([[[0.2] * 6 + [0.25]]], dtype=np.float32)
        raw_delta_actions = raw_actions - raw_base_actions

        prepared = prepare_actions(
            raw_chunk_actions={
                "actions": raw_actions.copy(),
                "base_actions": raw_base_actions.copy(),
                "delta_actions": raw_delta_actions.copy(),
            },
            env_type="dataset_env",
            model_type="openvla_oft",
            num_action_chunks=1,
            action_dim=7,
            wm_env_type="libero",
        )

        self.assertEqual(prepared["actions"].shape, raw_actions.shape)
        self.assertEqual(prepared["base_actions"].shape, raw_base_actions.shape)
        self.assertAlmostEqual(float(prepared["actions"][0, 0, -1]), -1.0, places=6)
        self.assertAlmostEqual(float(prepared["base_actions"][0, 0, -1]), 1.0, places=6)
        self.assertTrue(np.allclose(prepared["delta_actions"], raw_delta_actions))


if __name__ == "__main__":
    unittest.main()
