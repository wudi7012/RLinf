# Copyright 2025 The RLinf Authors.
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

import copy
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.algorithms.offline_rl import (
    get_offline_rl_algo_name,
    iql_advantage_weights,
    iql_expectile_loss,
    is_pure_offline_dataset_enabled,
)
from rlinf.config import SupportedModel
from rlinf.data.datasets.libero_offline_rl import (
    build_libero_chunk_offline_dataset_from_cfg,
)
from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedSACFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # SAC-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.entropy_temp = None
        self.conservative_temp = None
        self.demo_buffer = None
        self.alpha_optimizer = None
        self.conservative_alpha_optimizer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))
        self.offline_rl_name = get_offline_rl_algo_name(cfg)
        self.use_pure_offline_dataset = is_pure_offline_dataset_enabled(cfg)

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_sac_components()
        self.soft_update_target_model(tau=1.0)
        if self.use_dsrl:
            self._init_target_shadow()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(
                self.model, mode="default"
            )  # max-autotune-no-cudagraphs
            self.target_model = torch.compile(self.target_model, mode="default")

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Setup model, lr_scheduler, optimizer and grad_scaler."""
        """Add initializing target model logic."""
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        # When precision is null (e.g. Pi0), detect actual dtype from wrapped model
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        self.use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        use_dsrl = self.use_dsrl
        if use_dsrl:
            # DSRL: separate actor/critic encoders into different optimizer groups
            param_filters = {
                "critic": ["critic_image_encoder", "critic_state_encoder", "q_head"]
            }
        else:
            param_filters = {"critic": ["encoders", "encoder", "q_head", "state_proj"]}
        if self.offline_rl_name == "iql":
            param_filters["critic"].append("value_head")
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]

        if self.offline_rl_name != "iql":
            alpha_type = self.cfg.algorithm.entropy_tuning.get(
                "alpha_type", "softplus"
            )
            self.entropy_temp = EntropyTemperature(
                initial_alpha=self.cfg.algorithm.entropy_tuning.get(
                    "initial_alpha", 0.01
                ),
                alpha_type=alpha_type,
                device=self.device,
                dtype=self.torch_dtype,
            )
            if alpha_type != "fixed_alpha":
                self.target_entropy = self.cfg.algorithm.entropy_tuning.get(
                    "target_entropy",
                    -self.cfg.actor.model.action_dim,
                )

                self.alpha_optimizer = torch.optim.Adam(
                    self.entropy_temp.parameters(),
                    lr=self.cfg.algorithm.entropy_tuning.optim.lr,
                )

        if self.offline_rl_name in {"cql", "calql"}:
            cql_cfg = self.cfg.algorithm.offline_rl.get("cql", {})
            conservative_alpha_type = cql_cfg.get("alpha_type", "exp")
            self.conservative_temp = EntropyTemperature(
                initial_alpha=cql_cfg.get("initial_alpha", 1.0),
                alpha_type=conservative_alpha_type,
                device=self.device,
                dtype=self.torch_dtype,
            )
            conservative_optim_cfg = cql_cfg.get("optim", None)
            if (
                conservative_alpha_type != "fixed_alpha"
                and conservative_optim_cfg is not None
                and conservative_optim_cfg.get("lr", 0.0) > 0
            ):
                self.conservative_alpha_optimizer = torch.optim.Adam(
                    self.conservative_temp.parameters(),
                    lr=conservative_optim_cfg.lr,
                )

        self.build_lr_schedulers()

        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.amp.use_grad_scaler
        )

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        if self.alpha_optimizer is not None:
            self.alpha_lr_scheduler = self.build_lr_scheduler(
                self.alpha_optimizer, self.cfg.algorithm.entropy_tuning.optim
            )
        else:
            self.alpha_lr_scheduler = None
        if self.conservative_alpha_optimizer is not None:
            self.conservative_alpha_lr_scheduler = self.build_lr_scheduler(
                self.conservative_alpha_optimizer,
                self.cfg.algorithm.offline_rl.cql.optim,
            )
        else:
            self.conservative_alpha_lr_scheduler = None

    def setup_sac_components(self):
        """Initialize SAC-specific components"""
        # Initialize replay buffer
        seed = self.cfg.actor.get("seed", 1234)
        returns_to_go_gamma = None
        if self.offline_rl_name == "calql":
            returns_to_go_gamma = (
                self.cfg.algorithm.offline_rl.get("calql", {}).get(
                    "returns_to_go_gamma", 1.0
                )
            )
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get(
                "trajectory_format", "pt"
            ),
            returns_to_go_gamma=returns_to_go_gamma,
        )

        min_demo_buffer_size = 0
        if self.cfg.algorithm.get("demo_buffer", None) is not None:
            auto_save_path = self.cfg.algorithm.demo_buffer.get("auto_save_path", None)
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path, f"demo_buffer/rank_{self._rank}"
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.demo_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.demo_buffer.enable_cache,
                cache_size=self.cfg.algorithm.demo_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.demo_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.demo_buffer.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format="pt",
                returns_to_go_gamma=returns_to_go_gamma,
            )
            min_demo_buffer_size = self.cfg.algorithm.demo_buffer.min_buffer_size
            if self.cfg.algorithm.demo_buffer.get("load_path", None) is not None:
                self.demo_buffer.load_checkpoint(
                    self.cfg.algorithm.demo_buffer.load_path,
                    is_distributed=True,
                    local_rank=self._rank,
                    world_size=self._world_size,
                )

        if self.cfg.algorithm.replay_buffer.get("enable_preload", False):
            buffer_dataset_cls = PreloadReplayBufferDataset
        else:
            buffer_dataset_cls = ReplayBufferDataset
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=self.cfg.algorithm.replay_buffer.min_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=self.cfg.algorithm.replay_buffer.get("prefetch_size", 10),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        if self.use_pure_offline_dataset:
            self._preload_pure_offline_replay_buffer()

        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        self.critic_sample_generator = torch.Generator(self.device)
        self.critic_sample_generator.manual_seed(seed)

        self.target_update_type = self.cfg.algorithm.get("target_update_type", "all")
        assert self.target_update_type in ["all", "q_head_only"], (
            f"{self.target_update_type=} is not suppported!"
        )

    def _pure_offline_dataset_cfg(self):
        return self.cfg.algorithm.offline_rl.dataset

    def _slice_env_obs_batch(
        self,
        env_obs: dict[str, Any],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        for key, value in env_obs.items():
            if isinstance(value, torch.Tensor):
                sliced[key] = value[start:end].clone()
            elif isinstance(value, list):
                sliced[key] = list(value[start:end])
            else:
                sliced[key] = copy.deepcopy(value)
        return sliced

    def _transition_obs_to_cpu(
        self, transition_obs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().contiguous()
            for key, value in transition_obs.items()
        }

    def _concat_transition_obs_parts(
        self, obs_parts: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        if not obs_parts:
            return {}
        return {
            key: torch.cat([part[key] for part in obs_parts], dim=0).contiguous()
            for key in obs_parts[0].keys()
        }

    def _build_pure_offline_transition_kwargs(self) -> dict[str, Any]:
        dataset_cfg = self._pure_offline_dataset_cfg()
        sampling_cfg = self.cfg.algorithm.sampling_params

        do_sample = bool(dataset_cfg.get("base_do_sample", sampling_cfg.get("do_sample", True)))
        temperature = float(
            dataset_cfg.get("base_temperature", sampling_cfg.get("temperature_train", 1.0))
        )
        if temperature <= 0:
            do_sample = False
            temperature = 1.0

        kwargs: dict[str, Any] = {"do_sample": do_sample}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs.update(
                {
                    "temperature": temperature,
                    "top_k": int(dataset_cfg.get("base_top_k", sampling_cfg.get("top_k", -1))),
                    "top_p": float(dataset_cfg.get("base_top_p", sampling_cfg.get("top_p", 1.0))),
                    "repetition_penalty": float(
                        dataset_cfg.get(
                            "base_repetition_penalty",
                            sampling_cfg.get("repetition_penalty", 1.0),
                        )
                    ),
                }
            )
        return kwargs

    def _build_pure_offline_trajectory(
        self,
        curr_obs: dict[str, torch.Tensor],
        next_obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        dones: torch.Tensor,
        max_episode_length: int,
    ) -> Trajectory:
        trajectory = Trajectory(
            max_episode_length=max_episode_length,
            model_weights_id="offline_dataset",
            actions=actions.unsqueeze(1).cpu().contiguous(),
            rewards=rewards.unsqueeze(1).cpu().contiguous(),
            terminations=terminations.unsqueeze(1).cpu().contiguous(),
            truncations=truncations.unsqueeze(1).cpu().contiguous(),
            dones=dones.unsqueeze(1).cpu().contiguous(),
            curr_obs={
                key: value.unsqueeze(1).cpu().contiguous()
                for key, value in curr_obs.items()
            },
            next_obs={
                key: value.unsqueeze(1).cpu().contiguous()
                for key, value in next_obs.items()
            },
        )
        return trajectory

    def _preload_pure_offline_replay_buffer(self):
        dataset = build_libero_chunk_offline_dataset_from_cfg(self.cfg)
        dataset_cfg = self._pure_offline_dataset_cfg()
        preprocess_batch_size = int(dataset_cfg.get("preprocess_batch_size", 32))
        shard_by_rank = bool(dataset_cfg.get("shard_by_rank", True))
        transition_kwargs = self._build_pure_offline_transition_kwargs()

        local_episode_count = 0
        local_transition_count = 0
        skipped_episode_count = 0

        self.log_on_first_rank(
            f"Prefilling replay buffer from pure offline LIBERO dataset at {dataset.dataset_root}."
        )
        self.model.eval()
        for episode_offset in range(len(dataset)):
            episode = dataset.load_episode(episode_offset)
            if episode is None:
                skipped_episode_count += 1
                continue

            local_store = (episode_offset % self._world_size == self._rank) if shard_by_rank else True
            curr_obs_parts: list[dict[str, torch.Tensor]] = []
            next_obs_parts: list[dict[str, torch.Tensor]] = []

            num_transitions = int(episode.actions.shape[0])
            for start in range(0, num_transitions, preprocess_batch_size):
                end = min(start + preprocess_batch_size, num_transitions)
                curr_env_obs = self._slice_env_obs_batch(episode.curr_env_obs, start, end)
                next_env_obs = self._slice_env_obs_batch(episode.next_env_obs, start, end)

                with torch.no_grad():
                    curr_transition_obs = self.model.build_transition_obs(
                        env_obs=curr_env_obs,
                        **transition_kwargs,
                    )
                    next_transition_obs = self.model.build_transition_obs(
                        env_obs=next_env_obs,
                        **transition_kwargs,
                    )

                if local_store:
                    curr_obs_parts.append(self._transition_obs_to_cpu(curr_transition_obs))
                    next_obs_parts.append(self._transition_obs_to_cpu(next_transition_obs))

            if not local_store:
                continue

            trajectory = self._build_pure_offline_trajectory(
                curr_obs=self._concat_transition_obs_parts(curr_obs_parts),
                next_obs=self._concat_transition_obs_parts(next_obs_parts),
                actions=episode.actions,
                rewards=episode.rewards,
                terminations=episode.terminations,
                truncations=episode.truncations,
                dones=episode.dones,
                max_episode_length=episode.max_episode_length,
            )
            self.replay_buffer.add_trajectories([trajectory])
            local_episode_count += 1
            local_transition_count += episode.max_episode_length

            if local_episode_count == 1 or local_episode_count % 100 == 0:
                self.log_info(
                    f"Pure offline preload: rank={self._rank} "
                    f"episodes={local_episode_count} transitions={local_transition_count}"
                )

        self.log_info(
            f"Pure offline preload finished on rank={self._rank}: "
            f"episodes={local_episode_count} transitions={local_transition_count} "
            f"skipped={skipped_episode_count}"
        )

    def _build_sac_forward_kwargs(self) -> dict:
        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs["temperature"] = self.cfg.algorithm.sampling_params.temperature_train
        if self.use_dsrl:
            kwargs["train"] = True
        return kwargs

    def _infer_batch_size(self, obs: dict) -> int:
        for value in obs.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, dict):
                return self._infer_batch_size(value)
        raise ValueError("Failed to infer batch size from observation dict.")

    def _repeat_nested_batch(self, batch, repeat_times: int):
        if isinstance(batch, torch.Tensor):
            return batch.repeat_interleave(repeat_times, dim=0)
        if isinstance(batch, dict):
            return {
                key: self._repeat_nested_batch(value, repeat_times)
                for key, value in batch.items()
            }
        raise TypeError(f"Unsupported batch type for repeat: {type(batch)}")

    def _sum_action_logprobs(self, log_pi: torch.Tensor) -> torch.Tensor:
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        return log_pi.sum(dim=-1, keepdim=True)

    def _get_cql_random_action_bounds(self) -> tuple[float, float]:
        cql_cfg = self.cfg.algorithm.offline_rl.get("cql", {})
        if "random_action_low" in cql_cfg and "random_action_high" in cql_cfg:
            return float(cql_cfg.random_action_low), float(cql_cfg.random_action_high)

        adapter_cfg = self.cfg.actor.model.get("adapter", None)
        if adapter_cfg is not None and adapter_cfg.get("enable", False):
            if adapter_cfg.get("offline_rl_action_space", "residual") == "residual":
                residual_bound = float(adapter_cfg.get("residual_bound", 1.0))
                return -residual_bound, residual_bound

        return -1.0, 1.0

    def _aggregate_q_values(self, q_values: torch.Tensor, agg_q: str) -> torch.Tensor:
        if agg_q == "min":
            return torch.min(q_values, dim=1, keepdim=True).values
        if agg_q == "mean":
            return torch.mean(q_values, dim=1, keepdim=True)
        raise NotImplementedError(f"{agg_q=} is not supported!")

    def _sample_policy_actions(
        self,
        model,
        policy_obs: dict,
        repeat_times: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = self._infer_batch_size(policy_obs)
        repeated_obs = (
            self._repeat_nested_batch(policy_obs, repeat_times)
            if repeat_times > 1
            else policy_obs
        )
        actions, log_pi, _ = model(
            forward_type=ForwardType.SAC,
            obs=repeated_obs,
            **self._build_sac_forward_kwargs(),
        )
        log_pi = self._sum_action_logprobs(log_pi)
        action_dim = actions.shape[-1]
        return (
            actions.reshape(batch_size, repeat_times, action_dim),
            log_pi.reshape(batch_size, repeat_times, 1),
        )

    def _init_target_shadow(self):
        """Create persistent float32 shadow of target model parameters.

        bfloat16 has only 7 mantissa bits (ULP ~0.002 at magnitude 0.3).
        With tau=0.005, per-step EMA delta can be smaller than ULP/2, so
        storing back to bf16 each step rounds away the update. The shadow
        keeps the accumulated EMA state in float32 (ULP ~3.6e-8) across
        steps, preventing precision loss.
        """
        self._target_shadow_f32 = {}
        for name, param in self.target_model.named_parameters():
            self._target_shadow_f32[name] = param.data.float().clone()

    def soft_update_target_model(self, tau: Optional[float] = None):
        """Soft update target model parameters.

        For DSRL (bfloat16 models), uses a persistent float32 shadow buffer
        to prevent EMA precision loss. For non-DSRL SAC, uses direct EMA
        on model parameters.
        """
        if tau is None:
            tau = self.cfg.algorithm.tau

        assert self.target_model_initialized

        with torch.no_grad():
            if not hasattr(self, "_target_shadow_f32"):
                # Non-DSRL path (or before shadow init): direct EMA update
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1:
                        if self.target_update_type == "all":
                            target_param.data.mul_(1.0 - tau)
                            target_param.data.add_(online_param.data * tau)
                        else:
                            target_param.data.mul_(0.0)
                            target_param.data.add_(online_param.data)
                    else:
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(online_param.data * tau)
            else:
                # DSRL path: float32 shadow buffer for bf16 precision
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1 and self.target_update_type != "all":
                        shadow = self._target_shadow_f32[name1]
                        shadow.copy_(online_param.data.float())
                        target_param.data.copy_(shadow.to(target_param.data.dtype))
                    else:
                        shadow = self._target_shadow_f32[name1]
                        shadow.mul_(1.0 - tau).add_(
                            online_param.data.float(), alpha=tau
                        )
                        target_param.data.copy_(shadow.to(target_param.data.dtype))

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []

        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.replay_buffer.add_trajectories(recv_list)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_traj = traj.extract_intervene_traj()
                if intervene_traj is not None:
                    intervene_traj_list.append(intervene_traj)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

    def _compute_standard_q_targets(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)

        if use_dsrl:
            num_action_chunks = self.cfg.actor.model.get("num_action_chunks", 1)
            discount = self.cfg.algorithm.gamma**num_action_chunks
            rewards_for_bootstrap = batch["rewards"][:, 0:1].to(self.torch_dtype)
        else:
            discount = self.cfg.algorithm.gamma
            rewards_for_bootstrap = (
                batch["rewards"].sum(dim=-1, keepdim=True).to(self.torch_dtype)
            )
        terminations = batch["terminations"].to(self.torch_dtype)
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=next_obs,
                **self._build_sac_forward_kwargs(),
            )
            next_state_log_pi = self._sum_action_logprobs(next_state_log_pi)

            if not use_crossq:
                dsrl_kwargs = {"train": True} if use_dsrl else {}
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_state_actions,
                    shared_feature=None,
                    **dsrl_kwargs,
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx
                    )

                qf_next_target = self._aggregate_q_values(all_qf_next_target, agg_q)
                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next_target = (
                        qf_next_target - self.entropy_temp.alpha * next_state_log_pi
                    )
                    qf_next_target = qf_next_target.to(dtype=self.torch_dtype)
                if bootstrap_type == "always":
                    target_q_values = rewards_for_bootstrap + discount * qf_next_target
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards_for_bootstrap
                        + (~(terminations.any(dim=-1, keepdim=True)))
                        * discount
                        * qf_next_target
                    )
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

                all_data_q_values = self.model(
                    forward_type=ForwardType.SAC_Q,
                    obs=curr_obs,
                    actions=actions,
                    **dsrl_kwargs,
                )
            else:
                all_data_q_values, all_qf_next = self.model(
                    forward_type=ForwardType.CROSSQ_Q,
                    obs=curr_obs,
                    actions=actions,
                    next_obs=next_obs,
                    next_actions=next_state_actions,
                )
                all_qf_next = all_qf_next.detach()
                qf_next = self._aggregate_q_values(all_qf_next, agg_q)
                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next = qf_next - self.entropy_temp.alpha * next_state_log_pi
                    qf_next = qf_next.to(dtype=self.torch_dtype)
                if bootstrap_type == "always":
                    target_q_values = rewards_for_bootstrap + discount * qf_next
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards_for_bootstrap
                        + (~(terminations.any(dim=-1, keepdim=True)))
                        * discount
                        * qf_next
                    )
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        return target_q_values, all_data_q_values, next_state_log_pi

    def _compute_cql_conservative_loss(
        self, batch, all_data_q_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        cql_cfg = self.cfg.algorithm.offline_rl.get("cql", {})
        n_action_samples = cql_cfg.get("n_action_samples", 10)
        conservative_weight = cql_cfg.get("conservative_weight", 5.0)
        alpha_threshold = cql_cfg.get("alpha_threshold", 10.0)
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        batch_size = self._infer_batch_size(curr_obs)

        curr_policy_actions, curr_log_pi = self._sample_policy_actions(
            self.model, curr_obs, repeat_times=n_action_samples
        )
        next_policy_actions, next_log_pi = self._sample_policy_actions(
            self.model, next_obs, repeat_times=n_action_samples
        )

        value_obs = self._repeat_nested_batch(curr_obs, n_action_samples)
        next_value_obs = self._repeat_nested_batch(next_obs, n_action_samples)
        curr_action_values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=value_obs,
            actions=curr_policy_actions.reshape(batch_size * n_action_samples, -1),
            **({"train": True} if self.use_dsrl else {}),
        )
        next_action_values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=next_value_obs,
            actions=next_policy_actions.reshape(batch_size * n_action_samples, -1),
            **({"train": True} if self.use_dsrl else {}),
        )

        curr_action_values = curr_action_values.view(
            batch_size, n_action_samples, -1
        ).permute(2, 0, 1)
        next_action_values = next_action_values.view(
            batch_size, n_action_samples, -1
        ).permute(2, 0, 1)
        curr_log_pi = curr_log_pi.permute(2, 0, 1)
        next_log_pi = next_log_pi.permute(2, 0, 1)

        random_action_low, random_action_high = self._get_cql_random_action_bounds()
        random_actions = torch.empty(
            batch_size * n_action_samples,
            batch["actions"].shape[-1],
            device=self.device,
            dtype=batch["actions"].dtype,
        ).uniform_(random_action_low, random_action_high)
        random_values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=value_obs,
            actions=random_actions,
            **({"train": True} if self.use_dsrl else {}),
        )
        random_values = random_values.view(batch_size, n_action_samples, -1).permute(
            2, 0, 1
        )
        random_log_prob = np.log(0.5**batch["actions"].shape[-1])

        if self.offline_rl_name == "calql":
            returns_to_go = batch.get("returns_to_go", None)
            if returns_to_go is None:
                raise RuntimeError(
                    "Cal-QL requires returns_to_go in replay batches, but it was missing."
                )
            returns_to_go = returns_to_go.to(curr_action_values.device)
            if returns_to_go.ndim == 1:
                returns_to_go = returns_to_go.unsqueeze(-1)
            returns_to_go = returns_to_go.transpose(0, 1).unsqueeze(-1)
            curr_action_values = torch.maximum(curr_action_values, returns_to_go)
            next_action_values = torch.maximum(next_action_values, returns_to_go)

        target_values = torch.cat(
            [
                curr_action_values - curr_log_pi,
                next_action_values - next_log_pi,
                random_values - random_log_prob,
            ],
            dim=2,
        )
        logsumexp = torch.logsumexp(target_values, dim=2)
        data_values = all_data_q_values.transpose(0, 1)
        conservative_loss_per_q = conservative_weight * (
            (logsumexp - data_values).mean(dim=1) - alpha_threshold
        )

        conservative_alpha = torch.tensor(
            1.0, device=self.device, dtype=all_data_q_values.dtype
        )
        if self.conservative_temp is not None:
            conservative_alpha = self.conservative_temp.compute_alpha().clamp(0, 1e6)

        conservative_loss = (conservative_alpha * conservative_loss_per_q).sum()
        metrics = {
            "cql_conservative_loss": conservative_loss.item(),
            "cql_alpha": conservative_alpha.item(),
            "cql_gap": (logsumexp - data_values).mean().item(),
        }
        return conservative_loss, conservative_loss_per_q, metrics

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        if self.offline_rl_name == "iql":
            curr_obs = batch["curr_obs"]
            next_obs = batch["next_obs"]
            actions = batch["actions"]
            rewards = batch["rewards"].sum(dim=-1, keepdim=True).to(self.torch_dtype)
            terminations = batch["terminations"].to(self.torch_dtype)
            iql_cfg = self.cfg.algorithm.offline_rl.get("iql", {})

            with torch.no_grad():
                target_v = self.model(
                    forward_type=ForwardType.IQL_V, obs=next_obs
                ).to(self.torch_dtype)
                target_q_values = rewards + (
                    ~(terminations.any(dim=-1, keepdim=True))
                ) * self.cfg.algorithm.gamma * target_v
                target_q_on_data = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=curr_obs,
                    actions=actions,
                )
                target_q_min = torch.min(target_q_on_data, dim=-1, keepdim=True).values

            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
            )
            q_loss = F.mse_loss(
                all_data_q_values,
                target_q_values.to(dtype=all_data_q_values.dtype).expand_as(
                    all_data_q_values
                ),
            )
            values = self.model(forward_type=ForwardType.IQL_V, obs=curr_obs)
            v_loss = iql_expectile_loss(
                target_q_min.detach() - values,
                expectile=iql_cfg.get("expectile", 0.7),
            )
            critic_loss = q_loss + v_loss
            return critic_loss, {
                "q_data": all_data_q_values.mean().item(),
                "iql_q_loss": q_loss.item(),
                "iql_v_loss": v_loss.item(),
                "iql_v": values.mean().item(),
            }

        target_q_values, all_data_q_values, _ = self._compute_standard_q_targets(batch)
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        metrics = {"q_data": all_data_q_values.mean().item()}

        if self.offline_rl_name in {"cql", "calql"}:
            conservative_loss, _, conservative_metrics = (
                self._compute_cql_conservative_loss(batch, all_data_q_values)
            )
            critic_loss = critic_loss + conservative_loss
            metrics.update(conservative_metrics)

        return critic_loss, metrics

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        if "actor_agg_q" in self.cfg.algorithm:
            agg_q = self.cfg.algorithm["actor_agg_q"]
        else:
            agg_q = self.cfg.algorithm.get("agg_q", "min")

        curr_obs = batch["curr_obs"]
        if self.offline_rl_name == "iql":
            iql_cfg = self.cfg.algorithm.offline_rl.get("iql", {})
            actions = batch["actions"]
            log_pi = self.model(
                forward_type=ForwardType.IQL_LOGPROB,
                obs=curr_obs,
                actions=actions,
            )
            log_pi = self._sum_action_logprobs(log_pi)
            with torch.no_grad():
                target_q = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=curr_obs,
                    actions=actions,
                )
                target_q = torch.min(target_q, dim=-1, keepdim=True).values
                values = self.model(forward_type=ForwardType.IQL_V, obs=curr_obs)
                weights = iql_advantage_weights(
                    target_q - values,
                    weight_temp=iql_cfg.get("weight_temp", 3.0),
                    max_weight=iql_cfg.get("max_weight", 100.0),
                )
            actor_loss = -(weights * log_pi).mean()
            entropy = -log_pi.mean()
            return actor_loss, entropy, {
                "iql_weight": weights.mean().item(),
                "iql_adv": (target_q - values).mean().item(),
                "iql_logprob": log_pi.mean().item(),
            }

        pi, log_pi, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            **self._build_sac_forward_kwargs(),
        )
        log_pi = self._sum_action_logprobs(log_pi)
        if not use_crossq:
            dsrl_kwargs = {"train": True} if self.use_dsrl else {}
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                shared_feature=None,
                detach_encoder=True,
                **dsrl_kwargs,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                shared_feature=None,
                detach_encoder=True,
            )
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(all_qf_pi.shape[-1])
        }
        qf_pi = self._aggregate_q_values(all_qf_pi, agg_q)
        metrics["q_pi"] = qf_pi.mean().item()
        actor_loss = ((self.entropy_temp.alpha * log_pi) - qf_pi).mean()

        entropy = -log_pi.mean()
        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        if self.offline_rl_name == "iql":
            raise RuntimeError("IQL does not use entropy temperature tuning.")
        curr_obs = batch["curr_obs"]
        with torch.no_grad():
            _, log_pi, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=curr_obs,
                **self._build_sac_forward_kwargs(),
            )
            log_pi = self._sum_action_logprobs(log_pi)

        alpha = self.entropy_temp.compute_alpha()
        alpha_loss = -alpha * (log_pi.mean() + self.target_entropy)
        return alpha_loss

    @Worker.timer("forward_conservative_alpha")
    def forward_conservative_alpha(self, batch):
        if self.offline_rl_name not in {"cql", "calql"}:
            raise RuntimeError(
                "Conservative alpha is only used by CQL / Cal-QL algorithms."
            )
        assert self.conservative_temp is not None
        with torch.no_grad():
            target_q_values, all_data_q_values, _ = self._compute_standard_q_targets(
                batch
            )
            del target_q_values
            _, conservative_loss_per_q, _ = self._compute_cql_conservative_loss(
                batch, all_data_q_values
            )
        conservative_alpha = self.conservative_temp.compute_alpha().clamp(0, 1e6)
        return -(conservative_alpha * conservative_loss_per_q.detach()).mean()

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        # move train_micro_batch_list to device and apply DRQ for critic/actor/alpha passes
        for i, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[i] = batch

        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for batch in train_micro_batch_list:
            critic_loss, critic_metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        }
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )

        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "sac/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            gbs_actor_loss = []
            gbs_entropy = []
            all_actor_metrics = {}
            for batch in train_micro_batch_list:
                actor_loss, entropy, q_metrics = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                gbs_entropy.append(entropy.item())
                append_to_dict(all_actor_metrics, q_metrics)
            all_actor_metrics = {
                f"actor/{key}": np.mean(value)
                for key, value in all_actor_metrics.items()
            }
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            gbs_alpha_loss = [0]
            alpha_grad_norm = 0
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                gbs_alpha_loss = []
                for batch in train_micro_batch_list:
                    alpha_loss = self.forward_alpha(batch) / self.gradient_accumulation
                    alpha_loss.backward()
                    gbs_alpha_loss.append(
                        alpha_loss.item() * self.gradient_accumulation
                    )
                torch.distributed.all_reduce(
                    self.entropy_temp.base_alpha.grad, op=torch.distributed.ReduceOp.AVG
                )
                alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.entropy_temp.base_alpha,
                    self.cfg.algorithm.entropy_tuning.optim.clip_grad,
                )
                self.alpha_optimizer.step()
                if self.alpha_lr_scheduler is not None:
                    self.alpha_lr_scheduler.step()

            gbs_conservative_alpha_loss = [0]
            conservative_alpha_grad_norm = 0
            if self.conservative_alpha_optimizer is not None:
                self.conservative_alpha_optimizer.zero_grad()
                gbs_conservative_alpha_loss = []
                for batch in train_micro_batch_list:
                    conservative_alpha_loss = (
                        self.forward_conservative_alpha(batch)
                        / self.gradient_accumulation
                    )
                    conservative_alpha_loss.backward()
                    gbs_conservative_alpha_loss.append(
                        conservative_alpha_loss.item() * self.gradient_accumulation
                    )
                torch.distributed.all_reduce(
                    self.conservative_temp.base_alpha.grad,
                    op=torch.distributed.ReduceOp.AVG,
                )
                cql_optim_cfg = self.cfg.algorithm.offline_rl.cql.optim
                conservative_alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.conservative_temp.base_alpha,
                    cql_optim_cfg.clip_grad,
                )
                self.conservative_alpha_optimizer.step()
                if self.conservative_alpha_lr_scheduler is not None:
                    self.conservative_alpha_lr_scheduler.step()

            actor_metrics = {
                "sac/actor_loss": np.mean(gbs_actor_loss),
                "actor/lr": self.optimizer.param_groups[0]["lr"],
                "actor/grad_norm": actor_grad_norm,
                "actor/entropy": np.mean(gbs_entropy),
                **all_actor_metrics,
            }
            if self.entropy_temp is not None:
                actor_metrics.update(
                    {
                        "sac/alpha_loss": np.mean(gbs_alpha_loss),
                        "sac/alpha": self.entropy_temp.alpha,
                        "alpha/grad_norm": alpha_grad_norm,
                    }
                )
            if self.conservative_temp is not None:
                actor_metrics.update(
                    {
                        "cql/alpha_loss": np.mean(gbs_conservative_alpha_loss),
                        "cql/alpha": self.conservative_temp.alpha,
                        "cql/alpha_grad_norm": conservative_alpha_grad_norm,
                    }
                )
            metrics_data.update(actor_metrics)
        # Soft update target network
        if (
            self.target_model_initialized
            and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0
        ):
            self.soft_update_target_model()

        return metrics_data

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        if self.demo_buffer is not None:
            demo_buffer_stats = self.demo_buffer.get_stats()
            demo_buffer_stats = {
                f"demo_buffer/{key}": value for key, value in demo_buffer_stats.items()
            }
            append_to_dict(metrics, demo_buffer_stats)
        # Average metrics across updates
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert tensor values to CPU and detach before computing mean
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                # Handle single values
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    @Worker.timer("run_training")
    def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        # Delay actor training until buffer has enough samples
        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
        train_actor_steps = max(min_buffer_size, train_actor_steps)
        train_actor = self.replay_buffer.is_ready(train_actor_steps)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # Save sac components
        # save alpha
        if self.alpha_optimizer is not None:
            alpha_save_path = os.path.join(save_base_path, "sac_components/alpha")
            self._strategy.save_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                save_path=alpha_save_path,
                save_full_model_weights=False,
            )
        if self.conservative_alpha_optimizer is not None:
            cql_alpha_save_path = os.path.join(
                save_base_path, "sac_components/conservative_alpha"
            )
            self._strategy.save_checkpoint(
                model=self.conservative_temp,
                optimizers=self.conservative_alpha_optimizer,
                lr_schedulers=self.conservative_alpha_lr_scheduler,
                save_path=cql_alpha_save_path,
                save_full_model_weights=False,
            )

        # save target model
        target_model_save_path = os.path.join(
            save_base_path, "sac_components/target_model"
        )
        os.makedirs(target_model_save_path, exist_ok=True)
        target_model_state_dict = self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=True
        )
        torch.save(
            target_model_state_dict,
            os.path.join(target_model_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )

        # save replay buffer
        buffer_save_path = os.path.join(
            save_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)

    def load_checkpoint(self, load_base_path):
        # load model
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # load alpha
        if self.alpha_optimizer is not None:
            alpha_load_path = os.path.join(load_base_path, "sac_components/alpha")
            self._strategy.load_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                load_path=alpha_load_path,
            )
        if self.conservative_alpha_optimizer is not None:
            cql_alpha_load_path = os.path.join(
                load_base_path, "sac_components/conservative_alpha"
            )
            self._strategy.load_checkpoint(
                model=self.conservative_temp,
                optimizers=self.conservative_alpha_optimizer,
                lr_schedulers=self.conservative_alpha_lr_scheduler,
                load_path=cql_alpha_load_path,
            )

        # load target model
        target_model_load_path = os.path.join(
            load_base_path, "sac_components/target_model"
        )
        target_model_state_dict = torch.load(
            os.path.join(target_model_load_path, f"checkpoint_rank_{self._rank}.pt")
        )
        self._strategy.load_model_with_state_dict(
            self.target_model,
            target_model_state_dict,
            cpu_offload=False,
            full_state_dict=True,
        )

        # load replay buffer
        buffer_load_path = os.path.join(
            load_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
