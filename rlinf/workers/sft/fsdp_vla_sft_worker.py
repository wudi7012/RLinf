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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils import _pytree

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    @staticmethod
    def _resolve_lerobot_dataset_root(data_paths: list[str] | str) -> Path:
        if isinstance(data_paths, Sequence) and not isinstance(
            data_paths, (str, bytes, Path)
        ):
            if len(data_paths) != 1:
                raise ValueError(
                    "OpenVLA-OFT SFT expects a single LeRobot dataset root path."
                )
            dataset_root = Path(data_paths[0])
        else:
            dataset_root = Path(data_paths)

        if (dataset_root / "meta").is_dir() and (dataset_root / "data").is_dir():
            return dataset_root

        if dataset_root.name == "data" and (dataset_root.parent / "meta").is_dir():
            return dataset_root.parent

        raise FileNotFoundError(
            f"Could not resolve a LeRobot dataset root from {dataset_root}. "
            "Expected either a dataset root containing meta/ and data/, "
            "or a .../data directory with a sibling meta/ directory."
        )

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)

        if model_type == SupportedModel.OPENPI:
            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()

        if model_type == SupportedModel.OPENVLA_OFT:
            from torch.utils.data import DataLoader
            from transformers import AutoProcessor

            from rlinf.data.lerobot_sft_dataset import (
                ActionTokenizer,
                LeRobotSFTDataset,
                PaddedCollatorForActionPrediction,
            )

            dataset_root = self._resolve_lerobot_dataset_root(data_paths)
            custom_val_ratio = float(self.cfg.data.get("custom_val_ratio", 0.0))
            if eval_dataset and custom_val_ratio <= 0.0:
                raise ValueError(
                    "OpenVLA-OFT SFT eval requires data.custom_val_ratio > 0.0."
                )

            processor = AutoProcessor.from_pretrained(
                self.cfg.actor.model.model_path,
                trust_remote_code=self.cfg.actor.model.get("trust_remote_code", True),
            )
            action_tokenizer = ActionTokenizer(processor.tokenizer)
            use_wrist_image = self.cfg.actor.model.get("num_images_in_input", 1) > 1
            use_proprio = self.cfg.actor.model.get("use_proprio", False)
            if use_proprio and self.cfg.actor.model.get("implement_version", "rlinf") != "official":
                raise ValueError(
                    "OpenVLA-OFT SFT with proprio is only supported with "
                    "actor.model.implement_version=official."
                )

            dataset = LeRobotSFTDataset(
                dataset_root=dataset_root,
                dataset_name=self.cfg.data.get("dataset_name", "custom_lerobot_libero"),
                action_tokenizer=action_tokenizer,
                base_tokenizer=processor.tokenizer,
                image_transform=processor.image_processor.apply_transform,
                num_action_chunks=self.cfg.actor.model.num_action_chunks,
                train=not eval_dataset,
                use_wrist_image=use_wrist_image,
                use_proprio=use_proprio,
                val_ratio=custom_val_ratio,
                seed=int(self.cfg.actor.seed) + self._rank,
            )
            collator = PaddedCollatorForActionPrediction(
                processor.tokenizer.model_max_length,
                processor.tokenizer.pad_token_id,
                padding_side="right",
            )
            data_loader = DataLoader(
                dataset,
                batch_size=self.micro_batch_size,
                sampler=None,
                collate_fn=collator,
                num_workers=0,
            )
            return data_loader, getattr(dataset, "dataset_statistics", None)

        raise KeyError(
            f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
        )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: dict[str, Any]):
        model_type = SupportedModel(self.cfg.actor.model.model_type)

        if model_type == SupportedModel.OPENPI:
            observation, actions = batch

            register_pytree_dataclasses(observation)
            observation = _pytree.tree_map(
                lambda x: torch.as_tensor(x, device=self.device).contiguous().clone()
                if x is not None
                else x,
                observation,
            )
            actions = actions.to(torch.float32)
            actions = actions.to(self.device)

            with self.amp_context:
                losses = self.model(
                    forward_type=ForwardType.SFT,
                    data={"observation": observation, "actions": actions},
                )

            return losses

        if model_type == SupportedModel.OPENVLA_OFT:
            batch = _pytree.tree_map(
                lambda x: x.to(self.device) if torch.is_tensor(x) else x,
                batch,
            )
            if batch.get("actions", None) is not None:
                batch["actions"] = batch["actions"].to(torch.float32)

            with self.amp_context:
                losses = self.model(forward_type=ForwardType.SFT, batch=batch)

            return losses

        raise KeyError(
            f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
        )
