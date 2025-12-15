# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=2)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "response_mask"]
        # select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")
        
        # Add proposer keys for joint training
        has_proposer_data = "proposer_log_probs" in data.batch and "proposer_advantages" in data.batch
        if has_proposer_data:
            select_keys.extend(["proposer_log_probs", "proposer_input_ids", "proposer_attention_mask", 
                              "proposer_advantages", "proposer_response_mask"])

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
        print("data size: ", len(data), len(mini_batches))
        print('Global batch Size per device:', self.config.global_batch_size_per_device)
        print('micro batch size per device for update:', self.config.micro_batch_size_per_device_for_update)
        print('Gradient accumulation:', self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=2)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=3)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    
                    # ========== Solver training ==========
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = model_inputs["response_mask"][:, 1:]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"][:, 1:]

                    # all return: (bsz, response_length)
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                    entropy_loss = -VF.masked_mean(log_probs, response_mask)  # estimator of entropy loss

                    pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl = core_algos.compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                    )
                    if "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = core_algos.compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = VF.masked_mean(kld, response_mask)
                        pg_loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef

                    total_loss = pg_loss
                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    
                    # ========== Proposer training (joint training) ==========
                    if "proposer_log_probs" in model_inputs and "proposer_advantages" in model_inputs:
                        proposer_old_log_probs = model_inputs["proposer_log_probs"]  # [bsz, question_length]
                        proposer_advantages = model_inputs["proposer_advantages"]  # [bsz, question_length]
                        proposer_response_mask = model_inputs.get("proposer_response_mask")
                        if proposer_response_mask is None:
                            # Fallback: create mask from log_probs shape
                            proposer_response_mask = torch.ones_like(proposer_old_log_probs, dtype=torch.float32)
                        
                        # Create proposer model inputs for forward pass
                        # Proposer input_ids and attention_mask contain the full sequence (prompt + question)
                        # We need to create position_ids if not available
                        proposer_model_inputs = {
                            "input_ids": model_inputs["proposer_input_ids"],
                            "attention_mask": model_inputs["proposer_attention_mask"],
                        }
                        
                        # Generate position_ids for proposer if not available
                        # Use the same format as solver (from original position_ids)
                        if "position_ids" in model_inputs:
                            # Use solver's position_ids format as template
                            solver_position_ids = model_inputs["position_ids"]
                            proposer_seq_len = model_inputs["proposer_input_ids"].shape[1]
                            
                            if solver_position_ids.dim() == 3:  # qwen2vl mrope format
                                # Create position_ids for proposer sequence
                                proposer_position_ids = torch.arange(proposer_seq_len, device=solver_position_ids.device, dtype=solver_position_ids.dtype)
                                proposer_position_ids = proposer_position_ids.unsqueeze(0).unsqueeze(0).expand(3, -1, -1)  # (3, 1, seqlen)
                                proposer_position_ids = proposer_position_ids.expand(-1, proposer_model_inputs["input_ids"].shape[0], -1)  # (3, bsz, seqlen)
                            else:
                                # 2D position_ids
                                proposer_position_ids = torch.arange(proposer_seq_len, device=solver_position_ids.device, dtype=solver_position_ids.dtype)
                                proposer_position_ids = proposer_position_ids.unsqueeze(0).expand(proposer_model_inputs["input_ids"].shape[0], -1)  # (bsz, seqlen)
                            
                            proposer_model_inputs["position_ids"] = proposer_position_ids
                        else:
                            # Fallback: create simple position_ids
                            proposer_seq_len = model_inputs["proposer_input_ids"].shape[1]
                            proposer_position_ids = torch.arange(proposer_seq_len, device=model_inputs["proposer_input_ids"].device)
                            proposer_position_ids = proposer_position_ids.unsqueeze(0).expand(proposer_model_inputs["input_ids"].shape[0], -1)
                            proposer_model_inputs["position_ids"] = proposer_position_ids
                        
                        # For proposer, we need to set "responses" to the generated question part
                        # The responses are the question tokens (last part of input_ids)
                        # Extract question length from old_log_probs
                        question_length = proposer_old_log_probs.shape[1]
                        proposer_responses = model_inputs["proposer_input_ids"][:, -question_length:]  # Extract question tokens
                        proposer_model_inputs["responses"] = proposer_responses
                        
                        if "multi_modal_inputs" in model_inputs:
                            proposer_model_inputs["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
                        
                        # Forward pass for proposer (generate question)
                        proposer_log_probs_new = self._forward_micro_batch(proposer_model_inputs, temperature=temperature)
                        
                        # Compute proposer policy loss
                        proposer_pg_loss, proposer_pg_clipfrac_higher, proposer_pg_clipfrac_lower, proposer_ppo_kl = core_algos.compute_policy_loss(
                            old_log_probs=proposer_old_log_probs,
                            log_probs=proposer_log_probs_new,
                            advantages=proposer_advantages,
                            response_mask=proposer_response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                        )
                        
                        proposer_entropy_loss = -VF.masked_mean(proposer_log_probs_new, proposer_response_mask)
                        
                        # Add proposer loss to total loss (joint training)
                        total_loss = total_loss + proposer_pg_loss
                        
                        batch_metrics.update({
                            "actor/proposer_pg_loss": proposer_pg_loss.detach().item(),
                            "actor/proposer_pg_clipfrac_higher": proposer_pg_clipfrac_higher.detach().item(),
                            "actor/proposer_pg_clipfrac_lower": proposer_pg_clipfrac_lower.detach().item(),
                            "actor/proposer_entropy_loss": proposer_entropy_loss.detach().item(),
                            "actor/proposer_ppo_kl": proposer_ppo_kl.detach().item(),
                        })
                    
                    loss = total_loss / gradient_accumulation
                    print(f'total_loss (solver + proposer): {total_loss}, solver_pg_loss: {pg_loss.detach().item()}')
                    if "proposer_log_probs" in model_inputs:
                        print(f'proposer_pg_loss: {proposer_pg_loss.detach().item()}')
                    loss.backward()
                    
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
