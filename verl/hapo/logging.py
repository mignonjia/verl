from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch

from verl import DataProto


def save_hapo_step(
    step: int,
    batch: DataProto,
    metrics: Dict[str, Any],
    log_dir: str,
) -> None:
    """
    Save rich per-rollout HAPO logs for later visualization.

    The output format follows the design document:
        {
          "step": int,
          "timestamp": str,
          "rollouts": [...],
          "metrics": {...}
        }
    """
    os.makedirs(log_dir, exist_ok=True)
    questions=batch.non_tensor_batch.get("raw_prompt")
    responses = batch.batch["responses"]  # (bs, resp_len)
    base_logprobs = batch.batch["old_log_probs"]
    critic_logprobs = batch.batch.get("critic_logprobs")
    advantages = batch.batch.get("advantages")
    response_mask = batch.batch["response_mask"]

    is_correct_tensor = None
    if "is_correct" in batch.batch:
        is_correct_tensor = batch.batch["is_correct"].bool()
    elif "is_correct" in batch.non_tensor_batch:
        is_correct_tensor = torch.as_tensor(batch.non_tensor_batch["is_correct"], dtype=torch.bool)

    # Get tokenizer for decoding tokens
    tokenizer = batch.meta_info.get("tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "Tokenizer not found in batch.meta_info. Please ensure the tokenizer is passed "
            "via reward_kwargs so it is available in batch.meta_info['tokenizer']."
        )

    bs, resp_len = responses.shape

    rollouts: List[Dict[str, Any]] = []

    # Derive rollout indices if not present
    n = int(metrics.get("actor_rollouts_per_question", 1))

    for i in range(bs):
        qid = f"sample_{i}"
        rollout_idx = i % max(n, 1)
        rollout_id = f"{qid}_r{rollout_idx}"

        mask = response_mask[i].bool()
        token_ids = responses[i][mask].tolist()
        base_lp = base_logprobs[i][mask].tolist()
        hind_lp = critic_logprobs[i][mask].tolist() if critic_logprobs is not None else None
        adv = advantages[i][mask].tolist() if advantages is not None else None

        # Create per-token entries
        tokens: List[Dict[str, Any]] = []
        for j, token_id in enumerate(token_ids):
            decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)
            token_entry = {
                "token_id": int(token_id),
                "decoded_id": decoded_token,
                "old_log_prob": float(base_lp[j]) if j < len(base_lp) else None,
                "critic_prob": float(hind_lp[j]) if hind_lp is not None and j < len(hind_lp) else None,
                "advantage": float(adv[j]) if adv is not None and j < len(adv) else None,
            }
            tokens.append(token_entry)

        total_adv = float(sum(adv)) if adv is not None else 0.0
        mean_adv = float(np.mean(adv)) if adv is not None else 0.0
        num_pos = int(sum(a > 0 for a in (adv or [])))
        num_neg = int(sum(a < 0 for a in (adv or [])))

        rollouts.append(
            {
                "rollout_id": rollout_id,
                "question": questions[i][0].get("content"),
                "question_id": qid,
                "is_correct": bool(is_correct_tensor[i]) if is_correct_tensor is not None else None,
                "tokens": tokens,
                "total_advantage": total_adv,
                "mean_advantage": mean_adv,
                "num_positive_adv": num_pos,
                "num_negative_adv": num_neg,
            }
        )

    payload = {
        "step": int(step),
        "timestamp": datetime.utcnow().isoformat(),
        "rollouts": rollouts,
        "metrics": {
            "loss": float(metrics.get("actor/pg_loss", 0.0)),
            "accuracy": float(metrics.get("train/accuracy", 0.0)),
            "mean_reward": float(metrics.get("train/mean_reward", 0.0)),
            "mean_advantage": float(torch.mean(batch.batch["advantages"]).item())
            if "advantages" in batch.batch
            else 0.0,
            "advantage_std": float(torch.std(batch.batch["advantages"]).item())
            if "advantages" in batch.batch
            else 0.0,
        },
    }

    filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
