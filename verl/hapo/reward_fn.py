from __future__ import annotations

from typing import Any, Dict

import torch

from verl import DataProto


def _extract_answer(text: str) -> str:
    """
    Very small helper to extract a final numeric answer from a solution string.

    This is intentionally simple and can be replaced with a more robust parser.
    """
    import re

    # Look for the last number in the string
    matches = re.findall(r"-?\d+(\.\d+)?", text)
    return matches[-1] if matches else text.strip()


def hapo_gsm8k_reward(batch: DataProto, return_dict: bool = False) -> Any:
    """
    Simple GSM8K-style verifier.

    Expects in `batch.non_tensor_batch`:
      - question_id: str
      - gold_answer: str

    And in `batch.batch`:
      - responses: Tensor[bs, response_len]

    The reward is 1.0 on the last valid token if the extracted answer matches
    the gold answer, otherwise 0.0.
    """
    tokenizer = batch.meta_info.get("tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "For `hapo_gsm8k_reward`, please pass the tokenizer via "
            "`reward_kwargs` so it is available in `batch.meta_info['tokenizer']`."
        )

    responses = batch.batch["responses"]
    bs, resp_len = responses.shape

    gold_answers = batch.non_tensor_batch.get("gold_answer")
    if gold_answers is None:
        raise KeyError("gold_answer field not found in non_tensor_batch, required for HAPO GSM8K reward.")

    # Decode responses and compare with gold answers
    decoded = [
        tokenizer.decode(responses[i].tolist(), skip_special_tokens=True) for i in range(bs)
    ]

    token_level_rewards = torch.zeros_like(responses, dtype=torch.float32)
    is_correct_list = []

    for i in range(bs):
        pred_ans = _extract_answer(decoded[i])
        gold_ans = str(gold_answers[i])
        correct = pred_ans == _extract_answer(gold_ans)
        is_correct_list.append(bool(correct))
        if correct:
            token_level_rewards[i, -1] = 1.0

    reward_tensor = token_level_rewards
    extra_info: Dict[str, Any] = {
        "is_correct": is_correct_list,
    }

    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": extra_info,
        }
    else:
        return reward_tensor
