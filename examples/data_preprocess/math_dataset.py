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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def check_in_samples(question):
    samples = [
        "Find the domain of the real-valued function",
        "The bank offers him a choice between two",
        "A fair coin is flipped 7 times. "
        "If six people decide to come to a basketball game, but three of th",
        "Arnold is studying the prevalence of three health risk factors",
        "blue balls. A second urn contains",
        "If Greg rolls four fair six-sided dice, ",
        "Suppose that for some "
    ]
    for sample in samples:
        if sample in question:
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/math", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--cot_eval", default=False, help="Whether to evaluate the teacher COT.")
    parser.add_argument("--samples", default=False, help="Whether to sample the dataset.")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path
    if args.samples:
        args.local_save_dir = args.local_save_dir + "_samples"
    if args.cot_eval:
        args.local_save_dir = args.local_save_dir + "_cot_eval"

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(
            local_dataset_path,
        )
    else:
        dataset = datasets.load_dataset(
            data_source,
        )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            raw_question = example.pop("problem")
            question = raw_question + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution(answer)
            level = example.pop("level")
            type = example.pop("type")

            teacher_question = (
                f"Here is an example problem with one of its solution:\n\n"
                f"Example Problem: {question}\n\n"
                f"Example Solution:\n{answer}\n\n"
                f"The solution is the correct one. You should rethink the problem in your own words when you solve it.\n"
                f"Problem: {question}"
            )

            if level == "Level 1":
                return None
            if args.samples:
                if not check_in_samples(raw_question):
                    return None
            data = {
                "data_source": data_source if not args.cot_eval else data_source + "_cot_eval",
                "prompt": [{"role": "user", "content": question if not args.cot_eval else teacher_question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx, "question": raw_question, "teacher_cots": [answer], "level": level, "type": type},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    
    if not args.samples:
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    if not args.samples:
        test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)

    if not args.samples:
        example = test_dataset[0]
        with open(os.path.join(local_dir, "test_example.json"), "w") as f:
            json.dump(example, f, indent=2)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
