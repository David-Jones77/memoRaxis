import json
import argparse
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.logger import get_logger
from src.config import get_config
from src.llm_interface import OpenAIClient

logger = get_logger()
# --- 官方 PROMPTS (直接复用) ---
FLUENCY_PROMPT = """Please act as an impartial judge and evaluate the fluency of the provided text. The text should be coherent, non-repetitive, fluent, and grammatically correct.

Below is your grading rubric:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish.
- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers.

Now, read the provided text, and evaluate the fluency using the rubric. Then output your score in the following json format: {{"fluency": 1}}.

Text: "{text}"
"""

RECALL_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel. It should discuss the plots and characters of the story. The text should contain all the given key points.

Below is your grading rubric:
Recall:
- Evaluate the provided summary by deciding if each of the key points is present in the provided summary. A key point is considered present if its factual information is mostly-supported by the provided summary.
- Score: the number of key points mostly-supported by the provided summary.

Now, read the provided summary and key points, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"supported_key_points": [1, 3], "recall": 2}}.

Key points:
{keypoints}

Summary: <start of summary>{summary}<end of summary>
"""

PRECISION_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel.

Below is your grading rubric:
Precision:
- Evaluate the provided summary by deciding if each sentence in the provided summary is supported by the information provided in the expert summary.
- Score: the number of sentences in the provided summary that are supported by the expert summary.

Now, read the provided summary and expert summary, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"precision": 7, "sentence_count": 20}}.

Expert summary: <start of summary>{expert_summary}<end of summary>

Provided summary: <start of summary>{summary}<end of summary>
"""

class SummarizationJudge:
    def __init__(self):
        conf = get_config()
        self.llm = OpenAIClient(
            api_key=conf.llm["api_key"],
            base_url=conf.llm["base_url"],
            model=conf.llm["model"]
        )

    def judge_fluency(self, text: str) -> int:
        prompt = FLUENCY_PROMPT.format(text=text)
        res = self.llm.generate_json(prompt)
        return int(res.get("fluency", 0))

    def judge_recall(self, summary: str, keypoints: List[str]) -> int:
        kp_str = "\n".join([f"{i+1}. {kp}" for i, kp in enumerate(keypoints)])
        prompt = RECALL_PROMPT.format(keypoints=kp_str, summary=summary)
        res = self.llm.generate_json(prompt)
        return int(res.get("recall", 0))

    def judge_precision(self, summary: str, expert_summary: str) -> tuple[int, int]:
        prompt = PRECISION_PROMPT.format(expert_summary=expert_summary, summary=summary)
        res = self.llm.generate_json(prompt)
        return int(res.get("precision", 0)), int(res.get("sentence_count", 1))

def load_ground_truth(instance_path: Path):
    """兼容 json / parquet"""
    if instance_path.suffix == ".parquet":
        df = pd.read_parquet(instance_path)
        # 此处假设 instance_idx 对应行
        return df
    else:
        with open(instance_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="LRU-A Summarization Evaluator")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--instance_folder", type=str, default="MemoryAgentBench/data")
    args = parser.parse_args()

    with open(args.results, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    instance_idx = results_data["instance_idx"]+1

    # --- 读取 GT ---
    instance_path = Path(args.instance_folder) / f"instance_{instance_idx}.json"
    if not instance_path.exists():
        # fallback: parquet
        parquet_files = Path(args.instance_folder) / f"Long_Range_Understanding-00000-of-0000{instance_idx}.parquet"
        if not parquet_files:
            raise FileNotFoundError("No json or parquet found.")
        instance_path = parquet_files

    gt = load_ground_truth(instance_path)

    if isinstance(gt, dict):
        keypoints = gt["metadata"].get("keypoints", [])
        if isinstance(keypoints, np.ndarray):
            keypoints = keypoints.tolist()

        answers = gt.get("answers", [])
        expert_summary = ""
        if answers:
            # 修复：answers 是二维数组
            if isinstance(answers[0], list):
                expert_summary = answers[0][0]
            else:
                expert_summary = answers[0]
    else:
        # parquet
        row = gt.iloc[instance_idx-1]
        keypoints = row["metadata"].get("keypoints", [])
        if isinstance(keypoints, np.ndarray):
            keypoints = keypoints.tolist()

        answers = row["answers"]
        if isinstance(answers, np.ndarray):
            answers = answers.tolist()
        expert_summary = answers[0] if answers else ""

    # --- 评测 ---
    judge = SummarizationJudge()
    final_eval = {"dataset": "LRU-A", "instance_idx": instance_idx, "metrics": {}}

    for adaptor_name, predictions in results_data['results'].items():
        if not predictions:
            continue
        prediction = predictions[0]["answer"]

        # debug 对齐检查
        print("Prediction:", prediction[:200])
        print("GT summary:", expert_summary[:200])
        print("keypoints len:", len(keypoints))

        f_score = judge.judge_fluency(prediction)
        r_found = judge.judge_recall(prediction, keypoints)
        p_found, p_total = judge.judge_precision(prediction, expert_summary)

        recall = r_found / len(keypoints) if len(keypoints) > 0 else 0
        precision = p_found / p_total if p_total > 0 else 0
        f1 = f_score * 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

        final_eval["metrics"][adaptor_name] = {
            "fluency": f_score,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "raw": {
                "recall_found": r_found,
                "recall_total": len(keypoints),
                "precision_found": p_found,
                "precision_total": p_total
            }
        }

    output_dir = Path("out/eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"eval_lru_a_{instance_idx}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_eval, f, indent=2, ensure_ascii=False)

    print(f"\n✅ LRU-A Evaluation saved to {output_file}")

if __name__ == "__main__":
    main()
