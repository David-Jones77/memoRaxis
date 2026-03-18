import json
import re
import string
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# --- 官方风格的文本处理函数 ---
def normalize_answer(s):
    s = "" if s is None else str(s)
    """ 去小写、去标点、去冠词 """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

# --- 读取 Ground Truth（支持 parquet）---
def load_ground_truth(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        return df
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

# --- 主逻辑 ---
def evaluate_conflict_results():
    results_files = glob.glob("out/zep_conflict_res_results_*.json")
    results_files.sort(key=lambda x: int(re.search(r'zep_conflict_res_results_(\d+).json', x).group(1)))
    
    if not results_files:
        print("No conflict result files found.")
        return

    output_dir = Path(r"D:\memoRaxis\memoRaxis-main\out\eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "zep_conflict_official_eval.txt"

    with open(output_file, "w", encoding="utf-8") as fout:
        header = f"=== Conflict Resolution Official Evaluation Report (N={len(results_files)}) ===\n"
        table_head = f"{ 'Inst':<5} | { 'Adaptor':<8} | { 'ExactMatch':<10} | { 'SubMatch':<10} | { 'F1 Score':<10}"
        separator = "-" * 55

        print(header)
        print(table_head)
        print(separator)

        fout.write(header + "\n")
        fout.write(table_head + "\n")
        fout.write(separator + "\n")

        global_stats = {}

        for fpath in results_files:
            match = re.search(r'zep_conflict_res_results_(\d+).json', fpath)
            idx = int(match.group(1)) + 1  # 注意：你之前 +1

            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            instance_gt_path = f"MemoryAgentBench/data/Conflict_Resolution-00000-of-0000{idx}.parquet"
            gt_data = load_ground_truth(instance_gt_path)

            # parquet 处理
            if isinstance(gt_data, pd.DataFrame):
                # 如果 parquet 里是单行，就取第一行
                row = gt_data.iloc[0]
                qa_map = {q: a for q, a in zip(row["questions"], row["answers"])}
            else:
                qa_map = {q: a for q, a in zip(gt_data["questions"], gt_data["answers"])}

            for adaptor, items in data["results"].items():
                if adaptor not in global_stats:
                    global_stats[adaptor] = {"em": [], "sub": [], "f1": []}

                em_hits, sub_hits = 0, 0
                f1_scores = []
                total = len(items)

                for item in items:
                    q = item.get("question")
                    pred = item.get("answer", "")
                    ref_list = qa_map.get(q, [])
                    ref = ref_list[0] if isinstance(ref_list, list) else ref_list

                    norm_pred = normalize_answer(pred)
                    norm_ref = normalize_answer(ref)

                    if norm_pred == norm_ref:
                        em_hits += 1
                    if norm_ref in norm_pred:
                        sub_hits += 1
                    f1_scores.append(f1_score(pred, ref))

                inst_em = em_hits / total if total > 0 else 0
                inst_sub = sub_hits / total if total > 0 else 0
                inst_f1 = np.mean(f1_scores) if f1_scores else 0

                global_stats[adaptor]["em"].append(inst_em)
                global_stats[adaptor]["sub"].append(inst_sub)
                global_stats[adaptor]["f1"].append(inst_f1)

                line = f"{idx:<5} | {adaptor:<8} | {inst_em:>10.2%} | {inst_sub:>10.2%} | {inst_f1:>10.4f}"
                print(line)
                fout.write(line + "\n")

        # Global summary
        print("\n## Global Summary")
        fout.write("\n## Global Summary\n")

        for ad, stats in global_stats.items():
            summary_lines = [
                f"Adaptor {ad}:",
                f"  - Avg Exact Match: {np.mean(stats['em']):.2%}",
                f"  - Avg SubMatch:    {np.mean(stats['sub']):.2%}",
                f"  - Avg F1 Score:    {np.mean(stats['f1']):.4f}"
            ]
            print("\n".join(summary_lines))
            fout.write("\n".join(summary_lines) + "\n")

    print(f"\n✅ 结果已保存到: {output_file}")

if __name__ == "__main__":
    evaluate_conflict_results()
