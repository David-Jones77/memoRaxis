import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any
import glob

import pandas as pd
import numpy as np

def normalize_text(text: str) -> str:
    """ 基础文本归一化：转小写，去除标点，去除多余空格 """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def is_correct_mechanical(prediction: str, references: List[str]) -> bool:
    """ 
    机械评测逻辑：
    1. 检查预测中是否包含拒答词
    2. 检查预测中是否包含标准答案中的任意一个关键词/短语
    """ 
    negative_patterns = [
        "does not contain any information",
        "insufficient information",
        "not mentioned in the context",
        "no information related to",
        "上下文没有提到",
        "没有找到相关信息",
        "信息不足"
    ]
    
    pred_norm = prediction.lower()
    for pattern in negative_patterns:
        if pattern in pred_norm:
            return False

    for ref in references:
        ref_norm = normalize_text(ref)
        if ref_norm in normalize_text(prediction):
            return True
            
    return False

def load_ground_truth(path: str) -> Dict[str, List[str]]:
    """支持 JSON 或 Parquet"""
    qa_map = {}
    p = Path(path)

    if p.suffix == ".parquet":
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            qs = row["questions"]
            ans = row["answers"]
            if isinstance(qs, np.ndarray):
                qs = qs.tolist()
            if isinstance(ans, np.ndarray):
                ans = ans.tolist()
            for q, a in zip(qs, ans):
                if isinstance(a, np.ndarray):
                    a = a.tolist()
                if isinstance(a, str):
                    a = [a]
                qa_map[q] = a
    else:
        with open(path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        for q, a in zip(ground_truth['questions'], ground_truth['answers']):
            qa_map[q] = a
    return qa_map

def evaluate_one(results_file: str, qa_map: Dict[str, List[str]]) -> str:
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    summary = {}
    
    for adaptor_name, predictions in results_data['results'].items():
        correct_count = 0
        total_count = len(predictions)
        
        for item in predictions:
            q = item.get('question', '')
            pred = item.get('answer', '')
            ref = qa_map.get(q, [])
            
            if is_correct_mechanical(pred, ref):
                correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        summary[adaptor_name] = {
            "accuracy": accuracy,
            "total_questions": total_count,
            "correct_count": correct_count
        }

    # 输出文本块（供 analyze_acc_ret.py 解析）
    lines = []
    lines.append(f"\n--- [Mechanical Evaluation Result: {Path(results_file).name}] ---")
    for adaptor, stats in summary.items():
        lines.append(f"Adaptor {adaptor:3}: Accuracy = {stats['accuracy']:.2%} ({stats['correct_count']}/{stats['total_questions']})")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Mechanical Scorer for Accurate_Retrieval")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results JSON / directory / wildcard")
    parser.add_argument("--instance", type=str, required=True,
                        help="Path to ground truth JSON / Parquet")
    parser.add_argument("--output", type=str, default="out/eval/acc_ret_summary_raw.txt",
                        help="Summary output file (default: out/eval/acc_ret_summary_raw.txt)")
    args = parser.parse_args()

    # ✅ 确保目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 解析 results 文件列表
    results_path = args.results
    results_files = []
    if "*" in results_path:
        results_files = glob.glob(results_path)
    else:
        p = Path(results_path)
        if p.is_dir():
            results_files = sorted(p.glob("*.json"))
        else:
            results_files = [p]

    if not results_files:
        print(f"未找到结果文件: {args.results}")
        return

    qa_map = load_ground_truth(args.instance)

    # 输出到文件（追加模式）
    with open(args.output, "w", encoding="utf-8") as fout:
        for rf in results_files:
            block = evaluate_one(str(rf), qa_map)
            print(block)
            fout.write(block + "\n")

    print(f"\n✅ 汇总完成，已保存到: {args.output}")

if __name__ == "__main__":
    main()
