#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 ZepMemorySystem 进行 Benchmark 评测
格式和输入输出与 simple_memory 保持一致
包含 evidence 信息
支持长文本分块
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to sys.path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import get_logger
from src.llm_interface import OpenAIClient
from src.adaptors import SingleTurnAdaptor, AdaptorResult
from src.zep_memory import ZepMemorySystem
from src.benchmark_utils import chunk_context

logger = get_logger()

def evaluate_adaptor(name: str, adaptor, questions: list, limit: int) -> list:
    results = []
    # limit -1 表示跑所有
    target_questions = questions if limit == -1 else questions[:limit]
    total = len(target_questions)
    
    for i, q in enumerate(target_questions):
        logger.info(f"[{name}] Running Q{i+1}/{total}: {q}")
        try:
            res: AdaptorResult = adaptor.run(q)
            # 提取evidence信息
            evidence_list = []
            for evidence in res.evidence_collected:
                evidence_list.append({
                    "content": evidence.content,
                    "metadata": evidence.metadata
                })
            
            results.append({
                "question": q,
                "answer": res.answer,
                "evidence": evidence_list,
                "steps": res.steps_taken,
                "tokens": res.token_consumption,
                "replan": res.replan_count
            })
        except Exception as e:
            logger.error(f"[{name}] Failed on Q{i+1}: {e}")
            results.append({"question": q, "error": str(e)})
    return results

def run_benchmark(limit: int = 1, output_suffix: str = "", chunk_size: int = 850, overlap: int = 50, text_file: str = None, benchmark_file: str = None):
    """
    运行评测
    
    Args:
        limit: 评测问题数量限制
        output_suffix: 输出文件后缀
        chunk_size: 分块大小
        overlap: 块之间的重叠大小
        text_file: 长文本文件路径
        benchmark_file: benchmark 文件路径
    """
    logger.info(f"=== Running ZepMemorySystem Benchmark ===")
    
    benchmark_data = None
    test_cases = []
    
    # 读取 benchmark 文件
    if benchmark_file:
        logger.info(f"Reading benchmark data from file: {benchmark_file}")
        import json
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        
        logger.info(f"Benchmark file reading completed. Dataset: {benchmark_data.get('dataset_name')}")
        logger.info(f"Number of test cases: {len(benchmark_data.get('test_cases', []))}")
        
        # 提取测试用例
        test_cases = benchmark_data.get('test_cases', [])
        # 限制测试用例数量
        if limit != -1:
            test_cases = test_cases[:limit]
    else:
        # 准备测试数据
        # 模拟一个简单的 benchmark 测试用例
        test_cases = [
            {
                "id": 1,
                "question": "什么是深度学习？它与机器学习有什么关系？",
                "context": "深度学习是机器学习的一种方法，使用多层神经网络来学习数据表示。深度学习在计算机视觉、自然语言处理和语音识别等领域取得了显著成果。机器学习是人工智能的一个子领域，通过数据训练模型来做出预测或决策。",
                "reference_answer": "深度学习是机器学习的一个分支，它使用多层神经网络（深度神经网络）来学习数据的表示。机器学习是人工智能的一个子领域，而深度学习是机器学习的一种方法，专注于使用深度神经网络来解决复杂问题。"
            }
        ]
    
    # 配置 DeepSeek-Chat 模型
    # 使用用户提供的 API Key
    api_key = "sk-8c36866613a445b9951aa367451c8f87"
    base_url = "https://api.deepseek.com/v1"
    model = "deepseek-chat"
    
    logger.info(f"Initializing DeepSeek-Chat client")
    llm = OpenAIClient(
        api_key=api_key,
        base_url=base_url,
        model=model
    )
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"\nProcessing test case {i+1}/{len(test_cases)}")
        logger.info(f"Question: {test_case['question']}")
        
        # 为每个测试用例创建新的记忆系统实例
        logger.info(f"Initializing ZepMemorySystem for test case {i+1}")
        memory = ZepMemorySystem()
        memory.reset()
        
        # 添加当前测试用例的上下文到记忆系统
        context = test_case.get('context', '')
        if context:
            logger.info(f"Adding context to ZepMemorySystem")
            # 对上下文进行分块
            chunks = chunk_context(context, chunk_size=chunk_size, overlap=overlap)
            logger.info(f"Context chunked into {len(chunks)} chunks")
            # 添加分块后的上下文
            for j, chunk in enumerate(chunks):
                memory.add_memory(chunk, {"topic": "人工智能", "source": "benchmark", "test_case_id": test_case['id'], "chunk_id": j})
                logger.info(f"Added chunk {j+1}/{len(chunks)}")
            # 给 Zep 更多时间来处理和索引消息
            logger.info("Giving Zep time to process messages...")
            import time
            time.sleep(10)
        
        # 使用 SingleTurnAdaptor 运行测试
        adaptor = SingleTurnAdaptor(llm, memory)
        try:
            res: AdaptorResult = adaptor.run(test_case['question'])
            # 提取evidence信息
            evidence_list = []
            for evidence in res.evidence_collected:
                evidence_list.append({
                    "content": evidence.content,
                    "metadata": evidence.metadata
                })
            
            result = {
                "question": test_case['question'],
                "answer": res.answer,
                "evidence": evidence_list,
                "steps": res.steps_taken,
                "tokens": res.token_consumption,
                "replan": res.replan_count
            }
            
            # 添加参考答案
            if 'reference_answer' in test_case:
                result['reference_answer'] = test_case['reference_answer']
            
            results.append(result)
            logger.info(f"Test case {i+1} completed successfully")
        except Exception as e:
            logger.error(f"Test case {i+1} failed: {e}")
            results.append({"question": test_case['question'], "error": str(e)})
        finally:
            # 清理资源
            del memory
    
    # 准备最终报告
    final_report = {
        "dataset": benchmark_data.get('dataset_name', 'DeepSeek_Benchmark') if benchmark_data else 'DeepSeek_Benchmark',
        "memory_system": "ZepMemorySystem",
        "llm_model": "deepseek-chat",
        "questions": len(test_cases),
        "results": {
            "R1": results
        }
    }
    
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)
    
    filename = f"zep_benchmark_results"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    output_file = output_dir / filename
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Benchmark completed. Results saved to {output_file}")
    
    # 打印结果
    print("\n=== Benchmark Results ===")
    print(f"Dataset: {final_report.get('dataset')}")
    print(f"Memory System: ZepMemorySystem")
    print(f"LLM Model: deepseek-chat")
    print(f"Questions: {len(test_cases)}")
    print(f"Results saved to: {output_file}")
    
    print(f"\nR1 Results:")
    for i, result in enumerate(results):
        if "error" in result:
            print(f"  Q{i+1}: ERROR - {result['error']}")
        else:
            print(f"  Q{i+1}: Success")
            print(f"    Question: {result['question']}")
            print(f"    Answer: {result['answer'][:100]}...")
            print(f"    Evidence count: {len(result['evidence'])}")
            for j, evidence in enumerate(result['evidence']):
                print(f"    Evidence {j+1}: {evidence['content'][:50]}...")
            print(f"    Steps: {result['steps']}")
            print(f"    Tokens: {result['tokens']}")
            print(f"    Replan: {result['replan']}")
    
    return final_report

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Evaluate ZepMemorySystem with DeepSeek-Chat and text chunking")
    parser.add_argument("--limit", type=int, default=1, help="Number of questions to run (-1 for all)")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix for output filename")
    parser.add_argument("--chunk-size", type=int, default=850, help="Size of each text chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap between chunks")
    parser.add_argument("--text-file", type=str, default=None, help="Path to long text file for testing")
    parser.add_argument("--benchmark-file", type=str, default=None, help="Path to benchmark file containing questions and answers")
    args = parser.parse_args()
    
    run_benchmark(
        limit=args.limit, 
        output_suffix=args.output_suffix,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        text_file=args.text_file,
        benchmark_file=args.benchmark_file
    )

if __name__ == "__main__":
    main()
