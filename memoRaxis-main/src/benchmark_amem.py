#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 AMemMemorySystem 进行 Benchmark 评测
格式和输入输出与 simple_memory 保持一致
包含 evidence 信息
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import get_logger
from src.llm_interface import OpenAIClient
from src.adaptors import SingleTurnAdaptor, AdaptorResult
from src.amem_memory import AMemMemorySystem
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
    logger.info(f"=== Running AMemMemorySystem Benchmark ===")
    
    benchmark_data = None
    questions = []
    contexts = []
    reference_answers = []
    
    # 读取 benchmark 文件
    if benchmark_file:
        logger.info(f"Reading benchmark data from file: {benchmark_file}")
        import json
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        
        logger.info(f"Benchmark file reading completed. Dataset: {benchmark_data.get('dataset_name')}")
        logger.info(f"Number of test cases: {len(benchmark_data.get('test_cases', []))}")
        
        # 提取问题、上下文和参考答案
        for test_case in benchmark_data.get('test_cases', []):
            questions.append(test_case.get('question'))
            contexts.append(test_case.get('context', ''))
            reference_answers.append(test_case.get('reference_answer', ''))
        
        # 将所有上下文合并为一个长文本
        long_text = '\n'.join(contexts)
        logger.info(f"Combined context length: {len(long_text)} characters")
    else:
        # 准备测试数据
        # 模拟一个简单的 benchmark 测试用例
        test_question = "什么是深度学习？它与机器学习有什么关系？"
        questions = [test_question]
        
        # 读取长文本数据
        if text_file:
            logger.info(f"Reading long text from file: {text_file}")
            with open(text_file, 'r', encoding='utf-8') as f:
                long_text = f.read().strip()
            logger.info(f"File reading completed. Length: {len(long_text)} characters")
        else:
            # 使用默认的长文本作为示例
            logger.info(f"Using default long text for testing")
            long_text = """
            机器学习是人工智能的一个子领域，通过数据训练模型来做出预测或决策。机器学习算法可以分为监督学习、无监督学习和强化学习等类型。监督学习包括分类和回归任务，使用标记数据进行训练。无监督学习则处理未标记的数据，旨在发现数据中的模式和结构。强化学习通过与环境的交互来学习最优策略，通过奖励和惩罚机制来引导学习过程。

            深度学习是机器学习的一种方法，使用多层神经网络来学习数据表示。深度学习在计算机视觉、自然语言处理和语音识别等领域取得了显著成果。深度学习模型通常包含多个隐藏层，每一层都学习数据的不同层次的表示。深度神经网络的训练需要大量的数据和计算资源，但通常能够达到更高的准确率。

            Transformer 架构是一种基于注意力机制的神经网络架构，广泛用于自然语言处理。Transformer 架构由 Google 在 2017 年提出，它使用自注意力机制来捕获输入序列中的依赖关系，不需要循环或卷积操作。这种架构在机器翻译、文本分类、问答系统等任务中取得了巨大成功。BERT 和 GPT 等著名模型都基于 Transformer 架构。

            RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术，用于增强 LLM 的知识能力。RAG 系统通常包括检索器、知识库和生成器三个主要组件。使用 RAG 可以提高 LLM 回答的准确性和可靠性，减少 hallucination 现象。RAG 技术在需要最新或特定领域知识的场景中特别有用，因为它可以从外部知识库中检索相关信息来辅助生成。

            大语言模型 (LLM) 是指参数量非常大的语言模型，通常包含数十亿甚至数千亿个参数。这些模型通过在海量文本数据上进行预训练，学习到了丰富的语言知识和世界知识。LLM 可以执行多种自然语言处理任务，如文本生成、翻译、摘要、问答等。然而，LLM 也存在一些局限性，如可能产生错误信息、偏见，以及对长上下文的处理能力有限等。

            为了克服 LLM 的局限性，研究人员提出了多种技术，如提示工程、微调、RAG 等。提示工程通过设计有效的提示来引导 LLM 生成更好的回答。微调则是在特定任务的数据集上对预训练模型进行进一步训练，以提高其在该任务上的性能。RAG 则通过检索外部知识来增强 LLM 的回答能力，减少错误信息的产生。
            """.strip()
    
    logger.info(f"Preparing long text for testing (length: {len(long_text)} characters)")
    
    # 对长文本进行分块
    chunks = chunk_context(long_text, chunk_size=chunk_size, overlap=overlap)
    logger.info(f"Text chunking completed: {len(chunks)} chunks created")
    
    logger.info(f"Initializing AMemMemorySystem")
    
    # Initialize Memory with AMemMemorySystem
    memory = AMemMemorySystem(enable_llm=False)
    memory.reset()  # 重置记忆系统
    
    # 添加分块后的文本到记忆系统
    logger.info(f"Adding chunked text to AMemMemorySystem")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, {"topic": "人工智能", "source": "sample", "chunk_id": i})
        if i % 10 == 0:
            logger.info(f"Added chunk {i+1}/{len(chunks)}")
    logger.info(f"All chunks added successfully: {len(chunks)} chunks")

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

    results = {}
    
    # 只运行 R1 适配器（单次检索直接生成）
    results["R1"] = evaluate_adaptor("R1", SingleTurnAdaptor(llm, memory), questions, limit)

    # 准备最终报告
    final_report = {
        "dataset": benchmark_data.get('dataset_name', 'DeepSeek_Benchmark') if benchmark_data else 'DeepSeek_Benchmark',
        "memory_system": "AMemMemorySystem",
        "llm_model": "deepseek-chat",
        "chunk_config": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "total_chunks": len(chunks)
        },
        "results": results
    }
    
    # 如果有 benchmark 数据，添加参考答案
    if benchmark_data and reference_answers:
        for i, (result, reference_answer) in enumerate(zip(results.get('R1', []), reference_answers)):
            if i < len(results.get('R1', [])):
                results['R1'][i]['reference_answer'] = reference_answer
    
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)
    
    filename = f"amem_benchmark_results"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    output_file = output_dir / filename
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Benchmark completed. Results saved to {output_file}")
    
    # 打印结果
    print("\n=== Benchmark Results ===")
    print(f"Dataset: DeepSeek_Benchmark")
    print(f"Memory System: AMemMemorySystem")
    print(f"LLM Model: deepseek-chat")
    print(f"Text Chunking: Enabled (chunk_size={chunk_size}, overlap={overlap})")
    print(f"Total Chunks: {len(chunks)}")
    print(f"Questions: {len(questions)}")
    print(f"Results saved to: {output_file}")
    
    for adaptor, adaptor_results in results.items():
        print(f"\n{adaptor} Results:")
        for i, result in enumerate(adaptor_results):
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate AMemMemorySystem with DeepSeek-Chat and text chunking")
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
