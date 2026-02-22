#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 DeepSeek-Chat 模型的 RAG 应用
结合 AMemMemorySystem 实现检索增强生成
"""

import sys
import argparse
from pathlib import Path

# 确保 src 目录在路径中
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_interface import OpenAIClient
from src.amem_memory import AMemMemorySystem

def load_documents_from_file(file_path):
    """
    从文件加载文档内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """
    将文本切分为片段
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start += chunk_size - overlap
    
    return chunks

class DeepSeekRAGApp:
    """
    DeepSeek-Chat 模型的 RAG 应用
    """
    
    def __init__(self, api_key, model="deepseek-chat", base_url="https://api.deepseek.com/v1"):
        """
        初始化应用
        
        Args:
            api_key: DeepSeek API Key
            model: 使用的模型名称
            base_url: API 基础 URL
        """
        # 初始化 LLM 客户端
        self.llm = OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        
        # 初始化记忆系统
        self.memory = AMemMemorySystem(enable_llm=False)
        self.memory.reset()
        
        print("✓ DeepSeek RAG 应用初始化成功")
    
    def add_document(self, content, metadata=None):
        """
        添加文档到记忆系统
        
        Args:
            content: 文档内容
            metadata: 文档元数据
        """
        if metadata is None:
            metadata = {}
        
        # 切分文档
        chunks = chunk_text(content)
        print(f"切分文档为 {len(chunks)} 个片段")
        
        # 添加上下文片段到记忆系统
        for i, chunk in enumerate(chunks):
            self.memory.add_memory(chunk, metadata={**metadata, "chunk_id": i})
            if i % 10 == 0:
                print(f"已添加 {i+1}/{len(chunks)} 个片段")
        
        print(f"✓ 文档添加完成，共 {len(chunks)} 个片段")
    
    def query(self, question, top_k=5):
        """
        基于记忆系统回答问题
        
        Args:
            question: 问题
            top_k: 检索结果数量
            
        Returns:
            回答内容
        """
        print(f"\n查询: {question}")
        
        # 从记忆系统检索相关信息
        retrieved_results = self.memory.retrieve(question, top_k=top_k)
        print(f"检索到 {len(retrieved_results)} 个相关结果")
        
        # 构建上下文
        context = "\n".join([f"[{i+1}] {result.content}" for i, result in enumerate(retrieved_results)])
        
        # 构建提示模板
        prompt_template = """
        你是一个基于检索增强生成 (RAG) 的智能助手。
        请基于以下上下文信息，回答用户的问题。
        如果上下文信息不足以回答问题，请明确说明。
        
        上下文信息：
        {context}
        
        用户问题：
        {question}
        
        回答：
        """
        
        # 填充提示
        prompt = prompt_template.format(context=context, question=question)
        
        # 使用 LLM 生成回答
        response = self.llm.generate(prompt)
        
        # 打印回答
        print(f"\n回答: {response}")
        
        # 打印 Token 消耗
        print(f"\nToken 消耗: {self.llm.total_tokens} tokens")
        
        return response
    
    def reset(self):
        """
        重置记忆系统
        """
        self.memory.reset()
        print("✓ 记忆系统已重置")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="DeepSeek-Chat RAG 应用")
    parser.add_argument("--api-key", type=str, required=True, help="DeepSeek API Key")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="使用的模型名称")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com/v1", help="API 基础 URL")
    parser.add_argument("--add-file", type=str, help="添加文档文件")
    parser.add_argument("--query", type=str, help="查询问题")
    parser.add_argument("--reset", action="store_true", help="重置记忆系统")
    
    args = parser.parse_args()
    
    # 初始化应用
    app = DeepSeekRAGApp(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url
    )
    
    # 处理命令行参数
    if args.reset:
        app.reset()
    
    if args.add_file:
        content = load_documents_from_file(args.add_file)
        if content:
            app.add_document(content, metadata={"source": args.add_file})
    
    if args.query:
        app.query(args.query)
    
    # 如果没有提供具体命令，进入交互式模式
    if not any([args.add_file, args.query, args.reset]):
        print("\n进入交互式模式...")
        print("输入 'exit' 退出，'reset' 重置记忆系统，'add <文件路径>' 添加文档")
        
        while True:
            user_input = input("\n请输入问题: ").strip()
            
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "reset":
                app.reset()
            elif user_input.lower().startswith("add "):
                file_path = user_input[4:].strip()
                content = load_documents_from_file(file_path)
                if content:
                    app.add_document(content, metadata={"source": file_path})
            else:
                app.query(user_input)

if __name__ == "__main__":
    main()
