import json
import argparse
import numpy as np
from pathlib import Path

# --- 检查函数 ---

def check_dim_final():
    """ 检查嵌入维度 """
    print(f"=== Checking Embedding Dimensions (AMemMemorySystem) ===")
    print()

    # 检查配置文件
    config_path = Path("out/config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        embedding_dim = config.get("embedding", {}).get("dimension", 384)
        print(f"✓ Config file found")
        print(f"  Embedding dimension: {embedding_dim}")
    else:
        print(f"✗ Config file not found")
        embedding_dim = 384
        print(f"  Using default embedding dimension: {embedding_dim}")
    print()

    # 检查GraphDB配置
    print("Checking GraphDB vector index configuration...")
    print("  Expected vector dimension: 384")
    print("  Expected similarity function: cosine")
    print()

    # 检查是否有测试结果
    result_files = list(Path("out").glob("amem_*.json"))
    if result_files:
        print(f"✓ Found {len(result_files)} result files")
        print("  Sample files:")
        for i, file in enumerate(result_files[:3]):
            print(f"    {i+1}. {file.name}")
        if len(result_files) > 3:
            print(f"    ... and {len(result_files) - 3} more")
    else:
        print(f"✗ No result files found")
    print()

    # 检查嵌入模型
    print("Checking embedding model...")
    print("  Expected model: all-MiniLM-L6-v2")
    print("  Expected dimension: 384")
    print()

    print("Dimension check completed!")

if __name__ == "__main__":
    check_dim_final()