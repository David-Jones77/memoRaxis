import json
import argparse
from pathlib import Path

# --- 初始化函数 ---

def init_data():
    """ 初始化数据目录 """
    print("=== Initializing Data Directory (AMemMemorySystem) ===")
    print()

    # 创建必要的目录
    directories = [
        "out",
        "out/analyze",
        "out/evaluate",
        "out/infer",
        "out/ingest"
    ]

    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

    # 创建示例配置文件
    config = {
        "memory_system": "AMemMemorySystem",
        "graphdb": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        },
        "embedding": {
            "model_name": "all-MiniLM-L6-v2",
            "dimension": 384
        },
        "chunking": {
            "chunk_size": 850,
            "chunk_overlap": 50
        },
        "retrieval": {
            "top_k": 5,
            "score_threshold": 0.5
        }
    }

    config_path = Path("out/config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created config file: {config_path}")
    print()

    # 检查必要的数据集
    print("Checking required datasets...")
    datasets = {
        "Accurate_Retrieval": "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet",
        "Conflict_Resolution": "MemoryAgentBench/preview_samples/Conflict_Resolution/instance_0.json",
        "Long_Range_Understanding": "MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_0.json",
        "Test_Time_Learning": "MemoryAgentBench/preview_samples/Test_Time_Learning/instance_0.json"
    }

    for dataset_name, dataset_path in datasets.items():
        if Path(dataset_path).exists():
            print(f"✓ {dataset_name} dataset found")
        else:
            print(f"✗ {dataset_name} dataset not found: {dataset_path}")

    print()
    print("Data initialization completed!")
    print("You can now run the ingest scripts to load data into AMemMemorySystem.")

if __name__ == "__main__":
    init_data()