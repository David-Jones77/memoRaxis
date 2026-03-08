import json
import argparse
from pathlib import Path

# --- 验证函数 ---

def verify_llm():
    """ 验证LLM配置 """
    print(f"=== Verifying LLM Configuration (AMemMemorySystem) ===")
    print()

    # 检查配置文件
    config_path = Path("out/config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Config file found")
    else:
        print(f"✗ Config file not found")
        print("  Using default LLM configuration")
        config = {}
    print()

    # 检查环境变量
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"✓ OPENAI_API_KEY environment variable found")
        print(f"  Key starts with: {api_key[:5]}...")
    else:
        print(f"✗ OPENAI_API_KEY environment variable not found")
    print()

    # 检查配置中的LLM设置
    llm_config = config.get("llm", {})
    if llm_config:
        print(f"✓ LLM configuration found in config file")
        print(f"  Model: {llm_config.get('model', 'Not specified')}")
        print(f"  Base URL: {llm_config.get('base_url', 'Not specified')}")
    else:
        print(f"✗ LLM configuration not found in config file")
    print()

    # 尝试导入LLM接口
    try:
        from src.llm_interface import OpenAIClient
        print(f"✓ Successfully imported OpenAIClient")
    except Exception as e:
        print(f"✗ Error importing OpenAIClient: {e}")
    print()

    print("LLM verification completed!")

if __name__ == "__main__":
    verify_llm()