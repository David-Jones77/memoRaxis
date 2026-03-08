import json
import argparse
from pathlib import Path

# --- 调试函数 ---

def debug_json_13(json_path: str):
    """ 调试JSON文件 """
    print(f"=== Debugging JSON File (AMemMemorySystem) ===")
    print(f"File: {json_path}")
    print()

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ JSON file loaded successfully")
        print()
        
        # 打印文件结构
        print("File structure:")
        print(f"  Type: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")
            
            # 检查常见字段
            if "results" in data:
                print(f"  Results type: {type(data['results']).__name__}")
                if isinstance(data['results'], dict):
                    print(f"  Adaptors: {list(data['results'].keys())}")
                    
                    for adaptor, items in data['results'].items():
                        print(f"  {adaptor} items: {len(items)}")
                        if items:
                            print(f"  First item keys: {list(items[0].keys())}")
                            
        elif isinstance(data, list):
            print(f"  Length: {len(data)}")
            if data:
                print(f"  First item type: {type(data[0]).__name__}")
                if isinstance(data[0], dict):
                    print(f"  First item keys: {list(data[0].keys())}")
        
        print()
        print("JSON debugging completed successfully!")
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON decode error: {e}")
        print(f"  Error position: {e.pos}")
        print(f"  Error line: {e.lineno}")
        print(f"  Error column: {e.colno}")
    
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug JSON file for AMemMemorySystem")
    parser.add_argument("--json_path", type=str, default="out/amem_acc_ret_results_0.json", help="Path to JSON file")
    args = parser.parse_args()

    debug_json_13(args.json_path)