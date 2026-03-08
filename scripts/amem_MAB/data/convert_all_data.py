import json
import argparse
from pathlib import Path

# --- 转换函数 ---

def convert_all_data():
    """ 转换所有数据集 """
    print("=== Converting All Data (AMemMemorySystem) ===")
    print()

    # 转换Conflict Resolution数据
    print("1. Converting Conflict Resolution data...")
    convert_conflict_resolution()
    print()

    # 转换Long Range Understanding数据
    print("2. Converting Long Range Understanding data...")
    convert_long_range()
    print()

    # 转换Test Time Learning数据
    print("3. Converting Test Time Learning data...")
    convert_test_time()
    print()

    print("All data conversion completed!")

def convert_conflict_resolution():
    """ 转换Conflict Resolution数据 """
    instances = [0, 1, 2, 3, 4]
    
    for instance_idx in instances:
        data_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{instance_idx}.json"
        output_path = f"out/processed_conflict_instance_{instance_idx}.json"
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理数据
            processed_data = {
                "instance_idx": instance_idx,
                "context": data.get("context", ""),
                "questions": data.get("questions", []),
                "answers": data.get("answers", []),
                "metadata": {
                    "context_length": len(data.get("context", "")),
                    "num_questions": len(data.get("questions", []))
                }
            }
            
            # 保存处理后的数据
            output_dir = Path("out")
            output_dir.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Converted instance {instance_idx}")
        except Exception as e:
            print(f"  ✗ Error converting instance {instance_idx}: {e}")

def convert_long_range():
    """ 转换Long Range Understanding数据 """
    instances = [0, 1, 2, 3, 4]
    
    for instance_idx in instances:
        data_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{instance_idx}.json"
        output_path = f"out/processed_long_range_instance_{instance_idx}.json"
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理数据
            processed_data = {
                "instance_idx": instance_idx,
                "context": data.get("context", ""),
                "questions": data.get("questions", []),
                "answers": data.get("answers", []),
                "metadata": {
                    "context_length": len(data.get("context", "")),
                    "num_questions": len(data.get("questions", []))
                }
            }
            
            # 保存处理后的数据
            output_dir = Path("out")
            output_dir.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Converted instance {instance_idx}")
        except Exception as e:
            print(f"  ✗ Error converting instance {instance_idx}: {e}")

def convert_test_time():
    """ 转换Test Time Learning数据 """
    instances = [0, 1, 2, 3, 4]
    
    for instance_idx in instances:
        data_path = f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{instance_idx}.json"
        output_path = f"out/processed_test_time_instance_{instance_idx}.json"
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理数据
            processed_data = {
                "instance_idx": instance_idx,
                "initial_context": data.get("initial_context", ""),
                "new_information": data.get("new_information", ""),
                "questions": data.get("questions", []),
                "answers": data.get("answers", []),
                "metadata": {
                    "initial_context_length": len(data.get("initial_context", "")),
                    "new_information_length": len(data.get("new_information", "")),
                    "num_questions": len(data.get("questions", []))
                }
            }
            
            # 保存处理后的数据
            output_dir = Path("out")
            output_dir.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Converted instance {instance_idx}")
        except Exception as e:
            print(f"  ✗ Error converting instance {instance_idx}: {e}")

if __name__ == "__main__":
    convert_all_data()