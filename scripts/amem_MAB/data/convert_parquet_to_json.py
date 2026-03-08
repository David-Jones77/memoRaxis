import json
import argparse
import pandas as pd
from pathlib import Path

# --- 转换函数 ---

def convert_parquet_to_json(parquet_path: str, output_path: str):
    """ 将parquet文件转换为json格式 """
    print(f"=== Converting Parquet to JSON (AMemMemorySystem) ===")
    print(f"Input: {parquet_path}")
    print(f"Output: {output_path}")
    print()

    try:
        # 读取parquet文件
        df = pd.read_parquet(parquet_path)
        print(f"✓ Read parquet file successfully")
        print(f"Number of rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print()

        # 转换为json
        records = df.to_dict('records')
        
        # 保存为json
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Converted to JSON successfully")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Error converting parquet to json: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert parquet file to json format for AMemMemorySystem")
    parser.add_argument("--input", type=str, default="MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet", help="Path to input parquet file")
    parser.add_argument("--output", type=str, default="out/accurate_retrieval.json", help="Path to output json file")
    args = parser.parse_args()

    convert_parquet_to_json(args.input, args.output)