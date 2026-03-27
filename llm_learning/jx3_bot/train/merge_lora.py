"""
合并 LoRA 权重到基座模型

将 LoRA adapter 合并到基座模型中，生成完整的模型文件用于部署。

使用方式：
    python merge_lora.py \
        --base-model Qwen/Qwen2.5-14B-Instruct \
        --lora-path ./output/jx3_lora \
        --output-dir ./output/jx3_merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--lora-path", type=str, default="./output/jx3_lora")
    parser.add_argument("--output-dir", type=str, default="./output/jx3_merged")
    args = parser.parse_args()

    print(f"加载基座模型: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",  # 合并在 CPU 上做，避免显存问题
    )

    print(f"加载 LoRA 权重: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("合并权重...")
    model = model.merge_and_unload()

    print(f"保存合并后的模型: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n合并完成！模型保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
