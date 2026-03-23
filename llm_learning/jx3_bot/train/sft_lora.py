"""
LoRA 微调 Qwen2.5-7B-Instruct

使用 peft + trl 进行参数高效微调，适配 RTX 3090/4090（24GB）。

使用方式：
    python sft_lora.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --train-data ../data/train.jsonl \
        --val-data ../data/val.jsonl \
        --output-dir ./output/jx3_lora

    # 显存不够可用更小的模型
    python sft_lora.py --model Qwen/Qwen2.5-3B-Instruct ...
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


def load_jsonl(path: str) -> list[dict]:
    """加载 JSONL 文件"""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_messages(example: dict, tokenizer) -> str:
    """将 messages 格式转为模型输入文本"""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调 Qwen2.5")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-data", type=str, default="../data/train.jsonl")
    parser.add_argument("--val-data", type=str, default="../data/val.jsonl")
    parser.add_argument("--output-dir", type=str, default="./output/jx3_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--use-4bit", action="store_true", help="使用 4bit 量化 (QLoRA)，进一步节省显存")
    args = parser.parse_args()

    print("=" * 60)
    print("剑网三问答模型 — LoRA 微调")
    print("=" * 60)
    print(f"基座模型: {args.model}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"4bit 量化: {'是' if args.use_4bit else '否'}")
    print()

    # ========================================
    # 1. 加载 Tokenizer
    # ========================================
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================
    # 2. 加载模型
    # ========================================
    print("加载模型...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if args.use_4bit:
        # QLoRA: 4bit 量化加载，显存占用更小
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False  # 训练时关闭 KV Cache

    # ========================================
    # 3. 配置 LoRA
    # ========================================
    print("配置 LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ========================================
    # 4. 加载数据
    # ========================================
    print("加载训练数据...")
    train_data = load_jsonl(args.train_data)
    train_dataset = Dataset.from_list(train_data)

    val_dataset = None
    if Path(args.val_data).exists():
        val_data = load_jsonl(args.val_data)
        val_dataset = Dataset.from_list(val_data)
        print(f"训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")
    else:
        print(f"训练集: {len(train_data)} 条, 无验证集")

    # 格式化为文本
    def formatting_func(example):
        return format_messages(example, tokenizer)

    # ========================================
    # 5. 训练配置
    # ========================================
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=100 if val_dataset else None,
        max_seq_length=args.max_seq_len,
        report_to="none",
        seed=42,
    )

    # ========================================
    # 6. 开始训练
    # ========================================
    print("开始训练...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )

    trainer.train()

    # ========================================
    # 7. 保存
    # ========================================
    print("保存 LoRA 权重...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n训练完成！LoRA 权重保存到: {args.output_dir}")
    print(f"下一步：运行 merge_lora.py 合并权重，或直接加载 adapter 推理")


if __name__ == "__main__":
    main()
