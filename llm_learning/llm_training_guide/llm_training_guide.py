"""
HuggingFace 大模型全流程实战
目标模型：GPT-2 Medium (345M) 或 Qwen2-0.5B
目标硬件：RTX 3080 (10GB) 或更小显卡

流程：
  Step 1  环境安装
  Step 2  数据准备
  Step 3  预训练（Causal LM）
  Step 4  指令微调（SFT + LoRA）
  Step 5  训练配置（显存估算 / 混合精度 / 梯度累积）
  Step 6  DeepSpeed 并行训练
  Step 7  推理优化（量化 + KV Cache）
  Step 8  模型评估（PPL + 基准测试）
"""

# ============================================================
# Step 1: 环境安装
# ============================================================
"""
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft accelerate trl
pip install bitsandbytes   # 量化
pip install deepspeed      # 分布式训练
pip install evaluate rouge_score  # 评估
"""

# ============================================================
# 统一数据目录：模型缓存、数据集、训练输出都放在 ~/data 下
# ============================================================
import os
DATA_ROOT = os.path.expanduser("~/data")
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"                # 模型缓存（数据盘）
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/datasets"         # 数据集缓存（数据盘）
OUTPUT_DIR = os.path.join(DATA_ROOT, "output")                        # 训练输出

# ============================================================
# Step 2: 数据准备
# ============================================================
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2-0.5B"   # 0.5B，3080完全够用
# MODEL_NAME = "openai-community/gpt2-medium"  # 备选

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT系列没有pad_token

# —— 2a. 预训练数据：用纯文本，做 Next Token Prediction ——
def prepare_pretrain_data():
    # 用 WikiText-2 做演示（真实场景换成 Common Crawl / C4）
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True
        )

    def group_texts(examples):
        """把短文本拼接成固定长度的块，充分利用每个 batch"""
        block_size = 512
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()  # CLM：输入即标签
        return result

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    lm_dataset = tokenized.map(group_texts, batched=True)
    return lm_dataset

# —— 2b. 指令微调数据：(instruction, input, output) 三元组 ——
def prepare_sft_data():
    # 用 Alpaca 格式数据集做演示
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

    PROMPT_TEMPLATE = """Below is an instruction. Write a response.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

    def format_prompt(example):
        text = PROMPT_TEMPLATE.format(
            instruction=example["instruction"],
            input=example.get("input", ""),
            output=example["output"]
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(format_prompt, remove_columns=dataset.column_names)


# ============================================================
# Step 3: 预训练（Causal Language Modeling）
# ============================================================
from transformers import (
    AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
import torch

def pretrain():
    lm_dataset = prepare_pretrain_data()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,   # 混合精度：bfloat16 更稳定
        device_map="auto"
    )

    # ---- 显存估算（3080 10GB） ----
    # 模型参数：0.5B × 2 bytes(bf16) = 1GB
    # 优化器状态(Adam)：参数量 × 8 bytes = 4GB
    # 梯度：参数量 × 2 bytes = 1GB
    # 激活值：batch_size × seq_len × hidden × layers ≈ 变量
    # 总计基础占用 ≈ 6GB，留给激活值约 4GB
    # → batch_size=2，梯度累积=8，等效 batch=16

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "pretrain"),
        num_train_epochs=3,

        # ---- 显存控制 ----
        per_device_train_batch_size=2,      # 实际每步 batch
        gradient_accumulation_steps=8,      # 等效 batch_size = 2×8 = 16
        gradient_checkpointing=True,        # 用时间换显存（激活值重计算）

        # ---- 混合精度 ----
        bf16=True,                          # 3080+ 推荐 bf16；旧卡用 fp16=True

        # ---- 优化器 ----
        optim="adamw_torch_fused",          # fused 版本更快
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        # ---- 日志与保存 ----
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",                   # 换成 "wandb" 可接 W&B 监控
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "pretrain/final"))


# ============================================================
# Step 4: 指令微调（SFT + LoRA）
# ============================================================
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

def sft_with_lora():
    """
    LoRA 原理：冻结原始权重 W，只训练低秩矩阵 A、B
      W' = W + ΔW = W + B×A   （r << d，r通常为8或16）
    
    可训练参数量：原来的 0.1%~1%，显存大幅下降
    
    适用场景：
      LoRA    → 通用指令微调，资源受限场景
      QLoRA   → 4bit量化基础上做LoRA，超低显存
      Full SFT → 数据量大、效果要求极高时
    """

    # 加载基础模型（4bit量化加载，进一步省显存）
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4 量化类型
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # 双重量化，再省10%显存
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # 训练时关闭 KV Cache

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                    # 秩，越大效果越好但参数越多
        lora_alpha=32,           # 缩放因子，通常=2r
        lora_dropout=0.05,
        target_modules=[         # 对哪些层加 LoRA
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 输出示例：trainable params: 3,407,872 || all params: 498,432,000
    #           trainable%: 0.68%  ← 只训练0.68%的参数！

    sft_dataset = prepare_sft_data()

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "sft_lora"),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,   # 等效 batch=16
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # 保存 LoRA 权重（只有几十MB，不是完整模型）
    model.save_pretrained(os.path.join(OUTPUT_DIR, "sft_lora/final"))

    # 合并 LoRA 权重到基础模型（推理时用）
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    merged_model = PeftModel.from_pretrained(base_model, os.path.join(OUTPUT_DIR, "sft_lora/final"))
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(os.path.join(OUTPUT_DIR, "sft_lora/merged"))


# ============================================================
# Step 5: 显存估算速查表
# ============================================================
def estimate_vram(param_billions, batch_size, seq_len, precision="bf16"):
    """
    快速显存估算

    规则：
      训练时：param × (2 + 2 + 8) bytes = param × 12 bytes  (bf16模型+梯度+Adam)
      推理时：param × 2 bytes  (bf16)

    示例：
      0.5B 模型训练：0.5B × 12B = 6GB（+ 激活值）
      8B   模型训练：8B × 12B  = 96GB（需要多卡！）
      8B   模型推理：8B × 2B   = 16GB（3080装不下，需量化到INT4→4GB）
    """
    bytes_per_param = {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1, "int4": 0.5}
    b = bytes_per_param[precision]

    model_gb = param_billions * 1e9 * b / 1e9
    # 激活值估算（粗略）
    hidden = 2048  # 假设 hidden_size
    activation_gb = batch_size * seq_len * hidden * 4 * 2 / 1e9  # ×4 layers ×2

    print(f"模型权重:   {model_gb:.1f} GB")
    print(f"优化器(Adam): {model_gb * 4:.1f} GB")
    print(f"梯度:       {model_gb:.1f} GB")
    print(f"激活值估算: {activation_gb:.1f} GB")
    print(f"总计(训练): {model_gb * 6 + activation_gb:.1f} GB")
    print(f"总计(推理): {model_gb:.1f} GB")
    print()
    print("RTX 3080 (10GB) 建议配置：")
    print(f"  batch_size=2, grad_accum=8 → 等效batch=16")
    print(f"  开启 gradient_checkpointing（省约30%激活值显存）")
    print(f"  使用 bf16/fp16 混合精度（省50%模型显存）")


# ============================================================
# Step 6: DeepSpeed 并行训练
# ============================================================
"""
DeepSpeed ZeRO 三个阶段（显存优化核心）：
  ZeRO-1: 分片优化器状态           → 省 4x 显存
  ZeRO-2: 分片优化器状态 + 梯度     → 省 8x 显存
  ZeRO-3: 分片优化器+梯度+模型参数  → 省 64x 显存（多机必备）

单卡也能用 ZeRO-2！把优化器状态卸载到 CPU 内存
"""

# ds_config.json（放到项目根目录）
DS_CONFIG = """
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true
  },
  "bf16": {
    "enabled": true
  },
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "steps_per_print": 50
}
"""
# 启动命令（单卡）：
#   deepspeed --num_gpus=1 train.py --deepspeed ds_config.json
# 启动命令（多卡）：
#   deepspeed --num_gpus=4 train.py --deepspeed ds_config.json


# ============================================================
# Step 7: 推理优化
# ============================================================
def inference_demo():
    """
    7a. INT4 量化推理（bitsandbytes）
    """
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(OUTPUT_DIR, "sft_lora/merged"),
        quantization_config=quantization_config,
        device_map="auto"
    )
    # 8B 模型 INT4 后约 4.5GB，3080 可以跑推理！

    """
    7b. KV Cache（HuggingFace 默认开启）
    原理：Transformer 每次生成 token 需要所有历史的 K、V 向量
          KV Cache 缓存已计算的 K/V，避免重复计算
          代价：显存随序列长度线性增长
    """
    model.config.use_cache = True  # 推理时一定要开

    inputs = tokenizer("介绍一下深度学习：", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,       # KV Cache
            repetition_penalty=1.1
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    """
    7c. 投机采样（Speculative Decoding）
    原理：用小模型（Draft Model）快速生成多个候选 token，
          大模型（Target Model）并行验证，接受或拒绝
    效果：理论上 2~3x 加速，且输出分布不变
    """
    # HuggingFace 内置支持
    from transformers import AutoModelForCausalLM as AMCL
    draft_model = AMCL.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype=torch.bfloat16).to("cuda")
    target_model = AMCL.from_pretrained("Qwen/Qwen2-1.5B", torch_dtype=torch.bfloat16).to("cuda")

    outputs = target_model.generate(
        **inputs,
        assistant_model=draft_model,   # 投机采样一行搞定
        max_new_tokens=200,
    )


# ============================================================
# Step 8: 模型评估
# ============================================================
import math
import evaluate
from torch.nn import CrossEntropyLoss

def evaluate_model(model_path: str):
    """
    8a. 困惑度 PPL（Perplexity）—— 衡量语言模型流畅度
    
    PPL = exp(平均负对数似然)
    PPL 越低 = 模型对文本"越不困惑" = 语言建模能力越强
    参考值：GPT-2 在 WikiText103 上 PPL ≈ 18
           好的 LLM 通常 PPL < 10
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    max_length = 512
    stride = 256
    nlls = []

    for begin_loc in range(0, encodings.input_ids.size(1), stride):
        end_loc = min(begin_loc + max_length, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nlls.append(outputs.loss)

    ppl = math.exp(torch.stack(nlls).mean())
    print(f"Perplexity (PPL): {ppl:.2f}")

    """
    8b. 指令跟随能力评估
    常用基准：
      MMLU    → 57项学术知识考试（0-shot / 5-shot）
      HumanEval → 代码生成通过率（pass@1）
      MT-Bench  → 多轮对话质量，GPT-4 打分
      C-Eval    → 中文版 MMLU
    """
    # 用 lm-evaluation-harness 一行跑标准榜
    # pip install lm-eval
    # lm_eval --model hf --model_args pretrained=./output/sft_lora/merged \
    #         --tasks mmlu --num_fewshot 5 --batch_size 4

    """
    8c. 生成质量评估
    """
    rouge = evaluate.load("rouge")
    predictions = ["模型生成的答案"]
    references  = ["标准参考答案"]
    result = rouge.compute(predictions=predictions, references=references)
    print(f"ROUGE-L: {result['rougeL']:.4f}")


# ============================================================
# 完整运行入口
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "sft", "eval", "infer", "estimate"])
    args = parser.parse_args()

    if args.stage == "pretrain":
        pretrain()
    elif args.stage == "sft":
        sft_with_lora()
    elif args.stage == "eval":
        evaluate_model(os.path.join(OUTPUT_DIR, "sft_lora/merged"))
    elif args.stage == "infer":
        inference_demo()
    elif args.stage == "estimate":
        print("=== 0.5B 模型（Qwen2-0.5B）===")
        estimate_vram(0.5, batch_size=4, seq_len=512)
        print("\n=== 8B 模型（LLaMA-3-8B）===")
        estimate_vram(8, batch_size=1, seq_len=512)
