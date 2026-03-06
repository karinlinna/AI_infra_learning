## 说明
目前环境 AutoDL RTX 5090 * 1卡
🗺️ 推荐执行顺序
第一步：跑通环境（1小时）
pip install transformers datasets peft accelerate trl bitsandbytes evaluate （第一次）
配置镜像：export HF_ENDPOINT=https://hf-mirror.com
python llm_training_guide.py --stage estimate

第二步：跑指令微调 SFT + LoRA（最有价值，直接上手）
python llm_training_guide.py --stage sft
用的是 Qwen2-0.5B + 4bit QLoRA，3080 完全够跑，可训练参数只有 0.68%。

第三步：评估模型
pip install rouge_score nltk （第一次）
python llm_training_guide.py --stage eval   # 看 PPL 有没有下降
第四步：推理优化
python llm_training_guide.py --stage infer  # 试试量化推理 + KV Cache

🔑 几个关键决策点
问题3080 的答案
用 fp16 还是 bf16？ bf16，3080 支持且更稳定不溢出
batch_size 设多少？2~4，
开梯度累积到 16要不要梯度检查点？要，省约 30% 显存，代价是慢 20%
全量微调还是 LoRA？LoRA，0.5B 模型全量也行，8B 必须 
LoRA要不要 DeepSpeed？单卡意义不大，多卡或内存不足时开

## 都有什么作用
DeepSpeed = 让多台电脑合力搬运一个超重的货物
Hugging Face下载 = 从应用商店下载一个预装好的AI模型
LoRA微调 = 给一个通才医生做专科培训，不用从头培养，只教他专科知识