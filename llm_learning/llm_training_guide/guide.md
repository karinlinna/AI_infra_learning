## 说明
目前环境 AutoDL RTX 5090 * 1卡
清理回收站 https://zhuanlan.zhihu.com/p/1978177366409900756
查看占用：source ~/.bashrc
1. 清理系统盘回收站
这是系统盘瘦身的第一步，通常能释放几百 MB 到几 GB 的空间。
确认目录存在 ls -la /root/.local/share/Trash/
强制清空系统盘回收站（注意 rm -rf 的威力，请确保路径无误） rm -rf /root/.local/share/Trash/*
再次检查大小，应显示为 0 或 4.0K du -sh /root/.local/share/Trash
2. 清理数据盘回收站（核心操作）
针对我们发现的那个 22GB 的巨型回收站，必须将其彻底清除才能恢复数据盘性能。
再次确认目标目录（切勿手滑删错） echo "准备清理目录: /root/autodl-tmp/.Trash-0"
执行删除 rm -rf /root/autodl-tmp/.Trash-0
验证清理结果（如果目录不存在说明清理成功）[ ! -d "/root/autodl-tmp/.Trash-0" ] && echo "数据盘回收站已清理完毕"

## 代码推荐执行顺序
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