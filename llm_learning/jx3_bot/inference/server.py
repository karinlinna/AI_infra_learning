"""
剑网三问答推理服务

提供 FastAPI REST API，支持：
- 本地模型推理（Qwen2.5 + LoRA）
- 联网搜索增强
- 流式输出 (SSE)

使用方式：
    # 使用合并后的模型
    python server.py --model ./output/jx3_merged

    # 使用基座 + LoRA adapter
    python server.py --model Qwen/Qwen2.5-7B-Instruct --lora ./output/jx3_lora

    # 测试
    curl -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"question": "纯阳适合新手吗？"}'
"""

import argparse
import json
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from web_search import should_search, search_and_summarize


# ============================================================
# 全局变量
# ============================================================

app = FastAPI(title="剑网三问答助手 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None

SYSTEM_PROMPT = (
    "你是「剑三小助手」，一个专门帮助剑网三萌新的游戏助手。"
    "你熟悉剑网三的所有门派、技能、装备、副本、PVP、日常玩法。"
    "回答要求：语气友好耐心，回答准确，给出具体建议。"
)


# ============================================================
# 请求/响应模型
# ============================================================

class ChatRequest(BaseModel):
    question: str
    enable_search: bool = True  # 是否启用联网搜索
    stream: bool = False        # 是否流式输出
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class ChatResponse(BaseModel):
    answer: str
    search_used: bool = False
    search_context: str | None = None


# ============================================================
# 核心推理逻辑
# ============================================================

def build_messages(question: str, search_context: str | None = None) -> list[dict]:
    """构建对话消息"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if search_context:
        user_content = (
            f"参考以下最新资料回答问题（如果资料与问题无关请忽略）：\n"
            f"{search_context}\n\n"
            f"问题：{question}"
        )
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})
    return messages


def generate_answer(
    question: str,
    enable_search: bool = True,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[str, bool, str | None]:
    """
    生成回答

    返回: (answer, search_used, search_context)
    """
    # 联网搜索
    search_context = None
    search_used = False
    if enable_search:
        search_context = search_and_summarize(question)
        search_used = search_context is not None

    # 构建输入
    messages = build_messages(question, search_context)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 解码（只取新生成的部分）
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return answer, search_used, search_context


def generate_stream(
    question: str,
    enable_search: bool = True,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """流式生成回答"""
    # 联网搜索
    search_context = None
    if enable_search:
        search_context = search_and_summarize(question)

    # 构建输入
    messages = build_messages(question, search_context)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 流式生成
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text_chunk in streamer:
        yield f"data: {json.dumps({'text': text_chunk}, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


# ============================================================
# API 路由
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """问答接口"""
    if req.stream:
        return StreamingResponse(
            generate_stream(
                req.question, req.enable_search,
                req.max_new_tokens, req.temperature, req.top_p,
            ),
            media_type="text/event-stream",
        )

    answer, search_used, search_context = generate_answer(
        req.question, req.enable_search,
        req.max_new_tokens, req.temperature, req.top_p,
    )

    return ChatResponse(
        answer=answer,
        search_used=search_used,
        search_context=search_context,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


# ============================================================
# 启动
# ============================================================

def load_model(model_path: str, lora_path: str | None = None):
    """加载模型"""
    global model, tokenizer

    print(f"加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path or model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path:
        print(f"加载 LoRA: {lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    print("模型加载完成！")


def main():
    parser = argparse.ArgumentParser(description="剑网三问答推理服务")
    parser.add_argument("--model", type=str, default="./output/jx3_merged")
    parser.add_argument("--lora", type=str, default=None, help="LoRA adapter 路径（可选）")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.model, args.lora)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
