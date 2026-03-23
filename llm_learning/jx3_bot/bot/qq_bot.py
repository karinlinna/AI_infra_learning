"""
剑网三 QQ 问答机器人

基于 NoneBot2 + OneBot V11 协议，对接 go-cqhttp。

前置要求：
1. 安装并配置 go-cqhttp（https://docs.go-cqhttp.org/）
2. 启动推理服务（inference/server.py）

使用方式：
    python qq_bot.py

群聊触发：
    @剑三小助手 纯阳适合新手吗？
    /jx3 怎么赚钱？

私聊触发：
    直接发送问题即可
"""

import sys
import asyncio

import httpx
import nonebot
from nonebot import on_command, on_message
from nonebot.adapters.onebot.v11 import (
    Adapter as OneBotV11Adapter,
    Bot,
    GroupMessageEvent,
    PrivateMessageEvent,
    MessageEvent,
    Message,
    MessageSegment,
)
from nonebot.rule import to_me

from config import (
    API_BASE_URL,
    COMMAND_PREFIX,
    ENABLE_PRIVATE_CHAT,
    ENABLE_GROUP_CHAT,
    ENABLE_SEARCH,
    MAX_REPLY_LENGTH,
    THINKING_MESSAGE,
    ERROR_MESSAGE,
    ADMIN_QQ_LIST,
    BLACKLIST_GROUPS,
    WHITELIST_GROUPS,
)


# ============================================================
# 初始化 NoneBot
# ============================================================

nonebot.init()
driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)


# ============================================================
# 工具函数
# ============================================================

async def call_api(question: str, enable_search: bool = True) -> str:
    """调用推理服务 API"""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{API_BASE_URL}/chat",
            json={
                "question": question,
                "enable_search": enable_search,
                "max_new_tokens": 512,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", ERROR_MESSAGE)

        # 截断过长的回复
        if len(answer) > MAX_REPLY_LENGTH:
            answer = answer[:MAX_REPLY_LENGTH] + "..."

        # 如果使用了搜索，添加提示
        if data.get("search_used"):
            answer += "\n\n（已参考最新网络资料）"

        return answer


def should_respond_group(event: GroupMessageEvent) -> bool:
    """判断是否应该响应群消息"""
    if not ENABLE_GROUP_CHAT:
        return False

    group_id = event.group_id
    if BLACKLIST_GROUPS and group_id in BLACKLIST_GROUPS:
        return False
    if WHITELIST_GROUPS and group_id not in WHITELIST_GROUPS:
        return False

    return True


def extract_question(event: MessageEvent) -> str:
    """从消息中提取问题文本"""
    text = event.get_plaintext().strip()

    # 去掉命令前缀
    if text.startswith(COMMAND_PREFIX):
        text = text[len(COMMAND_PREFIX):].strip()

    return text


# ============================================================
# 命令处理器：/jx3 <问题>
# ============================================================

jx3_cmd = on_command("jx3", priority=5, block=True)


@jx3_cmd.handle()
async def handle_jx3_command(bot: Bot, event: MessageEvent):
    """处理 /jx3 命令"""
    # 群聊检查
    if isinstance(event, GroupMessageEvent) and not should_respond_group(event):
        return

    question = extract_question(event)
    if not question:
        await jx3_cmd.finish("请输入你的问题，例如：/jx3 纯阳适合新手吗？")

    # 发送思考提示
    await jx3_cmd.send(THINKING_MESSAGE)

    try:
        answer = await call_api(question, ENABLE_SEARCH)
        await jx3_cmd.finish(answer)
    except Exception as e:
        print(f"API 调用失败: {e}")
        await jx3_cmd.finish(ERROR_MESSAGE)


# ============================================================
# @机器人 触发
# ============================================================

at_me = on_message(rule=to_me(), priority=10, block=True)


@at_me.handle()
async def handle_at_me(bot: Bot, event: MessageEvent):
    """处理 @ 机器人的消息"""
    if isinstance(event, GroupMessageEvent) and not should_respond_group(event):
        return

    question = extract_question(event)
    if not question:
        await at_me.finish("有什么剑三问题尽管问我~")

    await at_me.send(THINKING_MESSAGE)

    try:
        answer = await call_api(question, ENABLE_SEARCH)
        await at_me.finish(answer)
    except Exception as e:
        print(f"API 调用失败: {e}")
        await at_me.finish(ERROR_MESSAGE)


# ============================================================
# 私聊处理
# ============================================================

private_chat = on_message(priority=99, block=False)


@private_chat.handle()
async def handle_private(bot: Bot, event: PrivateMessageEvent):
    """处理私聊消息"""
    if not ENABLE_PRIVATE_CHAT:
        return

    question = extract_question(event)
    if not question or len(question) < 2:
        return

    try:
        answer = await call_api(question, ENABLE_SEARCH)
        await private_chat.finish(answer)
    except Exception as e:
        print(f"API 调用失败: {e}")
        await private_chat.finish(ERROR_MESSAGE)


# ============================================================
# 管理命令
# ============================================================

admin_cmd = on_command("jx3admin", priority=1, block=True)


@admin_cmd.handle()
async def handle_admin(bot: Bot, event: MessageEvent):
    """管理命令"""
    user_id = event.get_user_id()
    if ADMIN_QQ_LIST and int(user_id) not in ADMIN_QQ_LIST:
        return

    text = extract_question(event)

    if text == "status":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{API_BASE_URL}/health")
                data = resp.json()
                await admin_cmd.finish(f"服务状态: {data}")
        except Exception as e:
            await admin_cmd.finish(f"服务异常: {e}")

    elif text == "help":
        await admin_cmd.finish(
            "管理命令：\n"
            "/jx3admin status - 查看服务状态\n"
            "/jx3admin help - 查看帮助"
        )


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    nonebot.run()
