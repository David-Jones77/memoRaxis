from memory_system import AgenticMemorySystem


memory_system = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend="deepseek",
    llm_model="deepseek-chat",
    api_key="sk-8c36866613a445b9951aa367451c8f87"
)

print("=== Agentic Memory Chat System ===")
print("输入 exit 退出。")
print("----------------------------------")

while True:
    user_input = input("你: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    # 1) 先把用户消息写入记忆库（自动生成 keywords/context/tags）
    mem_id = memory_system.add_note(user_input)

    # 2) 检索相关记忆（用 embedding + metadata）
    related = memory_system.search_agentic(user_input, k=5)

    # 整理历史记忆内容作为提示
    memory_context = "\n".join([
        f"- {m['content']}  (tags={m['tags']})"
        for m in related
    ])

    if memory_context.strip():
        prompt = (
            "你是一名具有长期记忆能力的助手。\n"
            f"以下是与你当前问题语义相关的记忆：\n{memory_context}\n\n"
            f"现在请回答用户：{user_input}"
        )
    else:
        prompt = user_input

    # 3) 调用 LLM 生成回复
    reply = memory_system.llm_controller.llm.get_completion(prompt, response_format=None)


    print("AI:", reply)
    print("----------------------------------")
