#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zep 记忆系统实现
适配 BaseMemorySystem 接口
"""

import asyncio
import uuid
from typing import Dict, Any, List

from src.memory_interface import BaseMemorySystem, Evidence
from src.logger import get_logger

# 导入 Zep 客户端
from zep_cloud.client import AsyncZep
from zep_cloud.types import Message


class ZepMemorySystem(BaseMemorySystem):
    """Zep 记忆系统实现"""

    def __init__(self, api_key: str = None, enable_llm: bool = False):
        """
        初始化 Zep 记忆系统

        Args:
            api_key: Zep API 密钥
            enable_llm: 是否启用 LLM 增强
        """
        self._logger = get_logger()
        self.api_key = "z_1dWlkIjoiNTMyNDk2NzItYjI5Mi00NmNiLWJhYTUtNzM3OGI0MmVjNmE0In0.I3Hl6-fEhigRhvxHRTFEaZ0BGZb6eogFIKGY8MPmRdriHWF8tniCT9pFKc30n8HbOUD6v3oE4UZr_VtSNxSbhg"
        self.enable_llm = enable_llm
        
        # 初始化事件循环
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # 初始化 Zep 客户端
        self.client = AsyncZep(api_key=self.api_key)
        
        # 创建默认用户和线程
        self.user_id = str(uuid.uuid4())
        self.thread_id = str(uuid.uuid4())
        
        # 初始化用户和线程
        self._init_user_and_thread()
        
        self._logger.info("ZepMemorySystem 初始化完成")

    def _run_async(self, coro):
        """运行异步协程

        Args:
            coro: 异步协程

        Returns:
            协程结果
        """
        try:
            if self.loop.is_running():
                # 如果事件循环正在运行，使用 create_task
                future = asyncio.run_coroutine_threadsafe(coro, self.loop)
                return future.result()
            else:
                # 如果事件循环未运行，使用 run_until_complete
                return self.loop.run_until_complete(coro)
        except Exception as e:
            self._logger.error(f"运行异步协程失败: {e}")
            return None

    def _init_user_and_thread(self):
        """初始化用户和线程"""
        async def init():
            try:
                # 添加用户
                await self.client.user.add(
                    user_id=self.user_id,
                    email="user@example.com",
                    first_name="Test",
                    last_name="User"
                )
                self._logger.info(f"Zep 用户创建成功: {self.user_id}")
                
                # 等待用户创建完成
                await asyncio.sleep(2)
                
                # 创建线程
                await self.client.thread.create(
                    thread_id=self.thread_id, 
                    user_id=self.user_id
                )
                self._logger.info(f"Zep 线程创建成功: {self.thread_id}")
                
                # 等待线程创建完成
                await asyncio.sleep(2)
                
                # 验证线程是否存在
                try:
                    thread = await self.client.thread.get(self.thread_id)
                    if thread:
                        self._logger.info("Zep 线程验证成功")
                    else:
                        self._logger.warning("Zep 线程验证失败")
                except Exception as e:
                    self._logger.warning(f"Zep 线程验证失败: {e}")
                    
            except Exception as e:
                self._logger.error(f"Zep 用户和线程初始化失败: {e}")
                # 尝试使用默认线程
                self.thread_id = "default-thread-" + self.user_id[:8]
                try:
                    await self.client.thread.create(
                        thread_id=self.thread_id, 
                        user_id=self.user_id
                    )
                    self._logger.info(f"使用默认线程: {self.thread_id}")
                except Exception as e2:
                    self._logger.error(f"创建默认线程失败: {e2}")
        
        self._run_async(init())

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        """添加记忆

        Args:
            data: 记忆内容
            metadata: 元数据
        """
        async def add():
            try:
                # 验证线程是否存在
                try:
                    thread = await self.client.thread.get(self.thread_id)
                    if not thread:
                        self._logger.warning(f"线程不存在: {self.thread_id}")
                        # 尝试重新创建线程
                        await self.client.thread.create(
                            thread_id=self.thread_id, 
                            user_id=self.user_id
                        )
                        await asyncio.sleep(2)
                except Exception as e:
                    self._logger.warning(f"验证线程失败: {e}")
                    # 尝试重新创建线程
                    try:
                        await self.client.thread.create(
                            thread_id=self.thread_id, 
                            user_id=self.user_id
                        )
                        await asyncio.sleep(2)
                        self._logger.info(f"线程重新创建成功: {self.thread_id}")
                    except Exception as e2:
                        self._logger.error(f"线程重新创建失败: {e2}")
                        return
                
                # 构建消息内容，参考官方示例的格式
                content = data
                if metadata:
                    content += f"\n\n元数据: {metadata}"
                
                # 创建消息，参考官方示例的格式
                message = Message(
                    role="user",
                    content=content
                )
                
                # 添加消息到线程，使用官方示例的方式
                await self.client.thread.add_messages(
                    thread_id=self.thread_id,
                    messages=[message]
                )
                self._logger.info(f"添加记忆到 Zep: {data[:50]}...")
                
                # 等待 Zep 处理消息（增加等待时间）
                await asyncio.sleep(15)
                
                # 添加助手回复，模拟对话过程，参考官方示例
                assistant_message = Message(
                    role="assistant",
                    content="I understand. Let me help you with that."
                )
                await self.client.thread.add_messages(
                    thread_id=self.thread_id,
                    messages=[assistant_message]
                )
                self._logger.info("添加助手回复到 Zep")
                
                # 等待 Zep 处理助手回复
                await asyncio.sleep(8)
                
                # 验证消息是否添加成功
                try:
                    messages = await self.client.thread.get_messages(self.thread_id)
                    if messages:
                        self._logger.info(f"消息添加成功，当前线程消息数: {len(messages)}")
                    else:
                        self._logger.warning("消息添加失败，线程消息数为0")
                except Exception as e:
                    self._logger.warning(f"验证消息失败: {e}")
                
            except Exception as e:
                self._logger.error(f"添加记忆失败: {e}")
        
        self._run_async(add())

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        """检索记忆

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            检索到的证据列表
        """
        async def get_context():
            try:
                # 验证线程是否存在
                try:
                    thread = await self.client.thread.get(self.thread_id)
                    if not thread:
                        self._logger.warning(f"线程不存在: {self.thread_id}")
                        return []
                except Exception as e:
                    self._logger.warning(f"验证线程失败: {e}")
                    # 尝试重新创建线程
                    try:
                        await self.client.thread.create(
                            thread_id=self.thread_id, 
                            user_id=self.user_id
                        )
                        await asyncio.sleep(2)
                        self._logger.info(f"线程重新创建成功: {self.thread_id}")
                    except Exception as e2:
                        self._logger.error(f"线程重新创建失败: {e2}")
                        return []
                
                # 首先添加查询作为用户消息，让 Zep 知道我们的查询内容
                # 参考官方示例的消息格式
                query_message = Message(
                    role="user",
                    content=query
                )
                await self.client.thread.add_messages(
                    thread_id=self.thread_id,
                    messages=[query_message]
                )
                self._logger.info(f"添加查询消息到 Zep: {query[:50]}...")
                
                # 等待 Zep 处理查询消息（增加等待时间）
                await asyncio.sleep(5)
                
                # 获取用户上下文，参考官方示例的方式
                try:
                    memory = await self.client.thread.get_user_context(self.thread_id)
                    context_text = memory.context or ""
                    
                    self._logger.info(f"从 Zep 获取用户上下文: {context_text[:100]}...")
                    
                    # 将上下文转换为 Evidence 对象
                    if context_text:
                        evidence = Evidence(
                            content=context_text,
                            metadata={"source": "zep_user_context", "thread_id": self.thread_id, "query": query}
                        )
                        return [evidence]
                except Exception as e:
                    self._logger.error(f"获取用户上下文失败: {e}")
                
                # 如果没有获取到上下文，尝试获取线程的最近消息
                try:
                    messages = await self.client.thread.get_messages(self.thread_id)
                    if messages:
                        self._logger.info(f"获取到 {len(messages)} 条线程消息")
                        # 提取最近的消息内容作为证据
                        recent_content = "\n".join([msg.content for msg in messages[-10:] if msg.content])
                        if recent_content:
                            evidence = Evidence(
                                content=recent_content,
                                metadata={"source": "zep_recent_messages", "thread_id": self.thread_id, "query": query}
                            )
                            return [evidence]
                        else:
                            self._logger.warning("没有找到有效的消息内容")
                    else:
                        self._logger.warning("线程中没有消息")
                except Exception as e:
                    self._logger.error(f"获取线程消息失败: {e}")
                
                # 尝试使用图搜索 API
                try:
                    self._logger.info("尝试使用图搜索 API")
                    # 搜索相关的事实和实体
                    edges_results = await self.client.graph.search(
                        user_id=self.user_id, query=query, limit=top_k
                    )
                    
                    # 构建上下文
                    facts = []
                    for edge in edges_results.edges or []:
                        start_date = edge.valid_at if edge.valid_at else "date unknown"
                        end_date = edge.invalid_at if edge.invalid_at else "present"
                        facts.append(f"  - {edge.fact} ({start_date} - {end_date})")
                    
                    if facts:
                        context_text = f"""
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges. If the fact is about an event, the event takes place during this time.
# format: FACT (Date range: from - to)
<FACTS>
{chr(10).join(facts)}
</FACTS>
"""
                        evidence = Evidence(
                            content=context_text,
                            metadata={"source": "zep_graph", "user_id": self.user_id, "query": query}
                        )
                        return [evidence]
                except Exception as e:
                    self._logger.warning(f"图搜索失败: {e}")
                
                return []
            except Exception as e:
                self._logger.error(f"检索记忆失败: {e}")
                return []
        
        results = self._run_async(get_context())
        # 限制返回数量
        return results[:top_k] if results else []

    def reset(self) -> None:
        """重置记忆系统"""
        async def reset_thread():
            try:
                # 创建新线程
                new_thread_id = str(uuid.uuid4())
                await self.client.thread.create(
                    thread_id=new_thread_id, 
                    user_id=self.user_id
                )
                
                # 等待线程创建完成
                await asyncio.sleep(2)
                
                # 验证新线程是否存在
                try:
                    thread = await self.client.thread.get(new_thread_id)
                    if thread:
                        self._logger.info(f"新线程验证成功: {new_thread_id}")
                        # 更新线程 ID
                        self.thread_id = new_thread_id
                        self._logger.info("ZepMemorySystem 已重置")
                    else:
                        self._logger.warning(f"新线程验证失败: {new_thread_id}")
                except Exception as e:
                    self._logger.warning(f"新线程验证失败: {e}")
                    # 仍然更新线程 ID，希望后续操作能够成功
                    self.thread_id = new_thread_id
                    self._logger.info("ZepMemorySystem 已重置（验证失败）")
                    
            except Exception as e:
                self._logger.error(f"重置记忆系统失败: {e}")
                # 尝试使用默认线程 ID
                try:
                    default_thread_id = "default-thread-" + str(int(time.time()))[-8:]
                    await self.client.thread.create(
                        thread_id=default_thread_id, 
                        user_id=self.user_id
                    )
                    await asyncio.sleep(2)
                    self.thread_id = default_thread_id
                    self._logger.info(f"使用默认线程重置成功: {default_thread_id}")
                except Exception as e2:
                    self._logger.error(f"使用默认线程重置失败: {e2}")
        
        self._run_async(reset_thread())

    def __del__(self):
        """析构函数，清理事件循环"""
        try:
            if not self.loop.is_closed():
                self.loop.close()
        except Exception:
            pass
