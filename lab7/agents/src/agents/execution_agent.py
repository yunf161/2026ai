"""
执行智能体

继承 BaseAgent，实现 ReAct 循环的执行节点。
工具以 (StructuredTool, raw_func) 元组存储，兼容 inspect.signature()。
"""
from typing import List, Dict, Any, Optional, Callable
import json

from .base_agent import BaseAgent
from config import config
from state import SubTask

MAX_REACT_STEPS = 10


class ExecutionAgent(BaseAgent):
    """执行智能体"""

    def __init__(self, name: str, system_prompt: str,
                 tools: Optional[List[Callable]] = None):
        super().__init__(name, "executor", system_prompt)
        # tools 存储为 {name: (tool_obj, raw_func)}
        self.tools: Dict[str, tuple] = {}
        if tools:
            for t in tools:
                self.bind_tool(t)

    # ================================================================
    # 工具管理
    # ================================================================

    def bind_tool(self, tool: Callable) -> None:
        tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown_tool")
        raw_func = getattr(tool, "func", tool)  # StructuredTool.func = 原始函数
        self.tools[tool_name] = (tool, raw_func)

    def invoke_tool(self, tool_name: str, *args, **kwargs) -> str:
        if tool_name not in self.tools:
            return f"[Error] 工具 '{tool_name}' 未绑定。可用: {list(self.tools.keys())}"
        tool_obj, _ = self.tools[tool_name]
        try:
            if kwargs:
                result = tool_obj.invoke(kwargs)
            elif args:
                result = tool_obj.invoke(args[0])
            else:
                result = tool_obj.invoke({})
            return str(result)
        except Exception as e:
            return f"[Error] 工具 {tool_name} 失败: {e}"

    def _get_tools_description(self) -> str:
        if not self.tools:
            return "（无可用工具）"
        lines = []
        for name, (tool_obj, _) in self.tools.items():
            desc = getattr(tool_obj, "description", name)
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    # ================================================================
    # ReAct 核心循环
    # ================================================================

    def react_loop(self, task: SubTask) -> str:
        task_desc = task.get("description", str(task))
        tools_desc = self._get_tools_description()
        self.log(f"开始 ReAct 执行: {task_desc[:60]}...")

        observations = []

        for step in range(MAX_REACT_STEPS):
            obs_text = "\n".join(
                f"[观察{i + 1}] {obs}" for i, obs in enumerate(observations)
            ) if observations else "（尚无观察结果）"

            think_prompt = f"""你是执行Agent，需要完成以下任务。你可以调用工具。

任务: {task_desc}

可用工具:
{tools_desc}

已有观察结果:
{obs_text}

请决定下一步操作。用以下格式回复:

如果还需要调用工具:
Action: <工具名>
Input: <工具参数>

如果任务已完成:
FinalAnswer: <最终结果摘要>

注意: 只用一种格式回复。"""

            response = self.call_llm(think_prompt)
            self.log(f"Step {step + 1}: {response[:100]}...")

            if response.startswith("FinalAnswer:"):
                final = response[len("FinalAnswer:"):].strip()
                self.remember_now("execution",
                                  f"Completed: {task_desc[:60]} → {final[:80]}")
                return final

            # Some tools/LLMs return the required structured JSON directly.
            # Treat raw JSON as a completed answer instead of a format error.
            stripped_response = response.strip()
            if stripped_response.startswith("{") or stripped_response.startswith("["):
                try:
                    json.loads(stripped_response)
                    self.remember_now("execution", f"Completed JSON: {task_desc[:60]}")
                    return stripped_response
                except Exception:
                    pass

            if "Action:" in response and "Input:" in response:
                action_line = ""
                input_line = ""
                for line in response.split("\n"):
                    if line.startswith("Action:"):
                        action_line = line[len("Action:"):].strip()
                    elif line.startswith("Input:"):
                        input_line = line[len("Input:"):].strip()

                if action_line and action_line in self.tools:
                    self.log(f"调用工具: {action_line}({input_line[:50]}...)")
                    parsed = self._parse_input(input_line, action_line)
                    if parsed:
                        result = self.invoke_tool(action_line, **parsed)
                    else:
                        result = self.invoke_tool(action_line, input_line)
                    observations.append(str(result))
                    continue

            self.log("无法解析响应格式，当作最终答案")
            return response

        self.log(f"ReAct 达到最大步数 {MAX_REACT_STEPS}，返回已有结果")
        return "\n\n".join(observations) if observations else "[ReAct] 未完成"

    def _parse_input(self, input_str: str, tool_name: str) -> Dict[str, str]:
        """解析 key=value 格式的参数"""
        s = str(input_str or "").strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return {str(k): str(v) for k, v in obj.items()}
            except Exception:
                pass

        params = {}
        for part in input_str.split(","):
            part = part.strip()
            if "=" in part:
                key, _, value = part.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                params[key] = value
            elif part:
                import inspect
                _, raw_func = self.tools.get(tool_name, (None, None))
                if raw_func and callable(raw_func):
                    try:
                        sig = inspect.signature(raw_func)
                        for pname in sig.parameters:
                            if pname not in params and pname != "self":
                                params[pname] = part
                                break
                    except (ValueError, TypeError):
                        pass
        return params

    # ================================================================
    # process
    # ================================================================

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state.get("plan", [])
        current_idx = state.get("current_task_index", 0)

        if current_idx >= len(plan):
            return state

        current_task = plan[current_idx]
        result = self.react_loop(current_task)

        result_key = f"_{self.name}_result"
        state[result_key] = result

        state.setdefault("execution_log", []).append(
            f"[{self.name}] result: {str(result)[:100]}..."
        )
        self.remember_now("execution", 
                          f"Executed '{current_task.get('description', '')[:60]}'")

        return state
