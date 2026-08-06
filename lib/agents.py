from typing import TypedDict, List, Optional, Union, Any
import json
import traceback

from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from lib.llm import LLM
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage
from lib.tooling import Tool
from lib.memory import ShortTermMemory


class AgentState(TypedDict, total=False):
    user_query: str
    instructions: str
    messages: List[Any]
    current_tool_calls: Optional[List[Any]]
    total_tokens: int
    session_id: str
    user_message_added: bool


class Agent:
    def __init__(
        self,
        model_name: str,
        instructions: str,
        tools: List[Tool] = None,
        temperature: float = 0.7,
    ):
        self.instructions = instructions
        self.tools = tools if tools else []
        self.model_name = model_name
        self.temperature = temperature

        self.llm = LLM(
            model=self.model_name,
            temperature=self.temperature,
            tools=self.tools
        )

        self.memory = ShortTermMemory()
        self.workflow = self._create_state_machine()

    def _prepare_messages_step(self, state: AgentState):
        messages = list(state.get("messages", []))

        if not messages:
            messages.append(SystemMessage(content=self.instructions))

        user_message_added = state.get("user_message_added", False)
        if not user_message_added:
            messages.append(UserMessage(content=state["user_query"]))
            user_message_added = True

        return {
            "messages": messages,
            "user_message_added": user_message_added
        }

    def _llm_step(self, state: AgentState):
        response = self.llm.invoke(state["messages"])
        tool_calls = getattr(response, "tool_calls", None)

        if tool_calls:
            ai_message = AIMessage(
                content=getattr(response, "content", None),
                tool_calls=tool_calls
            )
        else:
            ai_message = AIMessage(
                content=getattr(response, "content", None)
            )

        messages = list(state["messages"])
        messages.append(ai_message)

        total_tokens = state.get("total_tokens", 0)
        usage = getattr(response, "token_usage", None)

        if usage:
            if isinstance(usage, dict):
                total_tokens += usage.get("total_tokens", 0)
            else:
                total_tokens += getattr(usage, "total_tokens", 0)

        return {
            "messages": messages,
            "current_tool_calls": tool_calls,
            "total_tokens": total_tokens
        }

    def _tool_step(self, state: AgentState):
        tool_calls = state.get("current_tool_calls") or []
        tool_messages = []

        for call in tool_calls:
            try:
                tool_name = call.function.name
                tool_args = json.loads(call.function.arguments)

                selected_tool = None
                for tool in self.tools:
                    if tool.name == tool_name:
                        selected_tool = tool
                        break

                if selected_tool is None:
                    result = {"error": f"Tool '{tool_name}' not found"}
                else:
                    result = selected_tool(**tool_args)

                if hasattr(result, "model_dump"):
                    result = result.model_dump()
                elif hasattr(result, "dict"):
                    result = result.dict()

                tool_messages.append(
                    ToolMessage(
                        tool_call_id=call.id,
                        name=tool_name,
                        content=json.dumps(result)
                    )
                )

            except Exception as e:
                traceback.print_exc()
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=getattr(call, "id", "unknown"),
                        name=getattr(getattr(call, "function", None), "name", "unknown_tool"),
                        content=json.dumps({"error": str(e)})
                    )
                )

        messages = list(state["messages"])
        messages.extend(tool_messages)

        return {
            "messages": messages,
            "current_tool_calls": None
        }

    def _create_state_machine(self) -> StateMachine[AgentState]:
        machine = StateMachine[AgentState](AgentState)

        entry = EntryPoint[AgentState]()
        message_prep = Step[AgentState]("message_prep", self._prepare_messages_step)
        llm_processor = Step[AgentState]("llm_processor", self._llm_step)
        tool_executor = Step[AgentState]("tool_executor", self._tool_step)
        termination = Termination[AgentState]()

        machine.add_steps([entry, message_prep, llm_processor, tool_executor, termination])

        machine.connect(entry, message_prep)
        machine.connect(message_prep, llm_processor)

        def check_tool_calls(state: AgentState) -> Union[Step[AgentState], str]:
            if state.get("current_tool_calls"):
                return tool_executor
            return termination

        machine.connect(llm_processor, [tool_executor, termination], check_tool_calls)
        machine.connect(tool_executor, llm_processor)

        return machine

    def invoke(self, user_query: str, session_id: str = "default"):
        if session_id not in self.memory.sessions:
            self.memory.create_session(session_id)

        memory_state = self.memory.get_last_object(session_id)

        if memory_state is None:
            state = AgentState(
                user_query=user_query,
                instructions=self.instructions,
                messages=[],
                current_tool_calls=None,
                total_tokens=0,
                session_id=session_id,
                user_message_added=False
            )
        else:
            state = AgentState(
                user_query=user_query,
                instructions=self.instructions,
                messages=memory_state.get("messages", []),
                current_tool_calls=None,
                total_tokens=memory_state.get("total_tokens", 0),
                session_id=session_id,
                user_message_added=False
            )

        run = self.workflow.run(state)
        final_state = run.get_final_state()

        self.memory.add(final_state, session_id)

        return run

    def get_session_runs(self, session_id: Optional[str] = None) -> List[Run]:
        session_id = session_id or "default"
        return self.memory.get_all_objects(session_id)

    def reset_session(self, session_id: Optional[str] = None):
        session_id = session_id or "default"
        self.memory.reset(session_id)
