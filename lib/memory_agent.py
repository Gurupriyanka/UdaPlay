from typing import TypedDict, List, Optional, Union
import json

from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from lib.llm import LLM
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage, BaseMessage
from lib.tooling import Tool, ToolCall
from lib.memory import ShortTermMemory


class AgentState(TypedDict, total=False):
    user_query: str
    instructions: str
    messages: List[BaseMessage]
    current_tool_calls: Optional[List[ToolCall]]
    session_id: str


class MemoryAgent:
    def __init__(self, model_name: str, instructions: str, tools: List[Tool] = None, temperature: float = 0.7):
        self.instructions = instructions
        self.tools = tools if tools else []
        self.model_name = model_name
        self.temperature = temperature
        self.memory = ShortTermMemory()
        self.workflow = self._create_state_machine()

    def _prepare_messages_step(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])

        if not messages:
            messages = [SystemMessage(content=state["instructions"])]

        messages.append(UserMessage(content=state["user_query"]))

        return {
            "messages": messages,
            "session_id": state["session_id"]
        }

    def _llm_step(self, state: AgentState) -> AgentState:
        llm = LLM(
            model=self.model_name,
            temperature=self.temperature,
            tools=self.tools
        )

        response = llm.invoke(state["messages"])
        tool_calls = response.tool_calls if response.tool_calls else None
        ai_message = AIMessage(content=response.content, tool_calls=tool_calls)

        return {
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "session_id": state["session_id"]
        }

    def _tool_step(self, state: AgentState) -> AgentState:
        tool_calls = state["current_tool_calls"] or []
        tool_messages = []

        for call in tool_calls:
            function_name = call.function.name
            function_args = json.loads(call.function.arguments)
            tool_call_id = call.id

            tool_obj = next((t for t in self.tools if t.name == function_name), None)
            if tool_obj:
                result = tool_obj(**function_args)
                tool_message = ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_call_id,
                    name=function_name,
                )
                tool_messages.append(tool_message)

        return {
            "messages": state["messages"] + tool_messages,
            "current_tool_calls": None,
            "session_id": state["session_id"]
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

    def invoke(self, query: str, session_id: Optional[str] = "default") -> Run:
        self.memory.create_session(session_id)

        previous_messages = []
        last_run = self.memory.get_last_object(session_id)

        if last_run:
            last_state = last_run.get_final_state()
            previous_messages = last_state.get("messages", [])

        initial_state: AgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "session_id": session_id,
        }

        run_object = self.workflow.run(initial_state)
        self.memory.add(run_object, session_id)
        return run_object

    def get_session_runs(self, session_id: str = "default") -> List[Run]:
        return self.memory.get_all_objects(session_id)

    def reset_session(self, session_id: str = "default"):
        self.memory.reset(session_id)
