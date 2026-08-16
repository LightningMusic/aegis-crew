from __future__ import annotations

from typing import Any, Callable, Literal, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])

Message = dict[str, Any]

class Agent:
    name: str
    def __init__(self, name: str, **kwargs: Any) -> None: ...

class ChatResult:
    messages: list[Message]

class ConversableAgent(Agent):
    messages: list[Message]
    def __init__(
        self,
        name: str,
        system_message: str | list[str] | None = ...,
        is_termination_msg: Callable[[dict[str, Any]], bool] | None = ...,
        max_consecutive_auto_reply: int | None = ...,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = ...,
        function_map: dict[str, Callable[..., Any]] | None = ...,
        code_execution_config: dict[str, Any] | Literal[False] = ...,
        llm_config: dict[str, Any] | Literal[False] | None = ...,
        default_auto_reply: str | dict[str, Any] = ...,
        description: str | None = ...,
        chat_messages: dict[Agent, list[Message]] | None = ...,
        silent: bool | None = ...,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def register_for_llm(
        self,
        *,
        name: str | None = ...,
        description: str | None = ...,
        api_style: Literal["function", "tool"] = ...,
    ) -> Callable[[F], F]: ...
    @overload
    def register_for_execution(self, name: str | None = ...) -> Callable[[F], F]: ...
    def register_reply(
        self,
        trigger: type[Agent] | str | Agent | Callable[[Agent], bool] | list[Any],
        reply_func: Callable[..., Any],
        position: int = ...,
        config: Any | None = ...,
        reset_config: Callable[..., Any] | None = ...,
        *,
        ignore_async_in_sync_chat: bool = ...,
        remove_other_reply_funcs: bool = ...,
    ) -> None: ...
    def initiate_chat(
        self,
        recipient: ConversableAgent,
        clear_history: bool = ...,
        silent: bool | None = ...,
        cache: Any | None = ...,
        max_turns: int | None = ...,
        summary_method: str | Callable[..., Any] | None = ...,
        summary_args: dict[str, Any] | None = ...,
        message: dict[str, Any] | str | Callable[..., Any] | None = ...,
        **kwargs: Any,
    ) -> ChatResult: ...

class AssistantAgent(ConversableAgent):
    def __init__(
        self,
        name: str,
        system_message: str | None = ...,
        llm_config: dict[str, Any] | Literal[False] | None = ...,
        is_termination_msg: Callable[[dict[str, Any]], bool] | None = ...,
        max_consecutive_auto_reply: int | None = ...,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = ...,
        description: str | None = ...,
        **kwargs: Any,
    ) -> None: ...

class UserProxyAgent(ConversableAgent):
    _function_map: dict[str, Callable[..., Any]]
    def __init__(
        self,
        name: str,
        is_termination_msg: Callable[[dict[str, Any]], bool] | None = ...,
        max_consecutive_auto_reply: int | None = ...,
        human_input_mode: Literal["ALWAYS", "TERMINATE", "NEVER"] = ...,
        function_map: dict[str, Callable[..., Any]] | None = ...,
        code_execution_config: dict[str, Any] | Literal[False] = ...,
        default_auto_reply: str | dict[str, Any] | None = ...,
        llm_config: dict[str, Any] | Literal[False] | None = ...,
        system_message: str | list[str] | None = ...,
        description: str | None = ...,
        **kwargs: Any,
    ) -> None: ...

class GroupChat:
    messages: list[Message]
    def __init__(
        self,
        agents: list[Agent],
        messages: list[Message],
        max_round: int = ...,
        admin_name: str = ...,
        func_call_filter: bool = ...,
        speaker_selection_method: str | Callable[..., Any] = ...,
        max_retries_for_selecting_speaker: int = ...,
        allow_repeat_speaker: bool | list[Agent] | None = ...,
        allowed_or_disallowed_speaker_transitions: dict[str, Any] | None = ...,
        speaker_transitions_type: Literal["allowed", "disallowed"] | None = ...,
        enable_clear_history: bool = ...,
        send_introductions: bool = ...,
        select_speaker_message_template: str = ...,
        select_speaker_prompt_template: str = ...,
        select_speaker_auto_multiple_template: str = ...,
        select_speaker_auto_none_template: str = ...,
        select_speaker_auto_verbose: bool | None = ...,
        role_for_select_speaker_messages: str | None = ...,
    ) -> None: ...

class GroupChatManager(ConversableAgent):
    def __init__(
        self,
        groupchat: GroupChat,
        name: str | None = ...,
        max_consecutive_auto_reply: int = ...,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = ...,
        system_message: str | list[str] | None = ...,
        silent: bool = ...,
        **kwargs: Any,
    ) -> None: ...

class OpenAIWrapper: ...

class AgentNameConflict(Exception): ...

class SenderRequired(Exception): ...

class NoEligibleSpeaker(Exception): ...

class UndefinedNextAgent(Exception): ...

class InvalidCarryOverType(Exception): ...

DEFAULT_MODEL: str
FAST_MODEL: str

def initiate_chats(*args: Any, **kwargs: Any) -> Any: ...

def register_function(*args: Any, **kwargs: Any) -> Any: ...
