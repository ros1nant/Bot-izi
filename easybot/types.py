from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Skill = Literal["invoice", "send_document", "balance", "taxes"]
Kind = Literal["silent", "do", "ask", "to_accountant"]


@dataclass(frozen=True)
class Message:
    id: str
    author_id: str
    text: str
    role: Literal["client", "accountant", "bot"] = "client"
    chat_id: str = "demo"
    reply_to: str | None = None

    def __post_init__(self) -> None:
        for name in ("id", "author_id", "text", "chat_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if self.role not in {"client", "accountant", "bot"}:
            raise ValueError("Unknown message role")
        if self.reply_to is not None and not isinstance(self.reply_to, str):
            raise ValueError("reply_to must be a string or null")


@dataclass(frozen=True)
class Decision:
    kind: Kind
    skill: Skill | None = None
    answer: str | None = None
    reason: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    question_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class Turn:
    message: Message
    decision: Decision
