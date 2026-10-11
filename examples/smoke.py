"""Optional real-model check: python -m examples.smoke (requires Ollama)."""

import json
from pathlib import Path

from easybot import AccountantBot, Message, Turn
from easybot.llm import OllamaClassifier


def main() -> None:
    bot = AccountantBot(OllamaClassifier())
    history = []
    expected = [
        ("silent", None), ("silent", None), ("do", "invoice"),
        ("ask", "invoice"), ("do", "invoice"), ("to_accountant", None),
        ("do", "send_document"), ("do", "balance"), ("do", "taxes"),
        ("to_accountant", None), ("to_accountant", None), ("silent", None),
    ]
    failures = []
    for line, wanted in zip(Path(__file__).with_name("chat.jsonl").read_text(encoding="utf-8").splitlines(), expected, strict=True):
        message = Message(**json.loads(line))
        decision = bot.decide(message, history)
        history.append(Turn(message, decision))
        print(json.dumps({"id": message.id, **decision.to_dict()}, ensure_ascii=False), flush=True)
        if (decision.kind, decision.skill) != wanted:
            failures.append(message.id)
    for index, number in ((2, 32), (4, 33), (6, 28)):
        if history[index].decision.data.get("number") != number:
            failures.append(f"number at {index + 1}")
    if history[7].decision.data.get("balance") != "552314.17":
        failures.append("balance")
    if history[8].decision.data.get("total") != "21675":
        failures.append("taxes")
    if failures:
        raise SystemExit("Unexpected results: " + ", ".join(failures))
    print("All 12 real-model examples passed.")


if __name__ == "__main__":
    main()
