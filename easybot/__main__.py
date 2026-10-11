"""JSON Lines runner: stdin messages -> stdout decisions, same chat/process."""

import json
import sys

from . import Message, Turn, decide


def main() -> None:
    history: list[Turn] = []
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = Message(**json.loads(line))
        except (ValueError, TypeError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
            continue
        decision = decide(message, history)
        history.append(Turn(message, decision))
        print(json.dumps(decision.to_dict(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
