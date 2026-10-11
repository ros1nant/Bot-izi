from copy import deepcopy
from functools import lru_cache
from threading import RLock
from typing import Sequence

from .ledger import Ledger, money
from .llm import Classifier, Extraction, ModelError, ModelInput, OllamaClassifier
from .policy import (
    DISMISSAL, UnsafeRequest, check_numeric_evidence, corroborate_intent,
    grounded, has_mention, parse_amount, resolve_counterparty, risk_reason,
)
from .types import Decision, Message, Turn


def handoff(reason: str) -> Decision:
    return Decision("to_accountant", reason=reason)


class AccountantBot:
    """One instance per client chat; history is trusted, chronological metadata.

    The model is replaceable. The ledger and decision cache last for this process.
    A lock covers both the idempotency check and invoice numbering.
    """

    def __init__(self, model: Classifier, ledger: Ledger | None = None, *, chat_id: str = "demo"):
        self.model = model
        self.ledger = ledger if ledger is not None else Ledger()
        self.chat_id = chat_id
        self._cache: dict[str, tuple[Message, Decision]] = {}
        self._lock = RLock()

    def decide(self, message: Message, history: Sequence[Turn]) -> Decision:
        with self._lock:
            if message.chat_id != self.chat_id:
                return handoff("Чат не связан с данными этого клиента.")
            if message.id in self._cache:
                previous, decision = self._cache[message.id]
                if previous != message:
                    return handoff("Содержимое сообщения с этим идентификатором изменилось.")
                return deepcopy(decision)
            decision = self._decide(message, history)
            self._cache[message.id] = (message, deepcopy(decision))
            return decision

    def _pending(self, message: Message, history: Sequence[Turn]) -> Turn | None:
        if not message.reply_to:
            return None
        # Redelivery of an old question must not reopen a completed interaction.
        unique: dict[tuple[str, str], Turn] = {}
        for turn in history:
            unique.setdefault((turn.message.chat_id, turn.message.id), turn)
        # Only this author's latest active bot interaction can be continued.
        for turn in reversed(list(unique.values())):
            if (turn.message.chat_id != message.chat_id
                    or turn.message.author_id != message.author_id
                    or turn.message.role != message.role
                    or turn.decision.kind == "silent"):
                continue
            if (turn.decision.kind == "ask"
                    and turn.decision.question_id == message.reply_to):
                return turn
            return None
        return None

    def _decide(self, message: Message, history: Sequence[Turn]) -> Decision:
        if message.role == "bot":
            return Decision("silent")
        pending = self._pending(message, history)
        mentioned = has_mention(message.text)
        if not mentioned and pending is None:
            return Decision("silent")
        if len(message.text) > 4000:
            return handoff("Сообщение слишком длинное для надёжного разбора.")
        if DISMISSAL.search(message.text):
            return handoff("Уточнение отменено.") if pending else Decision("silent")
        if message.reply_to and message.reply_to.startswith("izi:") and pending is None:
            return handoff("Этот вопрос уже закрыт или задан другому участнику.")

        sources = (pending.message.text, message.text) if pending else (message.text,)
        for source in sources:
            reason = risk_reason(source)
            if reason:
                return handoff(reason)
        request = ModelInput(
            text=message.text,
            original_request=pending.message.text if pending else None,
            pending_skill=pending.decision.skill if pending else None,
        )
        try:
            parsed = Extraction.parse(self.model.classify(request))
        except (ModelError, TimeoutError, ConnectionError, OSError):
            return handoff("Не удалось надёжно разобрать сообщение моделью.")

        if not parsed.addressed_to_bot or parsed.intent == "none":
            return handoff("После уточнения запрос всё ещё непонятен.") if pending else Decision("silent")
        if not parsed.certain or parsed.scope != "client_business":
            return handoff("Запрос неоднозначен или относится не к бизнесу клиента.")
        if parsed.intent == "unsupported":
            return handoff("Бот не умеет выполнять эту просьбу.")
        if pending and parsed.intent != pending.decision.skill:
            return handoff("После уточнения изменилось действие; нужен бухгалтер.")

        try:
            corroborate_intent(parsed.intent, "\n".join(sources))
            for quote in (parsed.counterparty_quote, parsed.amount_quote, parsed.number_quote):
                grounded(quote, sources)
            return self._execute(message, parsed, sources, pending is not None)
        except UnsafeRequest as exc:
            return handoff(str(exc))

    def _ask(self, message: Message, skill: str, question: str, already_asked: bool) -> Decision:
        if already_asked:
            return handoff("После одного уточнения всё ещё не хватает данных.")
        return Decision("ask", skill=skill, answer=question, question_id=f"izi:{message.id}")

    def _execute(self, message: Message, parsed: Extraction, sources: tuple[str, ...], already_asked: bool) -> Decision:
        skill = parsed.intent
        if skill == "invoice":
            if parsed.number_quote is not None:
                raise UnsafeRequest("Для нового счёта нельзя использовать номер существующего.")
            counterparty = (
                resolve_counterparty(parsed.counterparty_quote, sources, self.ledger.counterparties)
                if parsed.counterparty_quote is not None else None
            )
            check_numeric_evidence(parsed.amount_quote, sources)
            amount = parse_amount(parsed.amount_quote) if parsed.amount_quote is not None else None
            if counterparty is None or amount is None:
                if counterparty is None and amount is None:
                    question = "На кого и на какую сумму выставить счёт?"
                elif counterparty is None:
                    question = "На кого выставить счёт?"
                else:
                    question = "На какую сумму выставить счёт?"
                return self._ask(message, skill, question, already_asked)
            invoice = self.ledger.issue(counterparty, amount)
            return Decision("do", skill=skill, answer=f"Выставлен счёт № {invoice.number}: {counterparty}, {money(amount)}.", data=invoice.to_dict())

        if skill == "send_document":
            if parsed.counterparty_quote is not None or parsed.amount_quote is not None:
                raise UnsafeRequest("Непонятно, какой документ запрошен.")
            check_numeric_evidence(parsed.number_quote, sources)
            if parsed.number_quote is None:
                return self._ask(message, skill, "Какой номер счёта прислать?", already_asked)
            if not parsed.number_quote.isascii() or not parsed.number_quote.isdigit():
                raise UnsafeRequest("Номер счёта должен состоять из цифр.")
            number = int(parsed.number_quote)
            invoice = self.ledger.invoices.get(number)
            if invoice is None:
                return handoff(f"Счёт № {number} не найден; передаю бухгалтеру.")
            return Decision("do", skill=skill, answer=f"Счёт № {number}: {invoice.counterparty}, {money(invoice.amount)}.", data=invoice.to_dict())

        if any(value is not None for value in (parsed.counterparty_quote, parsed.amount_quote, parsed.number_quote)):
            raise UnsafeRequest("В запросе есть лишние или противоречивые реквизиты.")
        check_numeric_evidence(None, sources)
        if skill == "balance":
            return Decision("do", skill=skill, answer=f"На счету {money(self.ledger.balance)}.", data={"balance": str(self.ledger.balance)})
        if skill == "taxes":
            total = sum(self.ledger.taxes.values())
            details = "; ".join(f"{name} — {money(value)}" for name, value in self.ledger.taxes.items())
            return Decision("do", skill=skill, answer=f"К уплате {money(total)}: {details}.", data={"total": str(total), "breakdown": {key: str(value) for key, value in self.ledger.taxes.items()}})
        return handoff("Бот не умеет выполнять эту просьбу.")


@lru_cache(maxsize=1)
def _default_bot() -> AccountantBot:
    return AccountantBot(OllamaClassifier())


def decide(message: Message, history: Sequence[Turn]) -> Decision:
    """Convenience entry point for the single demo chat; inject models via the class."""
    return _default_bot().decide(message, history)
