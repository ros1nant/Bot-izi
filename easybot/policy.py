"""Conservative vetoes, evidence checks and exact value parsing; no LLM calls."""

import re
from decimal import Decimal

MENTION = re.compile(r"(?<!\w)изи(?!\w)", re.IGNORECASE)
QUOTES = re.compile(r'```[\s\S]*?```|`[^`]*`|«[^»]*»|“[^”]*”|"[^"\n]*"|(?m:^\s*>.*$)')
DISMISSAL = re.compile(r"\bизи\b\s+(?:тут\s+|здесь\s+)?(?:не\s+нужен|не\s+надо|молчи|не\s+отвечай)", re.I)
NUMBER = re.compile(r"(?<![\w.,])[-+]?\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?(?!\w)")


class UnsafeRequest(ValueError):
    pass


def has_mention(text: str) -> bool:
    # A mention inside quoted/forwarded material is not an invitation to act.
    return bool(MENTION.search(QUOTES.sub(" ", text)))


def normalize_name(value: str) -> str:
    words = re.findall(r"\w+", value.casefold().replace("ё", "е"))
    if words and words[0] == "тоо":
        words = words[1:]
    return " ".join(words)


def risk_reason(text: str) -> str | None:
    checks = (
        (r"\b(?:удал\w*|переоформ\w*|аннулир\w*|измен\w*|исправ\w*|перевыстав\w*|сторнир\w*|отмен\w*|замен\w*|оплати\w*|переведи\w*)\b",
         "Запрошено действие, которого бот не умеет."),
        (r"\b(?:обслуживан\w*|абонент\w*|бухгалтерск\w*\s+услуг\w*|ваш\w*\s+сч[её]т\w*)\b",
         "Вопрос об услугах бухгалтерской фирмы требует бухгалтера."),
        (r"\b(?:не|нет|без|нельзя|если|либо|или|наверное|потом|завтра|передай\w*|игнорир\w*|инструкц\w*|промпт\w*|пример\w*|цитат\w*|написал\w*|просил\w*)\b",
         "В запросе есть отрицание, условие или неоднозначность."),
        (r"\b(?:доллар\w*|евро|рубл\w*|usd|eur|rub)\b|[$€₽]",
         "Бот работает только с суммами в тенге."),
        (r"\b(?:тыс|тысяч\w*|млн|миллион\w*|млрд|миллиард\w*)\b",
         "Сумма с сокращением или множителем требует бухгалтера."),
        (r"\b(?:рассчита\w*|пересчита\w*|прогноз\w*)\b|\bза\s+(?:прошл\w*|следующ\w*|квартал\w*|месяц\w*|год\w*)\b",
         "Расчёт и выбор другого периода требуют бухгалтера."),
    )
    for pattern, reason in checks:
        if re.search(pattern, text, re.I):
            return reason
    return None


def corroborate_intent(intent: str, request: str) -> None:
    """Veto unsupported/multiple actions even if the classifier says 'certain'.

    These are conservative lexical witnesses, not a fallback intent classifier:
    without a valid LLM selection they can never authorize an action.
    """
    invoice_word = bool(re.search(r"\bсч[её]т(?:а|у|ом|ы|ов)?\b", request, re.I))
    witnesses = {
        "invoice": invoice_word and bool(re.search(r"\b(?:выстав\w*|созда\w*|сдела\w*|подготов\w*|сформир\w*)\b", request, re.I)),
        "send_document": invoice_word and bool(re.search(r"\b(?:пришл\w*|отправ\w*|покаж\w*|скин\w*|найд\w*|вышл\w*|дайте)\b", request, re.I)),
        "balance": bool(re.search(r"\b(?:баланс\w*|остат\w*|денег|деньги)\b", request, re.I)),
        "taxes": bool(re.search(r"\bналог\w*\b", request, re.I)),
    }
    if not witnesses.get(intent) or sum(witnesses.values()) != 1:
        raise UnsafeRequest("Не удалось однозначно подтвердить одно действие в сообщении.")


def grounded(quote: str | None, sources: tuple[str, ...]) -> None:
    if quote is not None and not any(quote in source for source in sources):
        raise UnsafeRequest("Модель предложила данные, которых нет в запросе.")


def parse_amount(quote: str) -> Decimal:
    if not re.fullmatch(r"(?:\d+|\d{1,3}(?:[ \u00a0\u202f]\d{3})+)(?:[.,]\d{1,2})?", quote):
        raise UnsafeRequest("Не удалось однозначно прочитать сумму.")
    value = Decimal(re.sub(r"[ \u00a0\u202f]", "", quote).replace(",", "."))
    if value <= 0 or value > Decimal("999999999999.99"):
        raise UnsafeRequest("Сумма должна быть положительной и в допустимом диапазоне.")
    return value


def check_numeric_evidence(quote: str | None, sources: tuple[str, ...]) -> None:
    numbers = [match.group() for source in sources for match in NUMBER.finditer(source)]
    if quote is None:
        if numbers:
            raise UnsafeRequest("В сообщении есть число, но его назначение непонятно.")
    elif not numbers or any(number != quote for number in numbers):
        raise UnsafeRequest("В запросе есть неоднозначные или несовпадающие числа.")


def resolve_counterparty(quote: str, sources: tuple[str, ...], known: tuple[str, ...]) -> str:
    matches = [name for name in known if normalize_name(name) == normalize_name(quote)]
    if len(matches) != 1:
        raise UnsafeRequest("Покупатель не найден однозначно в справочнике.")
    mentioned = {
        name for name in known
        if any(re.search(r"(?<!\w)" + re.escape(normalize_name(name)) + r"(?!\w)", normalize_name(source)) for source in sources)
    }
    if mentioned != {matches[0]}:
        raise UnsafeRequest("В запросе указан неоднозначный покупатель.")
    return matches[0]
