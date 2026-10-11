import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data.json"


def money(value: Decimal) -> str:
    precision = 0 if value == value.to_integral() else 2
    return f"{value:,.{precision}f}".replace(",", " ").replace(".", ",") + " ₸"


@dataclass(frozen=True)
class Invoice:
    number: int
    counterparty: str
    amount: Decimal

    def to_dict(self) -> dict:
        return {"number": self.number, "counterparty": self.counterparty, "amount": str(self.amount)}


class Ledger:
    """One client's data. Mutations are serialized by AccountantBot's lock."""

    def __init__(self, path: str | Path = DEFAULT_DATA):
        with open(path, encoding="utf-8") as source:
            data = json.load(source, parse_float=Decimal)
        self.counterparties = tuple(data["counterparties"])
        self.invoices = {
            row["number"]: Invoice(row["number"], row["counterparty"], Decimal(row["amount"]))
            for row in data["invoices"]
        }
        self.balance = Decimal(data["balance"])
        self.taxes = {name: Decimal(value) for name, value in data["taxes_due"].items()}

    def issue(self, counterparty: str, amount: Decimal) -> Invoice:
        number = max(self.invoices, default=0) + 1
        invoice = Invoice(number, counterparty, amount)
        self.invoices[number] = invoice
        return invoice
