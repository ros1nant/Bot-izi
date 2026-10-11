from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from easybot.llm import ModelInput


def prediction(intent="invoice", *, counterparty=None, amount=None, number=None, addressed=True, certain=True, scope="client_business"):
    return {
        "intent": intent,
        "addressed_to_bot": addressed,
        "certain": certain,
        "scope": scope,
        "counterparty_quote": counterparty,
        "amount_quote": amount,
        "number_quote": number,
    }


class FakeModel:
    def __init__(self, result=None):
        self.result = result if result is not None else prediction()
        self.calls: list[ModelInput] = []

    def classify(self, request):
        self.calls.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return deepcopy(self.result)


class OfflineTestCase(TestCase):
    def setUp(self):
        super().setUp()
        # Any accidental HTTP call fails the test instead of depending on the network.
        for name in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex"):
            guard = patch(name, side_effect=AssertionError("Network is forbidden in tests"))
            guard.start()
            self.addCleanup(guard.stop)
