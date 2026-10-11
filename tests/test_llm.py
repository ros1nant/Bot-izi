import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from easybot.llm import ModelError, ModelInput, OllamaClassifier, SCHEMA
from tests.helpers import OfflineTestCase, prediction


class AdapterTests(OfflineTestCase):
    def response(self, body):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = body
        return response

    def test_real_adapter_builds_structured_request_with_mocked_http(self):
        content = prediction("balance")
        response = self.response(json.dumps({"done": True, "message": {"content": json.dumps(content)}}).encode())
        with patch("easybot.llm.urlopen", return_value=response) as http:
            result = OllamaClassifier(model="qwen3:8b").classify(ModelInput("Изи, баланс"))
        self.assertEqual(result, content)
        request = http.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.method, "POST")
        self.assertEqual(payload["format"], SCHEMA)
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(http.call_args.kwargs["timeout"], 60)

    def test_http_errors_become_model_errors(self):
        for error in (URLError("connection refused"), TimeoutError(), OSError()):
            with self.subTest(error=error), patch("easybot.llm.urlopen", side_effect=error):
                with self.assertRaises(ModelError):
                    OllamaClassifier().classify(ModelInput("Изи, баланс"))

    def test_malformed_or_truncated_response_is_rejected(self):
        bodies = [b"not json", b"{}", b"null", b"[]", b'{"done":false}', b'{"done":true,"done_reason":"length"}', b'{"done":true,"message":{"content":"no JSON"}}']
        for body in bodies:
            with self.subTest(body=body), patch("easybot.llm.urlopen", return_value=self.response(body)):
                with self.assertRaises(ModelError):
                    OllamaClassifier().classify(ModelInput("Изи, баланс"))
