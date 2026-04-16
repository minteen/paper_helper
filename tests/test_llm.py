import json
import unittest

from core.llm import LLMClient, LLMResponse


class FakeTransport:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    def chat_completion(self, payload):
        self.calls.append(payload)
        result = self._outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class TestLLMClient(unittest.TestCase):
    def test_chat_retries_then_success(self) -> None:
        transport = FakeTransport([RuntimeError("temporary"), "ok"])
        client = LLMClient(transport=transport, max_retries=2, retry_delay=0.0)

        response = client.chat("hello")
        self.assertEqual(response.content, "ok")
        self.assertEqual(len(transport.calls), 2)

    def test_chat_passes_parameters(self) -> None:
        transport = FakeTransport(["ok"])
        client = LLMClient(transport=transport)

        client.chat("hello", temperature=0.2, max_tokens=256)
        payload = transport.calls[0]
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["messages"][0]["content"], "hello")

    def test_structured_output_parses_json(self) -> None:
        transport = FakeTransport(['{"topic": "RAG"}'])
        client = LLMClient(transport=transport)

        output = client.chat_json("return json")
        self.assertEqual(output["topic"], "RAG")

    def test_structured_output_raises_for_invalid_json(self) -> None:
        transport = FakeTransport(["not-json"])
        client = LLMClient(transport=transport)

        with self.assertRaises(json.JSONDecodeError):
            client.chat_json("return json")


class TestLLMResponse(unittest.TestCase):
    def test_response_keeps_raw_payload(self) -> None:
        response = LLMResponse(content="x", raw={"id": "abc"})
        self.assertEqual(response.raw["id"], "abc")


if __name__ == "__main__":
    unittest.main()
