from unittest import mock

import pytest

from docmind.llm import OpenAICompatLLM


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._b, self.text = status, body or {}, str(body)

    def json(self):
        return self._b


def test_openai_compat_parses_response_and_sends_payload():
    body = {"choices": [{"message": {"content": "  hello [1]  "}}]}
    with mock.patch("requests.post", return_value=Resp(200, body)) as post:
        llm = OpenAICompatLLM("llama3.2", "http://localhost:11434/v1/", api_key="k")
        assert llm.generate("sys", "usr") == "hello [1]"
    url = post.call_args.args[0]
    assert url == "http://localhost:11434/v1/chat/completions"
    sent = post.call_args.kwargs
    assert sent["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["headers"]["Authorization"] == "Bearer k"


def test_http_error_raises_clear_message():
    with mock.patch("requests.post", return_value=Resp(401, {"error": "bad key"})):
        with pytest.raises(RuntimeError):
            OpenAICompatLLM("m", "http://x/v1").generate("s", "u")


def test_missing_model_or_url():
    with pytest.raises(RuntimeError):
        OpenAICompatLLM("", "http://x")
    with pytest.raises(RuntimeError):
        OpenAICompatLLM("m", "")
