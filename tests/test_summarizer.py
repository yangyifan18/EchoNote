from unittest.mock import MagicMock, patch

import pytest

from echonote.summarizer import (
    ClaudeProvider,
    OpenAIProvider,
    build_prompt,
    split_transcript,
    summarize,
    summarize_with_provider,
)


class FakeProvider:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"response-{len(self.prompts)}"


def test_build_prompt_includes_transcript():
    transcript = "这是一段测试转写文本。"
    prompt = build_prompt(transcript)
    assert "这是一段测试转写文本。" in prompt
    assert "摘要" in prompt
    assert "详细内容" in prompt
    assert "Q&A" in prompt
    assert "行动项" in prompt


def test_build_prompt_supports_meeting_template():
    prompt = build_prompt("测试会议内容", template_name="meeting")
    assert "测试会议内容" in prompt
    assert "决策" in prompt
    assert "风险与阻塞" in prompt


def test_build_prompt_supports_custom_template(tmp_path):
    template_path = tmp_path / "custom_prompt.md"
    template_path.write_text("请整理成简明摘要：\n{transcript}", encoding="utf-8")

    prompt = build_prompt("自定义内容", prompt_path=str(template_path))

    assert prompt == "请整理成简明摘要：\n自定义内容"


def test_build_prompt_requires_transcript_placeholder(tmp_path):
    template_path = tmp_path / "broken_prompt.md"
    template_path.write_text("没有占位符", encoding="utf-8")

    with pytest.raises(ValueError, match="\\{transcript\\}"):
        build_prompt("内容", prompt_path=str(template_path))


def test_split_transcript_breaks_long_text_into_chunks():
    transcript = "第一行内容\n第二行内容\n第三行内容"

    chunks = split_transcript(transcript, max_chars_per_chunk=8)

    assert len(chunks) >= 3
    assert "".join(chunks).replace("\n", "") == transcript.replace("\n", "")


def test_summarize_with_provider_uses_chunking_for_long_transcripts():
    provider = FakeProvider()
    transcript = "A" * 25

    result = summarize_with_provider(
        transcript=transcript,
        provider=provider,
        max_chars_per_chunk=10,
    )

    assert result == "response-4"
    assert len(provider.prompts) == 4
    assert "当前片段" in provider.prompts[0]
    assert "分段笔记" in provider.prompts[-1]


@patch("echonote.summarizer.anthropic")
def test_claude_provider_calls_api(mock_anthropic):
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="## 摘要\n测试总结")]
    mock_client.messages.create.return_value = mock_response

    provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-6")
    result = provider.generate("测试 prompt")

    assert result == "## 摘要\n测试总结"
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["messages"][0]["content"] == "测试 prompt"


@patch("echonote.summarizer.openai")
def test_openai_provider_calls_api(mock_openai):
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="## 摘要\n测试总结"))]
    mock_client.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
    result = provider.generate("测试 prompt")

    assert result == "## 摘要\n测试总结"
    mock_client.chat.completions.create.assert_called_once()


@patch("echonote.summarizer.anthropic")
def test_summarize_end_to_end(mock_anthropic):
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="---\nspeaker: 张教授\n---\n\n## 摘要\n测试")]
    mock_client.messages.create.return_value = mock_response

    result = summarize(
        transcript="完整的转写文本",
        provider_name="claude",
        model="claude-sonnet-4-6",
        api_key="test-key",
    )

    assert "张教授" in result
    assert "摘要" in result
