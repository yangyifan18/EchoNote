from unittest.mock import patch, MagicMock
from echonote.summarizer import (
    build_prompt,
    summarize,
    ClaudeProvider,
    OpenAIProvider,
)


def test_build_prompt_includes_transcript():
    transcript = "这是一段测试转写文本。"
    prompt = build_prompt(transcript)
    assert "这是一段测试转写文本。" in prompt
    assert "摘要" in prompt
    assert "详细内容" in prompt
    assert "Q&A" in prompt
    assert "行动项" in prompt


def test_build_prompt_includes_structural_instructions():
    prompt = build_prompt("test")
    assert "跟随演讲者的原始组织方式" in prompt
    assert "不要过度压缩" in prompt


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
