"""LLM summarization — provider abstraction and prompt construction."""

from abc import ABC, abstractmethod

import anthropic
import openai


SUMMARIZE_PROMPT = """你是一个专业的学术会议记录整理助手。请根据以下会议转写文本，生成结构化的会议笔记。

## 要求

### 元数据
请从转写内容中提取以下信息（如无法确定，标记为"未知"）：
- 演讲者姓名和机构
- 报告主题
- 相关标签（3-5个，用于知识库分类）

### 摘要
用2-3句话概括本次报告的核心内容和主要贡献。

### 详细内容
这是最重要的部分。请遵循以下原则：
1. **跟随演讲者的原始组织方式** — 如果演讲者按项目/论文时间线讲解，就按项目分块；如果按专题讲解，就按专题分块
2. **不要过度压缩** — 保留具体的数字、方法名、工具名、论文名、实验结果等细节
3. **标注论文外信息** — 用「📌 论文外」标记演讲者分享的、在已发表论文中未涉及的实践经验
4. **保持足够上下文** — 使后续读者（或AI助手）能基于这份笔记进行深入问答

每个分块用 ### 标题标记，内容用 bullet points 组织。

### Q&A
如果转写中包含提问环节，按 Q1/Q2/... 格式整理每个问答对。如果没有明显的提问环节，省略此部分。

### 行动项
提取可执行的TODO事项（如需要调研的工具、需要联系的人、需要阅读的论文等），用 `- [ ]` 格式。如果没有明确的行动项，省略此部分。

## 输出格式

请直接输出 Markdown 格式。以 YAML frontmatter 开头：

---
speaker: 姓名 (机构)
topic: 主题
tags: [tag1, tag2, ...]
---

## 摘要
...

## 详细内容
### 分块标题1
...

## Q&A
### Q1: 问题
...

## 行动项
- [ ] ...

---

## 转写文本

{transcript}"""


def build_prompt(transcript: str) -> str:
    """Construct the summarization prompt with the transcript inserted."""
    return SUMMARIZE_PROMPT.replace("{transcript}", transcript)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to LLM and return the response text."""


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def _make_provider(provider_name: str, model: str, api_key: str) -> LLMProvider:
    if provider_name == "claude":
        return ClaudeProvider(api_key=api_key, model=model)
    elif provider_name == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")


def summarize(
    transcript: str,
    provider_name: str,
    model: str,
    api_key: str,
) -> str:
    """Summarize a transcript using the configured LLM provider."""
    provider = _make_provider(provider_name, model, api_key)
    prompt = build_prompt(transcript)
    return provider.generate(prompt)
